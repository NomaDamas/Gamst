"""Enrolled-voice verification: was this stretch of speech said by the enrolled person?

Math in this module (no model): an enrollment centroid (mean of L2-normalised embeddings), cosine
scores of speech segments against it, a threshold with hysteresis across consecutive segments, an
energy gate so silence is never scored, and utterance re-checks (the audio of one final transcript).

Embedding backends (real models, local files only; :data:`BACKENDS`):

- ``wespeaker``: the WeSpeaker toolkit, ``wespeaker.load_model(local_dir)``. Default checkpoint is
  the w2v-BERT 2.0 LoRA-MFA model trained on VoxCeleb only (weights follow the VoxCeleb licence,
  CC BY 4.0, per WeSpeaker); a VoxBlink2-trained checkpoint is opt-in (``allow_noncommercial``).
- ``sherpa-onnx``: ``sherpa_onnx.SpeakerEmbeddingExtractor`` with a CAM++ ONNX export, the light
  CPU option.

A ``voice`` reference (``Pipeline.add_reference(name, 'voice', value, consent=...)``) is either the
embeddings ``gamst recipe enroll`` saved (``{"model": id, "embeddings": [[...], ...]}``; only
embeddings, never audio) or samples to embed at bind time (``{"samples": [array, ...],
"sample_rate": 16000}``). Embeddings of another model are rejected: cosine scores of two models are
not comparable.

Two kinds of output, one capability (``audio.speaker_match``):

- ``kind="segment"``: every ``hop_ms`` the last ``segment_ms`` of audio (if above the energy gate)
  gets a score; ``match`` follows the hysteresis state, so a flicker below the threshold inside one
  person's speech does not split it. Segment spans also drive the SAM-Audio span prompt
  (:mod:`gamst.providers.target_speech`).
- ``kind="utterance"``: for each final transcript (``speech.transcript_update`` or
  ``speech.transcript_words`` value) the utterance's own audio (its first to last word when word times are given)
  is scored and the transcript is passed through with a ``speaker`` object, so :class:`gamst.providers.term_events.TermEvents` downstream
  counts only what the enrolled person said.

Overlapping voices (``sources``): one window of audio where two people talk at once embeds as a mixture
of both. With ``sources`` naming the derived input of a
:class:`gamst.providers.target_speech.SpeakerSources` module, each word is also scored on every separated
source that is active at that word (its level over the word is above ``min_dbfs``; ``source_gate_db``
optionally also drops sources that much quieter than the loudest, which drops a quiet person under a loud
voice: off by default); the word's ``speaker_score`` is the best of those, so the
enrolled voice decides as before, on the separated sound. ``mixture_score`` keeps the score on the room
audio and ``sources_active`` how many sources spoke at the word (2 or more: overlap). A word with no
active source keeps the mixture score (``scored_on: "mixture"``). Utterances wait (``pending_since_ms``)
until the separated blocks that cover their words have arrived.

Speaker turns (``turns``): ``audio.speaker_turns`` records of the same reference (for example hosted
identification, :mod:`gamst.providers.pyannoteai`) add evidence to each word, not a decision:
``turn_match`` (a turn matched to the enrolled voice overlaps the word, or the turns cover the word and
none does), ``turn_speakers`` (how many turns overlap it) and ``turn_confidence``.

L1 policy (``l1``, only for ``provisional`` words, i.e. when a verification lane settles them later; owner decision
26). ``{"policy": "threshold"}`` (the old behaviour): a word counts at once when its score reaches ``threshold``, else it
is another voice's (shown, not counted) until verification gives it to the person. ``{"policy": "instant",
"clear_other": 0.2, "clear_margin": null}``: every heard word counts at once, also without a score, and is excluded
only when its voice is *clearly* someone else's: its score is below ``clear_other`` (a low floor, far under
``threshold``), or (with speaker decision v2's bank of negatives) a voice known not to be the person is closer than
the person by ``clear_margin``. Verification revokes the rest later, visibly. Measured on the owner's four clips
(run m1, CAM++ against the same-clip reference): the person's words scored 0.26-0.82 at L1, other voices 0.36-0.79,
so a threshold cannot separate them at L1 and 0.2 drops none of the person's words.

Scores are uncalibrated cosine similarities. The threshold must be set on the owner's own audio
(``gamst recipe run --truth``); the research notes report EER rising several-fold on 1-3 s segments.
"""
from __future__ import annotations

import hashlib
import inspect
import math

from ..capabilities import CapabilitySpec, OutputField
from .audio_ring import AudioRing, mono_float, rms_dbfs

SAMPLE_RATE = 16000
L1 = dict(policy='threshold', clear_other=.2, clear_margin=None)
L1_POLICIES = ('threshold', 'instant')


# ---------------------------------------------------------------- math

def l2_normalize(vector):
    norm = math.sqrt(sum(float(v) * float(v) for v in vector))
    if not norm or not math.isfinite(norm):
        raise ValueError('embedding has zero or non-finite norm')
    return [float(v) / norm for v in vector]


def cosine(a, b):
    if len(a) != len(b):
        raise ValueError(f'embedding sizes differ: {len(a)} vs {len(b)}')
    a, b = l2_normalize(a), l2_normalize(b)
    return sum(x * y for x, y in zip(a, b))


def centroid(embeddings):
    """Normalised mean of normalised embeddings (each enrollment clip weighs the same)."""
    embeddings = [l2_normalize(item) for item in embeddings]
    if not embeddings:
        raise ValueError('no enrollment embeddings')
    size = len(embeddings[0])
    if any(len(item) != size for item in embeddings):
        raise ValueError('enrollment embeddings have different sizes')
    return l2_normalize([sum(item[i] for item in embeddings) / len(embeddings) for i in range(size)])


class Hysteresis:
    """``enter`` <= score turns the state on; it turns off below ``exit`` (``exit <= enter``)."""

    def __init__(self, enter, exit):
        if not exit <= enter:
            raise ValueError('require exit <= enter')
        self.enter, self.exit = float(enter), float(exit)
        self.active = False

    def update(self, score):
        if score is None:
            return self.active
        if self.active and score < self.exit:
            self.active = False
        elif not self.active and score >= self.enter:
            self.active = True
        return self.active


# ---------------------------------------------------------------- embedding backends

class SherpaOnnxEmbedder:
    """CAM++ (3D-Speaker or WeSpeaker export) through sherpa-onnx's speaker embedding extractor."""

    def __init__(self, extractor, model_id):
        self.extractor, self.model_id = extractor, model_id
        self.dim = int(extractor.dim)

    @classmethod
    def from_local(cls, path, *, sha256=None, num_threads=1, provider='cpu'):
        import sherpa_onnx
        digest = _verify(path, sha256)
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(path), num_threads=int(num_threads),
                                                              provider=provider)
        if not config.validate():
            raise ValueError(f'sherpa-onnx rejected the speaker model config for {path}')
        return cls(sherpa_onnx.SpeakerEmbeddingExtractor(config), f'sherpa-onnx:{digest[:16]}')

    def embed(self, samples, sample_rate=SAMPLE_RATE):
        stream = self.extractor.create_stream()
        stream.accept_waveform(sample_rate=int(sample_rate), waveform=mono_float(samples))
        stream.input_finished()
        if not self.extractor.is_ready(stream):
            raise ValueError('segment too short for the speaker model')
        return [float(v) for v in self.extractor.compute(stream)]


class WeSpeakerEmbedder:
    """A WeSpeaker model directory (``config.yaml`` + ``avg_model.pt``) through ``wespeaker.load_model``.

    ``wespeaker.load_model`` treats a path that is not a hub key as a local directory, so nothing is
    downloaded here. The w2v-BERT 2.0 models also load the ``facebook/w2v-bert-2.0`` backbone named in
    ``config.yaml`` (``dataset_args.w2vbert_args.model_name``); it must point at a local directory
    (setup writes it), otherwise transformers would fetch it.
    """

    def __init__(self, model, model_id, pcm_scale):
        self.model, self.model_id, self.pcm_scale = model, model_id, float(pcm_scale)
        self.dim = None

    @classmethod
    def from_local(cls, directory, *, sha256=None, device='cpu'):
        from pathlib import Path
        import yaml
        import wespeaker
        directory = Path(directory)
        checkpoint, config_path = directory / 'avg_model.pt', directory / 'config.yaml'
        if not checkpoint.is_file() or not config_path.is_file():
            raise ValueError(f'{directory} needs avg_model.pt and config.yaml (a WeSpeaker model directory)')
        config = yaml.safe_load(config_path.read_text())
        dataset = config.get('dataset_args') or {}
        if dataset.get('frontend', 'fbank') == 'w2vbert':
            backbone = (dataset.get('w2vbert_args') or {}).get('model_name')
            if not backbone or not Path(backbone).is_dir():
                raise ValueError(f'{config_path}: w2vbert_args.model_name must be a local facebook/w2v-bert-2.0 '
                                 f'directory, not {backbone!r} (it would be downloaded)')
        digest = _verify(checkpoint, sha256)
        model = wespeaker.load_model(str(directory))
        model.set_device(device)
        # extract_embedding(path) reads wavs with torchaudio.load(normalize=wavform_norm): int16-scale
        # values unless wavform_norm is set. Feeding PCM at the same scale keeps both paths identical.
        scale = 1. if getattr(model, 'wavform_norm', False) else 32768.
        return cls(model, f'wespeaker:{digest[:16]}', scale)

    def embed(self, samples, sample_rate=SAMPLE_RATE):
        import torch
        pcm = torch.from_numpy(mono_float(samples) * self.pcm_scale).unsqueeze(0)
        embedding = self.model.extract_embedding_from_pcm(pcm, int(sample_rate))
        if embedding is None:
            raise ValueError('segment too short or silent for the speaker model')
        values = [float(v) for v in embedding.detach().cpu().reshape(-1).tolist()]
        self.dim = len(values)
        return values


BACKENDS = {'sherpa-onnx': SherpaOnnxEmbedder, 'wespeaker': WeSpeakerEmbedder}


def _verify(path, expected):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            digest.update(block)
    value = digest.hexdigest()
    if expected is not None and value != expected:
        raise ValueError(f'{path} does not match its pinned sha256 {expected}')
    return value


def load_embedder(backend, path, **options):
    if backend not in BACKENDS:
        raise ValueError(f'speaker backend must be one of {sorted(BACKENDS)}')
    return BACKENDS[backend].from_local(path, **options)


async def embed(embedder, *args):
    """An in-process embedder, or one in its own worker environment (``RemoteEmbedder``)."""
    value = embedder.embed(*args)
    return await value if inspect.isawaitable(value) else value


async def enroll(embedder, clips, sample_rate=SAMPLE_RATE, *, min_dbfs=-45.):
    """Enrollment embeddings of the owner's clips (silent clips are skipped). Audio is not kept."""
    embeddings = []
    for clip in clips:
        clip = mono_float(clip)
        if rms_dbfs(clip) < min_dbfs:
            continue
        embeddings.append(await embed(embedder, clip, sample_rate))
    if not embeddings:
        raise ValueError('no enrollment clip had speech above the energy gate')
    return dict(kind='voice', model=embedder.model_id, dim=len(embeddings[0]), embeddings=embeddings)


# ---------------------------------------------------------------- pipeline module

class SpeakerVerify:
    """Pipeline module: audio frames (16 kHz mono) and final transcripts in, speaker matches out."""
    owns_process = True    # the embedding model runs in its worker

    def __init__(self, embedder, *, reference='voice', threshold=.5, hysteresis=.05, segment_ms=1500,
                 hop_ms=500, min_dbfs=-45., min_utterance_ms=400, ring_seconds=30.,
                 segments=True, capability_id='audio.speaker_match', context_ms=0, words=False, sources=None,
                 source_gate_db=None, turns=False, provisional=False, audio_memory=None, relative=None, bank=None,
                 l1=None):
        # speaker decision v2 (gamst/providers/voice_bank.py): each word against the person's references AND the
        # voices known not to be the person, as the bank holds them when the word is committed
        self.relative = None if relative is None else dict(dict(floor=float(threshold), margin=0.), **relative)
        if self.relative is not None and set(self.relative) - {'floor', 'margin'}:
            raise ValueError('relative is {floor, margin}')
        self.bank_settings, self.bank = bank, None
        self.l1 = dict(L1, **(l1 or {}))
        if set(self.l1) - set(L1) or self.l1['policy'] not in L1_POLICIES:
            raise ValueError(f"l1 is {{policy: {' | '.join(L1_POLICIES)}, clear_other, clear_margin}}")
        if self.l1['policy'] == 'instant' and not provisional:
            raise ValueError('the instant L1 policy counts words a later lane settles: it needs provisional words')
        self.provisional = bool(provisional)    # a later module settles each word: give it an ID and say so
        self.audio_memory, self._memory = audio_memory, None   # an input whose kept audio covers what the ring lost
        if self.provisional and not words:
            raise ValueError('provisional words need word scores: set words')
        self.words = bool(words)     # also score each word, with the sound around it: people share a phrase
        if self.words and not context_ms:
            raise ValueError('word scores need context_ms')
        if (sources is not None or turns) and not self.words:
            raise ValueError('separated sources and speaker turns are read per word: set words')
        self.turns = bool(turns)
        self.sources = sources
        self.source_gate_db = None if source_gate_db is None else float(source_gate_db)
        if type(context_ms) is not int or not 0 <= context_ms <= 5000:
            raise ValueError('context_ms must be integer milliseconds 0..5000')
        self.context_ms = context_ms     # an utterance shorter than this is scored with the sound around it
        if not 200 <= segment_ms <= 10000 or not 100 <= hop_ms <= segment_ms:
            raise ValueError('require 200 <= segment_ms <= 10000 and 100 <= hop_ms <= segment_ms')
        self.embedder, self.reference_name = embedder, reference
        self.threshold, self.hysteresis = float(threshold), float(hysteresis)
        self.segment_ms, self.hop_ms, self.min_dbfs = int(segment_ms), int(hop_ms), float(min_dbfs)
        self.min_utterance_ms = int(min_utterance_ms)
        self.segments = bool(segments)
        self.target = None
        self.ring = AudioRing(SAMPLE_RATE, ring_seconds)
        self.spec = CapabilitySpec(
            capability_id, capability_id, 'audio', 'gamst.speaker_verify',
            f'1.{embedder.model_id}'.replace(':', '.'),
            (OutputField('kind', 'string'), OutputField('reference', 'string'), OutputField('start_ms', 'integer'),
             OutputField('end_ms', 'integer'), OutputField('score', 'number', required=False),
             OutputField('match', 'boolean'), OutputField('state', 'string'),
             OutputField('utterance_id', 'string', required=False), OutputField('status', 'string', required=False),
             OutputField('text', 'string', required=False), OutputField('words', 'array', required=False),
             OutputField('language', 'string', required=False), OutputField('speaker', 'object', required=False),
             OutputField('model', 'object')),
            'window', window_ms=self.segment_ms, stateful=True, reset_on_gap=True, freshness_ms=5000,
            interpretation='estimate', devices=('cpu',), license='Apache-2.0 adapter; model weights per backend')
        self._samples = None
        self.reset()

    def bind(self, pipeline):
        reference = pipeline.references.get(self.reference_name)
        if reference is None or reference['kind'] != 'voice':
            raise ValueError(f'speaker_verify needs a voice reference named {self.reference_name!r}')
        self.set_reference(reference['value'])
        if self.relative is not None:
            from .voice_bank import shared
            if 'embeddings' not in (reference['value'] or {}):
                raise ValueError('a relative speaker decision needs an enrolled voice (embeddings)')
            self.bank = shared(pipeline, self.reference_name, reference['value']['embeddings'], self.bank_settings)
        if self.audio_memory is not None:
            self._memory = pipeline.keep_audio(self.audio_memory)

    def _covers(self, first, last):
        return self.ring.covers(first, last) or self._memory is not None and self._memory.covers(first, last)

    def _span(self, first, last):
        """Samples of [first, last): the ring's, else the input's kept audio (words a catch-up worker gave late)."""
        if self.ring.covers(first, last) or self._memory is None or not self._memory.covers(first, last):
            return self.ring.span(first, last)
        return self._memory.span(first, last, SAMPLE_RATE)

    def _settled_later(self, result):
        """Each word gets an ID and ``state: provisional``: a verification module gives its verdict later."""
        if not self.provisional or not isinstance(result.get('words'), list):
            return result
        words = [dict(word, word_id=f"{result['utterance_id']}#{index}", state='provisional')
                 if isinstance(word, dict) and type(word.get('start_ms')) is int else word
                 for index, word in enumerate(result['words'])]
        return dict(result, words=words)

    def set_reference(self, value):
        if not isinstance(value, dict):
            raise ValueError('a voice reference is {"model", "embeddings"} or {"samples", "sample_rate"}')
        if 'embeddings' in value:
            if value.get('model') != self.embedder.model_id:
                raise ValueError(f"voice reference was enrolled with {value.get('model')!r}; this module runs "
                                 f'{self.embedder.model_id!r} (re-enroll with the same model)')
            embeddings = value['embeddings']
            self.target, self._samples = centroid(embeddings), None
        elif value.get('samples'):
            self.target, self._samples = None, value   # embedded once, at the first frame, then dropped
        else:
            raise ValueError('a voice reference is {"model", "embeddings"} or {"samples", "sample_rate"}')

    def reset(self):
        self.ring.reset()
        self.state = Hysteresis(self.threshold, self.threshold - self.hysteresis)
        self._next_segment_ms = None
        self._checked = {}
        self._blocks, self._pending, self._sources_until = [], [], None
        self._turns, self._turns_until = [], None

    async def score(self, samples):
        """Cosine score of samples against the enrolled centroid, None below the energy gate."""
        return (await self._scored(samples))[0]

    async def _scored(self, samples):
        """(score, embedding) of :meth:`score`; (None, None) below the energy gate."""
        if self._samples is not None:
            try:
                enrolled = await enroll(self.embedder, self._samples['samples'],
                                        int(self._samples.get('sample_rate', SAMPLE_RATE)), min_dbfs=self.min_dbfs)
            except ValueError as error:
                raise RuntimeError(f'voice reference: {error}') from error
            self.target, self._samples = centroid(enrolled['embeddings']), None
        if self.target is None:
            raise RuntimeError('no voice reference bound')
        if rms_dbfs(samples) < self.min_dbfs:
            return None, None
        embedding = await embed(self.embedder, samples, SAMPLE_RATE)
        if len(embedding) != len(self.target):
            raise RuntimeError('the voice reference has another embedding size than this model')
        return round(cosine(embedding, self.target), 5), embedding

    async def close(self):
        close = getattr(self.embedder, 'close', None)
        value = close() if callable(close) else None
        if inspect.isawaitable(value):
            await value

    def _model(self):
        return dict(id=self.embedder.model_id, threshold=self.threshold, hysteresis=self.hysteresis,
                    **({'l1': dict(self.l1)} if self.l1['policy'] != 'threshold' else {}))

    @property
    def pending_since_ms(self):
        """Utterances that wait for separated sources (not complete yet)."""
        return min(item[0]['start_ms'] for item in self._pending) if self._pending else None

    async def finish(self):
        """End of input: score what still waits on whatever separated blocks arrived."""
        return await self._release(final=True)

    async def process(self, frame):
        from .target_speech import SourceBlock
        if isinstance(frame.data, SourceBlock):
            if self.sources is None:
                return None
            self._blocks = self._blocks[-63:] + [frame.data]
            self._sources_until = max(frame.data.end_ms, self._sources_until or frame.data.end_ms)
            return await self._release()
        if isinstance(frame.data, dict) and frame.data.get('kind') == 'turns':
            if not self.turns or frame.data.get('reference') != self.reference_name:
                return None
            self._turns = self._turns[-1023:] + list(frame.data.get('turns') or ())
            self._turns_until = max(frame.data['end_ms'], self._turns_until or frame.data['end_ms'])
            return await self._release()
        if isinstance(frame.data, dict):
            result = await self._utterance(frame)
            if not self._waits() or result is None:
                return result
            waiting = bool(self._pending) and self._pending[-1][0] is result
            released = await self._release() or []
            return released if waiting else [result, *released]
        if frame.modality != 'audio' or frame.data is None:
            return None
        if frame.sample_rate != SAMPLE_RATE:
            raise ValueError('speaker_verify needs 16 kHz mono audio')
        end_ms = int(frame.timestamp_ms)
        if self.ring.append(frame.data, end_ms, SAMPLE_RATE):   # capture gap: no history across it
            self.state.active = False
            self._next_segment_ms = None
        if not self.segments:
            return None
        end_ms = self.ring.end_ms
        if self._next_segment_ms is None:
            self._next_segment_ms = self.ring.start_ms + self.segment_ms
        outputs = []
        while self._next_segment_ms <= end_ms:
            segment_end = self._next_segment_ms
            self._next_segment_ms += self.hop_ms
            start = segment_end - self.segment_ms
            if not self.ring.covers(start, segment_end):
                continue
            try:
                score = await self.score(self.ring.span(start, segment_end))
            except ValueError:
                score = None   # too short for the model: no evidence either way
            active = self.state.update(score)
            value = dict(kind='segment', reference=self.reference_name, start_ms=int(start),
                         end_ms=int(segment_end), match=bool(active and score is not None),
                         state='silent' if score is None else ('match' if active else 'other'),
                         model=self._model())
            if score is not None:
                value['score'] = score
            outputs.append(self.spec.validate(value))
        return outputs or None

    def _around(self, first, last):
        """At least ``context_ms`` of sound with [first, last] in its middle, as far as it is held."""
        if last - first >= self.context_ms or self.ring.start_ms is None:
            return first, last
        extra = self.context_ms - (last - first)
        held = self.ring
        if self._memory is not None and self._memory.start_ms is not None and not self.ring.covers(first, last):
            held = self._memory       # a late word: around it in the kept audio
        return max(first - extra // 2, int(held.start_ms) + 1), min(last + extra - extra // 2, int(held.end_ms))

    async def _word(self, item):
        if not isinstance(item, dict) or not isinstance(item.get('start_ms'), int) or not isinstance(item.get('end_ms'), int):
            return item
        first, last = self._around(item['start_ms'], item['end_ms'])
        try:
            score, vector = await self._scored(self._span(first, last)) if self._covers(first, last) else (None, None)
        except ValueError:
            score = vector = None
        if self.bank is not None and vector is not None and not self._waits():
            match, pos, neg = self.bank.decide(vector, **self.relative)
            if self.l1['policy'] == 'instant':     # excluded only when clearly another voice
                margin = self.l1['clear_margin']
                match = pos >= self.l1['clear_other'] and not (
                    margin is not None and neg is not None and neg - pos >= margin)
            return dict(item, match=match, speaker_score=pos, decision='relative', **self._l1_fields(),
                        **({} if neg is None else {'negative_score': neg}))
        if self._waits():   # decided once the separated sources / turns of this word are here
            return dict(item, **({} if score is None else {'mixture_score': score})), (first, last)
        return dict(item, match=self._l1_match(score), **self._l1_fields(),
                    **({} if score is None else {'speaker_score': score}))

    def _waits(self):
        return self.sources is not None or self.turns

    def _l1_match(self, score):
        """Whether a word counts at once (see the module doc, L1 policy)."""
        if self.l1['policy'] == 'instant':
            return score is None or score >= self.l1['clear_other']
        return score is not None and score >= self.threshold

    def _l1_fields(self):
        return {'l1': 'instant'} if self.l1['policy'] == 'instant' else {}

    def _on_turns(self, item):
        if not self.turns or self._turns_until is None:
            return item
        start, end = item['start_ms'], max(item['end_ms'], item['start_ms'] + 1)
        over = [turn for turn in self._turns if turn['start_ms'] < end and turn['end_ms'] > start]
        matched = [turn for turn in over if turn.get('match')]
        extra = dict(turn_match=bool(matched), turn_speakers=len({turn['speaker'] for turn in over}))
        confidence = [turn['confidence'] for turn in matched if isinstance(turn.get('confidence'), (int, float))]
        if confidence:
            extra['turn_confidence'] = max(confidence)
        return dict(item, **extra)

    def _block(self, first, last):
        """The first block whose window covers [first, last]: the same one whatever arrived since, so the
        verdict does not depend on how far separation ran ahead of the transcript."""
        return next((block for block in sorted(self._blocks, key=lambda block: block.end_ms)
                     if block.covers(first, last)), None)

    async def _on_sources(self, item, first, last):
        """One word decided on the separated sources active at it (else on the mixture)."""
        block, scores, active = self._block(first, last), [], 0
        if block is not None:
            levels = [rms_dbfs(block.span(index, item['start_ms'], max(item['end_ms'], item['start_ms'] + 1)))
                      for index in range(block.sources.shape[0])]
            loudest = max(levels, default=float('-inf'))
            for index, level in enumerate(levels):
                if level < self.min_dbfs or self.source_gate_db is not None and level < loudest - self.source_gate_db:
                    continue
                active += 1
                try:
                    score = await self.score(block.span(index, first, last))
                except ValueError:
                    score = None
                if score is not None:
                    scores.append(score)
        if scores:
            score, scored_on = max(scores), 'sources'
        else:
            score, scored_on = item.get('mixture_score'), 'mixture'
        return dict(item, match=self._l1_match(score), sources_active=active, **self._l1_fields(),
                    scored_on=scored_on, **({} if score is None else {'speaker_score': score}))

    async def _decide(self, item, first, last):
        if self.sources is not None:
            item = await self._on_sources(item, first, last)
        else:
            score = item.get('mixture_score')
            item = dict({key: value for key, value in item.items() if key != 'mixture_score'},
                        match=self._l1_match(score), **self._l1_fields(),
                        **({} if score is None else {'speaker_score': score}))
        return self._on_turns(item)

    async def _release(self, final=False):
        """Utterances whose words the separated blocks now cover (all of them at the end)."""
        outputs = []
        while self._pending:
            result, spans = self._pending[0]
            needed = max((last for _, (_, last) in spans), default=result['end_ms'])
            waiting = [until for bound, until in ((self.sources is not None, self._sources_until),
                                                  (self.turns, self._turns_until)) if bound]
            if not final and any(until is None or until < needed for until in waiting):
                break
            self._pending.pop(0)
            words = list(result['words'])
            for index, (first, last) in spans:
                words[index] = await self._decide(words[index], first, last)
            outputs.append(self.spec.validate(self._settled_later(dict(result, words=words))))
        oldest = min((first for _, spans in self._pending for _, (first, _) in spans), default=None)
        if oldest is not None:
            self._blocks = [block for block in self._blocks if block.end_ms > oldest]
            self._turns = [turn for turn in self._turns if turn['end_ms'] > oldest]
        return outputs or None

    async def _utterance(self, frame):
        value = frame.data
        if value.get('status') != 'final' or not value.get('text'):
            return None
        start, end = value.get('start_ms'), value.get('end_ms')
        if not isinstance(start, int) or not isinstance(end, int) or end <= start:
            return None
        key = (value.get('utterance_id'), value.get('revision'))
        if key in self._checked:
            return None
        self._checked[key] = True
        while len(self._checked) > 4096:
            self._checked.pop(next(iter(self._checked)))
        score, first, last = None, start, end
        words = [item for item in value.get('words') or () if isinstance(item, dict)
                 and isinstance(item.get('start_ms'), int) and isinstance(item.get('end_ms'), int)]
        if words and max(item['end_ms'] for item in words) - min(item['start_ms'] for item in words) \
                >= self.min_utterance_ms:
            first, last = min(item['start_ms'] for item in words), max(item['end_ms'] for item in words)
        first, last = self._around(first, last)
        # Up to 500 ms of the tail may not have reached this module yet; score what is held.
        if last - first >= self.min_utterance_ms and (self.ring.covers(
                first, min(last, (self.ring.end_ms or 0) + 500) - 500) or not self.ring.covers(first, last) and self._covers(first, last)):
            try:
                score = await self.score(self._span(first, last))
            except ValueError:
                score = None
        match = score is not None and score >= self.threshold
        speaker = dict(reference=self.reference_name, match=match, model=self.embedder.model_id)
        if score is not None:
            speaker['score'] = score
        result = dict(kind='utterance', reference=self.reference_name, start_ms=start, end_ms=end, match=match,
                      state='match' if match else ('unscored' if score is None else 'other'),
                      utterance_id=str(value.get('utterance_id') or f'window:{end}'), status='final',
                      text=str(value['text']), speaker=speaker, model=self._model())
        if score is not None:
            result['score'] = score
        if isinstance(value.get('words'), list):
            result['words'] = value['words'] if not self.words else [await self._word(item) for item in value['words']]
        if isinstance(value.get('language'), str):
            result['language'] = value['language']
        if self._waits() and isinstance(value.get('words'), list):
            spans = [(index, item[1]) for index, item in enumerate(result['words']) if isinstance(item, tuple)]
            result['words'] = [item[0] if isinstance(item, tuple) else item for item in result['words']]
            self._pending.append((result, spans))
            return result
        return self.spec.validate(self._settled_later(result))

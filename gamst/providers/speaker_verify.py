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
  ``speech.transcript_words`` value) the utterance's own audio is scored and the transcript is passed
  through with a ``speaker`` object, so :class:`gamst.providers.term_events.TermEvents` downstream
  counts only what the enrolled person said.

Scores are uncalibrated cosine similarities. The threshold must be set on the owner's own audio
(``gamst recipe run --truth``); the research notes report EER rising several-fold on 1-3 s segments.
"""
from __future__ import annotations

import hashlib
import math
import threading

from ..capabilities import CapabilitySpec, OutputField
from .audio_ring import AudioRing, mono_float, rms_dbfs

SAMPLE_RATE = 16000


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


def enroll(embedder, clips, sample_rate=SAMPLE_RATE, *, min_dbfs=-45.):
    """Enrollment embeddings of the owner's clips (silent clips are skipped). Audio is not kept."""
    embeddings = []
    for clip in clips:
        clip = mono_float(clip)
        if rms_dbfs(clip) < min_dbfs:
            continue
        embeddings.append(embedder.embed(clip, sample_rate))
    if not embeddings:
        raise ValueError('no enrollment clip had speech above the energy gate')
    return dict(kind='voice', model=embedder.model_id, dim=len(embeddings[0]), embeddings=embeddings)


# ---------------------------------------------------------------- pipeline module

class SpeakerVerify:
    """Pipeline module: audio frames (16 kHz mono) and final transcripts in, speaker matches out."""

    def __init__(self, embedder, *, reference='voice', threshold=.5, hysteresis=.05, segment_ms=1500,
                 hop_ms=500, min_dbfs=-45., min_utterance_ms=400, max_gap_ms=1000, ring_seconds=30.):
        if not 200 <= segment_ms <= 10000 or not 100 <= hop_ms <= segment_ms:
            raise ValueError('require 200 <= segment_ms <= 10000 and 100 <= hop_ms <= segment_ms')
        self.embedder, self.reference_name = embedder, reference
        self.threshold, self.hysteresis = float(threshold), float(hysteresis)
        self.segment_ms, self.hop_ms, self.min_dbfs = int(segment_ms), int(hop_ms), float(min_dbfs)
        self.min_utterance_ms, self.max_gap_ms = int(min_utterance_ms), int(max_gap_ms)
        self.target = None
        self.ring = AudioRing(SAMPLE_RATE, ring_seconds)
        self.spec = CapabilitySpec(
            'audio.speaker_match', 'audio.speaker_match', 'audio', 'gamst.speaker_verify',
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
        self._lock = threading.RLock()
        self.reset()

    def bind(self, pipeline):
        reference = pipeline.references.get(self.reference_name)
        if reference is None or reference['kind'] != 'voice':
            raise ValueError(f'speaker_verify needs a voice reference named {self.reference_name!r}')
        self.set_reference(reference['value'])

    def set_reference(self, value):
        if not isinstance(value, dict):
            raise ValueError('a voice reference is {"model", "embeddings"} or {"samples", "sample_rate"}')
        if 'embeddings' in value:
            if value.get('model') != self.embedder.model_id:
                raise ValueError(f"voice reference was enrolled with {value.get('model')!r}; this module runs "
                                 f'{self.embedder.model_id!r} (re-enroll with the same model)')
            embeddings = value['embeddings']
        else:
            embeddings = enroll(self.embedder, value.get('samples') or (),
                                int(value.get('sample_rate', SAMPLE_RATE)), min_dbfs=self.min_dbfs)['embeddings']
        self.target = centroid(embeddings)

    def reset(self):
        with self._lock:
            self.ring.reset()
            self.state = Hysteresis(self.threshold, self.threshold - self.hysteresis)
            self._next_segment_ms = None
            self._last_audio_ms = None
            self._checked = {}

    def score(self, samples):
        """Cosine score of samples against the enrolled centroid, None below the energy gate."""
        if self.target is None:
            raise RuntimeError('no voice reference bound')
        if rms_dbfs(samples) < self.min_dbfs:
            return None
        return round(cosine(self.embedder.embed(samples, SAMPLE_RATE), self.target), 5)

    def _model(self):
        return dict(id=self.embedder.model_id, threshold=self.threshold, hysteresis=self.hysteresis)

    def process(self, frame):
        with self._lock:
            if isinstance(frame.data, dict):
                return self._utterance(frame)
            if frame.modality != 'audio' or frame.data is None:
                return None
            if frame.sample_rate != SAMPLE_RATE:
                raise ValueError('speaker_verify needs 16 kHz mono audio')
            end_ms = int(frame.timestamp_ms)
            if self._last_audio_ms is not None and end_ms - self._last_audio_ms > self.max_gap_ms:
                self.state.active = False
                self._next_segment_ms = None
            self._last_audio_ms = end_ms
            self.ring.append(frame.data, end_ms, SAMPLE_RATE)
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
                    score = self.score(self.ring.span(start, segment_end))
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

    def _utterance(self, frame):
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
        score = None
        if end - start >= self.min_utterance_ms and self.ring.covers(start, end):
            try:
                score = self.score(self.ring.span(start, end))
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
            result['words'] = value['words']
        if isinstance(value.get('language'), str):
            result['language'] = value['language']
        return self.spec.validate(result)

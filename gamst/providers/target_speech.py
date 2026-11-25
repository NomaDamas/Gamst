"""One enrolled person's voice separated from the room, as a derived input (``voice:<reference>``).

The separated audio is a :class:`gamst.pipeline.Derived` input: it is routed to the modules that
consume it (ASR, prosody, speaker verification) with the original capture time, and never enters the
Store or any context. Only records do.

Two ways to condition on the person:

1. **Enrollment audio** (:class:`TargetSpeech`): a target-speaker-extraction model takes the mixture
   and a clip of the enrolled voice.
   - ``wesep``: WeSep BSRNN (``wesep_2024`` branch, pretrained English model ``bsrnn_ecapa_vox1``),
     Apache-2.0 code. ``wesep.load_model_local(dir)`` + ``extract_speech_from_pcm``. 16 kHz.
   - ``usef``: USEF-TFGridNet (ZBang/USEF-TSE). Best SI-SDRi in the research notes, but code and
     weights are **CC BY-NC 4.0**: opt-in only (``allow_noncommercial=True``). It is trained on
     WSJ0-2mix/WHAM!/WHAMR! at **8 kHz**, so the mixture is resampled to 8 kHz and the estimate back to
     16 kHz (:func:`gamst.providers.audio_ring.resample`); content above 4 kHz is lost.
   Both are offline (non-causal) models; :class:`TargetSpeech` runs them on blocks of ``block_ms`` with
   ``context_ms`` of preceding audio and keeps only the new block, so each output sample has left
   context and the latency is one block.

2. **Speaker-match spans** (:class:`SAMSpanSpeech`): no enrollment audio at separation time. The spans
   where :class:`gamst.providers.speaker_verify.SpeakerVerify` matched the enrolled voice become
   SAM-Audio span anchors (``+`` for the person, ``-`` for scored speech that is someone else) on
   complete 48 kHz windows, using GAMST's existing SAM-Audio loader
   (:class:`gamst.providers.sam_audio_native.SAMAudioNative`) and window contract
   (:func:`gamst.sam_audio_windows.prepare_window`). SAM-Audio's span prompt "can be inherently
   ambiguous when multiple sound events occur simultaneously" (SAM Audio paper).

The enrollment clip for (1) is the owner's own WAV file named in the ``voice`` reference
(``{"wav": path}``) or samples given in process (``{"samples": [...], "sample_rate": 16000}``); the
module keeps it in memory only. A person reference needs a consent record (``Pipeline.add_reference``).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from .audio_ring import AudioRing, mono_float, resample

SAMPLE_RATE = 16000


def _np():
    import numpy as np
    return np


def _verify(path, expected):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            digest.update(block)
    value = digest.hexdigest()
    if expected is not None and value != expected:
        raise ValueError(f'{path} does not match its pinned sha256 {expected}')
    return value


def read_wav(path):
    """Mono float32 samples and rate of a 16/32-bit PCM WAV."""
    import wave
    np = _np()
    with wave.open(str(path), 'rb') as source:
        rate, width, channels = source.getframerate(), source.getsampwidth(), source.getnchannels()
        raw = source.readframes(source.getnframes())
    if width not in (2, 4):
        raise ValueError('16- or 32-bit PCM WAV required')
    samples = np.frombuffer(raw, np.int16 if width == 2 else np.int32).astype(np.float32)
    samples /= 32768. if width == 2 else 2147483648.
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, rate


def enrollment_audio(value):
    """16 kHz enrollment samples from a voice reference value."""
    if isinstance(value, dict) and value.get('wav'):
        samples, rate = read_wav(value['wav'])
    elif isinstance(value, dict) and value.get('samples') is not None:
        np = _np()
        clips = value['samples']
        clips = clips if isinstance(clips, (list, tuple)) and clips and hasattr(clips[0], '__len__') else [clips]
        samples, rate = np.concatenate([mono_float(clip) for clip in clips]), int(value.get('sample_rate', SAMPLE_RATE))
    else:
        raise ValueError('target speech needs the enrolled voice clip: a voice reference with "wav" or "samples" '
                         '(saved embeddings alone cannot condition an extraction model)')
    return resample(samples, rate, SAMPLE_RATE) if rate != SAMPLE_RATE else samples


# ---------------------------------------------------------------- extraction backends

class WeSepExtractor:
    """WeSep (``wesep_2024`` branch) BSRNN target speaker extraction from a local model directory."""

    sample_rate = SAMPLE_RATE

    def __init__(self, model, model_id):
        self.model, self.model_id = model, model_id

    @classmethod
    def from_local(cls, directory, *, sha256=None, device='cpu'):
        import yaml
        import wesep
        directory = Path(directory)
        checkpoint, config = directory / 'avg_model.pt', directory / 'config.yaml'
        if not checkpoint.is_file() or not config.is_file():
            raise ValueError(f'{directory} needs avg_model.pt and config.yaml (a WeSep model directory)')
        model_args = ((yaml.safe_load(config.read_text()) or {}).get('model_args') or {}).get('tse_model') or {}
        if not model_args.get('joint_training'):
            raise ValueError(f'{config}: extract_speech_from_pcm returns nothing unless joint_training is true')
        digest = _verify(checkpoint, sha256)
        model = wesep.load_model_local(str(directory))
        model.set_device(device)
        model.set_vad(False)
        model.set_output_norm(False)   # keep the mixture's level; no peak normalisation
        return cls(model, f'wesep:{digest[:16]}')

    def extract(self, mixture, enrollment):
        import torch
        mix = torch.from_numpy(mono_float(mixture)).unsqueeze(0)
        enroll = torch.from_numpy(mono_float(enrollment)).unsqueeze(0)
        estimate = self.model.extract_speech_from_pcm(mix, SAMPLE_RATE, enroll, SAMPLE_RATE)
        if estimate is None:
            raise ValueError('WeSep returned no estimate')
        return _fit(estimate.detach().cpu().numpy().reshape(-1), len(mixture))


class USEFExtractor:
    """USEF-TFGridNet (CC BY-NC 4.0) from a pinned ZBang/USEF-TSE source checkout and checkpoint."""

    sample_rate = 8000
    SOURCE_REVISION = 'bdaeb3609735b3a9dbedd1b0625343826f2cb5fe'

    def __init__(self, model, model_id, device):
        self.model, self.model_id, self.device = model, model_id, device

    @classmethod
    def from_local(cls, source, checkpoint, *, sha256=None, device='cpu', allow_noncommercial=False):
        if allow_noncommercial is not True:
            raise PermissionError('USEF-TSE code and weights are CC BY-NC 4.0: pass allow_noncommercial=True '
                                  'to opt in (non-commercial use only)')
        import subprocess
        import sys
        import torch
        from hyperpyyaml import load_hyperpyyaml
        source = Path(source)
        head = subprocess.run(['git', '-C', str(source), 'rev-parse', 'HEAD'], capture_output=True,
                              text=True).stdout.strip()
        if head != cls.SOURCE_REVISION:
            raise ValueError(f'{source} must be ZBang/USEF-TSE at {cls.SOURCE_REVISION}, found {head or "no git"}')
        digest = _verify(checkpoint, sha256)
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))   # the repo's modules (models.*, utils.*) are imported by the config
        with open(source / 'chkpt' / 'USEF-TFGridNet' / 'config.yaml') as handle:
            config = load_hyperpyyaml(handle)
        model = config['modules']['masknet']
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)['model_state_dict']
        state = {key.replace('module.', '').replace('convolution_', 'convolution_module.'): value
                 for key, value in state.items()}
        model.load_state_dict(state)
        model.eval().to(device)
        return cls(model, f'usef-tfgridnet:{digest[:16]}', device)

    def extract(self, mixture, enrollment):
        import torch
        mix = resample(mixture, SAMPLE_RATE, self.sample_rate)
        aux = resample(enrollment, SAMPLE_RATE, self.sample_rate)
        with torch.inference_mode():
            estimate = self.model(torch.from_numpy(mix).unsqueeze(0).to(self.device),
                                  torch.from_numpy(aux).unsqueeze(0).to(self.device))
        estimate = estimate.detach().cpu().numpy().reshape(-1)
        return _fit(resample(estimate, self.sample_rate, SAMPLE_RATE), len(mixture))


EXTRACTORS = {'wesep': WeSepExtractor, 'usef': USEFExtractor}


def _fit(values, length):
    np = _np()
    values = np.asarray(values, dtype=np.float32)
    if len(values) >= length:
        return values[:length]
    return np.concatenate([values, np.zeros(length - len(values), np.float32)])


# ---------------------------------------------------------------- block scheduling (math)

class Blocks:
    """Emit [end - context - block, end) windows every ``block_ms`` of new audio; keep the last block."""

    def __init__(self, *, block_ms=2000, context_ms=2000, ring_seconds=20.):
        if block_ms < 200 or context_ms < 0:
            raise ValueError('require block_ms >= 200 and context_ms >= 0')
        self.block_ms, self.context_ms = int(block_ms), int(context_ms)
        self.ring = AudioRing(SAMPLE_RATE, max(ring_seconds, 2 * (block_ms + context_ms) / 1000))
        self.reset()

    def reset(self):
        self.ring.reset()
        self._next_ms = None

    def push(self, samples, end_ms):
        """Returns [(block_start_ms, block_end_ms, window_samples, keep_samples)]."""
        if self.ring.append(samples, end_ms, SAMPLE_RATE) or self._next_ms is None:
            if self.ring.start_ms is None:
                return []
            self._next_ms = self.ring.start_ms + self.block_ms
        blocks = []
        while self._next_ms <= self.ring.end_ms + .5:
            blocks.append(self._block(self._next_ms - self.block_ms, self._next_ms))
            self._next_ms += self.block_ms
        return [block for block in blocks if block[3]]

    def flush(self):
        """The last, shorter block at the end of the input."""
        if self._next_ms is None:
            return []
        start, end = self._next_ms - self.block_ms, self.ring.end_ms
        self._next_ms = None
        return [self._block(start, end)] if end - start >= 100 else []

    def _block(self, start, end):
        window = self.ring.span(max(self.ring.start_ms, start - self.context_ms), end)
        keep = min(len(window), int(round((end - start) * SAMPLE_RATE / 1000)))
        return int(round(start)), int(round(end)), window, keep


def derived_chunks(name, samples, start_ms, chunk_ms=100):
    """Split separated audio into chunks stamped with their own capture time, like a microphone's."""
    from ..pipeline import Derived
    size = SAMPLE_RATE * chunk_ms // 1000
    return [Derived(name, samples[i:i + size].copy(), 'audio', SAMPLE_RATE,
                    timestamp_ms=int(round(start_ms + 1000 * min(i + size, len(samples)) / SAMPLE_RATE)))
            for i in range(0, len(samples), size)]


class TargetSpeech:
    """Pipeline module: mixture audio in, ``Derived(emits, separated block)`` out (enrollment-audio TSE)."""

    def __init__(self, extractor, *, reference='voice', emits=None, block_ms=2000, context_ms=2000):
        self.extractor, self.reference_name = extractor, reference
        self.emits = emits or f'voice:{reference}'
        self.blocks = Blocks(block_ms=block_ms, context_ms=context_ms)
        self.enrollment = None
        self.name, self.version = 'target_speech', getattr(extractor, 'model_id', 'unspecified')

    def bind(self, pipeline):
        reference = pipeline.references.get(self.reference_name)
        if reference is None or reference['kind'] != 'voice':
            raise ValueError(f'target_speech needs a voice reference named {self.reference_name!r}')
        self.enrollment = enrollment_audio(reference['value'])

    def reset(self):
        self.blocks.reset()

    def process(self, frame):
        if frame.modality != 'audio' or frame.data is None or isinstance(frame.data, dict):
            return []
        if frame.sample_rate != SAMPLE_RATE:
            raise ValueError('target_speech needs 16 kHz mono audio')
        if self.enrollment is None:
            raise RuntimeError('no enrolled voice clip bound')
        return self._extract(self.blocks.push(frame.data, int(frame.timestamp_ms)))

    def finish(self):
        return self._extract(self.blocks.flush())

    def _extract(self, blocks):
        output = []
        for start_ms, _, window, keep in blocks:
            estimate = self.extractor.extract(window, self.enrollment)
            output += derived_chunks(self.emits, estimate[-keep:], start_ms)
        return output


class SpeakerGate:
    """Pipeline module: the room audio, silenced wherever the enrolled voice was not matched (no model).

    Consumes the audio and the ``audio.speaker_match`` segments of :class:`SpeakerVerify`. Each 100 ms
    chunk takes the verdict of the segment whose middle is nearest to it, so output lags by one segment.
    This does not separate overlapping voices: while the enrolled person speaks, everything audible
    passes. Use a :class:`TargetSpeech` model where voices overlap.
    """

    def __init__(self, *, reference='voice', emits=None, segment_ms=1500, chunk_ms=100):
        self.reference_name, self.emits = reference, emits or f'voice:{reference}'
        self.segment_ms, self.chunk_ms = int(segment_ms), int(chunk_ms)
        self.name, self.version = 'speaker_gate', '1'
        self.reset()

    def reset(self):
        self.blocks = Blocks(block_ms=200, context_ms=0)
        self.pending, self.segments = [], []

    def process(self, frame):
        value = frame.data
        if isinstance(value, dict):
            if value.get('kind') == 'segment' and value.get('reference') == self.reference_name:
                self.segments = self.segments[-63:] + [(value['start_ms'], value['end_ms'], bool(value['match']))]
            return []
        if frame.modality != 'audio' or value is None:
            return []
        if frame.sample_rate != SAMPLE_RATE:
            raise ValueError('speaker gate needs 16 kHz mono audio')
        self.pending += [(start, end, window) for start, end, window, _ in
                         self.blocks.push(value, int(frame.timestamp_ms))]
        return self._emit(self.blocks.ring.end_ms - self.segment_ms)

    def finish(self):
        self.pending += [(start, end, window) for start, end, window, _ in self.blocks.flush()]
        return self._emit(None)

    def _emit(self, until_ms):
        np = _np()
        output = []
        while self.pending and (until_ms is None or self.pending[0][1] <= until_ms):
            start, end, window = self.pending.pop(0)
            middle = (start + end) / 2
            covering = [item for item in self.segments if item[0] <= middle <= item[1]] or \
                (self.segments[-1:] if until_ms is None else [])
            keep = bool(covering) and min(covering, key=lambda item: abs((item[0] + item[1]) / 2 - middle))[2]
            output += derived_chunks(self.emits, window if keep else np.zeros_like(window), start, self.chunk_ms)
        return output


# ---------------------------------------------------------------- SAM-Audio span prompt path

def span_anchors(spans, window_start_ms, window_end_ms, *, include_other=True, min_ms=100):
    """Speaker-match spans -> non-overlapping SAM-Audio anchors (token, start_s, end_s) in window seconds.

    ``spans``: ``(start_ms, end_ms, match)``; overlapping spans of the same token are merged, a ``+``
    span wins over an overlapping ``-`` span, and pieces shorter than ``min_ms`` are dropped.
    """
    duration = window_end_ms - window_start_ms
    clipped = {'+': [], '-': []}
    for start, end, match in spans:
        start, end = max(start, window_start_ms), min(end, window_end_ms)
        if end - start >= min_ms and (match or include_other):
            clipped['+' if match else '-'].append([start, end])
    merged = {}
    for token, items in clipped.items():
        items.sort()
        out = []
        for start, end in items:
            if out and start <= out[-1][1]:
                out[-1][1] = max(out[-1][1], end)
            else:
                out.append([start, end])
        merged[token] = out
    negatives = []
    for start, end in merged['-']:
        pieces = [[start, end]]
        for p_start, p_end in merged['+']:
            next_pieces = []
            for a, b in pieces:
                if p_end <= a or p_start >= b:
                    next_pieces.append([a, b])
                    continue
                if p_start > a:
                    next_pieces.append([a, p_start])
                if p_end < b:
                    next_pieces.append([p_end, b])
            pieces = next_pieces
        negatives += [piece for piece in pieces if piece[1] - piece[0] >= min_ms]
    anchors = [('+', s, e) for s, e in merged['+']] + [('-', s, e) for s, e in negatives]
    anchors.sort(key=lambda item: item[1])
    result = []
    for token, start, end in anchors[:32]:
        start_s = round((start - window_start_ms) / 1000, 3)
        end_s = round(min(duration, end - window_start_ms) / 1000, 3)
        if end_s > start_s:
            result.append((token, start_s, end_s))
    return result


class SAMSpanSpeech:
    """Pipeline module: 16 kHz mixture + ``audio.speaker_match`` segments in, Derived voice blocks out.

    The SAM-Audio backend is GAMST's existing :class:`SAMAudioNative` (complete 48 kHz windows of
    1..4 s). Windows with no ``+`` span emit silence for the person (nothing of theirs to keep).
    """

    def __init__(self, backend, *, prompt='speech', reference='voice', emits=None, window_ms=2000,
                 include_other=True):
        if not 1000 <= window_ms <= 4000:
            raise ValueError('SAM-Audio windows are 1..4 s')
        self.backend, self.prompt, self.reference_name = backend, prompt, reference
        self.emits = emits or f'voice:{reference}'
        self.include_other = include_other
        self.blocks = Blocks(block_ms=window_ms, context_ms=0)
        self.spans = []
        self.name, self.version = 'sam_span_speech', 'sam-audio.span'

    @classmethod
    def from_local(cls, config, **options):
        from .sam_audio_context import configuration
        from .sam_audio_native import SAMAudioNative
        config = configuration(config)
        options.setdefault('prompt', config['prompt'])
        return cls(SAMAudioNative.from_local(config), **options)

    def reset(self):
        self.blocks.reset()
        self.spans = []

    def process(self, frame):
        from ..sam_audio_windows import SAMPLE_RATE as SAM_RATE, prepare_window
        np = _np()
        value = frame.data
        if isinstance(value, dict):
            if value.get('kind') == 'segment' and value.get('reference') == self.reference_name \
                    and value.get('state') != 'silent':
                self.spans.append((value['start_ms'], value['end_ms'], bool(value.get('match'))))
                self.spans = self.spans[-256:]
            return []
        if frame.modality != 'audio' or value is None:
            return []
        if frame.sample_rate != SAMPLE_RATE:
            raise ValueError('sam span speech needs 16 kHz mono audio')
        output = []
        for start_ms, end_ms, window, keep in self.blocks.push(value, int(frame.timestamp_ms)):
            anchors = span_anchors(self.spans, start_ms, end_ms, include_other=self.include_other)
            if not any(token == '+' for token, _, _ in anchors):
                output += derived_chunks(self.emits, np.zeros(keep, np.float32), start_ms)
                continue
            pcm = np.clip(resample(window[-keep:], SAMPLE_RATE, SAM_RATE), -1., 1.).astype(np.float32)
            prepared = prepare_window(pcm, sample_rate=SAM_RATE, prompt=self.prompt, anchors=list(anchors),
                                      hop_length=self.backend.hop_length)
            target, _, _, _ = self.backend.predict(prepared)
            estimate = np.frombuffer(target, dtype=np.float32)
            output += derived_chunks(self.emits, _fit(resample(estimate, SAM_RATE, SAMPLE_RATE), keep), start_ms)
            self.spans = [span for span in self.spans if span[1] > end_ms - 4000]
        return output

    def close(self):
        close = getattr(self.backend, 'close', None)
        if callable(close):
            close()

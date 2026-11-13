"""Bounded native chunk adapter for pinned Nemotron3 diarization artifacts."""
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import threading
import uuid

from ..capabilities import CapabilitySpec, OutputField, SensorMetadata, identifier
from ..plugins import InputFrame
from .vocal_expression import _read

REVISION = '52a432672732425e4c19765b9fec9b5810e6f4ba'
ARTIFACTS = {
    'config.json': (1458, 'dd795ec07ffa09acf18d231aa3fa8642aa5c521760953754e02559a7413b6873'),
    'processor_config.json': (623, '91726746788a024a2f0f2f699a6c3667f0d74e6a0e9288fc44fe3018e23af68e'),
    'model.safetensors': (396954592, 'c074d86335b3b794f8fa5edc25594558f128bdb3914d27806a3a5a2e44963cb6'),
}
MODES = {'ultra_low_latency': (3, 1), 'very_low_latency': (6, 2), 'low_latency': (9, 4)}


class _NativeBackend:
    def __init__(self, model, processor, torch, device):
        self.model, self.processor, self.torch, self.device = model, processor, torch, device

    def infer(self, samples, *, first, last, cache):
        torch = self.torch
        with torch.inference_mode():
            inputs = self.processor(samples, sampling_rate=16000, is_streaming=True,
                                    is_first_audio_chunk=first, is_last_audio_chunk=last,
                                    return_tensors='pt')
            if last:
                # A short first-and-last chunk must still use the streaming cache
                # configuration, even though the processor omits this argument.
                inputs['num_lookahead_frames'] = 0
            inputs = inputs.to(self.device)
            output = self.model(**inputs, speaker_cache=cache)
            logits, state = output.logits, output.speaker_cache
            if (type(logits) is not torch.Tensor or logits.dtype != torch.float32 or
                    logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[2] != 8 or
                    not 1 <= logits.shape[1] <= 104 or not torch.isfinite(logits).all().item()):
                raise ValueError('invalid native diarization logits')
            for name, shape in (('embeds', (1, 264, 512)), ('fifo', (1, 264, 512)), ('probs', (1, 264, 8))):
                value = getattr(state, name, None)
                if (type(value) is not torch.Tensor or tuple(value.shape) != shape or
                        value.dtype != torch.float32 or value.device.type != self.device or
                        not torch.isfinite(value).all().item()):
                    raise ValueError('invalid or unbounded native speaker cache')
            for name in ('num_cache_frames', 'num_fifo_frames'):
                value = getattr(state, name, None)
                if type(value) is not int or not 0 <= value <= 264:
                    raise ValueError('invalid native speaker cache cursor')
            return logits.sigmoid()[0].cpu().numpy().copy(), state


class NemotronDiarization:
    """One source/session; backend doubles do not establish trained inference."""
    def __init__(self, backend, *, mode='ultra_low_latency', threshold=.5, device='cpu'):
        self.spec = self.capability(mode=mode, threshold=threshold, device=device)
        if not callable(getattr(backend, 'infer', None)):
            raise ValueError('explicit native backend required')
        import numpy as np
        self.np, self.backend = np, backend
        self._artifacts_verified = False
        self.mode, self.threshold = mode, threshold
        chunk, right = MODES[mode]
        self.step_frames, self.right_frames = chunk * 8, right * 8
        self.first_samples = ((chunk + right) * 8 - 1) * 160 + 200
        self.later_samples = (chunk + right) * 8 * 160 + 400
        self._lock = threading.RLock()
        self.closed = False
        self.reset()

    @staticmethod
    def capability(*, mode='ultra_low_latency', threshold=.5, device='cpu'):
        if type(mode) is not str or mode not in MODES or device not in ('cpu', 'cuda'):
            raise ValueError('select a supported streaming mode and cpu/cuda device')
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or not .01 <= threshold <= .99:
            raise ValueError('activity threshold must be .01.. .99')
        chunk, right = MODES[mode]
        digest = hashlib.sha256(json.dumps([REVISION, mode, threshold, device]).encode()).hexdigest()
        return CapabilitySpec('audio.nemotron3.diarization', 'audio.speaker_activity', 'audio',
            'nemotron3.diarization', '1.'+digest, (OutputField('diarization', 'object'),),
            'native', window_ms=chunk*80, lookahead_ms=right*80, freshness_ms=2000,
            stateful=True, reset_on_gap=True, interpretation='classification', devices=(device,),
            sensor_requirements=('16000hz-mono-pcm', 'whole-10ms-capture-blocks'),
            license='weights:OpenMDW-1.1-declared;adapter/Transformers:Apache-2.0')

    @classmethod
    def from_local(cls, path, *, mode='ultra_low_latency', threshold=.5, device='cpu'):
        cls.capability(mode=mode, threshold=threshold, device=device)
        for package, expected in (('transformers', '5.23.0'), ('torch', '2.8.0'), ('safetensors', '0.8.0')):
            try:
                installed = version(package)
            except PackageNotFoundError as exc:
                raise RuntimeError(f'install {package}=={expected} in the selected Python>=3.11 runtime') from exc
            if installed.split('+')[0] != expected:
                raise RuntimeError(f'selected Python>=3.11 runtime requires {package}=={expected}')
        root = Path(path).expanduser()
        blobs = {name: _read(root / name, size, digest) for name, (size, digest) in ARTIFACTS.items()}
        import torch
        from safetensors.torch import load
        from transformers import (Nemotron3DiarizationConfig,
                                  Nemotron3DiarizationForAudioFrameClassification,
                                  Nemotron3DiarizationProcessor,
                                  NemotronAsrStreamingFeatureExtractor)
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('selected CUDA device is unavailable')
        config = Nemotron3DiarizationConfig(**json.loads(blobs['config.json']))
        config._attn_implementation = 'eager'
        model = Nemotron3DiarizationForAudioFrameClassification(config).float().cpu().eval()
        weights = load(blobs.pop('model.safetensors'))
        expected = model.state_dict()
        if set(weights) != set(expected):
            raise ValueError('checkpoint must match every model tensor without missing/extra keys')
        for name, value in weights.items():
            if value.shape != expected[name].shape or value.dtype != expected[name].dtype or not torch.isfinite(value).all().item():
                raise ValueError('checkpoint tensor shape/dtype/finite mismatch')
        model.load_state_dict(weights, strict=True)
        del weights, expected
        processor_config = json.loads(blobs['processor_config.json'])
        extractor_config = processor_config.pop('feature_extractor')
        extractor_config.pop('feature_extractor_type')
        processor_config.pop('processor_class')
        processor = Nemotron3DiarizationProcessor(NemotronAsrStreamingFeatureExtractor(**extractor_config),
                                                **processor_config)
        processor.set_streaming_mode(mode)
        provider = cls(_NativeBackend(model.to(device), processor, torch, device),
                       mode=mode, threshold=threshold, device=device)
        if (processor.num_samples_first_audio_chunk != provider.first_samples or
                processor.num_samples_per_audio_chunk != provider.later_samples or
                processor.num_mel_frames_per_step != provider.step_frames or
                processor.audio_chunk_start(provider.step_frames) != provider.step_frames*160-256):
            raise ValueError('native processor framing contract changed')
        provider._artifacts_verified = True
        return provider

    def _clear(self):
        self._buffer = self.np.empty(0, self.np.float32)
        self._cache = self._owner = self._last = self._origin = None
        self._cursor = self._received = self._buffer_start = 0
        self._session = uuid.uuid4().hex

    def reset(self):
        with self._lock:
            self._clear()
            self._finished = False

    def close(self):
        with self._lock:
            self._clear()
            self.backend = None
            self.closed = True

    def _pcm(self, frame):
        np = self.np
        if (not isinstance(frame, InputFrame) or frame.modality != 'audio' or
                type(frame.sample_rate) is not int or frame.sample_rate != 16000 or
                not isinstance(frame.sensor, SensorMetadata) or len(frame.sensor.channels) != 1):
            raise ValueError('explicit16kHz mono audio frame/channel metadata required')
        identifier(frame.subject_id)
        data = frame.data
        if type(data) is bytes:
            if not 320 <= len(data) <= 32000 or len(data) % 320:
                raise ValueError('PCM16 must be10..1000ms in whole10ms blocks')
            values = np.frombuffer(data, dtype='<i2').astype(np.float32) / 32768
        else:
            if (type(data) is not np.ndarray or data.dtype not in (np.dtype('int16'), np.dtype('float32')) or
                    data.ndim not in (1, 2) or data.ndim == 2 and data.shape[1] != 1 or
                    not 160 <= len(data) <= 16000 or len(data) % 160):
                raise ValueError('mono int16/float32 of10..1000ms in whole10ms blocks required')
            values = data.reshape(-1).astype(np.float32, copy=True)
            if data.dtype == np.int16:
                values /= 32768
        if not np.isfinite(values).all() or np.max(np.abs(values)) > 1:
            raise ValueError('PCM must be finite and normalized')
        ts, duration = frame.timestamp_ms, len(values) // 16
        if type(ts) is not int or not duration <= ts <= 2**53-1:
            raise ValueError('integer capture-end timestamp must cover block duration')
        return values, ts, duration

    def process(self, frame):
        with self._lock:
            if self.closed or self._finished:
                raise RuntimeError('diarization session is closed/finished; reset before reuse')
            try:
                values, ts, duration = self._pcm(frame)
                sensor = frame.sensor
                owner = (frame.subject_id, sensor.source_id, sensor.sensor_id, sensor.stream_epoch, sensor.channels,
                         sensor.calibration_id, sensor.coordinate_frame, sensor.capture_sequence is not None)
                seq = sensor.capture_sequence if sensor.capture_sequence is not None else sensor.sequence
                if owner == self._owner and self._last:
                    if ts <= self._last[0] or seq <= self._last[1]:
                        raise ValueError('capture order must increase')
                    if ts-duration != self._last[0] or seq != self._last[1]+1:
                        self._clear()
                elif self._owner is not None:
                    self._clear()
                if self._origin is None:
                    self._origin = ts-duration
                self._buffer = self.np.concatenate((self._buffer, values))
                self._received += len(values)
                self._owner, self._last = owner, (ts, seq)
                chunks = []
                while len(self._buffer) >= (self.first_samples if self._cursor == 0 else self.later_samples):
                    count = self.first_samples if self._cursor == 0 else self.later_samples
                    chunks.append(self._infer(count, last=False))
                return self._result(chunks, final=False) if chunks else None
            except Exception:
                self._clear()
                raise

    def _infer(self, count, *, last):
        np = self.np
        first = self._cursor == 0
        expected = (count // 160 if first else (count-256)//160) if last else self.step_frames
        probabilities, cache = self.backend.infer(self._buffer[:count].copy(), first=first, last=last, cache=self._cache)
        if (type(probabilities) is not np.ndarray or probabilities.dtype != np.float32 or
                probabilities.shape != (expected, 8) or not 1 <= expected <= 104 or cache is None or
                not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1)):
            raise ValueError('invalid native frame probability grid/cache')
        start = self._origin+self._cursor*10
        activity = []
        for slot in range(8):
            scores = probabilities[:, slot]
            active = np.r_[False, scores > self.threshold, False].astype(np.int8)
            edges = np.diff(active)
            intervals = [[start+int(a)*10, start+int(b)*10]
                         for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]
            activity.append(dict(slot=slot, intervals_ms=intervals,
                                 mean_probability=float(scores.mean()), max_probability=float(scores.max())))
        self._cache = cache
        self._cursor += expected
        if not last:
            next_start = self._cursor*160-256
            consumed = next_start-self._buffer_start
            self._buffer = self._buffer[consumed:].copy()
            self._buffer_start = next_start
        return dict(start_ms=start, end_ms=start+expected*10, frame_ms=10, activity=activity)

    def _result(self, chunks, *, final):
        return self.spec.validate({'diarization': dict(session_id=self._session, chunks=chunks,
            capture_end_ms=self._last[0], final=final, mode=self.mode, threshold=self.threshold,
            slot_count=8, identity_verified=False, slot_continuity_verified=False,
            lookahead_ms=self.right_frames*10, source_revision=REVISION,
            checkpoint_sha256=ARTIFACTS['model.safetensors'][1],
            artifact_hashes_verified=self._artifacts_verified, quality_verified=False,
            turn_boundaries_final=False, overlap_supported=True)})

    def finish(self):
        with self._lock:
            if self.closed:
                raise RuntimeError('diarization provider is closed')
            if self._finished:
                return None
            try:
                if self._origin is None:
                    return None
                if self._received//160 <= self._cursor:
                    return None
                chunk = self._infer(len(self._buffer), last=True)
                if self._cursor != self._received//160:
                    raise ValueError('final diarization frames do not cover captured input')
                return self._result([chunk], final=True)
            finally:
                self._clear()
                self._finished = True

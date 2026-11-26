"""NVIDIA Streaming Sortformer v2.1 (four speaker slots, NeMo) as a live ``audio.diarization`` candidate.

The model card (https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1) names Nemotron-3-Diarization
as its successor; this candidate is kept to compare the two on the owner's clips. Weights: NVIDIA Open Model
License; NeMo: Apache-2.0.

Streaming here is the model's own speaker-cache step (``forward_streaming_step``) fed as capture blocks arrive:
log-mel features (no normalisation, so each frame depends on 25 ms of sound only) are computed on the buffered
sound with a margin of four frames on each side and cut to the chunk with its left/right context, exactly the
frames NeMo's own streaming loop feeds when it holds the whole recording (checked by ``tests/test_sortformer.py``
with a model double, and on the real model in ``docs/measurements/diarization-2026-11-26.md``). Output frames are
80 ms; the record is the one every ``audio.diarization`` candidate emits (:func:`.diarization.slot_record`), with
slots 4-7 never active.
"""
import hashlib
import json
import math
import threading
import uuid
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from ..capabilities import CapabilitySpec, OutputField
from .diarization import SlotStream, slot_activity, slot_record

REVISION = 'cd03eee90fbec18297ac31b8c21546e596b7f71c'
FILE = 'diar_streaming_sortformer_4spk-v2.1.nemo'
SIZE, SHA256 = 471367680, '8abd32832159c6ac1148c926b7276f35ba34582c444e559dce1f1253fea42ef8'
PACKAGES = (('nemo_toolkit', '3.0.0'), ('torch', '2.8.0'))
# Model card table, in 80 ms frames: (chunk, right context, fifo, cache update period, cache size).
MODES = {'ultra_low_latency': (3, 1, 188, 144, 188), 'low_latency': (6, 7, 188, 144, 188),
         'high_latency': (124, 1, 124, 124, 188)}
FRAME, SUB, MARGIN = 160, 8, 4      # samples per feature frame, feature frames per output frame, margin frames


class _NeMoBackend:
    def __init__(self, model, torch, device):
        self.model, self.torch, self.device = model, torch, device

    def features(self, samples):
        torch = self.torch
        with torch.inference_mode():
            signal = torch.from_numpy(samples).to(self.device)[None]
            length = torch.tensor([len(samples)], device=self.device)
            feats, lengths = self.model.preprocessor(input_signal=signal, length=length)
            return feats[0, :, :int(lengths[0])].cpu().numpy()

    def initial_state(self):
        return self.model.sortformer_modules.init_streaming_state(batch_size=1, async_streaming=False,
                                                                  device=self.model.device)

    def step(self, feats, state, *, left, right):
        torch = self.torch
        with torch.inference_mode():
            chunk = torch.from_numpy(feats.T.copy()).to(self.device)[None]
            length = torch.tensor([feats.shape[1]], device=self.device)
            empty = torch.zeros((1, 0, self.model.sortformer_modules.n_spk), device=self.device)
            state, preds = self.model.forward_streaming_step(processed_signal=chunk, processed_signal_length=length,
                                                             streaming_state=state, total_preds=empty,
                                                             left_offset=left, right_offset=right)
            return preds[0].float().cpu().numpy().copy(), state


class SortformerDiarization(SlotStream):
    """One source/session of Streaming Sortformer slots; a backend double does not establish trained inference."""

    def __init__(self, backend, *, mode='low_latency', threshold=.5, device='cpu'):
        self.spec = self.capability(mode=mode, threshold=threshold, device=device)
        if not all(callable(getattr(backend, name, None)) for name in ('features', 'initial_state', 'step')):
            raise ValueError('explicit Sortformer backend required')
        import numpy as np
        self.np, self.backend, self.mode, self.threshold = np, backend, mode, threshold
        self.chunk, self.right = MODES[mode][0], MODES[mode][1]
        self._artifacts_verified = False
        self._lock = threading.RLock()
        self.closed = False
        self.reset()

    @staticmethod
    def capability(*, mode='low_latency', threshold=.5, device='cpu'):
        if type(mode) is not str or mode not in MODES or device not in ('cpu', 'cuda'):
            raise ValueError(f"mode is one of {', '.join(MODES)}; device cpu or cuda")
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or not .01 <= threshold <= .99:
            raise ValueError('activity threshold must be .01.. .99')
        chunk, right = MODES[mode][:2]
        digest = hashlib.sha256(json.dumps([REVISION, mode, threshold, device]).encode()).hexdigest()
        return CapabilitySpec('audio.sortformer.diarization', 'audio.speaker_activity', 'audio',
                              'sortformer.diarization', '1.' + digest, (OutputField('diarization', 'object'),),
                              'native', window_ms=chunk * 80, lookahead_ms=right * 80, freshness_ms=2000,
                              stateful=True, reset_on_gap=True, interpretation='classification', devices=(device,),
                              sensor_requirements=('16000hz-mono-pcm', 'whole-10ms-capture-blocks'),
                              license='weights:NVIDIA-Open-Model-License;NeMo:Apache-2.0;adapter:Apache-2.0')

    @classmethod
    def from_local(cls, path, *, mode='low_latency', threshold=.5, device='cpu'):
        cls.capability(mode=mode, threshold=threshold, device=device)
        for package, expected in PACKAGES:
            try:
                installed = version(package)
            except PackageNotFoundError as error:
                raise RuntimeError(f'install {package}=={expected} in the sortformer environment') from error
            if installed.split('+')[0] != expected:
                raise RuntimeError(f'the sortformer environment requires {package}=={expected}')
        root = Path(path).expanduser()
        nemo = root / FILE if root.is_dir() else root
        digest, size = hashlib.sha256(), 0
        with open(nemo, 'rb') as handle:
            for block in iter(lambda: handle.read(1 << 22), b''):
                digest.update(block)
                size += len(block)
        if size != SIZE or digest.hexdigest() != SHA256:
            raise ValueError(f'{nemo} is not the pinned {FILE} (size/SHA256 mismatch)')
        import torch
        from nemo.collections.asr.models import SortformerEncLabelModel
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('selected CUDA device is unavailable')
        model = SortformerEncLabelModel.restore_from(restore_path=str(nemo), map_location=device, strict=False)
        model.eval()
        chunk, right, fifo, update, cache = MODES[mode]
        modules = model.sortformer_modules
        modules.chunk_len, modules.chunk_right_context, modules.fifo_len = chunk, right, fifo
        modules.spkcache_update_period, modules.spkcache_len = update, cache
        modules._check_streaming_parameters()
        model.streaming_mode, model.async_streaming = True, False
        if modules.subsampling_factor != SUB or modules.n_spk != 4 or modules.chunk_left_context != 1:
            raise ValueError('Sortformer framing contract changed')
        provider = cls(_NeMoBackend(model, torch, device), mode=mode, threshold=threshold, device=device)
        provider._artifacts_verified = True
        return provider

    def _clear(self):
        self._buffer = self.np.empty(0, self.np.float32)
        self._state = self._owner = self._last = self._origin = None
        self._feat = self._received = self._buffer_start = 0     # next chunk's first feature frame; samples
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

    def _ready(self):
        end = self._feat + self.chunk * SUB
        return self._received >= (end + self.right * SUB + MARGIN) * FRAME

    def _infer(self, *, last):
        np = self.np
        total = self._received // FRAME if last else None      # NeMo's feature count for the whole capture
        left = min(SUB, self._feat)
        end = self._feat + self.chunk * SUB if not last else min(self._feat + self.chunk * SUB, total)
        right = self.right * SUB if not last else min(self.right * SUB, total - end)
        first_sample = max(0, (self._feat - left - MARGIN) * FRAME)
        stop = self._received if last else (end + right + MARGIN) * FRAME
        feats = self.backend.features(self._buffer[first_sample - self._buffer_start:stop - self._buffer_start].copy())
        offset = first_sample // FRAME
        feats = feats[:, self._feat - left - offset:end + right - offset]
        if self._state is None:
            self._state = self.backend.initial_state()
        probabilities, self._state = self.backend.step(feats, self._state, left=left, right=right)
        frames = math.ceil((end - self._feat) / SUB)
        if last and type(probabilities) is np.ndarray and probabilities.ndim == 2 and abs(len(probabilities) - frames) <= 1:
            frames = len(probabilities)   # the encoder's own length of a short final chunk (as NeMo's loop has it)
        if (type(probabilities) is not np.ndarray or probabilities.ndim != 2 or probabilities.shape[1] != 4 or
                probabilities.shape[0] != frames or not np.isfinite(probabilities).all() or
                np.any(probabilities < 0) or np.any(probabilities > 1)):
            raise ValueError('invalid Sortformer slot probabilities')
        start = self._origin + self._feat * 10
        chunk = dict(start_ms=start, end_ms=start + frames * 80, frame_ms=80,
                     activity=slot_activity(np, probabilities.astype(np.float32), start, 80, self.threshold))
        self._feat = end
        keep = max(0, (self._feat - SUB - MARGIN) * FRAME)       # the next chunk's left context and margin
        if keep > self._buffer_start:
            self._buffer = self._buffer[keep - self._buffer_start:].copy()
            self._buffer_start = keep
        return chunk

    def _result(self, chunks, *, final):
        return slot_record(self.spec, session_id=self._session, chunks=chunks, capture_end_ms=self._last[0],
                           final=final, mode=self.mode, threshold=self.threshold, lookahead_ms=self.right * 80,
                           source_revision=REVISION, checkpoint_sha256=SHA256,
                           artifact_hashes_verified=self._artifacts_verified)

    def process(self, frame):
        with self._lock:
            if self.closed or self._finished:
                raise RuntimeError('diarization session is closed/finished; reset before reuse')
            try:
                values = self._accept(frame)
                self._buffer = self.np.concatenate((self._buffer, values))
                self._received += len(values)
                chunks = []
                while self._ready():
                    chunks.append(self._infer(last=False))
                return self._result(chunks, final=False) if chunks else None
            except Exception:
                self._clear()
                raise

    def finish(self):
        with self._lock:
            if self.closed:
                raise RuntimeError('diarization provider is closed')
            if self._finished:
                return None
            try:
                if self._origin is None:
                    return None
                chunks = []
                while self._feat < self._received // FRAME:
                    chunks.append(self._infer(last=True))
                return self._result(chunks, final=True) if chunks else None
            finally:
                self._clear()
                self._finished = True

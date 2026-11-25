"""Optional TEN-VAD speech activity; Silero remains the default.

Implemented adapter; real-weight/native acceptance pending. Source:
https://github.com/TEN-framework/ten-vad/tree/9a3f4b55467a3418f896044370cc7689f4b3b14a

The upstream license is Apache-2.0 with additional Agora conditions, NOT plain
Apache-2.0: deployment may not compete with Agora or enable others to compete,
and is limited to the licensee's applications and their direct end users.
Derivative works retain those conditions; bundled BSD notices also apply.
Review the exact native artifact before deployment. This module uses the public
C ABI only and does not copy upstream code or load anything at import time.
"""
import ctypes
import hashlib
import math
from pathlib import Path
import re

from ..capabilities import CapabilitySpec, OutputField, identifier
from ..plugins import _pcm_floats

TEN_VAD_SOURCE_REVISION = '9a3f4b55467a3418f896044370cc7689f4b3b14a'
TEN_VAD_PACKAGE_VERSION = '1.0.6.9'
TEN_VAD_LICENSE = ('Apache-2.0 with Agora additional deployment/non-compete conditions; '
                   'applications and direct end users only; BSD notices')


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


class _TenVADNative:
    """Owned native handle with explicit recreate-on-reset semantics."""
    def __init__(self, path, hop_samples, threshold):
        self.path, self.hop_samples, self.threshold = path, hop_samples, threshold
        self.library = None
        self.handle = None
        self.closed = False
        self._open()

    def _open(self):
        library = ctypes.CDLL(str(self.path))
        library.ten_vad_create.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_float]
        library.ten_vad_create.restype = ctypes.c_int
        library.ten_vad_destroy.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        library.ten_vad_destroy.restype = ctypes.c_int
        library.ten_vad_process.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int32)]
        library.ten_vad_process.restype = ctypes.c_int
        handle = ctypes.c_void_p()
        if library.ten_vad_create(ctypes.byref(handle), self.hop_samples, self.threshold) != 0:
            raise RuntimeError('TEN-VAD native handle creation failed')
        self.library, self.handle = library, handle

    def process(self, samples):
        if self.closed or self.handle is None:
            raise RuntimeError('TEN-VAD native runtime is closed')
        probability, flag = ctypes.c_float(), ctypes.c_int32()
        result = self.library.ten_vad_process(self.handle, samples.ctypes.data_as(ctypes.c_void_p),
            len(samples), ctypes.byref(probability), ctypes.byref(flag))
        if result != 0:
            raise RuntimeError('TEN-VAD native inference failed')
        return probability.value, flag.value

    def reset(self):
        if self.closed:
            raise RuntimeError('TEN-VAD native runtime is closed')
        self._destroy()
        self._open()

    def _destroy(self):
        if self.handle is not None:
            if self.library.ten_vad_destroy(ctypes.byref(self.handle)) != 0:
                self.handle = None
                raise RuntimeError('TEN-VAD native handle destruction failed')
            self.handle = None

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                self._destroy()
            finally:
                self.handle = self.library = None


class TENActivity:
    """Bounded 10/16 ms TEN-VAD decisions with GAMST hysteresis and continuity."""
    def __init__(self, runtime, *, version, hop_ms=16, start_threshold=.5, stop_threshold=.35):
        if hop_ms not in (10, 16) or type(hop_ms) is not int:
            raise ValueError('TEN-VAD hop_ms must be 10 or 16')
        for value in (start_threshold, stop_threshold):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('thresholds must be finite numbers')
        if not 0 <= stop_threshold < start_threshold <= 1:
            raise ValueError('require 0 <= stop_threshold < start_threshold <= 1')
        if any(not callable(getattr(runtime, name, None)) for name in ('process', 'reset', 'close')):
            raise ValueError('an explicit resettable TEN-VAD native runtime is required')
        identifier(version, 'model version')
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError('Install numpy in the selected TEN-VAD runtime') from exc
        self.np, self.runtime = np, runtime
        self.hop_ms, self.hop_samples = hop_ms, hop_ms*16
        self.start_threshold, self.stop_threshold = start_threshold, stop_threshold
        self.spec = self.capability(version, hop_ms=hop_ms)
        self.closed = False
        self._clear()

    @staticmethod
    def capability(version, *, hop_ms=16):
        identifier(version, 'model version')
        if hop_ms not in (10, 16) or type(hop_ms) is not int:
            raise ValueError('TEN-VAD hop_ms must be 10 or 16')
        return CapabilitySpec('audio.ten_vad.activity', 'speech.activity', 'audio', 'ten_vad.native', version,
            (OutputField('decisions', 'array'), OutputField('state', 'object')), 'native',
            window_ms=hop_ms, lookahead_ms=0, freshness_ms=1000, stateful=True, reset_on_gap=True,
            interpretation='classification', devices=('cpu',),
            sensor_requirements=('16000hz-mono-normalized-pcm',), license=TEN_VAD_LICENSE)

    @staticmethod
    def artifact_version(sha256):
        if not isinstance(sha256, str) or not re.fullmatch('[0-9a-f]{64}', sha256):
            raise ValueError('supply lowercase TEN-VAD native library SHA256')
        return (TEN_VAD_PACKAGE_VERSION+'.'+TEN_VAD_SOURCE_REVISION[:12]+
                '.native.sha256.'+sha256)

    @classmethod
    def verify_local(cls, path, sha256):
        version = cls.artifact_version(sha256)
        path = Path(path).expanduser().resolve()
        if not path.is_file() or not 0 < path.stat().st_size <= 32*1024*1024:
            raise ValueError('supply a trusted local TEN-VAD native library, at most 32 MiB')
        if _sha256(path) != sha256:
            raise ValueError('TEN-VAD native library SHA256 mismatch')
        return path, version

    @classmethod
    def from_local(cls, path, sha256, **kwargs):
        path, version = cls.verify_local(path, sha256)
        hop_ms = kwargs.get('hop_ms', 16)
        if hop_ms not in (10, 16) or type(hop_ms) is not int:
            raise ValueError('TEN-VAD hop_ms must be 10 or 16')
        threshold = kwargs.get('start_threshold', .5)
        try:
            import numpy as np  # Fail clearly before loading the native library.
            del np
            runtime = _TenVADNative(path, hop_ms*16, threshold)
        except ImportError as exc:
            raise RuntimeError('Install numpy in the selected TEN-VAD runtime') from exc
        except OSError as exc:
            raise RuntimeError('TEN-VAD native library could not be loaded; select the matching platform binary') from exc
        try:
            return cls(runtime, version=version, **kwargs)
        except BaseException:
            runtime.close()
            raise

    def _clear(self):
        self.buffer = []
        self.active = False
        self.owner = self.last_ts = self.expected_end = None

    def reset(self):
        if self.closed:
            return
        self._clear()
        self.runtime.reset()

    def close(self):
        if not self.closed:
            self.closed = True
            self._clear()
            self.runtime.close()

    def process(self, frame):
        if self.closed:
            raise RuntimeError('TEN-VAD provider is closed')
        if frame.modality != 'audio' or type(frame.sample_rate) is not int or frame.sample_rate != 16000:
            raise ValueError('TEN-VAD requires 16 kHz mono PCM; select/resample upstream')
        sensor, ts = frame.sensor, frame.timestamp_ms
        if sensor is None or len(sensor.channels) > 1:
            raise ValueError('explicit source metadata and mono audio are required')
        if type(ts) is not int or ts < 0 or self.last_ts is not None and ts <= self.last_ts:
            raise ValueError('capture end timestamps must strictly increase')
        owner = (frame.subject_id, sensor.source_id, sensor.sensor_id, sensor.stream_epoch, sensor.channels)
        if self.owner is not None and owner != self.owner:
            raise ValueError('reset before switching source or channel layout')
        data = frame.data
        if isinstance(data, (bytes, bytearray, memoryview)):
            size = data.nbytes if isinstance(data, memoryview) else len(data)
            if size > 32000:
                raise ValueError('audio block must be at most one second')
            data = bytes(data)
        elif hasattr(data, 'ndim'):
            if data.ndim != 1 or len(data) > 16000:
                raise ValueError('audio must be a bounded mono sample vector')
        elif not isinstance(data, (list, tuple)) or len(data) > 16000:
            raise ValueError('audio must be a bounded mono sample vector')
        try:
            values = _pcm_floats(data)
        except (TypeError, OverflowError) as exc:
            raise ValueError('audio must contain finite normalized mono samples') from exc
        if not values:
            return None
        expected = ts if self.expected_end is None else self.expected_end+len(values)/16
        if abs(ts-expected) > 1:
            self.reset()
            expected = ts
        self.owner, self.last_ts, self.expected_end = owner, ts, expected
        pending = self.buffer+values
        decisions = []
        try:
            consumed = 0
            while len(pending)-consumed >= self.hop_samples:
                block = pending[consumed:consumed+self.hop_samples]
                samples = self.np.clip(self.np.asarray(block, dtype=self.np.float32)*32768.,
                    -32768, 32767).astype(self.np.int16)
                raw = self.runtime.process(samples)
                if not isinstance(raw, (tuple, list)) or len(raw) != 2:
                    raise ValueError('TEN-VAD must return probability and native flag')
                value, native_flag = raw
                if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError('invalid TEN-VAD activity score')
                if type(native_flag) not in (int, bool) or native_flag not in (0, 1):
                    raise ValueError('invalid TEN-VAD native activity flag')
                previous = self.active
                self.active = value >= (self.stop_threshold if previous else self.start_threshold)
                consumed += self.hop_samples
                end = expected-(len(pending)-consumed)/16
                decisions.append(dict(start_ms=end-self.hop_ms, end_ms=end, score=value,
                    active=self.active, transition=('start' if self.active else 'stop')
                    if previous != self.active else 'none'))
            self.buffer = pending[consumed:]
            if not decisions:
                return None
            return self.spec.validate(dict(decisions=decisions, state=dict(active=self.active,
                pending_samples=len(self.buffer), hop_ms=self.hop_ms,
                start_threshold=self.start_threshold, stop_threshold=self.stop_threshold,
                sample_rate=16000, interpretation='speech_activity_candidate')))
        except Exception:
            self.reset()
            raise

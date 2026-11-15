"""Stateful Silero speech activity, preserving every 32 ms decision."""
import hashlib
import math
from pathlib import Path
import re

from ..capabilities import CapabilitySpec, OutputField, identifier
from ..plugins import _pcm_floats


class SileroActivity:
    def __init__(self, model, score, *, version, start_threshold=.6, stop_threshold=.35):
        for value in (start_threshold, stop_threshold):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('thresholds must be finite numbers')
        if not 0 <= stop_threshold < start_threshold <= 1:
            raise ValueError('require 0 <= stop_threshold < start_threshold <= 1')
        if not callable(score) or not callable(getattr(model, 'reset_states', None)):
            raise ValueError('a score function and state-resettable model are required')
        self.model, self.score = model, score
        self.start_threshold, self.stop_threshold = start_threshold, stop_threshold
        self.spec = self.capability(version)
        self.closed = False
        self.reset()

    @staticmethod
    def capability(version):
        identifier(version, 'model version')
        return CapabilitySpec('audio.silero.activity', 'speech.activity', 'audio', 'silero.jit', version,
            (OutputField('decisions', 'array'), OutputField('state', 'object')), 'native',
            window_ms=32, lookahead_ms=0, freshness_ms=1000, stateful=True, reset_on_gap=True,
            interpretation='classification', devices=('cpu',),
            sensor_requirements=('16000hz-mono-normalized-pcm',), license='MIT')

    @staticmethod
    def artifact_version(sha256):
        if not isinstance(sha256, str) or not re.fullmatch('[0-9a-f]{64}', sha256):
            raise ValueError('supply lowercase Silero checkpoint SHA256')
        return 'jit.sha256.'+sha256

    @classmethod
    def verify_local(cls, path, sha256):
        version = cls.artifact_version(sha256)
        path = Path(path).expanduser().resolve()
        if not path.is_file() or not 0 < path.stat().st_size <= 32*1024*1024:
            raise ValueError('supply a trusted local Silero JIT checkpoint, at most 32 MiB')
        digest = hashlib.sha256()
        with path.open('rb') as source:
            for block in iter(lambda: source.read(1024*1024), b''):
                digest.update(block)
        if digest.hexdigest() != sha256:
            raise ValueError('Silero checkpoint SHA256 mismatch')
        return path, version

    @classmethod
    def from_local(cls, path, sha256, **kwargs):
        path, version = cls.verify_local(path, sha256)
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError('Install humansense-sdk[vad] in the selected runtime') from exc
        model = torch.jit.load(str(path), map_location='cpu').eval()

        def score(values):
            with torch.inference_mode():
                return float(model(torch.tensor(values, dtype=torch.float32), 16000).item())

        return cls(model, score, version=version, **kwargs)

    def reset(self):
        self.buffer = []
        self.active = False
        self.owner = self.last_ts = self.expected_end = None
        self.model.reset_states()

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                self.reset()
            finally:
                self.model = self.score = None

    def process(self, frame):
        if self.closed:
            raise RuntimeError('Silero provider is closed')
        if frame.modality != 'audio' or type(frame.sample_rate) is not int or frame.sample_rate != 16000:
            raise ValueError('Silero requires 16 kHz mono PCM; select/resample upstream')
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
            while len(pending)-consumed >= 512:
                value = self.score(pending[consumed:consumed+512])
                if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError('invalid Silero activity score')
                previous = self.active
                self.active = value >= (self.stop_threshold if previous else self.start_threshold)
                consumed += 512
                end = expected-(len(pending)-consumed)/16
                decisions.append(dict(start_ms=end-32, end_ms=end, score=value, active=self.active,
                    transition=('start' if self.active else 'stop') if previous != self.active else 'none'))
            self.buffer = pending[consumed:]
            if not decisions:
                return None
            return self.spec.validate(dict(decisions=decisions, state=dict(active=self.active,
                pending_samples=len(self.buffer), start_threshold=self.start_threshold,
                stop_threshold=self.stop_threshold, sample_rate=16000,
                interpretation='speech_activity_candidate')))
        except Exception:
            self.reset()
            raise

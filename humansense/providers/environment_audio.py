"""Bounded YAMNet windows on an explicit ambient channel, separate from ASR."""
import hashlib
import math
from pathlib import Path

from ..capabilities import CapabilitySpec, OutputField, identifier
from ..plugins import _pcm_floats

YAMNET_SHA256 = '4d8b4a53282dc83ef04e3e7dbc4fbc98082e34e44ed798e16c3a0cdd4c584faf'
MEDIAPIPE_VERSION = '0.10.35'
MODEL_VERSION = 'mediapipe.'+MEDIAPIPE_VERSION+'.sha256.'+YAMNET_SHA256


class YAMNetEvents:
    """975ms trailing windows; no neural recurrent state or speech cleanup."""
    def __init__(self, classifier, make_audio, *, version, hop_ms=480, top_k=8,
                 score_threshold=.05, channel=None):
        if type(hop_ms) is not int or not 20 <= hop_ms <= 975:
            raise ValueError('hop_ms must be 20..975')
        if type(top_k) is not int or not 1 <= top_k <= 32:
            raise ValueError('top_k must be 1..32')
        if type(score_threshold) not in (int, float) or not math.isfinite(score_threshold) or not 0 <= score_threshold <= 1:
            raise ValueError('score_threshold must be in [0, 1]')
        if channel is not None and (type(channel) is not int or not 0 <= channel < 64):
            raise ValueError('channel must be None or 0..63')
        if not callable(getattr(classifier, 'classify', None)) or not callable(make_audio):
            raise ValueError('a YAMNet classifier and audio container factory are required')
        identifier(version, 'model version')
        import numpy as np
        self.np, self.classifier, self.make_audio = np, classifier, make_audio
        self.hop_samples, self.top_k = hop_ms*16, top_k
        self.threshold, self.channel = score_threshold, channel
        self.spec = self.capability(version=version)
        self.closed = False
        self.reset()

    @staticmethod
    def capability(*, version=MODEL_VERSION):
        return CapabilitySpec('audio.yamnet.events', 'audio.environment_events', 'audio',
            'yamnet.mediapipe', version, (OutputField('events', 'array'), OutputField('window', 'object')),
            'window', window_ms=975, freshness_ms=2000, stateful=True, reset_on_gap=True,
            interpretation='classification', devices=('cpu',),
            sensor_requirements=('16000hz-normalized-pcm', 'explicit-ambient-channel'), license='Apache-2.0')

    @classmethod
    def from_local(cls, path, **kwargs):
        """Load the pinned official MediaPipe YAMNet v1 artifact, without download."""
        path = Path(path).expanduser().resolve()
        if not path.is_file() or not 0 < path.stat().st_size <= 32*1024*1024:
            raise ValueError('supply an existing local YAMNet v1 TFLite file (max 32 MiB)')
        digest = hashlib.sha256()
        with path.open('rb') as source:
            for block in iter(lambda: source.read(1024*1024), b''):
                digest.update(block)
        if digest.hexdigest() != YAMNET_SHA256:
            raise ValueError('YAMNet v1 artifact SHA256 mismatch')
        try:
            from importlib.metadata import version
            if version('mediapipe') != MEDIAPIPE_VERSION:
                raise RuntimeError('YAMNet requires validated mediapipe=='+MEDIAPIPE_VERSION)
            from mediapipe.tasks import python
            from mediapipe.tasks.python import audio
            from mediapipe.tasks.python.components.containers.audio_data import AudioData
        except ImportError as exc:
            raise RuntimeError('Install humansense-sdk[environment-audio] in the selected runtime') from exc
        classifier = audio.AudioClassifier.create_from_options(audio.AudioClassifierOptions(
            base_options=python.BaseOptions(model_asset_path=str(path)),
            running_mode=audio.RunningMode.AUDIO_CLIPS, max_results=521))
        try:
            return cls(classifier, lambda array: AudioData.create_from_array(array, sample_rate=16000),
                       version=MODEL_VERSION, **kwargs)
        except BaseException:
            classifier.close()
            raise

    def reset(self):
        self.buffer = []
        self.last_ts = self.expected_end = self.owner = None
        self.since_output = 0

    def close(self):
        if not self.closed:
            self.closed = True
            self.reset()
            self.classifier.close()

    def process(self, frame):
        if self.closed:
            raise RuntimeError('YAMNet provider is closed')
        if frame.modality != 'audio' or frame.sample_rate != 16000 or type(frame.sample_rate) is not int:
            raise ValueError('YAMNet requires 16000 Hz PCM; resample explicitly before submission')
        ts, sensor = frame.timestamp_ms, frame.sensor
        if type(ts) is not int or ts < 0 or self.last_ts is not None and ts <= self.last_ts:
            raise ValueError('audio capture end timestamps must strictly increase')
        if sensor is None:
            raise ValueError('explicit source/sensor metadata is required')
        owner = (frame.subject_id, sensor.source_id, sensor.sensor_id, sensor.stream_epoch, sensor.channels)
        if self.owner is not None and owner != self.owner:
            raise ValueError('reset before switching audio streams or channel layouts')
        data, np = frame.data, self.np
        if isinstance(data, (bytes, bytearray, memoryview)):
            size = data.nbytes if isinstance(data, memoryview) else len(data)
            if size > 32000 or self.channel not in (None, 0) or len(sensor.channels) > 1:
                raise ValueError('byte input must be at most one second of mono PCM16')
            values = _pcm_floats(bytes(data))
        elif isinstance(data, np.ndarray):
            if data.ndim not in (1, 2) or len(data) > 16000:
                raise ValueError('audio block must be at most one second, samples by channels')
            if data.ndim == 2:
                if self.channel is None or not 1 <= data.shape[1] <= 64 or self.channel >= data.shape[1]:
                    raise ValueError('multichannel audio requires explicit channel selection')
                if sensor.channels and len(sensor.channels) != data.shape[1]:
                    raise ValueError('channel names do not match audio layout')
                values = _pcm_floats(data[:, self.channel])
            else:
                if self.channel not in (None, 0) or len(sensor.channels) > 1:
                    raise ValueError('mono audio cannot satisfy array channel configuration')
                values = _pcm_floats(data)
        else:
            if not isinstance(data, (list, tuple)) or len(data) > 16000 or any(isinstance(v, (list, tuple)) for v in data):
                raise ValueError('use a bounded mono sample list or numeric audio array')
            if self.channel not in (None, 0) or len(sensor.channels) > 1:
                raise ValueError('mono list cannot satisfy array channel configuration')
            values = _pcm_floats(data)
        if not values:
            return None
        expected = ts if self.expected_end is None else self.expected_end+len(values)/16
        if abs(ts-expected) > 1:
            self.reset()
            expected = ts
        self.last_ts, self.expected_end, self.owner = ts, expected, owner
        self.buffer = (self.buffer+values)[-15600:]
        self.since_output += len(values)
        if len(self.buffer) < 15600 or self.since_output < self.hop_samples:
            return None
        self.since_output = 0
        try:
            raw = self.classifier.classify(self.make_audio(np.asarray(self.buffer, dtype=np.float32)))
            if len(raw) != 1 or len(raw[0].classifications) != 1:
                raise ValueError('expected one YAMNet score head for one 975ms window')
            categories = raw[0].classifications[0].categories
            if len(categories) != 521:
                raise ValueError('YAMNet must return its complete 521-class vocabulary')
            events, indices = [], set()
            for category in categories:
                index, name, score = category.index, category.category_name, float(category.score)
                if type(index) is not int or not 0 <= index <= 520 or index in indices:
                    raise ValueError('invalid or duplicate YAMNet category index')
                if not isinstance(name, str) or not 1 <= len(name) <= 256 or not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError('invalid YAMNet category label/score')
                indices.add(index)
                if score >= self.threshold:
                    events.append(dict(index=index, label=name, score=score))
            events.sort(key=lambda item: (-item['score'], item['index']))
            return self.spec.validate(dict(events=events[:self.top_k], window=dict(
                start_ms=ts-975, end_ms=ts, sample_rate=16000, sample_count=15600,
                channel=self.channel if self.channel is not None else 0, hop_ms=self.hop_samples/16,
                returned_count=min(len(events), self.top_k), above_threshold_count=len(events),
                score_threshold=self.threshold, exclusive=False, interpretation='event_candidates')))
        except Exception:
            self.reset()
            raise

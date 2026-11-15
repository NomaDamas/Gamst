"""Optional CED-Tiny/CED-Mini AudioSet tagging; YAMNet remains the default.

Implemented adapter; real-weight acceptance pending. CED (ICASSP 2024) source:
https://github.com/RicherMans/CED/tree/4b2149dc9ba1ed25c6348083f82dba03f00a5022
Upstream CED code is GPL-3.0. It MUST remain an explicitly selected optional
extra/worker and is never imported by GAMST default discovery. Review copyleft
obligations before distributing a combined work; model weights, AudioSet labels,
ONNX Runtime and Torch have separate terms. No CED code is vendored here.

The loader accepts the official ``export_onnx.py`` ``feats`` graph and a local
527-label JSON mapping. It never downloads weights or labels.
"""
import hashlib
import json
import math
from pathlib import Path
import re

from ..capabilities import CapabilitySpec, OutputField, identifier
from ..plugins import _pcm_floats

CED_SOURCE_REVISION = '4b2149dc9ba1ed25c6348083f82dba03f00a5022'
CED_LICENSE = 'GPL-3.0 upstream CED code; optional extra only; model/data/runtime terms separate'
CED_MODEL_IDS = {'tiny': 'mispeech/ced-tiny', 'mini': 'mispeech/ced-mini'}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _labels(path):
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError('CED labels must be a readable local JSON document') from exc
    if isinstance(value, dict):
        value = value.get('id2label')
        if isinstance(value, dict):
            if set(value) != {str(index) for index in range(527)}:
                raise ValueError('CED id2label must contain exactly indices 0..526')
            value = [value[str(index)] for index in range(527)]
    if not isinstance(value, list) or len(value) != 527:
        raise ValueError('CED labels must be a 527-entry array or id2label object')
    if any(not isinstance(label, str) or not 1 <= len(label) <= 256 for label in value):
        raise ValueError('CED labels must be bounded nonempty strings')
    return tuple(value)


class _CEDONNXRuntime:
    """Official CED mel convention and ONNX ``feats`` graph, CPU only."""
    def __init__(self, session, frontend, torch):
        self.session, self.frontend, self.torch = session, frontend, torch
        self.closed = False

    def classify(self, waveform):
        if self.closed:
            raise RuntimeError('CED runtime is closed')
        with self.torch.inference_mode():
            features = self.frontend(self.torch.from_numpy(waveform)).unsqueeze(0).numpy()
            output = self.session.run(None, {'feats': features})
        if len(output) != 1 or output[0].shape != (1, 527):
            raise ValueError('CED ONNX must return one [1, 527] probability output')
        return output[0][0]

    def close(self):
        self.closed = True
        self.session = self.frontend = self.torch = None


class CEDEvents:
    """Trailing CED windows with non-exclusive AudioSet event candidates."""
    def __init__(self, classifier, labels, *, model, version, window_ms=5000,
                 hop_ms=1000, top_k=8, score_threshold=.05, channel=None):
        if model not in CED_MODEL_IDS:
            raise ValueError("model must be 'tiny' or 'mini'")
        if type(window_ms) is not int or not 1000 <= window_ms <= 10000:
            raise ValueError('window_ms must be 1000..10000')
        if type(hop_ms) is not int or not 20 <= hop_ms <= window_ms:
            raise ValueError('hop_ms must be 20..window_ms')
        if type(top_k) is not int or not 1 <= top_k <= 32:
            raise ValueError('top_k must be 1..32')
        if type(score_threshold) not in (int, float) or not math.isfinite(score_threshold) or not 0 <= score_threshold <= 1:
            raise ValueError('score_threshold must be in [0, 1]')
        if channel is not None and (type(channel) is not int or not 0 <= channel < 64):
            raise ValueError('channel must be None or 0..63')
        if not callable(getattr(classifier, 'classify', None)) or not callable(getattr(classifier, 'close', None)):
            raise ValueError('an explicit CED classifier with close lifecycle is required')
        if not isinstance(labels, (tuple, list)) or len(labels) != 527 or any(
                not isinstance(label, str) or not 1 <= len(label) <= 256 for label in labels):
            raise ValueError('exactly 527 bounded AudioSet labels are required')
        identifier(version, 'model version')
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError('Install numpy in the selected CED runtime') from exc
        self.np, self.classifier, self.labels = np, classifier, tuple(labels)
        self.model = model
        self.window_ms, self.window_samples = window_ms, window_ms*16
        self.hop_samples, self.top_k = hop_ms*16, top_k
        self.threshold, self.channel = score_threshold, channel
        self.spec = self.capability(model=model, version=version, window_ms=window_ms, hop_ms=hop_ms)
        self.closed = False
        self.reset()

    @staticmethod
    def capability(*, model, version, window_ms=5000, hop_ms=1000):
        if model not in CED_MODEL_IDS:
            raise ValueError("model must be 'tiny' or 'mini'")
        identifier(version, 'model version')
        if type(window_ms) is not int or not 1000 <= window_ms <= 10000:
            raise ValueError('window_ms must be 1000..10000')
        if type(hop_ms) is not int or not 20 <= hop_ms <= window_ms:
            raise ValueError('hop_ms must be 20..window_ms')
        return CapabilitySpec('audio.ced_'+model+'.events', 'audio.environment_events', 'audio',
            'ced.onnx', version, (OutputField('events', 'array'), OutputField('window', 'object')),
            'window', window_ms=window_ms, freshness_ms=max(2000, 2*hop_ms), stateful=True,
            reset_on_gap=True, interpretation='classification', devices=('cpu',),
            sensor_requirements=('16000hz-normalized-pcm', 'explicit-ambient-channel'),
            license=CED_LICENSE)

    @staticmethod
    def artifact_version(model, sha256, labels_sha256):
        if model not in CED_MODEL_IDS:
            raise ValueError("model must be 'tiny' or 'mini'")
        for value in (sha256, labels_sha256):
            if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
                raise ValueError('supply lowercase CED ONNX and label SHA256 values')
        combined = hashlib.sha256((sha256+':'+labels_sha256).encode()).hexdigest()
        return 'ced_'+model+'.'+CED_SOURCE_REVISION[:12]+'.onnx.sha256.'+combined

    @classmethod
    def verify_local(cls, path, labels_path, sha256, labels_sha256, model='tiny'):
        version = cls.artifact_version(model, sha256, labels_sha256)
        path = Path(path).expanduser().resolve()
        labels_path = Path(labels_path).expanduser().resolve()
        if not path.is_file() or not 0 < path.stat().st_size <= 512*1024*1024:
            raise ValueError('supply an existing local CED ONNX file, at most 512 MiB')
        if not labels_path.is_file() or not 0 < labels_path.stat().st_size <= 1024*1024:
            raise ValueError('supply an existing local CED label JSON file, at most 1 MiB')
        if _sha256(path) != sha256:
            raise ValueError('CED ONNX SHA256 mismatch')
        if _sha256(labels_path) != labels_sha256:
            raise ValueError('CED label JSON SHA256 mismatch')
        return path, labels_path, _labels(labels_path), version

    @classmethod
    def from_local(cls, path, labels_path, sha256, labels_sha256, model='tiny', **kwargs):
        """Load an official local CED ONNX export without hub/network APIs."""
        path, labels_path, labels, version = cls.verify_local(
            path, labels_path, sha256, labels_sha256, model)
        try:
            import numpy as np
            import onnxruntime as ort
            import torch
            import torchaudio.transforms as audio_transforms
        except ImportError as exc:
            raise RuntimeError('Install the optional CED runtime (numpy, onnxruntime, torch, torchaudio)') from exc
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        try:
            session = ort.InferenceSession(str(path), sess_options=options,
                providers=['CPUExecutionProvider'])
            frontend = torch.nn.Sequential(
                audio_transforms.MelSpectrogram(f_min=0, sample_rate=16000,
                    win_length=512, center=False, n_fft=512, f_max=8000,
                    hop_length=160, n_mels=64),
                audio_transforms.AmplitudeToDB(top_db=120))
            runtime = _CEDONNXRuntime(session, frontend, torch)
            return cls(runtime, labels, model=model, version=version, **kwargs)
        except (OSError, RuntimeError) as exc:
            raise RuntimeError('CED ONNX runtime/model loading failed; use the official feats export') from exc

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
            raise RuntimeError('CED provider is closed')
        if frame.modality != 'audio' or frame.sample_rate != 16000 or type(frame.sample_rate) is not int:
            raise ValueError('CED requires 16000 Hz PCM; resample explicitly before submission')
        ts, sensor = frame.timestamp_ms, frame.sensor
        if type(ts) is not int or ts < 0 or self.last_ts is not None and ts <= self.last_ts:
            raise ValueError('audio capture end timestamps must strictly increase')
        if sensor is None:
            raise ValueError('explicit source/sensor metadata is required')
        owner = (frame.subject_id, sensor.source_id, sensor.sensor_id, sensor.stream_epoch, sensor.channels)
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
        channel_count = data.shape[1] if isinstance(data, np.ndarray) and data.ndim == 2 else 1
        owner += (channel_count,)
        if self.owner is not None and owner != self.owner:
            raise ValueError('reset before switching audio streams or channel layouts')
        if not values:
            return None
        expected = ts if self.expected_end is None else self.expected_end+len(values)/16
        if abs(ts-expected) > 1:
            self.reset()
            expected = ts
        self.last_ts, self.expected_end, self.owner = ts, expected, owner
        self.buffer = (self.buffer+values)[-self.window_samples:]
        self.since_output += len(values)
        if len(self.buffer) < self.window_samples or self.since_output < self.hop_samples:
            return None
        self.since_output = 0
        try:
            raw = self.classifier.classify(np.asarray(self.buffer, dtype=np.float32))
            if not isinstance(raw, np.ndarray) or raw.shape != (527,):
                raise ValueError('CED must return its complete 527-class probability vector')
            events = []
            for index, value in enumerate(raw):
                score = float(value)
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError('invalid CED category score')
                if score >= self.threshold:
                    events.append(dict(index=index, label=self.labels[index], score=score))
            events.sort(key=lambda item: (-item['score'], item['index']))
            return self.spec.validate(dict(events=events[:self.top_k], window=dict(
                start_ms=ts-self.window_ms, end_ms=ts, sample_rate=16000,
                sample_count=self.window_samples, channel=self.channel if self.channel is not None else 0,
                hop_ms=self.hop_samples/16, returned_count=min(len(events), self.top_k),
                above_threshold_count=len(events), score_threshold=self.threshold,
                model=CED_MODEL_IDS[self.model], ontology='AudioSet-527', exclusive=False,
                interpretation='event_candidates')))
        except Exception:
            self.reset()
            raise

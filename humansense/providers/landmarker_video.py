"""Stateful local MediaPipe VIDEO inference with explicit continuity boundaries."""
from dataclasses import replace
import hashlib
from pathlib import Path
import threading
from types import SimpleNamespace
import uuid

from ..capabilities import SensorMetadata, identifier
from .body_context import MediaPipeBodyContext
from .hand_context import MediaPipeHandContext
from .face_context import MediaPipeFaceContext

_FORMATTERS = {'body': (MediaPipeBodyContext, 'max_poses'),
               'hand': (MediaPipeHandContext, 'max_hands'),
               'face': (MediaPipeFaceContext, 'max_faces')}


class MediaPipeVideoContext:
    def __init__(self, kind, detector_factory, image_api, *, sha256,
                 max_entities=4, max_gap_ms=1000):
        self.spec = self.capability(kind, sha256, max_entities=max_entities, max_gap_ms=max_gap_ms)
        if not callable(detector_factory):
            raise ValueError('a fresh VIDEO detector factory is required')
        self.factory = detector_factory
        self.kind, self.max_gap_ms = kind, max_gap_ms
        self.lock = threading.RLock()
        self.closed, self.detector = False, None
        self.owner = self.last_sequence = self.last_ts = self.shape = None
        self.origin_ts = self.current_ts = self.segment_id = None
        formatter, count_key = _FORMATTERS[kind]
        # Reuse bounded geometry conversion; this proxy owns no native resource.
        self.formatter = formatter(SimpleNamespace(detect=self._detect, close=lambda: None),
                                   image_api, sha256=sha256, **{count_key: max_entities})

    @staticmethod
    def capability(kind, sha256, *, max_entities=4, max_gap_ms=1000):
        if type(kind) is not str or kind not in _FORMATTERS:
            raise ValueError('kind must be body, hand or face')
        if type(max_gap_ms) is not int or not 1 <= max_gap_ms <= 60000:
            raise ValueError('max_gap_ms must be integer milliseconds in 1..60000')
        formatter, count_key = _FORMATTERS[kind]
        base = formatter.capability(sha256, **{count_key: max_entities})
        return replace(base, id=f'vision.mediapipe.{kind}_video_context',
            provider=f'mediapipe.{kind}.video', version=base.version+'.g'+str(max_gap_ms),
            streaming_mode='native', stateful=True, reset_on_gap=True)

    @classmethod
    def from_local(cls, kind, path, sha256, *, max_entities=4, max_gap_ms=1000):
        cls.capability(kind, sha256, max_entities=max_entities, max_gap_ms=max_gap_ms)
        path = Path(path).expanduser()
        if not path.is_absolute() or not path.is_file():
            raise ValueError('absolute existing local task path required')
        with path.open('rb') as source:
            payload = source.read(64*1024*1024+1)
        if not 0 < len(payload) <= 64*1024*1024 or hashlib.sha256(payload).hexdigest() != sha256:
            raise ValueError('task must match SHA256 and be at most 64 MiB')
        try:
            import importlib.metadata
            if importlib.metadata.version('mediapipe') != '0.10.35':
                raise RuntimeError('select a runtime with mediapipe==0.10.35')
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
            raise RuntimeError('install mediapipe==0.10.35 in the explicitly selected runtime') from exc
        name, count_key = {'body': ('Pose', 'num_poses'), 'hand': ('Hand', 'num_hands'),
                           'face': ('Face', 'num_faces')}[kind]
        options_type = getattr(vision, name+'LandmarkerOptions')
        detector_type = getattr(vision, name+'Landmarker')
        extra = {'output_segmentation_masks': False} if kind == 'body' else (
            {'output_face_blendshapes': True, 'output_facial_transformation_matrixes': False}
            if kind == 'face' else {})
        def factory():
            return detector_type.create_from_options(options_type(
                base_options=python.BaseOptions(model_asset_buffer=payload, delegate=python.BaseOptions.Delegate.CPU),
                running_mode=vision.RunningMode.VIDEO, **{count_key: max_entities}, **extra))
        return cls(kind, factory, mp, sha256=sha256, max_entities=max_entities, max_gap_ms=max_gap_ms)

    def _detect(self, image):
        if self.detector is None:
            detector = self.factory()
            if not callable(getattr(detector, 'detect_for_video', None)) or not callable(getattr(detector, 'close', None)):
                if callable(getattr(detector, 'close', None)): detector.close()
                raise ValueError('factory must return a fresh VIDEO detector with close')
            self.detector = detector
            self.origin_ts, self.segment_id = self.current_ts, uuid.uuid4().hex
        return self.detector.detect_for_video(image, self.current_ts-self.origin_ts)

    def _reset(self):
        detector, self.detector = self.detector, None
        self.owner = self.last_sequence = self.last_ts = self.shape = None
        self.origin_ts = self.current_ts = self.segment_id = None
        if detector is not None:
            try:
                detector.close()
            except BaseException:
                # Failed native teardown is not a recoverable continuity reset.
                self.closed = True
                self.factory = None
                raise

    def reset(self):
        with self.lock:
            if self.closed: raise RuntimeError('video provider is closed')
            self._reset()

    def close(self):
        with self.lock:
            if not self.closed:
                self.closed = True
                try:
                    self._reset()
                finally:
                    self.factory = None  # release the retained verified model bytes
                    self.formatter.close()

    def process(self, frame):
        with self.lock:
            if self.closed: raise RuntimeError('video provider is closed')
            try:
                sensor, ts = frame.sensor, frame.timestamp_ms
                if not isinstance(sensor, SensorMetadata):
                    raise ValueError('video inference requires capture SensorMetadata')
                identifier(frame.subject_id, 'subject_id')
                if type(ts) is not int or not 0 <= ts <= 2**53-1:
                    raise ValueError('capture timestamp must be nonnegative integer milliseconds at most 2**53-1')
                owner = (frame.subject_id, sensor.source_id, sensor.sensor_id, sensor.stream_epoch,
                         sensor.coordinate_frame, sensor.calibration_id, sensor.channels)
                shape = getattr(frame.data, 'shape', None)
                if self.owner is not None:
                    if owner == self.owner and (sensor.sequence <= self.last_sequence or ts <= self.last_ts):
                        raise ValueError('capture sequence and timestamps must strictly increase')
                    if (owner != self.owner or sensor.sequence != self.last_sequence+1 or
                            ts-self.last_ts > self.max_gap_ms or shape != self.shape):
                        self._reset()
                self.current_ts = ts
                result = self.formatter.process(frame)
                result['summary'].update(running_mode='VIDEO', temporal_tracking=True,
                    persistent_track_ids='unavailable', smoothing='runtime_default',
                    state_segment=self.segment_id, native_timestamp_ms=ts-self.origin_ts)
                result = self.spec.validate(result)
                self.owner, self.last_sequence, self.last_ts, self.shape = owner, sensor.sequence, ts, shape
                return result
            except BaseException:
                self._reset()
                raise

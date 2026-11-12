"""Local ONNX adapters with explicit preprocessing/output contracts.

No model downloads. Metadata describes an interface, not model accuracy or
license suitability. Arbitrary raw YOLO/SAM exports are not auto-decoded.
"""
from __future__ import annotations
import asyncio
import hashlib
import math
from pathlib import Path
import threading
from .plugins import PerceptionPlugin, _signal, _version
from .tracking import PersonTracker


def _model(path, providers, contract):
    path = Path(path)
    if not path.is_file() or path.suffix.lower() != '.onnx':
        raise ValueError('Supply an explicit local .onnx model file')
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError as exc:
        raise RuntimeError('Install humansense-sdk[onnx]') from exc
    selected = ['CPUExecutionProvider'] if providers is None else list(providers)
    if not selected or any(not isinstance(p, str) for p in selected):
        raise ValueError('providers must be non-empty ONNX provider names')
    if set(selected) - set(ort.get_available_providers()):
        raise ValueError('Requested ONNX execution providers are unavailable')
    session = ort.InferenceSession(str(path), providers=selected)
    if session.get_modelmeta().custom_metadata_map.get('humansense.contract') != contract:
        raise ValueError(f'Model metadata must declare humansense.contract={contract}')
    return session, np, digest.hexdigest()


class _LocalPlugin(PerceptionPlugin):
    def _setup(self):
        self._native_lock = threading.Lock()
        self._generation = 0
        self._closed = False

    def reset(self):
        self._generation += 1

    async def close(self):
        self.reset()
        def release():
            with self._native_lock:
                self._closed = True
                self.session = None
        await asyncio.to_thread(release)

    async def _run(self, function):
        generation = self._generation
        def guarded():
            with self._native_lock:
                if self._closed:
                    raise RuntimeError(f'{self.name} is closed')
                if generation != self._generation:
                    return None
                return function()
        result = await asyncio.to_thread(guarded)
        return result if generation == self._generation else None

    @property
    def model_version(self):
        return f"{_version('onnxruntime')}:sha256:{self.model_hash}"


class ONNXPersonPlugin(_LocalPlugin):
    """Postprocessed person detector; explicitly exported model required.

    Metadata: humansense.contract=person_xyxy_score_class_v1. One float32 RGB
    input [1,3,H,W], values [0,1], stretched resize to input_size=(width,height),
    with no letterbox. One output [N,6] or [1,N,6], each row x1,y1,x2,y2 in
    resized input pixels, score [0,1], integer class ID. Model includes NMS.
    Geometric tracks never establish biometric identity or speaker ownership.
    """
    name = 'onnx_person'
    modalities = ('video',)

    def __init__(self, model_path, input_size=(640, 640), person_class_id=0,
                 score_threshold=.5, providers=None):
        if not isinstance(input_size, (tuple, list)) or len(input_size) != 2 or any(
                isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 8192 for v in input_size):
            raise ValueError('input_size must be positive integer (width,height), <=8192')
        if isinstance(person_class_id, bool) or not isinstance(person_class_id, int) or person_class_id < 0:
            raise ValueError('person_class_id must be a nonnegative integer')
        if isinstance(score_threshold, bool) or not math.isfinite(score_threshold) or not 0 <= score_threshold <= 1:
            raise ValueError('score_threshold must be in [0,1]')
        self.session, self.np, self.model_hash = _model(model_path, providers, 'person_xyxy_score_class_v1')
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError('Install humansense-sdk[onnx] for image preprocessing') from exc
        self.cv2 = cv2
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].type != 'tensor(float)' or len(inputs[0].shape) != 4:
            raise ValueError('Person model requires one float32 NCHW input')
        width, height = input_size
        if any(isinstance(actual, int) and actual != expected
               for actual, expected in zip(inputs[0].shape, [1, 3, height, width])):
            raise ValueError('Model shape does not match [1,3,height,width]')
        if len(outputs) != 1:
            raise ValueError('Person model requires one postprocessed output')
        self.input_name, self.output_name = inputs[0].name, outputs[0].name
        self.input_size = tuple(input_size)
        self.person_class_id, self.score_threshold = person_class_id, score_threshold
        self.tracker = PersonTracker()
        self._setup()

    def _detect(self, image):
        np = self.np
        if getattr(image, 'ndim', None) != 3 or image.shape[2] != 3 or image.shape[0] <= 0 or image.shape[1] <= 0:
            raise ValueError('Video must be a non-empty HWC BGR image')
        if image.dtype != np.uint8:
            raise ValueError('Video must use uint8 BGR pixels')
        resized = self.cv2.resize(image, self.input_size)
        tensor = np.ascontiguousarray(resized[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.
        raw = self.session.run([self.output_name], {self.input_name: tensor})
        if len(raw) != 1:
            raise ValueError('Unexpected model output count')
        return self._parse(raw[0])

    def _parse(self, raw):
        values = self.np.asarray(raw)
        if values.ndim == 3 and values.shape[0] == 1:
            values = values[0]
        if values.ndim != 2 or values.shape[1] != 6:
            raise ValueError('Expected postprocessed Nx6 or1xNx6; raw detection heads unsupported')
        if values.shape[0] > 10000 or not self.np.isfinite(values).all():
            raise ValueError('Detections must be finite with at most 10000 rows')
        width, height = self.input_size
        boxes, scores = [], []
        for row in values.tolist():
            x1, y1, x2, y2, score, category = [float(v) for v in row]
            if not 0 <= score <= 1 or category < 0 or not category.is_integer():
                raise ValueError('Detection score must be [0,1] and class ID an integer')
            if score < self.score_threshold or category != self.person_class_id:
                continue
            if x2 <= x1 or y2 <= y1:
                raise ValueError('Selected detection boxes must have positive area')
            x1, x2 = max(0., min(1., x1 / width)), max(0., min(1., x2 / width))
            y1, y2 = max(0., min(1., y1 / height)), max(0., min(1., y2 / height))
            if x2 > x1 and y2 > y1:
                boxes.append(dict(x=x1, y=y1, width=x2-x1, height=y2-y1))
                scores.append(score)
        return boxes, scores

    async def process(self, frame):
        result = await self._run(lambda: self._detect(frame.data))
        if result is None:
            return []
        boxes, scores = result
        tracks = self.tracker.update(boxes, frame.timestamp_ms, stream_id=frame.subject_id)
        return [_signal(frame, name, value, self.name, self.model_version) for name, value in (
            ('scene.person_count', len(boxes)), ('scene.person_boxes', boxes),
            ('scene.person_detection_scores', dict(model_scores=scores, probability=None, calibrated=False)),
            ('scene.person_tracks', tracks))]

    def reset(self):
        super().reset()
        self.tracker.reset()


class ONNXFeaturePlugin(_LocalPlugin):
    """Explicit expression/acoustic model interface; no invented semantic labels.

    Metadata: humansense.contract=feature_model_scores_v1. preprocessor(frame)
    returns the complete model input-feed dict. output_labels maps ONNX output
    names to signal-name lists. Selected outputs must be [K] or [1,K] finite
    scores in [0,1]. Signals contain model_score, probability=None, calibrated=False.
    Labels must reflect the model's documented target, not inner emotion claims.
    """
    name = 'onnx_features'

    def __init__(self, model_path, *, modality, output_labels, preprocessor, providers=None):
        if modality not in ('video', 'audio', 'text') or not callable(preprocessor):
            raise ValueError('Explicit modality and callable preprocessor required')
        if not isinstance(output_labels, dict) or not output_labels:
            raise ValueError('output_labels must map ONNX output names to signal-name lists')
        labels = []
        for output, names in output_labels.items():
            if not isinstance(output, str) or not isinstance(names, (tuple, list)) or not names or any(
                    not isinstance(name, str) or not name.strip() for name in names):
                raise ValueError('Each output must declare a non-empty signal-name list')
            labels.extend(names)
        if len(labels) != len(set(labels)):
            raise ValueError('Declared signal names must be unique')
        self.session, self.np, self.model_hash = _model(model_path, providers, 'feature_model_scores_v1')
        if not set(output_labels) <= {output.name for output in self.session.get_outputs()}:
            raise ValueError('Declared output names do not exist in model')
        self.input_names = {node.name for node in self.session.get_inputs()}
        self.output_labels = {name: tuple(labels) for name, labels in output_labels.items()}
        self.preprocessor, self.modalities = preprocessor, (modality,)
        self._setup()

    def _predict(self, frame):
        feed = self.preprocessor(frame)
        if not isinstance(feed, dict) or set(feed) != self.input_names:
            raise ValueError('Preprocessor must return exactly the model input names')
        names = list(self.output_labels)
        outputs = self.session.run(names, feed)
        if len(outputs) != len(names):
            raise ValueError('Unexpected model output count')
        predictions = []
        for name, raw in zip(names, outputs):
            values = self.np.asarray(raw)
            if values.ndim == 2 and values.shape[0] == 1:
                values = values[0]
            if values.ndim != 1 or values.shape[0] != len(self.output_labels[name]):
                raise ValueError('Feature shape must match explicitly declared label vector')
            if not self.np.isfinite(values).all():
                raise ValueError('Feature scores must be finite')
            for label, value in zip(self.output_labels[name], values.tolist()):
                score = float(value)
                if not 0 <= score <= 1:
                    raise ValueError('Feature model scores must be in [0,1]')
                predictions.append((label, score))
        return predictions

    async def process(self, frame):
        if frame.modality not in self.modalities:
            raise ValueError('Frame modality differs from declared model modality')
        predictions = await self._run(lambda: self._predict(frame))
        if predictions is None:
            return []
        return [_signal(frame, name, dict(model_score=score, probability=None, calibrated=False),
                        self.name, self.model_version) for name, score in predictions]

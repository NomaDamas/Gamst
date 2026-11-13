"""Local RF-DETR frame inference with bounded, conservative object continuity."""
import hashlib
import math
from pathlib import Path
import re

from ..capabilities import CapabilitySpec, OutputField, identifier
from ..tracking import intersection_over_union


class RFDETRObjects:
    """Use one instance per stream; SensorStreams serializes native calls.

    The detector is frame based. IoU continuity is stateful, class constrained,
    and never establishes physical identity or a location outside the image.
    """
    def __init__(self, predictor, *, version, threshold=.5, max_objects=32,
                 max_age_ms=1000, iou_threshold=.3, ambiguity_margin=.1):
        if not callable(getattr(predictor, 'predict', None)):
            raise ValueError('predictor must expose RF-DETR predict')
        identifier(version, 'model version')
        for value in (threshold, iou_threshold, ambiguity_margin):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('thresholds must be finite numbers')
        if not 0 < threshold <= 1 or not 0 < iou_threshold <= 1 or not 0 <= ambiguity_margin <= 1:
            raise ValueError('invalid detection/association threshold')
        if type(max_objects) is not int or not 1 <= max_objects <= 64:
            raise ValueError('max_objects must be 1..64')
        if type(max_age_ms) is not int or not 1 <= max_age_ms <= 10000:
            raise ValueError('max_age_ms must be 1..10000')
        self.predictor, self.version = predictor, version
        self.threshold, self.max_objects, self.max_age_ms = threshold, max_objects, max_age_ms
        self.iou_threshold, self.ambiguity_margin = iou_threshold, ambiguity_margin
        self.counter, self.closed = 0, False
        self.reset()

    @staticmethod
    def capability(version='installed'):
        return CapabilitySpec('vision.rfdetr.objects', 'vision.objects', 'video',
            'rfdetr.nano', version, (OutputField('objects', 'array'),
            OutputField('summary', 'object')), 'frame', stateful=True,
            reset_on_gap=True, freshness_ms=1000, devices=('cpu', 'cuda', 'mps'),
            sensor_requirements=('uint8-bgr-camera', 'normalized-image-coordinate-frame'),
            license='Apache-2.0', interpretation='estimate')

    @classmethod
    def from_local(cls, path, *, sha256, device='cpu', **kwargs):
        """Explicit local loading only; checkpoint digest is required provenance."""
        path = Path(path).expanduser().resolve()
        if not path.is_file() or not isinstance(sha256, str) or not re.fullmatch('[0-9a-f]{64}', sha256):
            raise ValueError('existing checkpoint and lowercase SHA256 required')
        digest = hashlib.sha256()
        with path.open('rb') as source:
            for block in iter(lambda: source.read(1024*1024), b''):
                digest.update(block)
        if digest.hexdigest() != sha256:
            raise ValueError('checkpoint SHA256 mismatch')
        if device not in ('cpu', 'cuda', 'mps'):
            raise ValueError('device must be cpu, cuda or mps')
        try:
            from importlib.metadata import version
            installed = version('rfdetr')
            if installed != '1.12.1':
                raise RuntimeError('RF-DETR adapter requires validated rfdetr==1.12.1')
            from rfdetr import RFDETRNano
        except ImportError as exc:
            raise RuntimeError('Install humansense-sdk[detection] in a dedicated runtime') from exc
        model = RFDETRNano(pretrain_weights=str(path), device=device)
        # An inference-only snapshot folds supported operations without compile
        # caches or CUDA graph side effects. Release the training/source model.
        predictor = model.inference(compile=False)
        return cls(predictor, version=installed+'.eager.sha256.'+sha256, **kwargs)

    def reset(self):
        self.tracks = {}
        self.last_ts = self.shape = self.owner = None

    def close(self):
        self.reset()
        self.predictor = None
        self.closed = True

    def process(self, frame):
        if self.closed:
            raise RuntimeError('object provider is closed')
        import numpy as np
        data, ts, sensor = frame.data, frame.timestamp_ms, frame.sensor
        if frame.modality != 'video' or not isinstance(data, np.ndarray) or data.dtype != np.uint8 or data.ndim != 3 or data.shape[2] != 3:
            raise ValueError('detector requires uint8 BGR image')
        h, w = data.shape[:2]
        if min(h, w) < 16 or max(h, w) > 4096 or data.nbytes > 16*1024*1024:
            raise ValueError('image dimensions exceed detector limits')
        if type(ts) is not int or ts < 0 or self.last_ts is not None and ts <= self.last_ts:
            raise ValueError('capture timestamps must strictly increase')
        if sensor is None or not sensor.coordinate_frame:
            raise ValueError('declare the normalized image coordinate frame')
        owner = (frame.subject_id, sensor.source_id, sensor.sensor_id, sensor.coordinate_frame, sensor.stream_epoch)
        if self.owner is not None and owner != self.owner:
            raise ValueError('reset before switching detector streams or coordinate frames')
        result = self.predictor.predict(data[:, :, ::-1].copy(), threshold=self.threshold,
                                        include_source_image=False)
        boxes, scores, classes = result.xyxy, result.confidence, result.class_id
        labels = result.data.get('class_name')
        if labels is None or not len(boxes) == len(scores) == len(classes) == len(labels) or len(boxes) > 4096:
            raise ValueError('invalid detector output layout')
        rows = []
        for raw, score, class_id, label in zip(boxes, scores, classes, labels):
            score = float(score)
            coords = [float(v) for v in raw]
            if len(coords) != 4 or not all(math.isfinite(v) for v in coords+[score]) or not 0 <= score <= 1:
                raise ValueError('invalid detector box or score')
            if not isinstance(class_id, (int, np.integer)) or isinstance(class_id, (bool, np.bool_)) or not 0 <= class_id <= 100000:
                raise ValueError('invalid detector class ID')
            if not isinstance(label, str) or not 1 <= len(label) <= 256:
                raise ValueError('invalid detector label')
            x0, y0, x1, y1 = coords
            x0, x1 = max(0., min(1., x0/w)), max(0., min(1., x1/w))
            y0, y1 = max(0., min(1., y0/h)), max(0., min(1., y1/h))
            if score < self.threshold or x1 <= x0 or y1 <= y0:
                continue
            rows.append(dict(label=str(label), class_id=int(class_id), score=score,
                             bbox=dict(x=x0, y=y0, width=x1-x0, height=y1-y0)))
        rows.sort(key=lambda row: -row['score'])
        count = len(rows)
        rows = rows[:self.max_objects]
        # Inference/validation precede state mutation. Image resizing is a break
        # in camera geometry, so existing associations cannot cross it.
        old = {} if self.shape != (h, w) else {key: value for key, value in self.tracks.items()
                                              if ts-value['last_seen_ms'] <= self.max_age_ms}
        candidates = []
        for index, row in enumerate(rows):
            for track_id, track in old.items():
                if row['class_id'] == track['class_id'] and row['label'] == track['label']:
                    iou = intersection_over_union(row['bbox'], track['bbox'])
                    if iou >= self.iou_threshold:
                        candidates.append((iou, index, track_id))
        ambiguous_rows, ambiguous_tracks = set(), set()
        for group in ([c for c in candidates if c[1] == i] for i in range(len(rows))):
            group.sort(reverse=True)
            if len(group) > 1 and group[0][0]-group[1][0] <= self.ambiguity_margin:
                ambiguous_rows.update(c[1] for c in group)
                ambiguous_tracks.update(c[2] for c in group)
        for track_id in old:
            group = sorted((c for c in candidates if c[2] == track_id), reverse=True)
            if len(group) > 1 and group[0][0]-group[1][0] <= self.ambiguity_margin:
                ambiguous_rows.update(c[1] for c in group)
                ambiguous_tracks.add(track_id)
        matched, used = {}, set()
        for _, index, track_id in sorted(candidates, reverse=True):
            if index not in ambiguous_rows and track_id not in ambiguous_tracks and index not in matched and track_id not in used:
                matched[index] = track_id
                used.add(track_id)
        objects = []
        # Retire ambiguous histories instead of later reconnecting through them.
        updated = {key: value for key, value in old.items() if key not in ambiguous_tracks}
        for index, row in enumerate(rows):
            track_id = matched.get(index)
            quality = 'matched' if track_id else 'uncertain_new' if index in ambiguous_rows else 'new'
            if track_id is None:
                self.counter += 1
                track_id = 'object-'+str(self.counter)
            updated[track_id] = dict(row, last_seen_ms=ts)
            objects.append(dict(track_id=track_id, label=row['label'],
                attributes=dict(class_id=row['class_id'], score=row['score'], association=quality,
                                identity_verified=False),
                location=dict(bbox=row['bbox'], unit='normalized_image')))
        # Retain a bounded number of missing associations, prioritizing visible.
        visible = {item['track_id'] for item in objects}
        ordered = sorted(updated, key=lambda key: (key in visible, updated[key]['last_seen_ms']), reverse=True)
        output = self.capability(self.version).validate(dict(objects=objects, summary=dict(detected_count=count,
            returned_count=len(objects), truncated=count > len(objects),
            complete_snapshot=count <= len(objects), tracking='class_constrained_iou',
            detector_mode='frame', width=w, height=h)))
        self.tracks = {key: updated[key] for key in ordered[:self.max_objects]}
        self.last_ts, self.shape, self.owner = ts, (h, w), owner
        return output

"""Capture-bound seeding and compact causal mask context."""
from dataclasses import replace
import hashlib
import json
import math
import threading
import uuid

from ..capabilities import CapabilitySpec, OutputField, bounded_json, identifier
from .efficienttam_stream import UPSTREAM_REVISION, _image
from .region_appearance import capture_binding
from .yoloe import _hash, mask_context, prompt_labels


class EfficientTAMContext:
    def __init__(self, backend, *, checkpoint_sha256, device='cpu', max_gap_ms=1000,
                 max_frames=1000000, artifact_verified=False):
        self.spec = self.capability(checkpoint_sha256=checkpoint_sha256, device=device,
            max_gap_ms=max_gap_ms, max_frames=max_frames)
        if any(not callable(getattr(backend, name, None)) for name in ('step', 'reset', 'close')):
            raise ValueError('owned streaming backend with step/reset/close required')
        if type(artifact_verified) is not bool: raise ValueError('artifact_verified must be boolean')
        self.backend, self.verified = backend, artifact_verified
        self.checkpoint_sha256, self.max_gap_ms, self.max_frames = checkpoint_sha256, max_gap_ms, max_frames
        self.lock = threading.RLock()
        self.closed = False
        self._clear()

    @staticmethod
    def capability(*, checkpoint_sha256, device='cpu', max_gap_ms=1000, max_frames=1000000):
        _hash(checkpoint_sha256)
        if type(device) is not str or device not in ('cpu', 'cuda'): raise ValueError('cpu or cuda required')
        if type(max_gap_ms) is not int or not 1 <= max_gap_ms <= 60000: raise ValueError('max_gap_ms must be1..60000')
        if type(max_frames) is not int or not 1 <= max_frames <= 1000000: raise ValueError('max_frames must be1..1000000')
        digest = hashlib.sha256(json.dumps([UPSTREAM_REVISION, checkpoint_sha256, device, max_gap_ms, max_frames]).encode()).hexdigest()
        return CapabilitySpec('vision.efficienttam.tracking', 'vision.mask_tracks', 'video',
            'efficienttam.tiny512', '1.'+digest,
            (OutputField('objects', 'array'), OutputField('summary', 'object')),
            'native', stateful=True, reset_on_gap=True, freshness_ms=1000,
            interpretation='estimate', devices=(device,),
            sensor_requirements=('uint8-bgr-camera', 'declared-image-coordinate-frame', 'exact-capture-box-seed'),
            license='adapter:Apache-2.0;EfficientTAM-code:Apache-2.0;checkpoint:verify-separately')

    def _clear(self):
        self.owner = self.last = self.binding = self.segment = self.seed = None
        self.rows = ()
        self.frames = 0

    def _reset(self):
        self._clear()
        try: self.backend.reset()
        except BaseException:
            self.closed = True
            self.backend.close()
            raise

    def reset(self):
        with self.lock:
            if self.closed: raise RuntimeError('EfficientTAM context closed')
            self._reset()

    def close(self):
        with self.lock:
            if not self.closed:
                self.closed = True
                self._clear()
                self.backend.close()

    def process(self, frame):
        return self._process(frame, None)

    def process_regions(self, frame, evidence):
        if evidence is None:
            with self.lock:
                if self.closed: raise RuntimeError('EfficientTAM context closed')
                self._reset()
                raise ValueError('exact-capture seed evidence required')
        return self._process(frame, evidence)

    def _process(self, frame, evidence):
        with self.lock:
            if self.closed: raise RuntimeError('EfficientTAM context closed')
            try:
                import numpy as np
                frame = replace(frame, data=_image(frame.data))
                binding = capture_binding(frame)
                sensor, ts = frame.sensor, frame.timestamp_ms
                seq = sensor.capture_sequence if sensor.capture_sequence is not None else sensor.sequence
                owner = (frame.subject_id, sensor.source_id, sensor.sensor_id, sensor.stream_epoch,
                    sensor.coordinate_frame, sensor.calibration_id, sensor.channels, sensor.capture_sequence is not None)
                # A needs_prompt response may be followed by a seed on that exact
                # capture. No other repeated capture/timestamp is accepted.
                reseed_same = evidence is not None and self.segment is None and binding == self.binding
                if self.owner == owner and self.last and not reseed_same:
                    if ts <= self.last[0] or seq <= self.last[1]: raise ValueError('capture time/sequence must advance')
                if self.last and (owner != self.owner or (not reseed_same and seq != self.last[1]+1) or
                        ts-self.last[0] > self.max_gap_ms or binding['shape'] != self.binding['shape']):
                    self._reset()
                if self.frames >= self.max_frames: self._reset()
                boxes = None
                if evidence is not None:
                    evidence = bounded_json(evidence)
                    if (type(evidence) is not dict or set(evidence) != {'capture','objects','detector_id','detector_version','complete_snapshot'} or
                            evidence['capture'] != binding or type(evidence['complete_snapshot']) is not bool):
                        raise ValueError('seed evidence must match exact capture')
                    identifier(evidence['detector_id']); identifier(evidence['detector_version'])
                    rows = evidence['objects']
                    if type(rows) is not list or not 1 <= len(rows) <= 4: raise ValueError('1..4 seed objects required')
                    boxes, identities, normalized = [], set(), []
                    h, w = frame.data.shape[:2]
                    for row in rows:
                        if type(row) is not dict or not {'track_id','label','location'} <= set(row): raise ValueError('seed track/label/location required')
                        track = identifier(row['track_id'])
                        label = prompt_labels([row['label']])[0]
                        if track in identities: raise ValueError('seed track IDs must be unique')
                        identities.add(track)
                        location = row['location']
                        if type(location) is not dict or location.get('unit') != 'normalized_image': raise ValueError('normalized image box required')
                        box = location.get('bbox')
                        if type(box) is not dict or set(box) != {'x','y','width','height'}: raise ValueError('xywh box required')
                        x,y,bw,bh = (box[k] for k in ('x','y','width','height'))
                        if (any(type(v) not in (int,float) or not math.isfinite(v) for v in (x,y,bw,bh)) or
                                min(x,y) < 0 or min(bw,bh) <= 0 or x+bw > 1 or y+bh > 1 or bw*w < 1 or bh*h < 1):
                            raise ValueError('bounded nonempty seed box required')
                        boxes.append([x*w,y*h,(x+bw)*w,(y+bh)*h])
                        normalized.append((track,label))
                    self._reset()
                    self.rows = tuple(normalized)
                    self.segment = uuid.uuid4().hex
                    self.seed = dict(capture_sha256=binding['sha256'],timestamp_ms=ts,
                        detector_id=evidence['detector_id'],detector_version=evidence['detector_version'])
                objects, empty_tracks = [], []
                if self.segment is not None:
                    masks = self.backend.step(frame.data, boxes=boxes)
                    if type(masks) is not np.ndarray or masks.dtype != np.bool_ or masks.shape != (len(self.rows), *frame.data.shape[:2]):
                        raise ValueError('one original-size boolean mask per seed required')
                    h,w = frame.data.shape[:2]
                    for i,(track,label) in enumerate(self.rows):
                        mask = masks[i]; compact = mask_context(mask)
                        ys,xs = np.flatnonzero(mask.any(1)),np.flatnonzero(mask.any(0))
                        bbox = None if not len(xs) else dict(x=int(xs[0])/w,y=int(ys[0])/h,
                            width=int(xs[-1]-xs[0]+1)/w,height=int(ys[-1]-ys[0]+1)/h)
                        if bbox is None:
                            empty_tracks.append(dict(track_id=self.segment+':'+str(i+1),label=label,
                                seed_track_id=track,mask_status='empty',presence_confirmed=False))
                            continue  # No observation: do not refresh object-memory last_seen.
                        item = dict(track_id=self.segment+':'+str(i+1),label=label,
                            attributes=dict(mask=compact,seed_track_id=track,identity_verified=False,
                                mask_status='estimated' if bbox is not None else 'empty',
                                presence_confirmed=False))
                        if bbox is not None: item['location'] = dict(bbox=bbox,unit='normalized_image')
                        objects.append(item)
                    self.frames += 1
                result = self.spec.validate(dict(objects=objects,summary=dict(
                    status='tracking' if self.segment is not None else 'needs_prompt',
                    state_segment=self.segment,seed=self.seed,processed_frames=self.frames,
                    capture_timestamp_ms=ts,capture_sequence=seq,capture_sha256=binding['sha256'],
                    coordinate_frame=sensor.coordinate_frame,complete_snapshot=False,
                    empty_tracks=empty_tracks,
                    identity_verified=False,artifact_integrity_verified=self.verified,model_quality_verified=False,
                    lookahead_ms=0,content_role='untrusted_observed_labels')))
                self.owner,self.last,self.binding = owner,(ts,seq),binding
                return result
            except BaseException:
                self._reset()
                raise

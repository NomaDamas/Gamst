"""Generic per-capture object rotation over GAMST's standard video intake.

Regions can come from any detector observation already in the Store, an
explicit normalized ROI, or detector-free frame motion. Detector labels are
optional evidence, never a requirement for tracking. The semantic scene model
can name the same timeline later; this provider only measures visible motion.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import threading
import uuid

from ..capabilities import CapabilitySpec, OutputField, SensorMetadata, bounded_json
from ..plugins import InputFrame
from .object_rotation_bank import ObjectRotationBank
from .object_spin import ObjectSpinCounter
from .shape_tracks import ShapeTracks
from .target_tracks import TargetTracks


_OPTIONS = {
    'region_source', 'roi', 'targets', 'expected_max_rev_s', 'max_tracks',
    'max_gap_ms', 'search_scale', 'max_points', 'min_quality', 'min_rev_s',
    'idle_ms', 'detector_max_age_ms', 'region_refresh_ms',
}


def _configuration(config):
    config = bounded_json(config)
    if type(config) is not dict or set(config) - _OPTIONS:
        raise ValueError('unknown vision.rotation option')
    values = dict(config)
    values.setdefault('region_source', 'auto')
    values.setdefault('roi', None)
    values.setdefault('targets', None)
    values.setdefault('expected_max_rev_s', 40.0)
    values.setdefault('max_tracks', 8)
    values.setdefault('max_gap_ms', 500)
    values.setdefault('search_scale', 2.2)
    values.setdefault('max_points', 24)
    values.setdefault('min_quality', .25)
    values.setdefault('min_rev_s', .02)
    values.setdefault('idle_ms', 200)
    values.setdefault('detector_max_age_ms', 1000)
    values.setdefault('region_refresh_ms', 250)
    if values['region_source'] not in ('auto', 'detections', 'roi', 'motion'):
        raise ValueError('region_source must be auto, detections, roi or motion')
    if values['region_source'] == 'roi' and values['roi'] is None:
        raise ValueError('region_source=roi requires a normalized ROI')
    roi = values['roi']
    if roi is not None:
        if type(roi) is not dict or set(roi) - {'x', 'y', 'width', 'height', 'label'}:
            raise ValueError('ROI contains unsupported fields')
        for key in ('x', 'y', 'width', 'height'):
            value = roi.get(key)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('ROI coordinates must be finite normalized numbers')
        if (not 0 <= roi['x'] < 1 or not 0 <= roi['y'] < 1 or
                not 0 < roi['width'] <= 1 - roi['x'] or
                not 0 < roi['height'] <= 1 - roi['y']):
            raise ValueError('ROI must be a nonempty normalized image rectangle')
        if 'label' in roi and (not isinstance(roi['label'], str) or not 1 <= len(roi['label']) <= 128):
            raise ValueError('ROI label must be a bounded string')
    targets = values['targets']
    if targets is not None:
        if (type(targets) is not list or not 1 <= len(targets) <= 32 or
                any(not isinstance(label, str) or not 1 <= len(label.strip()) <= 128 for label in targets)):
            raise ValueError('targets must contain 1..32 nonempty labels')
        values['targets'] = [label.strip() for label in targets]
    for key, upper in (('max_tracks', 32), ('max_gap_ms', 5000), ('max_points', 128),
                       ('idle_ms', 5000), ('detector_max_age_ms', 10000),
                       ('region_refresh_ms', 10000)):
        value = values[key]
        minimum = 4 if key == 'max_points' else 1
        if type(value) is not int or not minimum <= value <= upper:
            raise ValueError(f'{key} is outside its integer bound')
    for key in ('expected_max_rev_s', 'search_scale', 'min_quality', 'min_rev_s'):
        value = values[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be positive and finite')
    if not 0 < values['expected_max_rev_s'] <= 1000 or not 1.2 <= values['search_scale'] <= 6:
        raise ValueError('expected_max_rev_s or search_scale is outside its bound')
    if values['min_quality'] > 1 or values['min_rev_s'] > 1000:
        raise ValueError('min_quality must be <=1 and min_rev_s <=1000')
    return values


def _rows(items, targets=None):
    """Extract the common normalized detector-box contract; ignore other objects."""
    if type(items) is not list:
        return []
    accepted = None if targets is None else {label.casefold() for label in targets}
    output = []
    for item in items[:64]:
        if not isinstance(item, dict):
            continue
        box = item.get('bbox')
        if type(box) is not dict or not {'x', 'y', 'width', 'height'} <= set(box):
            continue
        try:
            box = {key: float(box[key]) for key in ('x', 'y', 'width', 'height')}
        except (TypeError, ValueError, OverflowError):
            continue
        if (not all(math.isfinite(value) for value in box.values()) or
                not 0 <= box['x'] < 1 or not 0 <= box['y'] < 1 or
                not 0 < box['width'] <= 1 - box['x'] or
                not 0 < box['height'] <= 1 - box['y']):
            continue
        raw_label = item.get('label')
        label = raw_label.strip() if isinstance(raw_label, str) and raw_label.strip() else 'unknown'
        if accepted is not None and label.casefold() not in accepted:
            continue
        score = item.get('score', 0.)
        if type(score) not in (int, float) or not math.isfinite(score):
            score = 0.
        row = dict(label=label, score=max(0., min(1., float(score))), bbox=box)
        size = item.get('size_px')
        if type(size) in (int, float) and math.isfinite(size) and size > 0:
            row['size_px'] = float(size)
        output.append(row)
    return output


def classify_trajectory(samples):
    """How a rotating object moved: ``free_fall`` (thrown: image-down constant acceleration,
    steady sideways drift), ``stationary`` (spinning in place) or ``moving``.

    ``samples`` are (timestamp_ms, x, y, major_axis) of observed frames. Units are object
    diameters, so the test needs no camera calibration; a 24 mm coin under gravity accelerates
    ~400 diameters/s^2. A flipping rigid disc keeps its major axis while the minor axis
    oscillates, so a varying major axis (hands, bodies, blur envelopes) is never classified as
    thrown or spinning in place.
    """
    import numpy as np
    if len(samples) < 10:
        return dict(kind='unknown', observed_frames=len(samples))
    t = np.array([(sample[0] - samples[0][0]) / 1000 for sample in samples])
    x, y = np.array([sample[1] for sample in samples]), np.array([sample[2] for sample in samples])
    major = np.array([sample[3] for sample in samples])
    size = max(1., float(np.median(major)))
    rigidity = float(np.std(major) / size)
    spread = float(max(np.ptp(x), np.ptp(y))) / size
    common = dict(travel_diameters=round(spread, 2), size_variation=round(rigidity, 3),
                  observed_frames=len(samples))
    if rigidity > .25:
        return dict(kind='moving', **common)
    if spread < .5:
        return dict(kind='stationary', **common)
    fit_y = np.polyfit(t, y, 2)
    residual = float(np.sum((y - np.polyval(fit_y, t)) ** 2))
    total = float(np.sum((y - y.mean()) ** 2)) or 1.
    r2 = 1 - residual / total
    drift = float(np.sqrt(np.mean((x - np.polyval(np.polyfit(t, x, 1), t)) ** 2))) / size
    accel = 2 * float(fit_y[0]) / size
    free_fall = r2 >= .95 and 150 <= accel <= 3000 and drift <= .5
    return dict(kind='free_fall' if free_fall else 'moving', accel_diameters_s2=round(accel, 1),
                fit_r2=round(r2, 3), **common)


class RotationProvider:
    """One isolated stream's tracker, in-plane counter and flip counter."""

    def __init__(self, **config):
        self.config = _configuration(config)
        options = self.config
        version = hashlib.sha256(json.dumps(options, sort_keys=True).encode()).hexdigest()[:16]
        self.spec = self.capability(**config)
        self.name, self.version = self.spec.provider, '1.' + version
        self.tracks = ShapeTracks(max_tracks=options['max_tracks'], max_age_ms=options['max_gap_ms'],
                                  search_scale=options['search_scale'], max_points=options['max_points'])
        self.spin = ObjectSpinCounter(max_tracks=options['max_tracks'], max_gap_ms=options['max_gap_ms'],
                                      expected_max_rev_s=options['expected_max_rev_s'], idle_ms=options['idle_ms'])
        self.rotation = ObjectRotationBank(max_tracks=options['max_tracks'], max_gap_ms=options['max_gap_ms'],
                                           max_points=options['max_points'], min_quality=options['min_quality'])
        self._lock = threading.RLock()
        self._target_size = None
        self._target_tracks = None
        self.closed = False
        self._clear()

    @staticmethod
    def capability(**config):
        options = _configuration(config)
        digest = hashlib.sha256(json.dumps(options, sort_keys=True).encode()).hexdigest()[:16]
        return CapabilitySpec(
            'vision.rotation', 'vision.rotation', 'video', 'gamst.rotation', '1.' + digest,
            (OutputField('objects', 'array'), OutputField('events', 'array')), 'frame',
            stateful=True, reset_on_gap=True, freshness_ms=options['max_gap_ms'],
            interpretation='measurement', devices=('cpu',),
            sensor_requirements=('uint8-bgr-camera', 'capture-timestamps'),
            license='Apache-2.0')

    def _clear(self):
        self._active = {}
        self._labels = {}
        self._paths = {}
        self._last_region_ts = None
        self._motion_previous = None
        self._motion_owner = None

    def _trajectory(self, track_id, start_ms, end_ms):
        return classify_trajectory([sample for sample in self._paths.get(track_id, ())
                                    if start_ms <= sample[0] <= end_ms])

    def reset(self):
        with self._lock:
            self.tracks.reset()
            if self._target_tracks is not None:
                self._target_tracks.reset()
            self.spin.reset()
            self.rotation.reset()
            self._clear()

    def close(self):
        with self._lock:
            if self.closed:
                return
            self.reset()
            self.spin.close()
            self.rotation.close()
            self.closed = True

    def set_target(self, size_px):
        """Follow only an object of an exemplar's size (pixels on the frames this provider receives).

        Motion alone proposes every moving thing (hands, arms, heads). With a target exemplar,
        e.g. a box the user drew around a coin, exemplar-sized motion proposals seed
        :class:`TargetTracks`, which follows that object frame to frame along a predicted
        path and ignores everything else. ``None`` returns to generic motion tracking.
        """
        if size_px is not None and (type(size_px) not in (int, float) or not math.isfinite(size_px) or
                                    not 4 <= size_px <= 4096):
            raise ValueError('target size must be 4..4096 pixels or None')
        with self._lock:
            self._target_size = None if size_px is None else float(size_px)
            self._target_tracks = None if size_px is None else TargetTracks(
                size_px, max_tracks=self.config['max_tracks'])
            self.tracks.reset()

    def _observe_target(self, frame):
        """Exemplar mode: motion proposals every refresh, frame-to-frame tracking in between."""
        import cv2
        stamp = frame.timestamp_ms
        proposals = []
        if self._last_region_ts is None or stamp - self._last_region_ts >= self.config['region_refresh_ms']:
            height, width = frame.data.shape[:2]
            proposals = [((row['bbox']['x'] + row['bbox']['width'] / 2) * width,
                          (row['bbox']['y'] + row['bbox']['height'] / 2) * height)
                         for row in self._motion_regions(frame.data, stamp)]
            self._last_region_ts = stamp
        gray = cv2.cvtColor(frame.data, cv2.COLOR_BGR2GRAY)
        return self._target_tracks.observe(gray, stamp, proposals)

    def _motion_regions(self, image, stamp):
        import cv2
        import numpy as np
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        previous = self._motion_previous
        self._motion_previous = gray
        if previous is None or previous.shape != gray.shape:
            return []
        blurred_previous, blurred = cv2.GaussianBlur(previous, (5, 5), 0), cv2.GaussianBlur(gray, (5, 5), 0)
        target = self._target_size
        if target is not None:
            # Lights flicker against a 1/1000 s shutter, so whole frames brighten and darken between
            # captures; remove the global change before differencing.
            shift = float(blurred.mean()) - float(blurred_previous.mean())
            difference = cv2.absdiff(cv2.convertScaleAbs(blurred_previous, beta=shift), blurred)
        else:
            difference = cv2.absdiff(blurred_previous, blurred)
        _, mask = cv2.threshold(difference, 24, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        grow = 9 if target is None else max(3, int(target // 4) | 1)
        mask = cv2.dilate(mask, np.ones((grow, grow), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = gray.shape
        if target is not None:
            # The exemplar fixes the scale: a moving coin's difference blob spans about one to a few
            # diameters (its motion smear); hands and arms are far larger. Closest to that size first.
            boxes = [cv2.boundingRect(contour) for contour in contours]
            boxes = [box for box in boxes if .5 * target <= max(box[2], box[3]) <= 5 * target
                     and min(box[2], box[3]) >= .3 * target]
            boxes.sort(key=lambda box: abs(math.log(max(box[2], box[3]) / (1.6 * target))))
        else:
            boxes = []
            for contour in sorted(contours, key=cv2.contourArea, reverse=True):
                if cv2.contourArea(contour) < max(144., height * width * .0005):
                    break
                x, y, w, h = cv2.boundingRect(contour)
                if w >= 12 and h >= 12 and w * h <= height * width * .5:
                    boxes.append((x, y, w, h))
        rows = []
        for x, y, w, h in boxes[:self.config['max_tracks']]:
            pad_x, pad_y = w * .1, h * .1
            x0, y0 = max(0., x - pad_x), max(0., y - pad_y)
            x1, y1 = min(float(width), x + w + pad_x), min(float(height), y + h + pad_y)
            row = dict(label='unknown', score=0., bbox=dict(
                x=x0 / width, y=y0 / height, width=(x1 - x0) / width, height=(y1 - y0) / height))
            if target is not None:
                row['size_px'] = target
            rows.append(row)
        return rows

    def _refresh_regions(self, frame, detections, detector_fresh):
        source = self.config['region_source']
        stamp = frame.timestamp_ms
        if detections is not None and source in ('auto', 'detections'):
            rows = _rows(detections, self.config['targets'])
            self.tracks.detections(rows, frame.data.shape, stamp)
            self._last_region_ts = stamp
            return
        if source == 'detections' or (source == 'auto' and detector_fresh):
            return
        refresh = self._last_region_ts is None or stamp - self._last_region_ts >= self.config['region_refresh_ms']
        if not refresh:
            return
        if source == 'roi':
            roi = self.config['roi']
            rows = _rows([dict(label=roi.get('label', 'unknown'), score=1., bbox=roi)], self.config['targets'])
        else:
            self.tracks.expire_unconfirmed(stamp, max(150, 3 * self.config['region_refresh_ms']))
            rows = _rows(self._motion_regions(frame.data, stamp), self.config['targets'])
        self.tracks.detections(rows, frame.data.shape, stamp)
        self._last_region_ts = stamp

    def process(self, frame, *, detections=None, detector_fresh=False):
        with self._lock:
            if self.closed:
                raise RuntimeError('rotation provider is closed')
            if (not isinstance(frame, InputFrame) or frame.modality != 'video' or
                    not isinstance(frame.sensor, SensorMetadata) or type(frame.timestamp_ms) is not int):
                raise ValueError('video InputFrame with capture sensor metadata required')
            owner = (frame.subject_id, frame.sensor.source_id, frame.sensor.sensor_id, frame.sensor.stream_epoch)
            if owner != self._motion_owner:
                self._motion_previous = None
                self._motion_owner = owner
                if self._target_tracks is not None:
                    self._target_tracks.reset()
            if self._target_tracks is not None:
                objects = self._observe_target(frame)
            else:
                if detections is not None:
                    detector_fresh = True
                self._refresh_regions(frame, detections, detector_fresh)
                objects = self.tracks.observe(frame.data, frame.timestamp_ms, owner)
            spin = self.spin.process(InputFrame({'objects': [
                {'track_id': item['track_id'], 'axes': item['axes'], 'observed': item['observed']}
                for item in objects]}, 'video', frame.timestamp_ms,
                subject_id=frame.subject_id, sensor=frame.sensor))
            rotation = self.rotation.process(InputFrame({'objects': [
                {'track_id': item['track_id'], 'center': item['center'], 'points': item['points'],
                 'points_epoch': item['points_epoch']} for item in objects]},
                'video', frame.timestamp_ms, subject_id=frame.subject_id, sensor=frame.sensor))
            return self.spec.validate(self._combine(frame, objects, spin, rotation))

    def _combine(self, frame, objects, spin, rotation):
        stamp = frame.timestamp_ms
        spin_by = {item['track_id']: item for item in spin['objects']}
        rotation_by = {item['track_id']: item for item in rotation['objects']}
        output, events = [], []
        seen = set()
        for item in objects:
            track_id = item['track_id']
            seen.add(track_id)
            label = None if item['label'] == 'unknown' else item['label']
            if label is not None:
                self._labels[track_id] = label
            if item['observed'] and item['axes']:
                path = self._paths.setdefault(track_id, collections.deque(maxlen=4096))
                path.append((stamp, item['center'][0], item['center'][1], item['axes'][0]))
            flip = spin_by[track_id]
            plane = rotation_by.get(track_id, dict(rotation_count=0., angular_velocity_deg_s=0., direction='unknown'))
            plane_rev_s = abs(plane['angular_velocity_deg_s']) / 360
            plane_active = item['observed'] and plane_rev_s >= self.config['min_rev_s'] and plane.get('confidence', 0.) > 0
            flip_status = flip['status']
            flip_active = flip_status != 'idle'
            fps = flip.get('capture_fps')
            plane_under = fps is not None and fps / 2 < self.config['expected_max_rev_s']
            state = self._active.get(track_id)
            if plane_active:
                if state is None:
                    state = self._active[track_id] = dict(started_ms=stamp, last_active_ms=stamp,
                        start_turns=plane['rotation_count'], turns=plane['rotation_count'], fps=fps)
                    events.append(dict(type='rotation.started', axis='in_plane', track_id=track_id,
                                       label=self._labels.get(track_id), timestamp_ms=stamp))
                state.update(last_active_ms=stamp, turns=plane['rotation_count'], fps=fps)
            elif state is not None and stamp - state['last_active_ms'] > self.config['idle_ms']:
                events.append(self._end_in_plane(track_id, state, 'stopped'))
                del self._active[track_id]
            if flip_status == 'undersampled' or (plane_active and plane_under):
                status = 'undersampled'
            elif flip_active or plane_active:
                status = 'counting'
            else:
                status = 'idle'
            output.append(dict(track_id=track_id, label=self._labels.get(track_id), observed=item['observed'],
                box=item['box'], status=status, in_plane_turns=round(plane['rotation_count'], 4),
                in_plane_rev_s=round(plane_rev_s, 4), in_plane_direction=plane['direction'],
                flip_turns=flip['turns'], flip_rev_s=flip['rev_s'], capture_fps=fps,
                max_countable_rev_s=flip.get('max_countable_rev_s')))
        for track_id, state in list(self._active.items()):
            if track_id not in seen and stamp - state['last_active_ms'] > self.config['idle_ms']:
                events.append(self._end_in_plane(track_id, state, 'lost'))
                del self._active[track_id]
        for event in spin['events']:
            label = self._labels.get(event['track_id'])
            if event['type'] == 'object.spin.started':
                events.append(dict(type='rotation.started', axis='flip', track_id=event['track_id'],
                                   label=label, timestamp_ms=event['timestamp_ms']))
            elif event['type'] == 'object.spin.ended':
                status = event['status'] if event['status'] in ('counting', 'undersampled') else 'idle'
                events.append(dict(type='rotation.ended', axis='flip', track_id=event['track_id'], label=label,
                    timestamp_ms=event['timestamp_ms'], reason=event['reason'], status=status,
                    turns=event['turns'], rev_s=event['rev_s'], duration_ms=event['duration_ms'],
                    fit_quality=event['fit_quality'], capture_fps=event['capture_fps'],
                    max_countable_rev_s=event['max_countable_rev_s'],
                    expected_max_rev_s=event['expected_max_rev_s'],
                    trajectory=self._trajectory(event['track_id'], event['timestamp_ms'] - event['duration_ms'],
                                                event['timestamp_ms'])))
        for track_id in [key for key in self._paths if key not in seen]:
            if stamp - self._paths[track_id][-1][0] > 2 * self.config['max_gap_ms']:
                del self._paths[track_id]
        return dict(objects=output, events=events)

    def _end_in_plane(self, track_id, state, reason):
        duration = max(0, state['last_active_ms'] - state['started_ms'])
        turns = abs(state['turns'] - state['start_turns'])
        fps = state.get('fps')
        status = 'undersampled' if fps is not None and fps / 2 < self.config['expected_max_rev_s'] else 'counting'
        return dict(type='rotation.ended', axis='in_plane', track_id=track_id,
            label=self._labels.get(track_id), timestamp_ms=state['last_active_ms'], reason=reason,
            status=status, turns=round(turns, 4), rev_s=round(turns / (duration / 1000), 4) if duration else None,
            duration_ms=duration, capture_fps=fps)


class RotationPlugin:
    """Store-composing adapter for ``human.register(...)`` + ``push_video``.

    ``push_video`` intentionally has no sensor argument, so this adapter gives
    each subject a monotonic local capture sequence. It commits the registered
    observation directly and returns no duplicate legacy signals to the engine.

    ``record_every_ms`` keeps high-fps sources affordable: every frame is still
    measured, but a record is committed only when an event fires or the cadence
    elapses. ``0`` commits every frame. ``latest`` holds the newest per-frame
    measurement for live overlays; records stay the context contract.
    """

    modalities = ('video',)

    def __init__(self, human, record_every_ms=0, **config):
        if type(record_every_ms) is not int or record_every_ms < 0:
            raise ValueError('record_every_ms must be a non-negative integer')
        self.human = human
        self.record_every_ms = record_every_ms
        self.provider = RotationProvider(**config)
        self.spec = self.provider.spec
        self.name, self.version = self.spec.provider, self.spec.version
        self.human.register_capability(self.spec)
        self._epoch = uuid.uuid4().hex
        self._sequences = {}
        self._captures = {}
        self._committed_at = {}
        self._detection_ids = {}
        self.latest = None
        self.closed = False

    def _latest_detections(self, frame):
        config = self.provider.config
        if config['region_source'] not in ('auto', 'detections'):
            return None, False
        records = self.human.store.query(frame.subject_id,
            start=max(0, frame.timestamp_ms - config['detector_max_age_ms']), end=frame.timestamp_ms,
            kind='observation.signal', limit=64, registered_only=True)
        for record in reversed(records):
            observation = record.get('observation', {})
            if observation.get('capability', {}).get('id') == self.spec.id:
                continue
            value = observation.get('value')
            if not isinstance(value, dict) or type(value.get('objects')) is not list:
                continue
            rows = _rows(value['objects'])
            if not rows and not value['objects']:
                continue
            previous = self._detection_ids.get(frame.subject_id)
            self._detection_ids[frame.subject_id] = record['id']
            return (value['objects'] if previous != record['id'] else None), True
        return None, False

    async def process(self, frame):
        if self.closed:
            raise RuntimeError('rotation plugin is closed')
        capture = self._captures.get(frame.subject_id, -1) + 1
        self._captures[frame.subject_id] = capture
        sequence = self._sequences.get(frame.subject_id, -1) + 1
        sensor = SensorMetadata(source_id='gamst-plugin', sensor_id='vision-rotation', sequence=sequence,
                                capture_sequence=capture, coordinate_frame='image', stream_epoch=self._epoch)
        detections, fresh = self._latest_detections(frame)
        value = self.provider.process(InputFrame(frame.data, 'video', frame.timestamp_ms,
            subject_id=frame.subject_id, sensor=sensor), detections=detections, detector_fresh=fresh)
        self.latest = value
        last = self._committed_at.get(frame.subject_id)
        if value['events'] or last is None or frame.timestamp_ms - last >= self.record_every_ms:
            self._sequences[frame.subject_id] = sequence
            self._committed_at[frame.subject_id] = frame.timestamp_ms
            await self.human.push_observation(self.spec.id, value, sensor=sensor,
                subject_id=frame.subject_id, timestamp_ms=frame.timestamp_ms)
        return []

    def set_target(self, size_px):
        """See :meth:`RotationProvider.set_target`; sizes are in pixels of the pushed frames."""
        self.provider.set_target(size_px)

    def reset(self):
        self.provider.reset()
        self._sequences.clear()
        self._captures.clear()
        self._committed_at.clear()
        self._detection_ids.clear()
        self._epoch = uuid.uuid4().hex

    def close(self):
        if not self.closed:
            self.provider.close()
            self._sequences.clear()
            self._captures.clear()
            self._committed_at.clear()
            self._detection_ids.clear()
            self.closed = True

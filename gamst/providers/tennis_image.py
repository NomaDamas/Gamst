"""Fixed-camera tennis ball-line candidates, within an explicit image envelope."""
from dataclasses import replace
import hashlib
import json
import math
import threading

from ..capabilities import OutputField, bounded_json, identifier
from .detection_trajectories import RFDETRTrajectories, configuration as trajectory_config
from .image_events import ImageEventRules


def configuration(config):
    config = bounded_json(config, max_bytes=8192)
    if (type(config) is not dict or not {'path', 'sha256', 'reference'} <= set(config)
            or set(config)-{'path', 'sha256', 'device', 'reference', 'envelope', 'compensation'}):
        raise ValueError('tennis requires local checkpoint and explicit fixed-image reference')
    reference = config['reference']
    if (type(reference) is not dict or set(reference) !=
            {'id', 'source_id', 'sensor_id', 'coordinate_frame', 'calibration_id', 'image_size', 'line'}):
        raise ValueError('exact tennis reference fields required')
    for name in ('id', 'source_id', 'sensor_id', 'coordinate_frame'):
        identifier(reference[name])
    if reference['calibration_id'] is not None:
        identifier(reference['calibration_id'])
    size = reference['image_size']
    if (type(size) is not list or len(size) != 2 or
            any(type(n) is not int or not 16 <= n <= 4096 for n in size) or size[0]*size[1]*3 > 16*1024**2):
        raise ValueError('bounded reference image dimensions required')
    line = reference['line']
    if type(line) is not dict or set(line) != {'start', 'end'}:
        raise ValueError('finite image line required')
    rule = dict(id='tennis-ball-line', kind='line', **line)
    ImageEventRules([rule])
    envelope = config.get('envelope', dict(max_people=2, max_rackets=4, min_ball_side_pixels=4,
        max_capture_gap_ms=100, max_endpoint_displacement_normalized=.1))
    if type(envelope) is not dict or set(envelope) != {'max_people', 'max_rackets',
            'min_ball_side_pixels', 'max_capture_gap_ms', 'max_endpoint_displacement_normalized'}:
        raise ValueError('exact bounded tennis envelope required')
    for key, low, high in (('max_people', 1, 4), ('max_rackets', 0, 4),
            ('min_ball_side_pixels', 4, 128), ('max_capture_gap_ms', 1, 100)):
        if type(envelope[key]) is not int or not low <= envelope[key] <= high:
            raise ValueError('unsupported tennis envelope bound')
    displacement = envelope['max_endpoint_displacement_normalized']
    if type(displacement) not in (int, float) or not math.isfinite(displacement) or not 0 < displacement <= .1:
        raise ValueError('bounded displacement envelope required')
    config['envelope'] = envelope
    args = trajectory_config({k: v for k, v in config.items() if k in ('path', 'sha256', 'device')} |
        dict(camera_motion='stationary', max_gap_ms=100, max_samples=2, event_rules=[rule]))
    if 'compensation' in config:
        from .compensated_trajectories import configuration as compensated_config
        compensation = config['compensation']
        if type(compensation) is not dict or set(compensation) != {'background_roi', 'operating_assumption', 'anchor'}:
            raise ValueError('explicit compensation and original anchor capture required')
        anchor = compensation['anchor']
        if type(anchor) is not dict or set(anchor) != {'timestamp_ms', 'capture_sequence', 'evidence_sha256'}:
            raise ValueError('exact anchor capture identity required')
        event_reference = {k: v for k, v in reference.items() if k != 'line'} | anchor | dict(rules=[rule])
        args = compensated_config({k: v for k, v in config.items() if k in ('path', 'sha256', 'device')} |
            {k: v for k, v in compensation.items() if k != 'anchor'} |
            dict(max_gap_ms=envelope['max_capture_gap_ms'], event_reference=event_reference))
    return config, args


class TennisImageTask:
    """Observed endpoint candidate only; no bounce, officiating or score inference."""
    @staticmethod
    def capability(config):
        config, args = configuration(config)
        if 'compensation' in config:
            from .compensated_trajectories import CompensatedTrajectories
            base = CompensatedTrajectories.capability(args)
        else:
            base = RFDETRTrajectories.capability('1.12.1.eager.sha256.'+args['sha256'],
                **{k: v for k, v in args.items() if k not in ('path', 'sha256')})
        digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        return replace(base, id='sports.tennis.image_line', output_name='sports.tennis.image_line',
            provider='gamst.tennis.image_line', version='1.'+digest,
            fields=base.fields+(OutputField('task', 'object'),),
            sensor_requirements=base.sensor_requirements+('fixed-reference-image', 'tennis-singles-population'))

    def __init__(self, trajectories, config):
        self.spec = self.capability(config)
        self.config, _ = configuration(config)
        self.paths = trajectories
        self._lock = threading.RLock()
        self.closed = False
        self._reference_owner = None
        self._invalidated = False

    @classmethod
    def from_local(cls, config):
        config, args = configuration(config)
        if 'compensation' in config:
            from .compensated_trajectories import CompensatedTrajectories
            return cls(CompensatedTrajectories.from_local(args), config)
        return cls(RFDETRTrajectories.from_local(**args), config)

    def reset(self):
        with self._lock:
            self.paths.reset()
            self._reference_owner = None
            self._invalidated = False

    def close(self):
        with self._lock:
            self.paths.close()
            self.closed = True

    def process(self, frame):
        with self._lock:
            if self.closed:
                raise RuntimeError('tennis image task closed')
            try:
                result = self.paths.process(frame)
                ref = self.config['reference']
                envelope = self.config['envelope']
                compensated = 'compensation' in self.config
                sensor = frame.sensor
                expected = tuple(ref[k] for k in ('source_id', 'sensor_id', 'coordinate_frame', 'calibration_id'))
                actual = (sensor.source_id, sensor.sensor_id, sensor.coordinate_frame, sensor.calibration_id)
                owner = actual+(frame.subject_id, sensor.stream_epoch, tuple(frame.data.shape))
                reason = None
                if (actual != expected or [frame.data.shape[1], frame.data.shape[0]] != ref['image_size']
                        or self._reference_owner is not None and owner != self._reference_owner):
                    self._invalidated = True
                if self._invalidated:
                    reason = 'reference_invalidated_explicit_reset_required'
                elif self._reference_owner is None:
                    self._reference_owner = owner
                if not result['geometry']['complete_snapshot']:
                    reason = 'partial_detection_population'
                if compensated and result['event_summary']['status'] not in ('bound_estimated_anchor', 'collecting_anchor'):
                    reason = reason or 'anchor_reference_unbound'
                counts = {'person': 0, 'sports ball': 0, 'tennis racket': 0}
                for item in result['detections']:
                    if item['label'] not in counts:
                        reason = reason or 'unsupported_detection_population'
                    else:
                        counts[item['label']] += 1
                if counts['person'] > envelope['max_people'] or counts['tennis racket'] > envelope['max_rackets'] or counts['sports ball'] > 1:
                    reason = reason or 'unsupported_detection_population'
                if counts['sports ball'] != 1:
                    reason = reason or 'ball_unavailable_or_ambiguous'
                ball_ids = set()
                for item in result['detections']:
                    if item['label'] == 'sports ball':
                        box = item['location']['bbox']
                        if min(box['width']*ref['image_size'][0], box['height']*ref['image_size'][1]) < envelope['min_ball_side_pixels']:
                            reason = reason or 'ball_below_four_pixels'
                        ball_ids.add(item['track_id'])
                for obj in result['objects']:
                    if obj['track_id'] in ball_ids and len(obj['observations']) >= 2:
                        a, b = obj['observations'][-2:]
                        point = 'anchor_projection' if compensated else 'centroid'
                        if b['timestamp_ms']-a['timestamp_ms'] > envelope['max_capture_gap_ms']:
                            reason = reason or 'ball_capture_gap_outside_envelope'
                        if math.dist(a[point], b[point]) > envelope['max_endpoint_displacement_normalized']:
                            reason = reason or 'ball_displacement_outside_envelope'
                paired = any(obj['track_id'] in ball_ids and len(obj['observations']) >= 2
                    for obj in result['objects'])
                result['events'] = [event for event in result['events']
                    if result['objects'][event['object_index']]['track_id'] in ball_ids]
                if reason:
                    # No continuity may bridge an unsupported/occluded capture.
                    result['events'] = []
                    result['objects'] = []
                    if compensated:
                        self.paths.discard_object_continuity()
                    else:
                        self.paths.reset()
                result['task'] = dict(id='tennis-image-line-v1',
                    status='abstained' if reason else 'observed_endpoints' if paired else 'collecting',
                    reason=reason, reference=ref, stream_epoch=sensor.stream_epoch,
                    population=counts, interpretation='unverified_ball_line_candidate',
                    camera_stationary_verified=False, quality_verified=False,
                    envelope=dict(envelope, max_balls=1, max_detections=8,
                        occlusion='unsupported_reset', moving_camera='conditional_similarity' if compensated else 'unsupported',
                        metric_geometry='unsupported'),
                    event_meaning='observed_centroid_endpoints_straddle_finite_image_line_not_bounce_in_out_or_score')
                result['event_summary']['task_envelope_met'] = reason is None
                result['event_summary']['event_count'] = len(result['events'])
                return self.spec.validate(result)
            except Exception:
                self.paths.reset()
                raise

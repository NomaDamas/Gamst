"""Explicit projective calibration for caller-declared on-plane image points."""
import hashlib
import json
import math

from ..capabilities import CapabilitySpec, OutputField, bounded_json, identifier
from ..capabilities import SensorMetadata
from ..plugins import InputFrame


def configuration(value):
    value = bounded_json(value, max_bytes=8192)
    if type(value) is not dict or set(value) != {'id', 'reference_id', 'source_id', 'sensor_id',
            'input_frame', 'output_frame', 'image_size', 'valid_roi', 'image_to_plane'}:
        raise ValueError('exact explicit projective plane calibration required')
    for key in ('id', 'reference_id', 'source_id', 'sensor_id', 'input_frame', 'output_frame'):
        identifier(value[key])
    size = value['image_size']
    if type(size) is not list or len(size) != 2 or any(type(x) is not int or not 16 <= x <= 4096 for x in size):
        raise ValueError('bounded calibration image dimensions required')
    roi = value['valid_roi']
    if (type(roi) is not list or len(roi) != 4 or any(type(x) not in (int,float) or
            not math.isfinite(x) or not 0 <= x <= 1 for x in roi) or roi[0] >= roi[2] or roi[1] >= roi[3]):
        raise ValueError('explicit positive normalized calibration ROI required')
    matrix = value['image_to_plane']
    if (type(matrix) is not list or len(matrix) != 3 or any(type(row) is not list or len(row) != 3 for row in matrix)
            or any(type(x) not in (int,float) or not math.isfinite(x) or abs(x) > 10**6 for row in matrix for x in row)):
        raise ValueError('bounded finite projective matrix required')
    scale = max(abs(x) for row in matrix for x in row)
    if scale == 0:
        raise ValueError('nonsingular calibration required')
    a,b,c = ([x/scale for x in row] for row in matrix)
    determinant = a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])
    denominators = [c[0]*x+c[1]*y+c[2] for x,y in
        ((roi[0],roi[1]),(roi[2],roi[1]),(roi[2],roi[3]),(roi[0],roi[3]))]
    if abs(determinant) < 1e-12 or min(denominators) <= 1e-8 <= max(denominators) or min(abs(x) for x in denominators) <= 1e-8:
        raise ValueError('singular or projective horizon inside calibrated ROI')
    return value


class GroundPlane:
    @staticmethod
    def capability(config):
        config = configuration(config)
        digest = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
        return CapabilitySpec('geometry.projective_plane','geometry.plane_points','video',
            'gamst.projective_plane','1.'+digest,(OutputField('points','array'),OutputField('geometry','object')),
            'frame',stateful=False,interpretation='estimate',
            sensor_requirements=('declared-on-plane-points','explicit-projective-calibration'),license='Apache-2.0')

    def __init__(self, config):
        self.config = configuration(config)
        self.spec = self.capability(self.config)
        scale = max(abs(x) for row in self.config['image_to_plane'] for x in row)
        self.matrix = [[x / scale for x in row] for row in self.config['image_to_plane']]
        self.closed = False

    def reset(self):
        pass

    def close(self):
        self.closed = True

    def process(self, frame):
        if self.closed:
            raise RuntimeError('ground-plane provider closed')
        c = self.config
        if not isinstance(frame, InputFrame):
            raise ValueError('explicit structured image-point frame required')
        sensor = frame.sensor
        if (frame.modality != 'video' or frame.sample_rate is not None or not isinstance(sensor, SensorMetadata) or
                (sensor.source_id,sensor.sensor_id,sensor.coordinate_frame,sensor.calibration_id) !=
                (c['source_id'],c['sensor_id'],c['input_frame'],c['id']) or sensor.stream_epoch is None):
            raise ValueError('explicit matching plane source/frame/calibration/epoch required')
        value = bounded_json(frame.data,max_bytes=8192)
        if (type(value) is not dict or set(value) != {'reference_id','image_size','points'} or
                value['reference_id'] != c['reference_id'] or value['image_size'] != c['image_size'] or
                type(value['points']) is not list or len(value['points']) > 8):
            raise ValueError('bounded exact-reference image points required')
        if type(frame.timestamp_ms) is not int or not 0 <= frame.timestamp_ms <= 2**53-1:
            raise ValueError('original capture timestamp required')
        points=[];ids=set();matrix=self.matrix;roi=c['valid_roi']
        for point in value['points']:
            if type(point) is not dict or set(point) != {'id','xy','on_plane'}:
                raise ValueError('point identity, normalized xy and explicit plane eligibility required')
            pid=identifier(point['id'])
            if pid in ids:raise ValueError('unique point identity required')
            ids.add(pid);xy=point['xy']
            if (type(xy) is not list or len(xy) != 2 or any(type(x) not in (int,float) or
                    not math.isfinite(x) or not 0 <= x <= 1 for x in xy) or type(point['on_plane']) is not bool):
                raise ValueError('finite normalized point and explicit on-plane declaration required')
            x,y=xy;reason=None;projected=None
            if not point['on_plane']:reason='off_plane_or_unknown'
            elif not roi[0] <= x <= roi[2] or not roi[1] <= y <= roi[3]:reason='outside_calibrated_region'
            else:
                denominator=matrix[2][0]*x+matrix[2][1]*y+matrix[2][2]
                projected=[(row[0]*x+row[1]*y+row[2])/denominator for row in matrix[:2]]
                if any(not math.isfinite(n) or abs(n)>10**6 for n in projected):
                    projected=None;reason='projection_outside_supported_extent'
            points.append(dict(id=pid,image_xy=xy,plane_xy_m=projected,reason=reason,
                on_plane_declared=point['on_plane'],plane_eligibility_verified=False))
        digest=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
        return self.spec.validate(dict(points=points,geometry=dict(
            calibration_id=c['id'],reference_id=c['reference_id'],input_frame=c['input_frame'],output_frame=c['output_frame'],
            units='m',image_units='normalized_image',image_size=c['image_size'],calibration_verified=False,
            plane_eligibility='caller_declaration_not_measurement',timestamp_ms=frame.timestamp_ms,
            source_id=sensor.source_id,sensor_id=sensor.sensor_id,stream_epoch=sensor.stream_epoch,
            capture_sequence=sensor.capture_sequence if sensor.capture_sequence is not None else sensor.sequence,
            evidence_sha256=digest,evidence_meaning='structured_point_input_integrity_only',
            airborne_ball_3d_supported=False)))

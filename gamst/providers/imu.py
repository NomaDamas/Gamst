"""BNO055 page-zero 0x08..0x19 burst decoding; no bus access or fusion claims."""
import math
import struct
from ..capabilities import CapabilitySpec, OutputField, SensorMetadata, identifier


_IDENTITY = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))


def _vector(value, label, bound):
    if type(value) not in (tuple, list) or len(value) != 3 or any(
        type(x) not in (int, float) or not math.isfinite(x) or abs(x) > bound for x in value):
        raise ValueError(f'{label} must be three finite bounded numbers')
    return tuple(float(x) for x in value)


class BNO055IMU:
    def __init__(self, *, unit_sel, operation_mode, acceleration_range_g,
                 gyroscope_range_dps, calibration_id, sensor_frame='imu', output_frame='imu',
                 acceleration_bias=(0, 0, 0), gyroscope_bias=(0, 0, 0), rotation_to_output=_IDENTITY):
        if type(unit_sel) is not int or not 0 <= unit_sel <= 255:
            raise ValueError('declare UNIT_SEL byte')
        if type(operation_mode) is not int or operation_mode not in (8, 11, 12):
            raise ValueError('only fusion modes 8, 11 and 12 have supported gyro scaling')
        if type(acceleration_range_g) is not int or acceleration_range_g != 4:
            raise ValueError('fusion acceleration range must be 4 g')
        if type(gyroscope_range_dps) is not int or gyroscope_range_dps != 2000:
            raise ValueError('fusion gyroscope range must be 2000 dps')
        for value in (calibration_id, sensor_frame, output_frame):
            identifier(value)
        self.acc_bias = _vector(acceleration_bias, 'acceleration bias in m/s2', 200)
        self.gyr_bias = _vector(gyroscope_bias, 'angular velocity bias in rad/s', 40)
        if type(rotation_to_output) not in (tuple, list) or len(rotation_to_output) != 3:
            raise ValueError('rotation must be a 3x3 matrix')
        rotation = tuple(_vector(row, 'rotation row', 1) for row in rotation_to_output)
        for i in range(3):
            for j in range(3):
                if abs(sum(rotation[i][k]*rotation[j][k] for k in range(3))-(1 if i == j else 0)) > 1e-6:
                    raise ValueError('rotation must be orthonormal')
        a, b, c = rotation
        determinant = a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])
        if abs(determinant-1) > 1e-6:
            raise ValueError('rotation must preserve handedness')
        if sensor_frame == output_frame and rotation != _IDENTITY:
            raise ValueError('a nonidentity rotation requires a distinct output frame name')
        self.rotation = rotation
        self.unit_sel, self.operation_mode = unit_sel, operation_mode
        self.acc_range = acceleration_range_g*9.80665
        self.gyr_range = math.radians(gyroscope_range_dps)
        self.calibration_id, self.sensor_frame, self.output_frame = calibration_id, sensor_frame, output_frame
        self.closed = False
        self.spec = self.capability()

    @staticmethod
    def capability():
        return CapabilitySpec('imu.bno055.motion', 'imu.motion', 'imu', 'gamst.bno055', '1',
            (OutputField('motion', 'object'),), 'frame', interpretation='measurement',
            devices=('cpu',), sensor_requirements=('bno055-page0-0x08-burst', 'declared-imu-calibration'),
            license='Apache-2.0')

    def reset(self):
        pass

    def close(self):
        self.closed = True

    def process(self, frame):
        if self.closed:
            raise RuntimeError('IMU provider is closed')
        if frame.modality != 'imu' or type(frame.data) is not bytes or len(frame.data) != 18:
            raise ValueError('supply exactly 18 BNO055 page-zero bytes starting at register 0x08')
        if not isinstance(frame.sensor, SensorMetadata) or (frame.sensor.calibration_id, frame.sensor.coordinate_frame) != (self.calibration_id, self.sensor_frame):
            raise ValueError('IMU calibration ID and sensor coordinate frame must match')
        if type(frame.timestamp_ms) not in (int, float) or not math.isfinite(frame.timestamp_ms) or frame.timestamp_ms < 0:
            raise ValueError('invalid acquisition timestamp')
        words = struct.unpack('<9h', frame.data)
        acc_scale = .00980665 if self.unit_sel & 1 else .01
        gyr_scale = 1/900 if self.unit_sel & 2 else math.pi/(180*16)
        acceleration = [v*acc_scale for v in words[:3]]
        gyroscope = [v*gyr_scale for v in words[6:9]]
        def transform(values, bias):
            corrected = [x-b for x, b in zip(values, bias)]
            return [sum(r[i]*corrected[i] for i in range(3)) for r in self.rotation]
        acc_near = [abs(v) >= .99*self.acc_range for v in acceleration]
        gyr_near = [abs(v) >= .99*self.gyr_range for v in gyroscope]
        endpoints = any(v in (-32768, 32767) for v in (*words[:3], *words[6:9]))
        return {'motion': dict(acceleration_m_s2=transform(acceleration, self.acc_bias),
            angular_velocity_rad_s=transform(gyroscope, self.gyr_bias),
            axes=['x', 'y', 'z'], coordinate_frame=self.output_frame, input_coordinate_frame=self.sensor_frame,
            calibration_id=self.calibration_id, unit_sel=self.unit_sel, operation_mode=self.operation_mode,
            rotation_to_output=[list(row) for row in self.rotation],
            residual_acceleration_bias_m_s2=list(self.acc_bias), residual_gyroscope_bias_rad_s=list(self.gyr_bias),
            configured_acceleration_range_m_s2=self.acc_range, configured_gyroscope_range_rad_s=self.gyr_range,
            input_axes_near_acceleration_range=acc_near, input_axes_near_gyroscope_range=gyr_near,
            wire_endpoint_detected=endpoints, possible_saturation=endpoints or any(acc_near) or any(gyr_near),
            gravity_removed=False, orientation_estimated=False, position_estimated=False,
            physical_calibration_verified=False)}

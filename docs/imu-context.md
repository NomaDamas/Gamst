# IMU motion context

`humansense.providers.imu.BNO055IMU` decodes an exact18-byte, page-zero burst beginning at0x08. It uses only Python's standard library, never accesses I2C/UART, and does not install firmware or models. SDK code is Apache-2.0; hardware/driver licensing is separate.

The source format is [Bosch BNO055 datasheet revision1.8](https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bno055-ds000.pdf), sections3.6/4.3: signed little-endian acceleration at0x08..0x0D and angular rate at0x14..0x19. Intermediate magnetometer bytes are ignored. UNIT_SEL bits0/1 select acceleration m/s² versus mg and angular rate degrees/s versus radians/s. Decoder factors are0.01m/s² or0.00980665m/s² per count, andπ/(180×16)rad/s or1/900rad/s per count. This reads acceleration registers, not the sensor's separate gravity-compensated linear-acceleration output.

Non-fusion modes 5 and 7 are rejected: shipped firmware can expose range-dependent gyro counts instead of the fixed units. See the [Linux BNO055 driver scaling notes](https://code.googlesource.com/linux/torvalds/linux/+/4bf54e47525dcacd4c6cdd97fb5902592414dd7a/drivers/iio/imu/bno055/bno055.c). Supporting those modes requires an explicit, hardware-validated firmware/scaling profile.

## Explicit acquisition profile

Construction requires `unit_sel`, `operation_mode`, `acceleration_range_g`, `gyroscope_range_dps` and `calibration_id`. Supported operation modes are 8, 11 and 12 (fusion modes with both channels enabled). Fusion profiles must declare the effective fixed ranges of 4 g and 2000 degrees/s; other values are rejected. The acquisition layer must verify page, units, actual effective ranges, mode, firmware axis remapping, sensor readiness/error state and burst consistency. Fusion modes can override requested settings: declare the effective hardware behavior, not an ignored register request. Never label disconnected/stale/default bytes as valid measurements. Each input carries its acquisition timestamp and matching calibration ID and sensor coordinate frame.

The profile must record device/firmware identity, register settings, onboard calibration status, acquisition rate/timing, axis definitions and reference measurements. Matching a profile ID is a consistency check, not validation of physical calibration. The adapter always reports `physical_calibration_verified=false`.

Optional residual biases are SI values in the input sensor frame. Apply only residual corrections appropriate to the actual operating mode; do not double-apply onboard offsets. The adapter subtracts each bias, then applies `rotation_to_output` to the vectors. The3×3 transform must be finite, orthonormal and right-handed. Nonidentity rotation requires a distinct output-frame name. This is an application-declared fixed mounting transform, not estimated orientation or world tracking. Configurations are copied into bounded tuples.

```python
from humansense import SensorStreams, SensorMetadata
from humansense.plugins import InputFrame
from humansense.providers.imu import BNO055IMU

async def configure_imu(human):
    streams = SensorStreams(human)
    spec = BNO055IMU.capability()
    streams.register(spec, lambda: BNO055IMU(
        unit_sel=0, operation_mode=8, acceleration_range_g=4,
        gyroscope_range_dps=2000, calibration_id='imu-profile-1'))
    await streams.open('device', 'imu', [spec.id])
    return streams

# After an explicit driver has obtained one validated burst:
# await streams.submit(InputFrame(burst18, 'imu', acquired_at_ms,
#     sensor=SensorMetadata('device', 'imu', sequence,
#         coordinate_frame='imu', calibration_id='imu-profile-1')))
```

## Output and validity limits

`motion` contains XYZ acceleration in m/s² and angular velocity in rad/s, input/output frame names, configured units/mode/ranges, residual biases and transform. Gravity is not removed, orientation is not estimated, and no velocity/position integration is performed. Frame-mode processing retains no history; SensorStreams owns sequencing, gap epochs, bounded queues and freshness.

Before residual correction/rotation, each input axis is flagged if its magnitude reaches99% of the declared range. Signed16-bit endpoint codes are also flagged. `possible_saturation` is a conservative warning, not proof of clipping or a completeness guarantee; wrong profile settings and hardware clipping below these thresholds need capture-side detection. Reported vectors remain present with flags for consumer inspection, and must not be treated as reliable quantitative measurements when flagged. No raw burst or magnetometer data enters context.

## Evidence and pending requirements

`python3 -m unittest discover -s tests -p test_imu_provider.py -v` checks actual byte decoding, signed values, both unit modes, register offsets, pre-rotation residual correction, handedness, range/endpoint flags, malformed inputs and SensorStreams-to-context output. Inputs are constructed register fixtures, not hardware recordings. No physical-device execution, reference calibration, acquisition latency or full family17 support is claimed. Real BNO055 capture/calibration validation and the separate DIGIT/GelSight/Sparsh tactile path remain required under #33.

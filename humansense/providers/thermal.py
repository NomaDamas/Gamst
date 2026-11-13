"""Bounded ROI measurements from explicitly configured TLinear uint16 payloads.

This decodes an assembled radiometric image, not SPI/VoSPI packets or a colour
preview. The capture integration must verify radiometry/TLinear configuration,
remove telemetry and suppress frames during FFC before submission.
"""
import math
import struct

from ..capabilities import CapabilitySpec, OutputField, SensorMetadata, identifier


class TLinearThermal:
    def __init__(self, *, width, height, kelvin_per_count, byteorder,
                 calibration_id, emissivity, rois=None):
        for value, maximum in ((width, 640), (height, 512)):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError('thermal width must be 1..640 and height 1..512')
        if type(kelvin_per_count) not in (int, float) or kelvin_per_count not in (.01, .1):
            raise ValueError('declare TLinear resolution: 0.01 or 0.1 kelvin/count')
        if byteorder not in ('little', 'big'):
            raise ValueError('declare payload byteorder: little or big')
        if type(emissivity) not in (int, float) or not math.isfinite(emissivity) or not 0 < emissivity <= 1:
            raise ValueError('emissivity must be the configured sensor value in (0, 1]')
        identifier(calibration_id, 'calibration_id')
        if rois is None:
            rois = {'scene': (0, 0, width, height)}
        if type(rois) is not dict or not 1 <= len(rois) <= 16:
            raise ValueError('declare 1..16 named pixel ROIs')
        checked = []
        for name, rect in rois.items():
            identifier(name, 'ROI name')
            if type(rect) not in (tuple, list) or len(rect) != 4 or any(type(v) is not int for v in rect):
                raise ValueError('ROI is (left, top, right, bottom), with exclusive upper bounds')
            left, top, right, bottom = rect
            if not 0 <= left < right <= width or not 0 <= top < bottom <= height:
                raise ValueError('ROI is outside the configured image')
            checked.append((name, tuple(rect)))
        self.width, self.height = width, height
        self.scale, self.byteorder = float(kelvin_per_count), byteorder
        self.calibration_id, self.emissivity = calibration_id, float(emissivity)
        self.rois = tuple(checked)
        self.closed = False
        self.spec = self.capability()

    @staticmethod
    def capability():
        return CapabilitySpec('thermal.tlinear.roi', 'thermal.temperature', 'thermal',
            'gamst.tlinear', '1', (OutputField('temperature', 'object', unit='degC'),),
            'frame', interpretation='measurement', devices=('cpu',),
            sensor_requirements=('radiometric-TLinear-uint16', 'declared-calibration'),
            license='Apache-2.0')

    def reset(self):
        # Each assembled frame is independent; no retained pixel/history buffer.
        pass

    def close(self):
        self.closed = True

    def process(self, frame):
        if self.closed:
            raise RuntimeError('thermal provider is closed')
        if frame.modality != 'thermal' or type(frame.data) is not bytes:
            raise ValueError('thermal input must be assembled uint16 bytes, not RGB or AGC imagery')
        if not isinstance(frame.sensor, SensorMetadata) or frame.sensor.calibration_id != self.calibration_id:
            raise ValueError('sensor calibration_id must match the configured TLinear profile')
        if type(frame.timestamp_ms) not in (int, float) or not math.isfinite(frame.timestamp_ms) or frame.timestamp_ms < 0:
            raise ValueError('invalid capture timestamp')
        if len(frame.data) != self.width*self.height*2:
            raise ValueError('payload must exactly match configured dimensions; strip transport headers/telemetry upstream')
        # One bounded decoded image, reused by at most 16 ROIs. Endianness is explicit.
        prefix = '<' if self.byteorder == 'little' else '>'
        pixels = struct.unpack(f'{prefix}{self.width*self.height}H', frame.data)
        regions = []
        for name, rect in self.rois:
            left, top, right, bottom = rect
            count, total, low, high = 0, 0, 65535, 0
            for y in range(top, bottom):
                for x in range(left, right):
                    value = pixels[y*self.width+x]
                    # Reserved/extreme codes cannot establish a temperature.
                    if value in (0, 65535):
                        continue
                    count += 1
                    total += value
                    low, high = min(low, value), max(high, value)
            size = (right-left)*(bottom-top)
            regions.append(dict(name=name, bounds=list(rect), valid_pixels=count,
                invalid_pixels=size-count, valid_fraction=count/size,
                min_celsius=low*self.scale-273.15 if count else None,
                max_celsius=high*self.scale-273.15 if count else None,
                mean_celsius=total/count*self.scale-273.15 if count else None,
                available=bool(count)))
        return {'temperature': dict(regions=regions, width=self.width, height=self.height,
            coordinate_frame='thermal_image_pixels', calibration_id=self.calibration_id,
            kelvin_per_count=self.scale, byteorder=self.byteorder,
            configured_emissivity=self.emissivity, emissivity_applied_by='sensor',
            input_mode='TLinear', physical_accuracy_verified=False,
            rgb_alignment_available=False)}

"""Bounded TI xWR6843 UART decoding; no serial access or device configuration.

Wire profile: Radar Toolbox 3.30.00.06 UART layout with SDK3 signed side-info
from TI mmwave_ti_ros revision a244861eec7638792c0e070cd846b23d43e4fbaf.
The explicit SDK version identifies a declaration, not verified firmware.
"""
import hashlib
import json
import math
import struct
import threading
import uuid

from ..capabilities import CapabilitySpec, OutputField, SensorMetadata, identifier
from ..plugins import InputFrame

MAGIC = bytes.fromhex('0201040306050807')
MAX_PACKET = 65536
HEADER = struct.Struct('<8s8I')


class TIRadarPackets:
    """Incremental UART framing. Feed <=4096 bytes; reset on reconnect/timeout.

    No checksum exists in this format. A plausible false header can hold a
    partial packet until more data or an explicit reset. Never infer validity
    from framing; pass each result to parse_ti_radar.
    """
    def __init__(self):
        self._buffer = bytearray()
        self.discarded_bytes = 0

    @property
    def buffered_bytes(self):
        return len(self._buffer)

    def reset(self):
        self._buffer.clear()
        self.discarded_bytes = 0

    def feed(self, chunk):
        if type(chunk) is not bytes or len(chunk) > 4096:
            raise ValueError('UART chunk must be bytes of at most4096 bytes')
        self._buffer.extend(chunk)
        output = []
        while True:
            start = self._buffer.find(MAGIC)
            if start < 0:
                count = max(0, len(self._buffer)-7)
                self.discarded_bytes += count
                del self._buffer[:count]
                break
            if start:
                self.discarded_bytes += start
                del self._buffer[:start]
            if len(self._buffer) < HEADER.size:
                break
            length = struct.unpack_from('<I', self._buffer, 12)[0]
            if not 64 <= length <= MAX_PACKET or length % 32:
                del self._buffer[0]
                self.discarded_bytes += 1
                continue
            if len(self._buffer) < length:
                break
            output.append(bytes(self._buffer[:length]))
            del self._buffer[:length]
        return output


def _version(value):
    if type(value) is not int or not 0x03000000 <= value <= 0x03ffffff:
        raise ValueError('explicit uint32 SDK3 version required')
    return value


def _finite(values, bound=1e12):
    if any(not math.isfinite(v) or abs(v) > bound for v in values):
        raise ValueError('nonfinite or out-of-bound radar value')


def parse_ti_radar(packet, *, sdk_version):
    """One complete padded packet, little-endian,40-byte header, payload lengths.

    Only subframe0/platform0xA6843. Unknown TLVs are counted and skipped by
    checked length, never interpreted. At most512 points and32 targets.
    """
    _version(sdk_version)
    if type(packet) is not bytes or not 64 <= len(packet) <= MAX_PACKET or len(packet) % 32:
        raise ValueError('one padded bounded TI UART packet required')
    magic, version, length, platform, number, cycles, detected, count, subframe = HEADER.unpack_from(packet)
    if (magic != MAGIC or version != sdk_version or length != len(packet) or
            platform != 0xa6843 or subframe != 0 or detected > 512 or count > 32):
        raise ValueError('packet does not match declared xWR6843 wire profile')
    payloads = {}
    cursor = HEADER.size
    for _ in range(count):
        if cursor+8 > len(packet):
            raise ValueError('truncated TLV header')
        kind, size = struct.unpack_from('<II', packet, cursor)
        cursor += 8
        if size > len(packet)-cursor or kind in payloads:
            raise ValueError('truncated or duplicate TLV')
        payloads[kind] = memoryview(packet)[cursor:cursor+size]
        cursor += size
    # Padding is unspecified data, not necessarily zero. Only its length matters.
    if len(packet)-cursor != (-cursor) % 32:
        raise ValueError('invalid packet padding length')
    if 1 in payloads and 1020 in payloads:
        raise ValueError('ambiguous point cloud encodings')
    points = None
    if 1 in payloads:
        data = payloads[1]
        if len(data) != detected*16:
            raise ValueError('Cartesian point count mismatch')
        points = []
        for x, y, z, doppler in struct.iter_unpack('<4f', data):
            _finite((x,y,z,doppler), 1e6)
            distance = math.hypot(x,y,z)
            points.append(dict(range_m=distance, azimuth_rad=math.atan2(x,y),
                elevation_rad=math.atan2(z, math.hypot(x,y)), radial_velocity_m_s=doppler))
    if 1020 in payloads:
        data = payloads[1020]
        if len(data) != 20+detected*8:
            raise ValueError('compressed point count mismatch')
        units = struct.unpack_from('<5f', data)
        _finite(units, 1e6)
        if any(v <= 0 for v in units):
            raise ValueError('point scales must be positive')
        points = []
        for item in struct.iter_unpack('<bbhHH', data[20:]):
            elevation, azimuth, doppler, distance, snr = (x*u for x,u in zip(item,units))
            if abs(elevation) > math.pi/2 or abs(azimuth) > math.pi:
                raise ValueError('invalid spherical angles')
            _finite((doppler,distance,snr), 1e6)
            points.append(dict(range_m=distance, azimuth_rad=azimuth, elevation_rad=elevation,
                               radial_velocity_m_s=doppler, snr_db=snr))
    if 7 in payloads:
        if points is None or len(payloads[7]) != detected*4 or 1020 in payloads:
            raise ValueError('side information requires matching Cartesian points')
        for point, (snr, noise) in zip(points, struct.iter_unpack('<hh', payloads[7])):
            point.update(snr_db=snr*.1, noise_db=noise*.1)
    tracks = None
    if 1010 in payloads:
        data = payloads[1010]
        if len(data) % 112 or len(data) > 32*112:
            raise ValueError('invalid target list size')
        tracks, seen = [], set()
        for values in struct.iter_unpack('<I27f', data):
            tid, *v = values
            _finite(v)
            if tid in seen or tid > 249:
                raise ValueError('duplicate or reserved target ID')
            seen.add(tid)
            _finite(v[:9], 1e6)
            position, velocity = v[:3], v[3:6]
            distance = math.hypot(*position)
            tracks.append(dict(track_id=str(tid), position_m=position, velocity_m_s=velocity,
                acceleration_m_s2=v[6:9], error_covariance=v[9:25], gating_gain=v[25],
                firmware_confidence=v[26], range_to_frame_origin_m=distance,
                speed_m_s=math.hypot(*velocity), identity_verified=False))
    heights = None
    if 1012 in payloads:
        data = payloads[1012]
        if len(data) % 12 or len(data) > 32*12:
            raise ValueError('invalid height list size')
        heights, seen = [], set()
        for tid, high, low in struct.iter_unpack('<I2f', data):
            _finite((high,low), 1e6)
            if tid > 249 or tid in seen or low > high:
                raise ValueError('invalid height target or bounds')
            seen.add(tid)
            heights.append(dict(track_id=str(tid), max_z_m=high, min_z_m=low))
    assignments = None
    if 1011 in payloads:
        data = payloads[1011]
        if len(data) > 512 or any(v in (250,251,252) for v in data):
            raise ValueError('invalid target index payload')
        assignments = dict(point_count=len(data), associated_count=sum(v < 250 for v in data),
            weak_snr_count=sum(v == 253 for v in data), outside_boundary_count=sum(v == 254 for v in data),
            noise_count=sum(v == 255 for v in data), joined_to_current_points=False)
    presence = None
    if 1021 in payloads:
        data = payloads[1021]
        if len(data) != 4:
            raise ValueError('invalid presence size')
        presence, = struct.unpack('<I', data)
        if presence not in (0,1):
            raise ValueError('invalid presence flag')
    cloud = dict(reported_point_count=detected, decoded=points is not None)
    if points is not None:
        nearest = sorted(points, key=lambda p:p['range_m'])[:16]
        cloud.update(point_count=len(points), nearest_points=nearest, retained_point_count=len(nearest),
            selection='nearest16', minimum_range_m=min((p['range_m'] for p in points),default=None),
            maximum_range_m=max((p['range_m'] for p in points),default=None),
            mean_radial_velocity_m_s=(sum(p['radial_velocity_m_s'] for p in points)/len(points) if points else None))
    return dict(frame_number=number, cpu_cycles=cycles, point_cloud=cloud, tracks=tracks,
                target_heights=heights, target_index_summary=assignments, firmware_presence=presence,
                ignored_tlv_types=sorted(set(payloads)-{1,7,1010,1011,1012,1020,1021}))


class TIRadar:
    def __init__(self, *, sdk_version, coordinate_frame, calibration_id, max_gap_ms=1000):
        self.spec = self.capability(sdk_version=sdk_version, coordinate_frame=coordinate_frame,
                                   calibration_id=calibration_id, max_gap_ms=max_gap_ms)
        self.version, self.coordinates, self.calibration = sdk_version, coordinate_frame, calibration_id
        self.max_gap_ms = max_gap_ms
        self._lock = threading.RLock()
        self._closed = False
        self._clear()

    @staticmethod
    def capability(*, sdk_version, coordinate_frame, calibration_id, max_gap_ms=1000):
        _version(sdk_version); identifier(coordinate_frame); identifier(calibration_id)
        if type(max_gap_ms) is not int or not 1 <= max_gap_ms <= 10000:
            raise ValueError('max_gap_ms must be1..10000')
        digest = hashlib.sha256(json.dumps([sdk_version,coordinate_frame,calibration_id,max_gap_ms]).encode()).hexdigest()
        return CapabilitySpec('radar.ti.tracks','radar.tracks','radar','ti.radar','1.'+digest,
            (OutputField('scene','object'),), 'native', freshness_ms=max_gap_ms, stateful=True,
            reset_on_gap=True, interpretation='estimate', devices=('cpu',),
            sensor_requirements=('xwr6843-uart40-payload-length','mapped-capture-clock','declared-radar-frame'),
            license='adapter:Apache-2.0;firmware:separate-TI-license')

    def _clear(self):
        self._owner = self._last = None
        self._session = uuid.uuid4().hex

    def reset(self):
        with self._lock: self._clear()

    def close(self):
        with self._lock:
            self._clear(); self._closed = True

    def process(self, frame):
        with self._lock:
            if self._closed: raise RuntimeError('radar provider closed')
            try:
                return self._process(frame)
            except Exception:
                self._clear()
                raise

    def _process(self, frame):
        if (not isinstance(frame,InputFrame) or frame.modality != 'radar' or
                not isinstance(frame.sensor,SensorMetadata) or
                (frame.sensor.coordinate_frame,frame.sensor.calibration_id) != (self.coordinates,self.calibration) or
                type(frame.timestamp_ms) is not int or not 0 <= frame.timestamp_ms <= 2**53-1):
            raise ValueError('radar packet requires matching coordinates/calibration and integer capture time')
        identifier(frame.subject_id)
        sensor, ts = frame.sensor, frame.timestamp_ms
        seq = sensor.capture_sequence if sensor.capture_sequence is not None else sensor.sequence
        owner = (frame.subject_id,sensor.source_id,sensor.sensor_id,sensor.stream_epoch,
                 sensor.capture_sequence is not None)
        scene = parse_ti_radar(frame.data,sdk_version=self.version)
        continuous = False
        if owner == self._owner and self._last is not None:
            old_ts, old_seq, old_number = self._last
            delta = (scene['frame_number']-old_number) & 0xffffffff
            if ts <= old_ts or seq <= old_seq or delta == 0:
                raise ValueError('capture time/sequence and firmware frame must advance')
            continuous = seq == old_seq+1 and delta == 1 and ts-old_ts <= self.max_gap_ms
        if not continuous: self._clear()
        scene.update(session_id=self._session, continuity=continuous, capture_timestamp_ms=ts,
            coordinate_frame=self.coordinates, calibration_id=self.calibration, sdk_version=self.version,
            hardware_verified=False, calibration_verified=False, clock_mapping_verified=False,
            identity_verified=False, through_wall_verified=False, confidence_calibrated=False,
            track_ids_session_scoped=True, point_track_association_performed=False)
        result = self.spec.validate({'scene':scene})
        self._owner, self._last = owner, (ts,seq,scene['frame_number'])
        return result

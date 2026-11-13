"""ESP32 classic CSI CSV decoding and bounded channel-variation DSP."""
from collections import deque
import csv
import hashlib
import io
import json
import math
import re
import threading

from ..capabilities import CapabilitySpec, OutputField, SensorMetadata, identifier
from ..plugins import InputFrame

REVISION = '8633d67152db2808f141cc1595970aa9cf406045'
COLUMNS = ('type,id,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,'
           'aggregation,stbc,fec_coding,sgi,noise_floor,ampdu_cnt,channel,'
           'secondary_channel,local_timestamp,ant,sig_len,rx_format,len,first_word,data').split(',')
CARRIERS = tuple(k for k in range(-26,27) if k not in (0,-21,-7,7,21))


def _mac(value):
    if type(value) is not str or not re.fullmatch(r'(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}', value):
        raise ValueError('explicit colon-separated transmitter MAC required')
    return value.lower()


def parse_esp32_csi(line):
    """One classic ESP32 row only; no log stripping, eval, or serial port access."""
    if type(line) is not bytes or not 1 <= len(line) <= 4096:
        raise ValueError('one bounded ASCII CSI CSV row required')
    try:
        text = line.decode('ascii').rstrip('\r\n')
        if '\n' in text or '\r' in text:
            raise ValueError('one CSI row per input required')
        rows = list(csv.reader(io.StringIO(text), strict=True))
        if len(rows) != 1 or len(rows[0]) != len(COLUMNS) or rows[0][0] != 'CSI_DATA':
            raise ValueError('selected classic ESP32 25-column format required')
        fields = dict(zip(COLUMNS, rows[0]))
        fields['mac'] = _mac(fields['mac'])
        for name in COLUMNS[1:-1]:
            if name == 'mac': continue
            raw = fields[name]
            if not re.fullmatch(r'-?\d{1,10}', raw):
                raise ValueError('CSI metadata must contain bounded decimal integers')
            fields[name] = int(raw)
        for name in ('id','local_timestamp'):
            if not -2**31 <= fields[name] <= 2**32-1:
                raise ValueError('invalid32-bit CSI counter')
            fields[name] &= 0xffffffff  # Upstream uses signed printf for uint32.
        for name in ('smoothing','not_sounding','aggregation','stbc','fec_coding','sgi','first_word'):
            if fields[name] not in (0,1): raise ValueError('invalid CSI flag')
        if not (-128 <= fields['rssi'] <= 127 and -128 <= fields['noise_floor'] <= 127 and
                0 <= fields['rate'] <= 31 and 0 <= fields['mcs'] <= 7 and
                0 <= fields['ampdu_cnt'] <= 255 and 1 <= fields['channel'] <= 14 and
                0 <= fields['ant'] <= 1 and 0 <= fields['sig_len'] <= 4095):
            raise ValueError('invalid CSI radio metadata')
        if (fields['sig_mode'] not in (0,1) or fields['rx_format'] != fields['sig_mode'] or
                fields['bandwidth'] != 0 or fields['secondary_channel'] != 0 or fields['stbc'] != 0 or
                fields['len'] != (128 if fields['sig_mode'] == 0 else 256)):
            raise ValueError('select non-STBC20MHz legacy LLTF or HT LLTF+HTLTF firmware')
        # This grammar disallows nested JSON, booleans, exponents and huge ints.
        raw = fields.pop('data')
        if not re.fullmatch(r'\[\s*-?\d{1,3}(?:\s*,\s*-?\d{1,3})*\s*\]', raw):
            raise ValueError('signed byte CSI sample list required')
        values = [int(v) for v in raw[1:-1].split(',')]
        if len(values) != fields['len'] or any(not -128 <= v <= 127 for v in values):
            raise ValueError('CSI sample count/range mismatch; gain-compensated chips unsupported')
        carriers, amplitudes, phases = [], [], []
        for carrier in CARRIERS:
            index = carrier if carrier >= 0 else 64+carrier
            if fields['first_word'] and index < 2: continue
            imag, real = values[2*index:2*index+2]
            carriers.append(carrier)
            amplitudes.append(math.hypot(real, imag))
            phases.append(math.atan2(imag, real) if real or imag else None)
        return fields, carriers, amplitudes, phases
    except (UnicodeError, csv.Error) as exc:
        raise ValueError('malformed CSI CSV row') from exc


class ESP32CSI:
    def __init__(self, *, transmitter_mac, clock_calibration_id, window_ms=1000,
                 max_gap_ms=250, variation_threshold=.05):
        self.spec = self.capability(transmitter_mac=transmitter_mac, clock_calibration_id=clock_calibration_id,
            window_ms=window_ms, max_gap_ms=max_gap_ms, variation_threshold=variation_threshold)
        self.mac, self.clock_id = _mac(transmitter_mac), clock_calibration_id
        self.window_ms, self.max_gap_ms, self.threshold = window_ms, max_gap_ms, variation_threshold
        self._lock = threading.RLock()
        self.closed = False
        self._clear()

    @staticmethod
    def capability(*, transmitter_mac, clock_calibration_id, window_ms=1000,
                   max_gap_ms=250, variation_threshold=.05):
        mac = _mac(transmitter_mac); identifier(clock_calibration_id)
        if type(window_ms) is not int or not 100 <= window_ms <= 5000:
            raise ValueError('CSI window must be100..5000ms')
        if type(max_gap_ms) is not int or not 1 <= max_gap_ms <= min(1000,window_ms):
            raise ValueError('CSI gap limit must be1..min(1000,window_ms)')
        if type(variation_threshold) not in (int,float) or not math.isfinite(variation_threshold) or not 0 < variation_threshold <= 2:
            raise ValueError('positive dimensionless variation threshold<=2 required')
        digest = hashlib.sha256(json.dumps([mac,clock_calibration_id,window_ms,max_gap_ms,variation_threshold]).encode()).hexdigest()
        return CapabilitySpec('csi.esp32.channel_variation', 'csi.channel_variation', 'csi',
            'esp32.csi', '1.'+digest, (OutputField('channel_response','object'),), 'window',
            window_ms=window_ms, lookahead_ms=0, freshness_ms=1000, stateful=True, reset_on_gap=True,
            interpretation='measurement', devices=('cpu',),
            sensor_requirements=('classic-esp32-gain-control-disabled','20mhz-non-stbc','mapped-capture-clock'),
            license='adapter:Apache-2.0;firmware:Espressif-Apache-2.0')

    def _clear(self):
        self._history = deque()
        self._owner = self._last = None
        self._elapsed = 0
        self._host_origin = None
        self._clipped = False

    def reset(self):
        with self._lock: self._clear()

    def close(self):
        with self._lock:
            self._clear(); self.closed = True

    def process(self, frame):
        with self._lock:
            if self.closed: raise RuntimeError('CSI provider closed')
            try:
                if (not isinstance(frame,InputFrame) or frame.modality != 'csi' or
                        not isinstance(frame.sensor,SensorMetadata) or frame.sensor.calibration_id != self.clock_id or
                        type(frame.timestamp_ms) is not int or not 0 <= frame.timestamp_ms <= 2**53-1):
                    raise ValueError('CSI frame with declared mapped capture-clock calibration required')
                identifier(frame.subject_id)
                fields, carriers, amplitudes, phases = parse_esp32_csi(frame.data)
                if fields['mac'] != self.mac: raise ValueError('unexpected CSI transmitter')
                sensor = frame.sensor
                seq = sensor.capture_sequence if sensor.capture_sequence is not None else sensor.sequence
                owner = (frame.subject_id,sensor.source_id,sensor.sensor_id,sensor.stream_epoch,sensor.channels,
                         sensor.coordinate_frame,sensor.calibration_id,sensor.capture_sequence is not None,
                         tuple(carriers),*(fields[k] for k in ('rate','sig_mode','mcs','bandwidth','smoothing',
                         'not_sounding','aggregation','stbc','fec_coding','sgi','channel','secondary_channel','ant')))
                reason = None
                if self._owner is not None and owner != self._owner:
                    self._clear(); reason = 'ownership_or_radio_layout_changed'
                elif self._last:
                    host_ts, oldseq, oldid, oldclock = self._last
                    if frame.timestamp_ms < host_ts or seq <= oldseq:
                        raise ValueError('CSI capture time/sequence moved backward or repeated')
                    delta = (fields['local_timestamp']-oldclock) & 0xffffffff
                    packet_delta = (fields['id']-oldid) & 0xffffffff
                    if delta == 0 or packet_delta == 0:
                        raise ValueError('duplicate CSI device timestamp or transmitter sequence')
                    if delta >= 2**31 or packet_delta >= 2**31:
                        self._clear(); reason = 'device_or_transmitter_restarted'
                    elif seq != oldseq+1 or packet_delta != 1 or delta > self.max_gap_ms*1000:
                        self._clear(); reason = 'capture_gap'
                    elif abs((frame.timestamp_ms-self._host_origin)*1000-(self._elapsed+delta)) > 2000:
                        self._clear(); reason = 'capture_clock_disagreement'
                    else:
                        self._elapsed += delta
                rms = math.sqrt(sum(v*v for v in amplitudes)/len(amplitudes))
                normalized = tuple(v/rms for v in amplitudes) if rms else None
                previous = self._history[-1][2] if self._history else None
                delta_rms = (math.sqrt(sum((a-b)**2 for a,b in zip(normalized,previous))/len(normalized))
                             if normalized is not None and previous is not None else None)
                if normalized is None:
                    self._clear(); reason = 'zero_channel_energy'
                else:
                    self._history.append((self._elapsed,frame.timestamp_ms,normalized))
                    while self._history and self._elapsed-self._history[0][0] > self.window_ms*1000:
                        self._history.popleft()
                    if len(self._history) > 256:
                        self._history.popleft(); self._clipped = True
                if self._host_origin is None:
                    self._host_origin = frame.timestamp_ms
                self._owner = owner
                self._last = (frame.timestamp_ms,seq,fields['id'],fields['local_timestamp'])
                n = len(self._history)
                span = (self._history[-1][0]-self._history[0][0])/1000 if n else 0
                ready = n >= 8 and self._elapsed >= self.window_ms*1000 and span >= .8*self.window_ms
                variation = None
                if n >= 2:
                    means = [sum(row[2][i] for row in self._history)/n for i in range(len(carriers))]
                    variation = math.sqrt(sum(sum((row[2][i]-means[i])**2 for i in range(len(carriers)))
                                              for row in self._history)/(n*len(carriers)))
                result = dict(transmitter_mac=self.mac, firmware_source_revision=REVISION, radio=fields,
                    ltf='LLTF', subcarrier_indices=carriers, amplitude=amplitudes, phase_rad=phases,
                    amplitude_unit='receiver_integer_scale', amplitude_rms=rms,
                    normalized_amplitude_delta_rms=delta_rms, normalized_amplitude_variation_rms=variation,
                    variation_threshold=self.threshold, threshold_exceeded=(variation > self.threshold if ready else None),
                    ready=ready, sample_count=n, requested_window_ms=self.window_ms, effective_span_ms=span,
                    window_start_ms=self._history[0][1] if n else None, window_end_ms=frame.timestamp_ms,
                    sample_limit_ever_hit=self._clipped, reset_reason=reason,
                    first_word_invalid=bool(fields['first_word']), phase_calibrated=False,
                    clock_calibration_id=self.clock_id, clock_calibration_verified=False,
                    hardware_verified=False, motion_verified=False, presence=None,
                    range_m=None, velocity_m_s=None, through_wall_object_verified=False)
                return self.spec.validate({'channel_response': result})
            except Exception:
                self._clear()
                raise

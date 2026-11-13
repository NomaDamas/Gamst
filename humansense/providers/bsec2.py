"""Bosch BSEC2 callback transport to bounded environment context.

BSEC runs in the explicitly configured sensor firmware. This host bridge does
not infer gas identity from resistance or claim to execute the embedded model.
"""
import hashlib
import json
import math
import re
import struct
import threading
import uuid
import zlib

from ..capabilities import CapabilitySpec, OutputField, SensorMetadata, identifier
from .yoloe import _hash, prompt_labels

REVISION = '4f559a6aa450f0ce436e602b42dc384f52128c66'
MAGIC = b'GBSC001\0'
HEADER = struct.Struct('<8s4BBI32s')
ROW = struct.Struct('<qfBBB')
MAX_OUTPUTS = 23
# name, unit, engineering sanity bound (not physical accuracy/operating range).
MEASUREMENTS = {
    6: ('sensor_temperature', 'degC', -100, 200),
    7: ('pressure', 'Pa', 1, 200000),
    8: ('relative_humidity', 'percent', 0, 100),
    9: ('gas_resistance', 'ohm', 0, 1e12),
}
ESTIMATES = {
    1: ('iaq_index', 'vendor_index', 0, 500),
    2: ('static_iaq', 'vendor_unscaled_index', 0, 1e9),
    3: ('co2_equivalent', 'ppm_equivalent', 0, 1e9),
    4: ('breath_voc_equivalent', 'ppm_equivalent', 0, 1e9),
    14: ('heat_compensated_temperature', 'degC', -100, 200),
    15: ('heat_compensated_humidity', 'percent', 0, 100),
    18: ('compensated_gas_vendor_signal', 'vendor_log_signal', -1e9, 1e9),
    21: ('filtered_gas_percentage', 'percent', 0, 100),
}


def _config(config_sha256, bsec_version, classes, calibration_id, sensor_origin_ns, capture_origin_ms, max_gap_ms):
    _hash(config_sha256); identifier(calibration_id)
    if bsec_version != '2.6.1.0':
        raise ValueError('this pinned callback bridge requires BSEC2.6.1.0')
    version = tuple(map(int,bsec_version.split('.')))
    if any(v > 255 for v in version): raise ValueError('BSEC version bytes must fit uint8')
    if type(classes) not in (tuple,list) or len(classes) not in (0,2,3,4):
        raise ValueError('declare no classifier or exact ordered2..4 trained class labels')
    labels = prompt_labels(classes) if classes else ()
    if type(sensor_origin_ns) is not str or not re.fullmatch(r'0|[1-9][0-9]{0,18}',sensor_origin_ns) or int(sensor_origin_ns)>2**63-1:
        raise ValueError('sensor_origin_ns must be decimal int64 text')
    if type(capture_origin_ms) is not int or not 0 <= capture_origin_ms <= 2**53-1:
        raise ValueError('capture_origin_ms must be exact nonnegative milliseconds')
    if type(max_gap_ms) is not int or not 1 <= max_gap_ms <= 3600000:
        raise ValueError('max_gap_ms must be1..3600000')
    return version,labels,int(sensor_origin_ns)


def decode_bsec2(packet):
    if type(packet) is not bytes or not HEADER.size+ROW.size+4 <= len(packet) <= HEADER.size+MAX_OUTPUTS*ROW.size+4:
        raise ValueError('bounded complete GBSC001 callback packet required')
    magic,a,b,c,d,count,sequence,digest=HEADER.unpack_from(packet)
    if magic != MAGIC or not 1 <= count <= MAX_OUTPUTS or len(packet) != HEADER.size+count*ROW.size+4:
        raise ValueError('invalid BSEC callback header/count/length')
    if zlib.crc32(packet[:-4]) != struct.unpack_from('<I',packet,len(packet)-4)[0]:
        raise ValueError('BSEC packet CRC mismatch')
    rows=[];seen=set()
    for i in range(count):
        ns,value,sensor_id,accuracy,dimensions=ROW.unpack_from(packet,HEADER.size+i*ROW.size)
        if ns < 0 or not math.isfinite(value) or sensor_id in seen or accuracy > 3 or dimensions != 1:
            raise ValueError('invalid or duplicate BSEC output')
        seen.add(sensor_id)
        if sensor_id in MEASUREMENTS or sensor_id in ESTIMATES:
            _,_,low,high=(MEASUREMENTS|ESTIMATES)[sensor_id]
            if not low <= value <= high or sensor_id == 9 and value == 0:
                raise ValueError('BSEC output outside declared engineering bounds')
        elif sensor_id in (12,13):
            if value not in (0,1):raise ValueError('BSEC status must be0or1')
        elif sensor_id == 26:
            if value != int(value) or not 0 <= value <= 9:raise ValueError('heater index must be0..9')
        elif 22 <= sensor_id <= 25:
            if not 0 <= value <= 1 or accuracy not in (0,3):raise ValueError('invalid gas prediction score/status')
        else:
            raise ValueError('unsupported BSEC output ID (regression units require a separate trained profile)')
        rows.append((ns,value,sensor_id,accuracy))
    return (a,b,c,d),sequence,digest.hex(),rows


class BSEC2Packets:
    """Bounded transport assembler; caller owns port, timeout and reconnect reset."""
    def __init__(self):self.buffer=bytearray()
    def reset(self):self.buffer.clear()
    def feed(self,chunk):
        if type(chunk) is not bytes or len(chunk)>4096:raise ValueError('at most4096 transport bytes per feed')
        self.buffer.extend(chunk);packets=[]
        while True:
            start=self.buffer.find(MAGIC)
            if start<0:
                del self.buffer[:-7];break
            if start:del self.buffer[:start]
            if len(self.buffer)<HEADER.size:break
            count=self.buffer[12]
            if not 1<=count<=MAX_OUTPUTS:del self.buffer[0];continue
            size=HEADER.size+count*ROW.size+4
            if len(self.buffer)<size:break
            packet=bytes(self.buffer[:size])
            if zlib.crc32(packet[:-4])!=struct.unpack_from('<I',packet,size-4)[0]:
                del self.buffer[0];continue
            packets.append(packet);del self.buffer[:size]
        return packets


class BoschBSEC2:
    def __init__(self, *, config_sha256, bsec_version, calibration_id, sensor_origin_ns,
                 capture_origin_ms, classes=(), max_gap_ms=120000):
        args=dict(config_sha256=config_sha256,bsec_version=bsec_version,calibration_id=calibration_id,
            sensor_origin_ns=sensor_origin_ns,capture_origin_ms=capture_origin_ms,classes=classes,max_gap_ms=max_gap_ms)
        self.spec=self.capability(**args)
        self.version,self.classes,self.origin_ns=_config(**args)
        self.digest,self.calibration_id,self.origin_ms,self.max_gap_ms=config_sha256,calibration_id,capture_origin_ms,max_gap_ms
        self.lock=threading.RLock();self.closed=False;self._clear()

    @staticmethod
    def capability(*,config_sha256,bsec_version,calibration_id,sensor_origin_ns,capture_origin_ms,
                   classes=(),max_gap_ms=120000):
        args=dict(config_sha256=config_sha256,bsec_version=bsec_version,calibration_id=calibration_id,
            sensor_origin_ns=sensor_origin_ns,capture_origin_ms=capture_origin_ms,classes=classes,max_gap_ms=max_gap_ms)
        _config(**args)
        digest=hashlib.sha256(json.dumps([REVISION,args],sort_keys=True).encode()).hexdigest()
        return CapabilitySpec('environment.bosch.bsec2','environment.gas_context','environment',
            'bosch.bsec2','1.'+digest,(OutputField('measurements','array'),OutputField('estimates','array'),
            OutputField('classification','object'),OutputField('summary','object')),'native',
            stateful=True,reset_on_gap=True,freshness_ms=120000,interpretation='estimate',devices=('cpu',),
            sensor_requirements=('BME688-with-explicit-BSEC2-firmware','GBSC001-callback-packet','declared-clock-calibration'),
            license='adapter:Apache-2.0;BSEC-binary-and-config:Bosch-license-separate')

    def _clear(self):self.owner=self.last=self.segment=None

    def reset(self):
        with self.lock:
            if self.closed:raise RuntimeError('BSEC bridge closed')
            self._clear()

    def close(self):
        with self.lock:self.closed=True;self._clear()

    def process(self,frame):
        with self.lock:
            if self.closed:raise RuntimeError('BSEC bridge closed')
            try:
                if frame.modality!='environment' or not isinstance(frame.sensor,SensorMetadata) or frame.sensor.calibration_id!=self.calibration_id:
                    raise ValueError('environment packet with matching declared clock/calibration required')
                identifier(frame.subject_id)
                if type(frame.timestamp_ms) is not int or not 0 <= frame.timestamp_ms <= 2**53-1:
                    raise ValueError('exact capture milliseconds required')
                version,firmware_seq,digest,rows=decode_bsec2(frame.data)
                if version!=self.version or digest!=self.digest:raise ValueError('firmware BSEC version/config declaration mismatch')
                mapped=[]
                for ns,value,sensor_id,accuracy in rows:
                    delta=ns-self.origin_ns
                    stamp=self.origin_ms+delta//1000000
                    if delta < 0 or stamp>2**53-1:raise ValueError('BSEC time precedes declared clock origin or overflows')
                    mapped.append((stamp,ns,value,sensor_id,accuracy))
                latest=max(r[0] for r in mapped);latest_ns=max(r[1] for r in mapped)
                if frame.timestamp_ms!=latest:raise ValueError('outer capture must equal latest calibrated output time')
                sensor=frame.sensor;seq=sensor.capture_sequence if sensor.capture_sequence is not None else sensor.sequence
                owner=(frame.subject_id,sensor.source_id,sensor.sensor_id,sensor.stream_epoch,sensor.calibration_id,
                       sensor.coordinate_frame,sensor.capture_sequence is not None)
                continuous=False
                if owner==self.owner and self.last:
                    old_ts,old_ns,old_seq,old_fw=self.last
                    fw_delta=(firmware_seq-old_fw)&0xffffffff
                    if latest<=old_ts or latest_ns<=old_ns or seq<=old_seq or fw_delta==0 or fw_delta>=2**31:
                        raise ValueError('BSEC capture/firmware sequence must advance')
                    continuous=seq==old_seq+1 and fw_delta==1 and latest-old_ts<=self.max_gap_ms
                if not continuous:self.segment=uuid.uuid4().hex
                measurements=[];estimates=[];statuses={};gas={}
                for stamp,ns,value,sensor_id,accuracy in mapped:
                    if sensor_id in MEASUREMENTS or sensor_id in ESTIMATES:
                        name,unit,_,_=(MEASUREMENTS|ESTIMATES)[sensor_id]
                        item=dict(name=name,value=value,unit=unit,capture_timestamp_ms=stamp,
                            age_within_packet_ms=latest-stamp,vendor_accuracy=accuracy)
                        (measurements if sensor_id in MEASUREMENTS else estimates).append(item)
                    elif sensor_id in (12,13,26):
                        statuses[{12:'stabilized',13:'run_in_complete',26:'heater_index'}[sensor_id]]=dict(value=int(value),capture_timestamp_ms=stamp)
                    else:gas[sensor_id]=(stamp,ns,value,accuracy)
                expected=set(range(22,22+len(self.classes)))
                if not self.classes:state='unconfigured'
                elif set(gas)-expected:state='unmapped_channels'
                elif set(gas)!=expected:state='incomplete'
                elif any(row[3]!=3 for row in gas.values()):state='collecting_features'
                elif len({row[1] for row in gas.values()})!=1 or any(row[1]!=latest_ns for row in gas.values()):state='stale_or_mixed_timestamps'
                else:state='prediction_available'
                candidates=[dict(label=label,score=gas[22+i][2]) for i,label in enumerate(self.classes)] if state=='prediction_available' else []
                result=self.spec.validate(dict(measurements=measurements,estimates=estimates,
                    classification=dict(status=state,candidates=candidates,trained_classes=list(self.classes),
                        calibrated_probabilities_verified=False,concentration_measured=False),
                    summary=dict(sensor_model='BME688',bsec_version='.'.join(map(str,version)),config_sha256=digest,
                        configuration_attested=False,physical_accuracy_verified=False,model_quality_verified=False,
                        firmware_sequence=firmware_seq,state_segment=self.segment,continuous=continuous,
                        firmware_state_reset=False,firmware_window_ms=None,clock_calibration_verified=False,
                        capture_timestamp_ms=latest,sensor_timestamp_ns=str(latest_ns),statuses=statuses,
                        complete_snapshot=False,missing_channels_not_carried_forward=True)))
                self.owner,self.last=owner,(latest,latest_ns,seq,firmware_seq)
                return result
            except BaseException:self._clear();raise

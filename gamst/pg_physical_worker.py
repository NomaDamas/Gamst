"""Bounded serial protocol inside the sole owned PG canonical worker."""
import hashlib
import sys

from .pg_artifact import manifest_digest
from .pg_generation import HEADER_LIMIT,PACKET_LIMIT,layout,crop_digest,infer,ANSWERS
from .sam_audio_artifact import _json,_encoded,_keys,_digest
from .capabilities import identifier,SensorMetadata


def header_value(value,generation_sha):
    _keys(value,('schema_version','operation','request_id','generation_sha256','binding','concept','shape','crop',
        'pixels_sha256','crop_sha256','payload_bytes'),'physical request')
    if type(value['schema_version']) is not int or value['schema_version']!=1 or value['operation']!='infer':raise ValueError('unsupported PG operation')
    identifier(value['request_id']);_digest(value['pixels_sha256']);_digest(value['crop_sha256'])
    if value['generation_sha256']!=generation_sha or value['concept'] not in ANSWERS:raise ValueError('unknown policy/concept')
    size=layout(value['shape'],value['crop'])
    if type(value['payload_bytes']) is not int or value['payload_bytes']!=size or size+HEADER_LIMIT>PACKET_LIMIT:raise ValueError('invalid raw packet length')
    binding=value['binding'];_keys(binding,('timestamp_ms','subject_id','sensor','worker_epoch','track_provenance'),'capture binding')
    identifier(binding['subject_id']);identifier(binding['worker_epoch']);identifier(binding['track_provenance'])
    ts=binding['timestamp_ms']
    if type(ts) is not int or not 0<=ts<=2**53-1:raise ValueError('capture timestamp required')
    sensor=binding['sensor'];_keys(sensor,('source_id','sensor_id','sequence','coordinate_frame','channels','calibration_id','stream_epoch','capture_sequence'),'capture sensor')
    if type(sensor['channels']) is not list:raise ValueError('declared RGB channel list required')
    SensorMetadata(**dict(sensor,channels=tuple(sensor['channels'])))
    if sensor['channels']!=['R','G','B'] or sensor['coordinate_frame'] is None or sensor['stream_epoch'] is None or sensor['capture_sequence'] is None:
        raise ValueError('explicit RGB/coordinate/epoch/capture sequence required')
    return value


def serve(model,generate,manifest,generation,receipt):
    generation_sha=receipt['generation_sha256'];stream=sys.stdin.buffer
    while True:
        raw=stream.readline(HEADER_LIMIT+1)
        if raw==b'':return
        if not raw.endswith(b'\n') or len(raw)>HEADER_LIMIT:raise ValueError('bounded request header required')
        header=header_value(_json(raw,HEADER_LIMIT),generation_sha)
        # BufferedReader.read obtains exactly this bounded owned payload or EOF.
        payload=stream.read(header['payload_bytes'])
        if (len(payload)!=header['payload_bytes'] or hashlib.sha256(payload).hexdigest()!=header['pixels_sha256'] or
            crop_digest(payload,header['shape'],header['crop'])!=header['crop_sha256']):
            raise ValueError('immutable capture pixels/crop mismatch')
        result=infer(model,generate,manifest,generation,header,payload)
        response=dict(status='generated',request_id=header['request_id'],request_sha256=manifest_digest(header),
            canonical_sha256=receipt['canonical_sha256'],generation_sha256=generation_sha,result=result)
        raw=_encoded(response)
        if len(raw)+1>HEADER_LIMIT:raise ValueError('bounded PG response required')
        sys.stdout.buffer.write(raw+b'\n');sys.stdout.buffer.flush()

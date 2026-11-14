"""Immutable PCM/RGB/mask request for explicit SAM visual conditioning.

Only capture declarations and payload integrity are checked here. Frozen policy,
outer-frame identity, native feature alignment and media ownership are checked
again by the consumer. A packet does not prove clock or segmentation accuracy.
"""
from dataclasses import asdict, replace
import hashlib
import json
import struct

from .capabilities import SensorMetadata, bounded_json, identifier
from .provider_wire import encode_frame
from .separation_media import frame_binding

MAGIC = b'GSAMVIS1'
LIMIT = 16*1024*1024
HEADER_LIMIT = 65536


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _integer(v, a, b):
    if type(v) is not int or not a <= v <= b:
        raise ValueError('visual packet integer outside bounds')
    return v


def _keys(value, names):
    if type(value) is not dict or set(value) != set(names.split()):
        raise ValueError('unexpected visual packet fields')


def _hash(v):
    if type(v) is not str or len(v) != 64 or any(c not in '0123456789abcdef' for c in v):
        raise ValueError('lowercase SHA256 required')


def _validate(h):
    _keys(h,'audio audio_sha256 width height visual_sensor track_id segment_id mask_origin mask_polarity clock frames')
    _keys(h['audio'],'metadata layout')
    _keys(h['audio']['metadata'],'modality timestamp_ms subject_id sample_rate sensor')
    _keys(h['audio']['layout'],'kind dtype shape')
    audio_sensor=h['audio']['metadata']['sensor']
    if type(audio_sensor) is not dict or set(audio_sensor)!=set(SensorMetadata.__dataclass_fields__):
        raise ValueError('exact audio SensorMetadata required')
    if type(audio_sensor['channels']) is not list:
        raise ValueError('audio channel list required')
    SensorMetadata(**dict(audio_sensor,channels=tuple(audio_sensor['channels'])))
    _hash(h['audio_sha256'])
    w,hgt = _integer(h['width'],16,640),_integer(h['height'],16,480)
    sensor=h['visual_sensor']
    if type(sensor) is not dict or set(sensor)!=set(SensorMetadata.__dataclass_fields__):
        raise ValueError('exact visual SensorMetadata required')
    if type(sensor['channels']) is not list or sensor['channels'] != ['R','G','B']:
        raise ValueError('explicit RGB channel order required')
    s=SensorMetadata(**dict(sensor,channels=tuple(sensor['channels'])))
    for name in ('stream_epoch','coordinate_frame','calibration_id'):
        if getattr(s,name) is None:
            raise ValueError('visual epoch, geometry and calibration required')
    for name in ('track_id','segment_id'):
        identifier(h[name])
    if h['mask_polarity']!='zero-kept':
        raise ValueError('explicit zero-kept mask polarity required')
    _keys(h['mask_origin'],'producer_id revision')
    for v in h['mask_origin'].values(): identifier(v)
    clock=h['clock']
    _keys(clock,'audio_clock_id camera_clock_id alignment_id camera_to_audio_offset_us clock_error_us max_age_us')
    for k in ('audio_clock_id','camera_clock_id','alignment_id'): identifier(clock[k])
    _integer(clock['camera_to_audio_offset_us'],-(2**53-1),2**53-1)
    _integer(clock['max_age_us'],1,250000)
    _integer(clock['clock_error_us'],0,clock['max_age_us'])
    records=h['frames']
    if type(records) is not list or not 1 <= len(records) <= 128:
        raise ValueError('one to128 captured frames required')
    previous=None
    for r in records:
        _keys(r,'capture_sequence interval_us rgb_sha256 mask_sha256 status')
        seq=_integer(r['capture_sequence'],0,2**53-1)
        v=r['interval_us']
        if type(v) is not list or len(v)!=2:
            raise ValueError('capture interval pair required')
        a,b=_integer(v[0],0,2**53-1),_integer(v[1],0,2**53-1)
        if not 0<=b-a<=clock['max_age_us'] or r['status']!='observed':
            raise ValueError('bounded observed exposure required')
        if previous is not None and (seq<=previous[0] or a<previous[1] or b<=previous[2]):
            raise ValueError('capture ordering regressed within packet')
        previous=seq,a,b
        _hash(r['rgb_sha256']); _hash(r['mask_sha256'])
    if s.capture_sequence != records[0]['capture_sequence']:
        raise ValueError('visual sensor must identify first original capture')
    shape=h['audio']['layout']['shape']
    if type(shape) is not list or len(shape)!=1:
        raise ValueError('mono PCM shape required')
    samples=_integer(shape[0],48000,192000)
    return samples,w*hgt,len(records)


def unpack_visual_packet(data):
    if type(data) is not bytes or not 12<len(data)<=LIMIT or data[:8]!=MAGIC:
        raise ValueError('bounded immutable SAM visual packet required')
    n,=struct.unpack('>I',data[8:12])
    if not 0<n<=HEADER_LIMIT or 12+n>=len(data):
        raise ValueError('visual header length mismatch')
    def pairs(items):
        value={}
        for k,v in items:
            if k in value: raise ValueError('duplicate visual field')
            value[k]=v
        return value
    def reject(v): raise ValueError('nonfinite visual JSON')
    try:
        header=json.loads(data[12:12+n].decode(),object_pairs_hook=pairs,parse_constant=reject)
    except (UnicodeError,RecursionError) as error:
        raise ValueError('invalid visual JSON') from error
    count,pixels,frames=_validate(header)
    if len(data)!=12+n+count*4+frames*pixels*4:
        raise ValueError('visual payload length mismatch')
    start=12+n
    pcm=data[start:start+count*4]
    binding=frame_binding(header['audio'],pcm,generation=0,config_id='sam-visual-packet')
    if _digest(pcm)!=header['audio_sha256']:
        raise ValueError('PCM hash mismatch')
    start+=count*4
    rgb=data[start:start+frames*pixels*3]
    masks=data[start+frames*pixels*3:]
    for i,r in enumerate(header['frames']):
        image=rgb[i*pixels*3:(i+1)*pixels*3]
        mask=masks[i*pixels:(i+1)*pixels]
        if _digest(image)!=r['rgb_sha256'] or _digest(mask)!=r['mask_sha256']:
            raise ValueError('visual capture hash mismatch')
        if any(v not in (0,1) for v in mask) or 0 not in mask:
            raise ValueError('binary mask with retained zero-valued support required')
    # Kept explicit for downstream outer InputFrame binding: no independent
    # audio_end_us field can disagree with the original integer timestamp_ms.
    _integer(binding['capture_end_ms']*1000,0,2**53-1)
    return header,pcm,rgb,masks


def visual_frame(audio, rgb, masks, *, visual_sensor, intervals_us, sequences,
                 track_id, segment_id, mask_origin, clock):
    """Snapshot plain NCHW RGB and N1HW binary masks without resizing them."""
    import numpy as np
    audio_header,pcm=encode_frame(audio)
    frame_binding(audio_header,pcm,generation=0,config_id='sam-visual-packet')
    if (type(rgb) is not np.ndarray or rgb.dtype!=np.uint8 or rgb.ndim!=4 or
            not 1<=rgb.shape[0]<=128 or rgb.shape[1]!=3 or
            not 16<=rgb.shape[2]<=480 or not 16<=rgb.shape[3]<=640):
        raise ValueError('bounded plain NCHW RGB uint8 required')
    shape=rgb.shape
    if (type(masks) is not np.ndarray or masks.dtype!=np.uint8 or
            masks.shape!=(shape[0],1,shape[2],shape[3])):
        raise ValueError('matching plain N1HW uint8 masks required')
    if len(pcm)+shape[0]*shape[2]*shape[3]*4+12>LIMIT:
        raise ValueError('visual packet exceeds wire budget before copy')
    rgb,masks=rgb.copy(order='C'),masks.copy(order='C')
    if (rgb.dtype!=np.uint8 or masks.dtype!=np.uint8 or rgb.shape!=shape or
            masks.shape!=(shape[0],1,shape[2],shape[3])):
        raise ValueError('visual geometry changed during copy')
    if not isinstance(visual_sensor,SensorMetadata):
        raise ValueError('visual SensorMetadata required')
    intervals_us=bounded_json(intervals_us,max_bytes=8192)
    sequences=bounded_json(sequences,max_bytes=4096)
    if type(intervals_us) is not list or type(sequences) is not list or len(intervals_us)!=shape[0] or len(sequences)!=shape[0]:
        raise ValueError('one interval and sequence per original frame required')
    records=[]
    for i in range(shape[0]):
        records.append(dict(capture_sequence=sequences[i],interval_us=intervals_us[i],
            rgb_sha256=_digest(rgb[i].tobytes()),mask_sha256=_digest(masks[i].tobytes()),status='observed'))
    sensor=asdict(visual_sensor); sensor['channels']=list(sensor['channels'])
    header=bounded_json(dict(audio=audio_header,audio_sha256=_digest(pcm),width=shape[3],height=shape[2],
        visual_sensor=sensor,track_id=track_id,segment_id=segment_id,mask_origin=mask_origin,
        clock=clock,mask_polarity='zero-kept',frames=records),max_bytes=HEADER_LIMIT)
    _validate(header)
    encoded=json.dumps(header,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if 12+len(encoded)+len(pcm)+rgb.nbytes+masks.nbytes>LIMIT:
        raise ValueError('complete visual packet exceeds wire budget')
    data=MAGIC+struct.pack('>I',len(encoded))+encoded+pcm+rgb.tobytes()+masks.tobytes()
    unpack_visual_packet(data)
    return replace(audio,data=data)


def visual_policy(header):
    """Extract declarations that must equal the provider's frozen visual policy."""
    def source(sensor):
        return {k:sensor[k] for k in ('source_id','sensor_id','coordinate_frame','calibration_id','channels')}
    return bounded_json(dict(clock=header['clock'],width=header['width'],height=header['height'],
        audio_sensor=source(header['audio']['metadata']['sensor']),
        visual_sensor=source(header['visual_sensor']),track_id=header['track_id'],
        segment_id=header['segment_id'],mask_origin=header['mask_origin'],
        mask_polarity=header['mask_polarity']))


def visual_request_binding(outer_header, payload, *, policy, generation,
                           config_id, owner_revision=0):
    """Independently bind an immutable request at each side of the process pipe.

    SensorStreams may assign a runtime epoch/sequence, but original capture ID,
    source, subject, PCM timing and geometry must still agree. Original capture
    epochs are separately retained. Cross-window capture reuse is not rejected.
    """
    header,pcm,_,_=unpack_visual_packet(payload)
    _keys(outer_header,'metadata layout')
    if outer_header['layout']!={'kind':'bytes'}:
        raise ValueError('visual request must be a composite bytes frame')
    original=header['audio']['metadata']
    outer=outer_header['metadata']
    _keys(outer,'modality timestamp_ms subject_id sample_rate sensor')
    if any(outer[k]!=original[k] or type(outer[k]) is not type(original[k])
           for k in ('modality','timestamp_ms','subject_id','sample_rate')):
        raise ValueError('outer audio identity or capture timing differs')
    osensor=outer['sensor']; original_sensor=original['sensor']
    if type(osensor) is not dict or set(osensor)!=set(SensorMetadata.__dataclass_fields__):
        raise ValueError('exact outer sensor required')
    if type(osensor['channels']) is not list:
        raise ValueError('outer channel list required')
    SensorMetadata(**dict(osensor,channels=tuple(osensor['channels'])))
    identity=('source_id','sensor_id','coordinate_frame','calibration_id','channels')
    if any(osensor[k]!=original_sensor[k] for k in identity):
        raise ValueError('outer audio source binding differs')
    capture=lambda s:s['capture_sequence'] if s['capture_sequence'] is not None else s['sequence']
    if capture(osensor)!=capture(original_sensor):
        raise ValueError('outer capture sequence differs')
    canonical=lambda value:json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
    if canonical(visual_policy(header))!=canonical(bounded_json(policy)):
        raise ValueError('visual request differs from frozen provider policy')
    audio_header=dict(metadata=outer,layout=header['audio']['layout'])
    binding=frame_binding(audio_header,pcm,generation=generation,config_id=config_id,
                          owner_revision=owner_revision)
    visual={k:v for k,v in header.items() if k not in ('audio','audio_sha256')}
    inventory=json.dumps(visual,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    binding.update(input_sha256=_digest(payload),audio_input_sha256=_digest(pcm),
        visual_inventory_sha256=_digest(inventory),
        original_audio_epoch=original_sensor['stream_epoch'],
        original_visual_epoch=header['visual_sensor']['stream_epoch'])
    return bounded_json(binding)


def validate_visual_policy(value):
    """Dependency-light validation before any worker is launched."""
    value=bounded_json(value)
    _keys(value,'clock width height audio_sensor visual_sensor track_id segment_id mask_origin mask_polarity')
    _integer(value['width'],16,640); _integer(value['height'],16,480)
    for name in ('track_id','segment_id'): identifier(value[name])
    if value['mask_polarity']!='zero-kept':
        raise ValueError('zero-kept visual policy required')
    _keys(value['mask_origin'],'producer_id revision')
    for v in value['mask_origin'].values(): identifier(v)
    clock=value['clock']
    _keys(clock,'audio_clock_id camera_clock_id alignment_id camera_to_audio_offset_us clock_error_us max_age_us')
    for k in ('audio_clock_id','camera_clock_id','alignment_id'): identifier(clock[k])
    _integer(clock['camera_to_audio_offset_us'],-(2**53-1),2**53-1)
    _integer(clock['max_age_us'],1,250000)
    _integer(clock['clock_error_us'],0,clock['max_age_us'])
    for name in ('audio_sensor','visual_sensor'):
        source=value[name]
        _keys(source,'source_id sensor_id coordinate_frame calibration_id channels')
        if type(source['channels']) is not list:
            raise ValueError('declared channel list required')
        SensorMetadata(**dict(source,channels=tuple(source['channels']),sequence=0))
        if name=='visual_sensor':
            if source['channels']!=['R','G','B'] or any(source[k] is None for k in ('coordinate_frame','calibration_id')):
                raise ValueError('calibrated RGB visual source required')
        elif len(source['channels'])!=1:
            raise ValueError('mono audio source required')
    return value

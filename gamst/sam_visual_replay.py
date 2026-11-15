"""Replay frozen SAM visual packets through an owned native worker, metadata only."""
import argparse
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import re
import sys

from .engine import HumanSense
from .process_provider import ProcessProvider
from .provider_wire import decode_frame,encode_frame
from .providers.sam_audio_context import configuration
from .sam_visual_packet import LIMIT,unpack_visual_packet,visual_request_binding
from .separation_media import PROFILE
from .stereo_replay import _read,_json,_sha


def recorded_frame(path,digest):
    data=_read(path,LIMIT,_sha(digest))
    header,pcm,_,_=unpack_visual_packet(data)
    frame=decode_frame(header['audio'],pcm)
    return replace(frame,data=data)


def recording(manifest,sha256):
    path=Path(manifest)
    doc=_json(_read(path,1024*1024,_sha(sha256)))
    if (type(doc) is not dict or set(doc)!={'schema_version','config','packets'} or
            type(doc['schema_version']) is not int or doc['schema_version']!=1):
        raise ValueError('exact SAM visual recording schema1 required')
    config=configuration(doc['config'])
    if 'visual_policy' not in config:raise ValueError('explicit visual configuration required')
    rows=doc['packets']
    if type(rows) is not list or not 1<=len(rows)<=1024:
        raise ValueError('one to1024 packets required')
    names=set();entries=[];total=0;subject=None
    for row in rows:
        if type(row) is not dict or set(row)!={'file','sha256'}:
            raise ValueError('exact recording file/hash entry required')
        name=row['file']
        if type(name) is not str or re.fullmatch(r'[a-zA-Z0-9_-]{1,100}\.gsamv',name) is None or name in names:
            raise ValueError('unique local visual packet basename required')
        names.add(name);target=path.parent/name
        frame=recorded_frame(target,row['sha256'])
        visual_request_binding(*encode_frame(frame),policy=config['visual_policy'],generation=0,config_id='preflight')
        if subject is not None and frame.subject_id!=subject:
            raise ValueError('recording mixes subjects')
        subject=frame.subject_id
        total+=len(frame.data)
        if total>128*1024*1024:raise ValueError('recording exceeds128MiB budget')
        entries.append((target,row['sha256']))
    return config,entries,subject


async def replay(manifest,sha256,*,python=sys.executable,consent=False):
    if consent is not True:raise PermissionError('explicit local replay consent required')
    config,entries,subject=recording(manifest,sha256)
    policy=config['visual_policy'];audio=policy['audio_sensor'];visual=policy['visual_sensor']
    async with HumanSense(consent=True) as human:
        owner=human.enable_separation_media(subject_id=subject,source_id=audio['source_id'],sensor_id=audio['sensor_id'],
            visual_source={'source_id':visual['source_id'],'sensor_id':visual['sensor_id']},max_bytes=64*1024*1024)
        worker=await ProcessProvider.open(python,PROFILE,config,separation_owner=owner)
        previous=None
        try:
            for path,digest in entries:
                frame=recorded_frame(path,digest)
                header,_,_,_=unpack_visual_packet(frame.data)
                epochs=(header['audio']['metadata']['sensor']['stream_epoch'],header['visual_sensor']['stream_epoch'])
                if previous is not None and epochs!=previous:await worker.reset()
                value=await worker.process(frame)
                if value is None:raise RuntimeError('replay result invalidated')
                # This metadata-only replay does not export transient waveform
                # lookup tokens that cease to work when its local runtime ends.
                owner.discard(value['media']['reference'])
                value['media']={};value=worker.spec.validate(value)
                yield dict(packet_sha256=digest,capture_timestamp_ms=frame.timestamp_ms,
                    capability_id=worker.spec.id,capability_version=worker.spec.version,
                    value=value,media_retained=False,trained_quality_verified=False)
                previous=epochs
        finally:await worker.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True);parser.add_argument('--sha256',required=True)
    parser.add_argument('--python',default=sys.executable)
    parser.add_argument('--consent',action='store_true',required=True)
    args=parser.parse_args()
    async def run():
        async for value in replay(args.manifest,args.sha256,python=args.python,consent=args.consent):
            print(json.dumps(value,sort_keys=True,allow_nan=False),flush=True)
    asyncio.run(run())


if __name__=='__main__':main()

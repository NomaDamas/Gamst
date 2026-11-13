"""Explicit two-runtime object appearance and memory smoke; no downloads/devices."""
import argparse
import asyncio
import hashlib
import io
import json
from pathlib import Path
import sys
import time

from humansense import HumanSense, SensorStreams, SensorMetadata, ObjectAppearancePipeline
from humansense.plugins import InputFrame
from humansense.schema import timestamp


async def run(args):
    import numpy as np
    from PIL import Image
    if not 1<=len(args.image)<=32:raise ValueError('provide1..32 ordered images')
    detector=dict(path=str(Path(args.detector).resolve()),sha256=args.detector_sha256,device='cpu')
    regions=dict(path=str(Path(args.embedding).resolve()),sha256=args.embedding_sha256)
    spec=ObjectAppearancePipeline.capability(detector,regions)
    async def factory():return await ObjectAppearancePipeline.open(args.detector_python,detector,args.embedding_python,regions)
    report=dict(kind='ordered_object_smoke_not_accuracy_acceptance',parent_python=sys.version,
        detector_python=args.detector_python,embedding_python=args.embedding_python,
        detector=detector,regions=regions,timestamp_origin='replay_wall_time_not_original_capture',
        image_orientation='encoded_pixels_no_exif_rotation',observations=[])
    async with HumanSense(consent=True) as human:
        streams=SensorStreams(human,max_packet_bytes=8*1024*1024);streams.register(spec,factory)
        start=time.perf_counter();await streams.open('object-replay','images',[spec.id])
        report['startup_ms']=(time.perf_counter()-start)*1000;last=-1
        for seq,name in enumerate(args.image):
            with Path(name).open('rb') as source:payload=source.read(16*1024*1024+1)
            if not 0<len(payload)<=16*1024*1024:raise ValueError('encoded image must be1..16MiB')
            with Image.open(io.BytesIO(payload)) as image:
                w,h=image.size
                if min(w,h)<32 or w>3840 or h>2160 or w*h>1920*1080:raise ValueError('image exceeds bounds')
                bgr=np.asarray(image.convert('RGB'))[:,:,::-1].copy()
            ts=max(timestamp(),last+1);last=ts
            f=InputFrame(bgr,'video',ts,sensor=SensorMetadata('object-replay','images',seq,coordinate_frame='camera.normalized'))
            start=time.perf_counter();await streams.submit(f);await streams.drain()
            records=human.store.latest_signals('local-user',0,timestamp())
            if not records or records[0]['timestamp_ms']!=ts:raise RuntimeError('pipeline did not publish this capture')
            record=records[0]
            await human.remember_objects(record['id'],complete_snapshot=record['observation']['value']['summary']['complete_snapshot'])
            report['observations'].append(dict(image_sha256=hashlib.sha256(payload).hexdigest(),
                elapsed_ms=(time.perf_counter()-start)*1000,record=record,memory=await human.get_object_memory()))
        await streams.clear()
    with Path(args.output).open('x') as target:json.dump(report,target,indent=2,allow_nan=False);target.write('\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('detector-python','embedding-python','detector','detector-sha256','embedding','embedding-sha256','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--image',action='append',required=True)
    asyncio.run(run(parser.parse_args()))


if __name__=='__main__':main()

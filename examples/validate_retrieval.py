"""Explicit local multi-image appearance retrieval smoke; no downloads."""
import argparse
import asyncio
import hashlib
import io
import json
from pathlib import Path
import sys
import time

from humansense import ProcessProvider, SensorMetadata
from humansense.plugins import InputFrame
from humansense.provider_wire import spec_value
from humansense.schema import timestamp


async def run(args):
    import numpy as np
    from PIL import Image
    if not 1<=len(args.image)<=32: raise ValueError('provide1..32 images in observation order')
    config=dict(path=str(Path(args.checkpoint).resolve()),sha256=args.sha256,
                capacity=args.capacity,retention_ms=args.retention_ms,top_k=args.top_k)
    start=time.perf_counter()
    provider=await ProcessProvider.open(args.python,'dinov3.retrieval',config)
    report=dict(schema_version=1,kind='ordered_image_smoke_not_retrieval_acceptance',config=config,
        startup_ms=(time.perf_counter()-start)*1000,worker_python=args.python,parent_python=sys.version,
        capability=spec_value(provider.spec),observations=[],timestamp_origin='replay_wall_time_not_original_capture',
        image_orientation='encoded_pixels_no_exif_rotation')
    last=-1
    try:
        for sequence,name in enumerate(args.image):
            with Path(name).open('rb') as source: payload=source.read(16*1024*1024+1)
            if not 0<len(payload)<=16*1024*1024: raise ValueError('encoded image must be1..16MiB')
            with Image.open(io.BytesIO(payload)) as image:
                width,height=image.size
                if (min(width,height)<32 or width>3840 or height>2160 or width*height>1920*1080 or
                        max(width,height)/min(width,height)>4): raise ValueError('image exceeds retrieval bounds')
                bgr=np.asarray(image.convert('RGB'))[:,:,::-1].copy()
            ts=max(timestamp(),last+1);last=ts
            frame=InputFrame(bgr,'video',ts,sensor=SensorMetadata('retrieval-validation','images',sequence,stream_epoch='replay'))
            start=time.perf_counter(); context=await provider.process(frame)
            report['observations'].append(dict(image_sha256=hashlib.sha256(payload).hexdigest(),
                timestamp_ms=ts,process_ms=(time.perf_counter()-start)*1000,context=context))
    finally:
        await provider.close()
    with Path(args.output).open('x') as target:
        json.dump(report,target,indent=2,allow_nan=False);target.write('\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','sha256','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--image',action='append',required=True)
    parser.add_argument('--python',default=sys.executable)
    parser.add_argument('--capacity',type=int,default=128)
    parser.add_argument('--retention-ms',type=int,default=600000)
    parser.add_argument('--top-k',type=int,default=5)
    asyncio.run(run(parser.parse_args()))


if __name__=='__main__':main()

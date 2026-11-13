"""Explicit local single-image gaze smoke evidence; no checkpoint downloads."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys
import time

from humansense import ProcessProvider, SensorMetadata
from humansense.plugins import InputFrame
from humansense.provider_wire import spec_value
from humansense.schema import timestamp


async def run(args):
    import cv2
    import numpy as np
    with Path(args.image).open('rb') as source:
        payload=source.read(16*1024*1024+1)
    if not 0<len(payload)<=16*1024*1024:
        raise ValueError('encoded image must be1..16MiB')
    image=cv2.imdecode(np.frombuffer(payload,dtype=np.uint8),cv2.IMREAD_COLOR)
    if image is None or image.shape[0]*image.shape[1]>1920*1080:
        raise ValueError('valid image of at most1920*1080 pixels required')
    config=dict(path=str(Path(args.gaze).resolve()),sha256=args.gaze_sha256,
        face_path=str(Path(args.face).resolve()),face_sha256=args.face_sha256,
        arch=args.arch,max_faces=args.max_faces)
    start=time.perf_counter()
    provider=await ProcessProvider.open(args.python,'l2cs.gaze',config)
    startup_ms=(time.perf_counter()-start)*1000
    try:
        frame=InputFrame(image,'video',timestamp(),sensor=SensorMetadata('gaze-validation','image',0))
        start=time.perf_counter()
        context=await provider.process(frame)
        elapsed=(time.perf_counter()-start)*1000
        report=dict(schema_version=1,kind='single_image_smoke_not_accuracy_acceptance',
            image_sha256=hashlib.sha256(payload).hexdigest(),config=config,
            parent_python=sys.version,worker_python=args.python,capability=spec_value(provider.spec),
            startup_ms=startup_ms,first_process_ms=elapsed,context=context)
    finally:
        await provider.close()
    with Path(args.output).open('x') as target:
        json.dump(report,target,allow_nan=False,indent=2)
        target.write('\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('gaze','gaze-sha256','face','face-sha256','image','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--python',default=sys.executable)
    parser.add_argument('--arch',choices=('ResNet18','ResNet50'),default='ResNet50')
    parser.add_argument('--max-faces',type=int,default=4)
    asyncio.run(run(parser.parse_args()))


if __name__=='__main__': main()

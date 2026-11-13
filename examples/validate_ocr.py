"""Explicit local single-image OCR smoke; no downloads."""
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
    with Path(args.image).open('rb') as source: payload=source.read(16*1024*1024+1)
    if not 0<len(payload)<=16*1024*1024: raise ValueError('encoded image must be1..16MiB')
    with Image.open(io.BytesIO(payload)) as image:
        width,height=image.size
        if (min(width,height)<32 or width>3840 or height>2160 or width*height>1920*1080 or
                max(width,height)/min(width,height)>8):
            raise ValueError('image must satisfy the documented OCR frame limits')
        bgr=np.asarray(image.convert('RGB'))[:,:,::-1].copy()
    config=dict(det_path=str(Path(args.det).resolve()),det_sha256=args.det_sha256,
        rec_path=str(Path(args.rec).resolve()),rec_sha256=args.rec_sha256,language=args.language,max_regions=args.max_regions)
    start=time.perf_counter()
    provider=await ProcessProvider.open(args.python,'ppocr.mobile',config)
    startup_ms=(time.perf_counter()-start)*1000
    try:
        frame=InputFrame(bgr,'video',timestamp(),sensor=SensorMetadata('ocr-validation','image',0))
        start=time.perf_counter(); context=await provider.process(frame)
        elapsed=(time.perf_counter()-start)*1000
        report=dict(schema_version=1,kind='single_image_smoke_not_accuracy_acceptance',
            image_sha256=hashlib.sha256(payload).hexdigest(),config=config,parent_python=sys.version,
            worker_python=args.python,capability=spec_value(provider.spec),startup_ms=startup_ms,
            first_process_ms=elapsed,context=context,image_orientation='encoded_pixels_no_exif_rotation')
    finally:
        await provider.close()
    with Path(args.output).open('x') as target:
        json.dump(report,target,indent=2,allow_nan=False); target.write('\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('det','det-sha256','rec','rec-sha256','image','output'): parser.add_argument('--'+name,required=True)
    parser.add_argument('--python',default=sys.executable)
    parser.add_argument('--language',choices=('en','ch','korean'),default='en')
    parser.add_argument('--max-regions',type=int,default=16)
    asyncio.run(run(parser.parse_args()))


if __name__=='__main__': main()

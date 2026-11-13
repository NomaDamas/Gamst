"""Replay predecoded local images through an explicit EfficientTAM worker.

This records runtime output, not automatic quality acceptance. The supplied box
belongs only to the first image; an injected gap intentionally requires a new
prompt and never reuses that box. No sensor or model download is started.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time

from humansense import ProcessProvider
from humansense.capabilities import SensorMetadata
from humansense.plugins import InputFrame
from humansense.providers.region_appearance import region_evidence


async def run(args):
    import numpy as np
    from PIL import Image
    if not 1 <= len(args.image) <= 1000 or not 1 <= args.period_ms <= 1000:
        raise ValueError('1..1000 ordered images and period_ms1..1000 required')
    if args.gap_at is not None and not 1 <= args.gap_at < len(args.image):
        raise ValueError('gap_at must identify a later image index')
    config = dict(source_dir=args.source_dir, checkpoint_path=args.checkpoint,
        checkpoint_sha256=args.sha256, runtime_root=args.runtime_root, device=args.device)
    provider = await ProcessProvider.open(args.python, 'efficienttam.tiny512', config,
        startup_timeout=120, inference_timeout=60)
    results = []
    try:
        for i, name in enumerate(args.image):
            path = Path(name).resolve(strict=True)
            if path.stat().st_size > 8*1024*1024: raise ValueError('input image exceeds8MiB')
            payload = path.read_bytes()
            import io
            with Image.open(io.BytesIO(payload)) as image:
                w,h = image.size
                if min(w,h) < 32 or max(w,h) > 2048 or w*h > 1048576:
                    raise ValueError('image must satisfy native input bounds')
                data = np.asarray(image.convert('RGB'))[:,:,::-1].copy()
            seq = i + (1 if args.gap_at is not None and i >= args.gap_at else 0)
            frame = InputFrame(data,'video',1000+seq*args.period_ms,
                sensor=SensorMetadata('replay','video',seq,coordinate_frame='replay.image',stream_epoch='replay1'))
            start = time.perf_counter()
            if i == 0:
                x,y,bw,bh = args.box
                seed = region_evidence(frame,[dict(track_id='manual.seed',label=args.label,
                    location=dict(unit='normalized_image',bbox=dict(x=x,y=y,width=bw,height=bh)))],
                    detector_id='manual.seed',detector_version='1',complete_snapshot=False)
                value = await provider.process_regions(frame,seed)
            else:
                value = await provider.process(frame)
            results.append(dict(index=i,input_sha256=hashlib.sha256(payload).hexdigest(),
                processing_ms=(time.perf_counter()-start)*1000,context=value))
    finally:
        await provider.close()
    report = dict(provider='efficienttam.tiny512',checkpoint_sha256=args.sha256,
        source_revision='abcd061ebd3cc6e7527d152d75b890126aaa53f6',device=args.device,
        timestamp_domain='synthetic_replay_ms',period_ms=args.period_ms,
        quality_acceptance=False,live_capture_latency_measured=False,observations=results)
    # Avoid overwriting an existing evidence report.
    with Path(args.output).open('x') as output: json.dump(report,output,ensure_ascii=False,indent=2)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('python','source-dir','checkpoint','sha256','runtime-root','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--image',action='append',required=True)
    parser.add_argument('--box',nargs=4,type=float,required=True,metavar=('X','Y','WIDTH','HEIGHT'))
    parser.add_argument('--label',default='object')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--period-ms',type=int,default=40)
    parser.add_argument('--gap-at',type=int)
    asyncio.run(run(parser.parse_args()))

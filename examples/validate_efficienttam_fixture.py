"""Exercise the real native graph with temporary UNTRAINED generated weights.

Not recognition/quality acceptance. Requires existing optional dependencies and
an explicit pinned local source checkout; never downloads or installs anything.
Run in its own process because upstream Hydra/module state is process-global.
"""
import argparse
import asyncio
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

from humansense import ProcessProvider
from humansense.capabilities import SensorMetadata
from humansense.plugins import InputFrame
from humansense.providers.efficienttam_native import _read
from humansense.providers.efficienttam_sources import SOURCES
from humansense.providers.region_appearance import region_evidence


async def run(args):
    import numpy as np
    import torch
    root=Path(args.source_dir).resolve(strict=True)
    blobs={name:_read(root/name,digest,2*1024*1024) for name,digest in SOURCES.items()}
    with tempfile.TemporaryDirectory(prefix='gamst-untrained-fixture-',dir=args.runtime_root) as directory:
        temp=Path(directory).resolve()
        snapshot=temp/'sources'
        for name,payload in blobs.items():
            target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload)
        sys.path.insert(0,str(snapshot))
        from efficient_track_anything.build_efficienttam import build_efficienttam_video_predictor
        torch.set_num_threads(2);torch.manual_seed(17)
        model=build_efficienttam_video_predictor('configs/efficienttam/efficienttam_ti_512x512.yaml',
            ckpt_path=None,device='cpu',mode='eval',hydra_overrides_extra=['++model.compile_image_encoder=False'],
            apply_postprocessing=False,vos_optimized=False)
        path=temp/'untrained.pt';torch.save({'model':model.state_dict()},path)
        parameters=sum(p.numel() for p in model.parameters());del model;gc.collect()
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        config=dict(source_dir=str(snapshot),checkpoint_path=str(path),checkpoint_sha256=digest,runtime_root=str(temp))
        start=time.perf_counter()
        worker=await ProcessProvider.open(sys.executable,'efficienttam.tiny512',config,
            startup_timeout=120,inference_timeout=60)
        try:
            frame=InputFrame(np.zeros((32,45,3),dtype=np.uint8),'video',1000,
                sensor=SensorMetadata('fixture','video',0,coordinate_frame='image',stream_epoch='fixture1'))
            seed=region_evidence(frame,[dict(track_id='seed',label='fixture',location=dict(
                unit='normalized_image',bbox=dict(x=.1,y=.1,width=.5,height=.5)))],
                detector_id='untrained.fixture',detector_version='1',complete_snapshot=False)
            results=[await worker.process_regions(frame,seed)]
            for i in (1,2):
                results.append(await worker.process(replace(frame,timestamp_ms=1000+40*i,
                    sensor=replace(frame.sensor,sequence=i))))
            await worker.reset();reset=await worker.process(frame)
            if [r['summary']['processed_frames'] for r in results] != [1,2,3] or reset['summary']['status']!='needs_prompt':
                raise RuntimeError('native forward/reset fixture check failed')
            report=dict(kind='UNTRAINED_GENERATED_FIXTURE_NOT_MODEL_QUALITY',parameters=parameters,
                checkpoint_bytes=path.stat().st_size,checkpoint_sha256=digest,
                elapsed_seconds=time.perf_counter()-start,processed=[1,2,3],
                integrity=[r['summary']['artifact_integrity_verified'] for r in results],
                masks=[dict(nonempty=len(r['objects']),empty=len(r['summary']['empty_tracks'])) for r in results],
                reset=reset['summary']['status'])
        finally:await worker.close()
    with Path(args.output).open('x') as output:json.dump(report,output,indent=2)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('source-dir','runtime-root','output'):parser.add_argument('--'+key,required=True)
    asyncio.run(run(parser.parse_args()))

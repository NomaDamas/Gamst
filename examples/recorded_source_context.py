"""Generate a tiny explicit PCM fixture and replay it through real SDK Context.

Run from the repository root: python3 -m examples.recorded_source_context NEW_DIR
No microphone, camera, network, model or dependency installation is activated.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import struct

from benchmarks.recorded_source import digest
from benchmarks.recorded_replay import run_source_context_recording


async def example(directory):
    root=Path(directory).absolute()
    root.mkdir(mode=0o700,parents=False,exist_ok=False)
    assets=root/'assets';assets.mkdir(mode=0o700)
    payload=struct.pack('<h',4096)*400
    with (assets/'level.pcm').open('xb') as stream:stream.write(payload)
    source=dict(source_id='recorded-mic',sensor_id='mono',stream_epoch='original-fixture',
        format='pcm_s16le_mono',modality='audio',calibration_id=None)
    records=[]
    for index in range(10):
        records.append(dict(source_id=source['source_id'],sensor_id=source['sensor_id'],
            stream_epoch=source['stream_epoch'],sequence=index,capture_ns=1_000_000_000+index*50_000_000,
            offset_ns=index*50_000_000,format='pcm_s16le_mono',sample_rate=8000,sample_count=400,
            payload_path='level.pcm',payload_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest()))
    manifest=dict(schema_version=1,recording_id='synthetic-pcm-level-example',
        origin_capture_ns=1_000_000_000,duration_ns=600_000_000,sources=[source],records=records,
        availability=[dict(source_id=source['source_id'],intervals=[dict(start_ns=0,end_ns=600_000_000,state='available')])])
    question={'question':'Report observed PCM energy only; no speech or emotion inference.'}
    tasks=[dict(task_id='early',input=question,input_sha256=digest(question),opens_ns=0,
                snapshot_ns=200_000_000,deadline_ns=400_000_000,window='1h'),
           dict(task_id='late',input=question,input_sha256=digest(question),opens_ns=100_000_000,
                snapshot_ns=450_000_000,deadline_ns=550_000_000,window='1h')]
    return await run_source_context_recording(manifest,tasks,
        {source['source_id']:dict(profile='pcm_level',delay_ms=0)},root/'replay',
        asset_directory=assets,retain_evaluation_artifacts=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('new_directory')
    args=parser.parse_args()
    print(json.dumps(asyncio.run(example(args.new_directory)),indent=2))

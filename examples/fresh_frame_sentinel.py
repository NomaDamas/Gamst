"""Finite original-time raw-frame example; no device/model installation or retiming.

Run: python -m examples.fresh_frame_sentinel --help
See docs/frame-sentinel.md. A controlled fixture is not a trained-model acceptance.
"""
import argparse
import asyncio
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import time

from humansense import FreshFrameSentinel, HumanSense, SensorMetadata
from humansense.plugins import InputFrame
from humansense.call_budget import CallBudget
from humansense.call_controller import CallController
from humansense.call_policy import CallPolicy
from humansense.schema import timestamp
from humansense.smolvlm_call import smolvlm_frame_sha256


def _read(path, maximum):
    path=Path(path).resolve(strict=True)
    with path.open('rb') as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError('regular application-owned input file required')
        data=stream.read(maximum+1)
    if len(data)>maximum:raise ValueError('example input byte limit exceeded')
    return data


def declaration(path):
    """Validate small manifest without decoding arbitrary media or retimestamping."""
    path=Path(path).resolve(strict=True)
    original=_read(path,131072);value=json.loads(original)
    if (type(value) is not dict or set(value)!={'schema_version','shape','subject_id','sensor','captures'} or
            type(value['schema_version']) is not int or value['schema_version']!=1):
        raise ValueError('exact manifest schema version 1 required')
    shape=value['shape']
    if (type(shape) is not list or len(shape)!=3 or any(type(n) is not int for n in shape) or shape[2]!=3 or
            min(shape[:2])<32 or max(shape[:2])>2160 or shape[0]*shape[1]>1920*1080):
        raise ValueError('bounded native HWC uint8 BGR geometry required')
    if type(value['sensor']) is not dict:raise ValueError('original sensor metadata required')
    sensor=dict(value['sensor'])
    if 'channels' in sensor and type(sensor['channels']) is not list:raise ValueError('original JSON channel array required')
    sensor['channels']=tuple(sensor.get('channels',()))
    sensor=SensorMetadata(**sensor)
    if sensor.stream_epoch is None or sensor.capture_sequence is None:raise ValueError('explicit capture epoch/index required')
    if type(value['captures']) is not list or not 1<=len(value['captures'])<=128:raise ValueError('1..128 captures required')
    previous=(-1,-1);paths=[]
    for capture in value['captures']:
        if (type(capture) is not dict or set(capture)!={'path','timestamp_ms','capture_index','frame_sha256'} or
                type(capture['path']) is not str or not 1<=len(capture['path'])<=256 or
                type(capture['timestamp_ms']) is not int or not 0<=capture['timestamp_ms']<=2**53-1 or
                type(capture['capture_index']) is not int or not 0<=capture['capture_index']<=2**53-1 or
                capture['timestamp_ms']<previous[0] or capture['capture_index']<=previous[1] or
                type(capture['frame_sha256']) is not str or len(capture['frame_sha256'])!=64 or
                any(c not in '0123456789abcdef' for c in capture['frame_sha256'])):
            raise ValueError('bounded original ordered capture declarations required')
        relative=Path(capture['path'])
        if relative.is_absolute():raise ValueError('manifest-relative raw file required')
        resolved=(path.parent/relative).resolve(strict=True)
        if not resolved.is_relative_to(path.parent):raise ValueError('raw file escapes manifest owner directory')
        paths.append(resolved);previous=(capture['timestamp_ms'],capture['capture_index'])
    if value['captures'][-1]['timestamp_ms']-value['captures'][0]['timestamp_ms']>60000:
        raise ValueError('example capture span limited to one minute')
    return value,sensor,paths,hashlib.sha256(original).hexdigest()


async def run(args):
    import numpy as np
    manifest,sensor,paths,manifest_digest=declaration(args.manifest)
    config=json.loads(_read(args.config,131072))
    if type(args.interval_ms) is not int or not 1<=args.interval_ms<=10000:
        raise ValueError('example interval must be 1..10000 ms')
    if manifest['captures'][-1]['timestamp_ms']>timestamp()+60000:
        raise ValueError('example future wait limited to one minute; capture timestamps are never rewritten')
    controller=CallController(policy=CallPolicy(mode='sentinel',semantic_interval_ms=args.interval_ms,cooldown_ms=0),
        budget=CallBudget(unit=args.unit,pricing_revision=args.pricing_revision,ceiling=args.ceiling,max_inflight=4))
    capture_results=[];polls=0
    async with HumanSense(consent=True) as human:
        human.register_call_controller('recorded',controller)
        sentinel=await FreshFrameSentinel.open(human,'recorded','recorded',args.python,config,
            sensor=sensor,subject_id=manifest['subject_id'],frame_shape=tuple(manifest['shape']),
            interval_ms=args.interval_ms,deadline_tolerance_ms=min(250,args.interval_ms-1),
            max_age_ms=args.max_age_ms,maximum_charge=args.maximum_charge)
        async def consumer():
            nonlocal polls
            while True:
                await human.get_context(manifest['subject_id']);polls+=1
                await asyncio.sleep(.5)  # Fixed read cadence, no LLM or cadence adaptation.
        reader=asyncio.create_task(consumer())
        try:
            sentinel.start()
            for capture,path in zip(manifest['captures'],paths):
                await asyncio.sleep(max(0,(capture['timestamp_ms']-timestamp())/1000))
                raw=_read(path,math.prod(manifest['shape']))
                if len(raw)!=math.prod(manifest['shape']):raise ValueError('exact raw BGR byte count required')
                pixels=np.frombuffer(raw,dtype=np.uint8).reshape(manifest['shape']).copy()
                raw=None
                try:
                    digest=smolvlm_frame_sha256(pixels)
                    if digest!=capture['frame_sha256']:raise ValueError('recorded shape/pixels differ from manifest hash')
                    frame=InputFrame(pixels,'video',capture['timestamp_ms'],subject_id=manifest['subject_id'],
                        sensor=replace(sensor,capture_sequence=capture['capture_index']))
                    result=sentinel.offer(frame)
                    capture_results.append(dict(timestamp_ms=capture['timestamp_ms'],capture_index=capture['capture_index'],
                        frame_sha256=digest,accepted=result['accepted'],reason=result.get('reason')))
                finally:pixels.fill(0)
            # One following sampled absence point; no replay of the final capture.
            await asyncio.sleep(args.interval_ms/1000+.02)
            started=time.monotonic()
            while sentinel.snapshot()['inflight']:
                if time.monotonic()-started>31:raise TimeoutError('bounded example drain exceeded')
                await asyncio.sleep(.01)
            result=dict(schema_version=1,original_capture_timestamps_preserved=True,
                manifest_sha256=manifest_digest,captures=capture_results,
                sentinel=sentinel.snapshot(),budget=controller.snapshot()['budget'],consumer_context_polls=polls,
                consumer_llm_invocations=0,actual_cost=None,trained_model_quality_verified=False)
        finally:
            reader.cancel();await asyncio.gather(reader,return_exceptions=True)
            await sentinel.close()
        result['worker_reaped']=sentinel.adapter.worker.closed
        return result


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True,help='bounded schema-v1 manifest; paths are raw uint8 HWC BGR files')
    p.add_argument('--config',required=True,help='existing strict local SmolVLM native configuration JSON')
    p.add_argument('--python',required=True,help='existing selected native worker interpreter')
    p.add_argument('--maximum-charge',type=int,default=None,help='verified ceiling; omitted leaves quote unknown and blocks dispatch')
    p.add_argument('--unit',required=True);p.add_argument('--pricing-revision',required=True)
    p.add_argument('--ceiling',type=int,required=True)
    p.add_argument('--interval-ms',type=int,default=5000);p.add_argument('--max-age-ms',type=int,default=1000)
    return p


if __name__=='__main__':print(json.dumps(asyncio.run(run(parser().parse_args())),allow_nan=False))

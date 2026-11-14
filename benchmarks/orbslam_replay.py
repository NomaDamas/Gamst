"""Bounded unpaced recorded-BGR replay; not a real-device/quality benchmark."""
import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import re
import sys
import time

from humansense import ProcessProvider, ORBSLAMProcess, SensorMetadata
from humansense.plugins import InputFrame
from humansense.providers.orbslam_context import configuration
from humansense.streams import _finish

MAX_MANIFEST=1024*1024
MAX_REPORT=32*1024*1024


def _pairs(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('duplicate manifest key')
        result[key]=value
    return result


def _capture_path(root,name):
    if type(name) is not str or not 1<=len(name)<=1024 or Path(name).is_absolute():
        raise ValueError('bounded relative capture path required')
    path=(root/name).resolve()
    if not path.is_relative_to(root):raise ValueError('capture must remain inside manifest directory')
    return path


def load_manifest(path):
    path=Path(path).resolve()
    if not path.is_file():raise ValueError('regular local manifest file required')
    with path.open('rb') as source:raw=source.read(MAX_MANIFEST+1)
    if not 1<=len(raw)<=MAX_MANIFEST:raise ValueError('manifest exceeds1MiB')
    def reject(value):raise ValueError('nonfinite manifest number')
    try:m=json.loads(raw,object_pairs_hook=_pairs,parse_constant=reject)
    except (RecursionError,UnicodeError,json.JSONDecodeError) as exc:raise ValueError('invalid manifest JSON') from exc
    if type(m) is not dict or set(m)!={'schema_version','config','sensor','shape','captures'}:
        raise ValueError('exact replay manifest fields required')
    if type(m['schema_version']) is not int or m['schema_version']!=1:raise ValueError('schema1 required')
    if type(m['config']) is not dict:raise ValueError('native configuration object required')
    m['config']=configuration(**m['config'])
    s=m['sensor']
    if type(s) is not dict or set(s)!={'source_id','sensor_id','coordinate_frame','stream_epoch'}:
        raise ValueError('explicit sensor identity/coordinate frame/epoch required')
    if any(type(v) is not str for v in s.values()):raise ValueError('sensor identifiers required')
    SensorMetadata(**s,sequence=0,calibration_id=m['config']['calibration_id'])
    shape=m['shape']
    if (type(shape) is not list or len(shape)!=3 or any(type(v) is not int for v in shape) or
            not 32<=shape[0]<=1080 or not 32<=shape[1]<=1920 or shape[2]!=3):
        raise ValueError('bounded calibrated HWC BGR shape required')
    captures=m['captures']
    if type(captures) is not list or not 1<=len(captures)<=3000:raise ValueError('1..3000 captures required')
    previous=None
    for row in captures:
        if type(row) is not dict or set(row)!={'path','sha256','capture_ms','capture_sequence','reset_before'}:
            raise ValueError('exact capture provenance and reset marker required')
        _capture_path(path.parent,row['path'])
        if type(row['sha256']) is not str or not re.fullmatch('[0-9a-f]{64}',row['sha256']):raise ValueError('capture SHA256 required')
        if type(row['reset_before']) is not bool:raise ValueError('explicit reset flag required')
        for key in ('capture_ms','capture_sequence'):
            if type(row[key]) is not int or not 0<=row[key]<=2**53-1:raise ValueError('bounded integer capture clock/index required')
        pair=(row['capture_sequence'],row['capture_ms'])
        if previous and not row['reset_before'] and (pair[0]<=previous[0] or pair[1]<=previous[1]):
            raise ValueError('ordered capture sequence/time required between explicit resets')
        previous=pair
    return path.parent,m,hashlib.sha256(raw).hexdigest()


def _environment():
    versions={}
    for name in ('numpy','opencv-python','humansense-sdk'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]='unavailable'
    root=Path(__file__).resolve().parents[1]
    files=('benchmarks/orbslam_replay.py','humansense/orbslam_process.py',
           'humansense/providers/orbslam_context.py','humansense/providers/orbslam_native.py',
           'humansense/process_provider.py','humansense/provider_wire.py')
    return dict(python=sys.version,platform=platform.platform(),packages=versions,
                source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files})


async def replay(manifest,output,*,rotate=False,max_starts_per_minute=6):
    if type(rotate) is not bool:raise ValueError('rotate must be boolean')
    if type(max_starts_per_minute) is not int or not 1<=max_starts_per_minute<=60:
        raise ValueError('startup limit must be1..60')
    root,m,digest=load_manifest(manifest)
    if not rotate and len(m['captures'])>m['config']['max_frames']:
        raise ValueError('capture count exceeds finite session budget; select rotation explicitly')
    output=Path(output)
    provider=None;elapsed=[];tracked=0;completed=0;error=None;written=0
    started=time.monotonic()
    # Existing files, including symlinks, are never overwritten.
    with output.open('x',encoding='utf-8') as sink:
        def emit(row):
            nonlocal written
            line=json.dumps(row,allow_nan=False,ensure_ascii=True,separators=(',',':'))+'\n'
            size=len(line.encode())
            if size>65536 or written+size>MAX_REPORT:raise ValueError('replay report exceeds bounded output')
            sink.write(line);sink.flush();written+=size
        emit(dict(kind='run',schema_version=1,status='incomplete',manifest_sha256=digest,
            config=m['config'],sensor=m['sensor'],shape=m['shape'],rotate=rotate,
            max_starts_per_minute=max_starts_per_minute,environment=_environment(),
            timing_mode='unpaced_recorded_input',actual_slam_verified=False))
        try:
            import numpy as np
            if rotate:provider=await ORBSLAMProcess.open(sys.executable,m['config'],max_starts_per_minute=max_starts_per_minute)
            else:provider=await ProcessProvider.open(sys.executable,'orbslam3.monocular',m['config'])
            size=math.prod(m['shape'])
            for index,row in enumerate(m['captures']):
                path=_capture_path(root,row['path'])
                if not path.is_file():raise ValueError('regular local capture file required')
                with path.open('rb') as source:payload=source.read(size+1)
                if len(payload)!=size or hashlib.sha256(payload).hexdigest()!=row['sha256']:
                    raise ValueError('capture size/hash differs from manifest')
                pixels=np.frombuffer(payload,dtype=np.uint8).reshape(m['shape']).copy()
                sensor=SensorMetadata(**m['sensor'],sequence=index,capture_sequence=row['capture_sequence'],
                                      calibration_id=m['config']['calibration_id'])
                if row['reset_before']:await provider.reset()
                start=time.monotonic()
                value=await provider.process(InputFrame(pixels,'video',row['capture_ms'],sensor=sensor))
                duration=(time.monotonic()-start)*1000
                emit(dict(kind='observation',index=index,capture_sha256=row['sha256'],
                          request_elapsed_ms=duration,value=value))
                elapsed.append(duration);completed+=1;tracked+=int(value['pose']['tracked'])
        except BaseException as exc:error=exc
        finally:
            if provider is not None:
                try:await _finish(asyncio.create_task(provider.close()))
                except BaseException as exc:
                    if error is None:error=exc
            ordered=sorted(elapsed)
            quantile=lambda q:ordered[math.ceil(q*len(ordered))-1] if ordered else None
            emit(dict(kind='summary',status='complete' if error is None else 'failed',
                error_type=type(error).__name__ if error else None,requested_captures=len(m['captures']),
                completed_observations=completed,tracked=tracked,abstained=completed-tracked,
                request_elapsed_ms_p50=quantile(.5),request_elapsed_ms_p95=quantile(.95),
                quantile_method='nearest_rank',replay_elapsed_ms=(time.monotonic()-started)*1000,
                rotation_diagnostics=provider.diagnostics() if rotate and provider else None,
                capture_to_context_latency_ms=None,native_peak_memory_bytes=None,
                reference_trajectory_error=None,actual_cost=None,quality_validated=False,
                continuous_coverage_verified=False,actual_slam_verified=False))
    if error is not None:raise error
    return dict(report=str(output),completed_observations=completed,status='complete')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--rotate',action='store_true')
    parser.add_argument('--max-starts-per-minute',type=int,default=6)
    args=parser.parse_args()
    result=asyncio.run(replay(args.manifest,args.output,rotate=args.rotate,max_starts_per_minute=args.max_starts_per_minute))
    print(json.dumps(result))


if __name__=='__main__':main()

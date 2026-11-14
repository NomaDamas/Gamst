"""Collect bounded paired command traces; scoring and measurement audit stay separate.

Only explicitly selected trusted local commands are executed. This is not a sandbox.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import signal
import time

from .perception_comparison import _read, plan_digest, validate_plan, _digest

MAX_INPUT_BYTES = 16*1024*1024
MAX_OUTPUT_BYTES = 65536
MAX_STDERR_BYTES = 4096


def _encoded(value):
    return json.dumps(value,sort_keys=True,separators=(',', ':'),ensure_ascii=True,allow_nan=False).encode()


def _decode(payload):
    def reject(value): raise ValueError('nonfinite JSON number')
    return json.loads(payload, parse_constant=reject)


def _keys(obj, expected, label):
    if type(obj) is not dict or set(obj)!=set(expected):
        raise ValueError('invalid '+label+' fields')


def _artifact(path, digest):
    _digest(digest)
    if type(path) is not str or not Path(path).is_absolute() or not Path(path).is_file():
        raise ValueError('artifact must be an absolute existing file')
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        # Explicit artifacts can include large local checkpoint/code files.
        for chunk in iter(lambda:stream.read(1024*1024),b''): h.update(chunk)
    if h.hexdigest()!=digest: raise ValueError('execution artifact hash mismatch: '+path)


def validate_command_declaration(command):
    """Validate portable command declarations without accessing historical assets."""
    _keys(command,('argv','cwd','artifacts','environment'),'command')
    argv=command['argv']
    if (type(argv) is not list or not 1<=len(argv)<=64 or
            any(type(a) is not str or not a or len(a)>4096 or '\0' in a for a in argv)):
        raise ValueError('argv must contain 1..64 bounded strings')
    if not Path(argv[0]).is_absolute(): raise ValueError('executable must be absolute')
    if (type(command['cwd']) is not str or '\0' in command['cwd'] or
            not Path(command['cwd']).is_absolute()):
        raise ValueError('cwd must be an absolute directory path')
    artifacts=command['artifacts']
    if type(artifacts) is not dict or not 1<=len(artifacts)<=256 or argv[0] not in artifacts:
        raise ValueError('hash all declared execution artifacts including executable')
    for path,digest in artifacts.items():
        if type(path) is not str or not Path(path).is_absolute() or '\0' in path:
            raise ValueError('artifact must be an absolute path')
        _digest(digest)
    env=command['environment']
    if (type(env) is not dict or len(env)>64 or
            any(type(k) is not str or not k or len(k)>128 or '=' in k or '\0' in k or
                type(v) is not str or len(v)>4096 or '\0' in v for k,v in env.items())):
        raise ValueError('invalid explicit environment')


def validate_command(command):
    validate_command_declaration(command)
    if not Path(command['cwd']).is_dir():
        raise ValueError('cwd must be an absolute existing directory')
    for path, digest in command['artifacts'].items():
        _artifact(path, digest)


def validate_collection(plan, inputs, expected_plan_sha256):
    _,tasks,_,_=validate_plan(plan,expected_plan_sha256=expected_plan_sha256)
    runner=plan.get('runner')
    _keys(runner,('schema_version','timeout_ms','order','commands'),'runner')
    if type(runner['schema_version']) is not int or runner['schema_version']!=1:
        raise ValueError('unsupported runner schema')
    if type(runner['timeout_ms']) is not int or not 1<=runner['timeout_ms']<=3600000:
        raise ValueError('runner timeout_ms must be 1..3600000')
    if runner['order']!='alternating': raise ValueError('runner order must be alternating')
    _keys(runner['commands'],('baseline','candidate'),'commands')
    for command in runner['commands'].values(): validate_command(command)
    if type(inputs) is not list or len(inputs)!=len(tasks):
        raise ValueError('supply exactly one input per frozen task')
    if len(_encoded(inputs))>MAX_INPUT_BYTES: raise ValueError('recorded input package exceeds 16 MiB')
    by_id={}
    for item in inputs:
        _keys(item,('task_id','input'),'input')
        name=item['task_id']
        if type(name) is not str or name not in tasks or name in by_id:
            raise ValueError('unknown or duplicate task input')
        if type(item['input']) is not dict or plan_digest(item['input'])!=tasks[name]['input_sha256']:
            raise ValueError('recorded input/prompt differs from frozen task digest')
        by_id[name]=item['input']
    return by_id


def _write_new(path, value):
    # Exclusive creation: existing source traces are never overwritten or resumed.
    with path.open('xb') as stream:
        os.chmod(path,0o600)
        stream.write(_encoded(value)+b'\n')
        stream.flush(); os.fsync(stream.fileno())


async def _drain(stream, limit):
    result=bytearray()
    while True:
        block=await stream.read(4096)
        if not block: return bytes(result)
        result.extend(block)
        if len(result)>limit: raise ValueError('pipeline output exceeds byte limit')


async def _stop_owned(process):
    # The child starts its own process group. Never signal an unrelated process.
    for attempt in range(10):
        try:
            os.killpg(process.pid,signal.SIGKILL)
            break
        except ProcessLookupError:
            break
        except PermissionError:
            # XNU excludes SZOMB group members and can transiently return EPERM
            # while the owned leader is being reaped. Never ignore persistent EPERM.
            if attempt==9: raise
            await asyncio.sleep(.01)
    if process.stdin: process.stdin.close()
    transport=process._transport
    for descriptor in (1,2):
        pipe=transport.get_pipe_transport(descriptor)
        if pipe is not None: pipe.close()
    try:
        await process.wait()
    finally:
        transport.close()


async def _finish_owned(task):
    while True:
        try: return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done(): return task.result()


async def _execute(command, request, timeout_ms):
    started=time.perf_counter_ns()
    proc=None; spawn=None; collection=None; jobs=[]; output=b''; error=b''; response=None
    status='error'; diagnostic=None; answer_ready_ms=None
    try:
        spawn=asyncio.create_task(asyncio.create_subprocess_exec(*command['argv'],cwd=command['cwd'],
            env=command['environment'],stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE,start_new_session=True))
        # Preserve caller cancellation when completion occurs in the same loop
        # turn (wait_for can lose it on Python3.10). Spawn remains owned.
        done,_=await asyncio.wait({spawn},timeout=timeout_ms/1000)
        if not done: raise asyncio.TimeoutError()
        proc=spawn.result()
        async def send():
            proc.stdin.write(_encoded(request)+b'\n')
            await proc.stdin.drain(); proc.stdin.close()
            await proc.stdin.wait_closed()
        async def read_response():
            nonlocal output,response,answer_ready_ms
            # One complete JSON line is the answer-ready boundary; no trailing protocol data.
            line=bytearray()
            while True:
                chunk=await proc.stdout.read(4096)
                if not chunk: break
                line.extend(chunk)
                if len(line)>MAX_OUTPUT_BYTES: raise ValueError('pipeline output exceeds 64 KiB')
                if b'\n' in line:
                    first,rest=bytes(line).split(b'\n',1)
                    if rest.strip(): raise ValueError('pipeline emitted trailing output')
                    response=_decode(first)
                    _keys(response,('schema_version','status','answer','measurements'),'response')
                    if (type(response['schema_version']) is not int or response['schema_version']!=1 or
                            response['status'] not in ('success','error') or
                            type(response['answer']) is not str or type(response['measurements']) is not dict):
                        raise ValueError('invalid pipeline response')
                    _encoded(response)  # also reject exponent-overflow/nonfinite nested values
                    # Measurements are source reports, never trusted automatic evaluator rows.
                    output=first
                    answer_ready_ms=(time.perf_counter_ns()-started)/1e6
                    trailing=await _drain(proc.stdout,MAX_OUTPUT_BYTES-len(line))
                    if trailing.strip(): raise ValueError('pipeline emitted trailing output')
                    return
            raise ValueError('pipeline did not emit a complete JSON response line')
        jobs=[asyncio.create_task(send()),asyncio.create_task(read_response()),
              asyncio.create_task(_drain(proc.stderr,MAX_STDERR_BYTES)),asyncio.create_task(proc.wait())]
        async def collect(): return await asyncio.gather(*jobs)
        remaining=max(0.,timeout_ms/1000-(time.perf_counter_ns()-started)/1e9)
        if remaining<=0: raise asyncio.TimeoutError()
        collection=asyncio.create_task(collect())
        done,_=await asyncio.wait({collection},timeout=remaining)
        if not done: raise asyncio.TimeoutError()
        results=collection.result()
        error=results[2]
        status=response['status'] if proc.returncode==0 else 'error'
        if proc.returncode!=0: diagnostic='nonzero_exit'
    except asyncio.TimeoutError:
        status,diagnostic='timeout','deadline_exceeded'
    except asyncio.CancelledError:
        diagnostic='cancelled'
        raise
    except Exception as exc:
        response=None  # malformed/unbounded responses never enter source traces
        diagnostic=type(exc).__name__  # no environment/credential text in public errors
    finally:
        async def cleanup():
            nonlocal proc
            if proc is None and spawn is not None:
                try: proc=await spawn
                except Exception: pass
            try:
                if proc is not None:
                    # Every invocation owns its group, including descendants with
                    # detached stdio after a successful direct-child exit.
                    await _stop_owned(proc)
            finally:
                # Retrieve every owned task even if process-group cleanup fails.
                if collection is not None and not collection.done(): collection.cancel()
                for job in jobs:
                    if not job.done(): job.cancel()
                owned=jobs+([collection] if collection is not None else [])
                if owned: await asyncio.gather(*owned,return_exceptions=True)
        cleanup_task=asyncio.create_task(cleanup())
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            await _finish_owned(cleanup_task)
            raise
    elapsed=(time.perf_counter_ns()-started)/1e6
    return dict(status=status,diagnostic=diagnostic,response=response,
        answer_ready_ms=answer_ready_ms,process_lifetime_ms=elapsed,
        returncode=proc.returncode if proc else None,
        response_sha256=hashlib.sha256(output).hexdigest() if output else None,
        stderr_sha256=hashlib.sha256(error).hexdigest() if error else None,
        measurement_status='pipeline_reported_unverified',quality_status='unscored')


async def run(plan, inputs, output_directory, *, expected_plan_sha256):
    """Run every frozen pair, retaining failures. No scores or pass claims are generated."""
    if os.name!='posix': raise RuntimeError('paired command runner currently requires POSIX process groups')
    # Validate a detached snapshot, not caller-owned dictionaries that may change.
    plan=_decode(_encoded(plan)); inputs=_decode(_encoded(inputs))
    by_id=validate_collection(plan,inputs,expected_plan_sha256)
    destination=Path(output_directory)
    destination.mkdir(mode=0o700,parents=False,exist_ok=False)
    _write_new(destination/'plan.json',plan)
    _write_new(destination/'inputs.json',[{'task_id':t['id'],'input':by_id[t['id']]} for t in plan['tasks']])
    _write_new(destination/'collection.json',dict(schema_version=1,plan_sha256=expected_plan_sha256,
        state='started',order='alternating',scoring='unscored',started_unix_ns=time.time_ns()))
    count=0; previous='0'*64
    with (destination/'source-traces.jsonl').open('xb') as stream:
        os.chmod(destination/'source-traces.jsonl',0o600)
        for index,task in enumerate(plan['tasks']):
            order=('baseline','candidate') if index%2==0 else ('candidate','baseline')
            for system in order:
                request=dict(schema_version=1,task_id=task['id'],input=by_id[task['id']],system=system,
                    plan_sha256=expected_plan_sha256,configuration=plan['systems'][system])
                result=await _execute(plan['runner']['commands'][system],request,plan['runner']['timeout_ms'])
                record=dict(ordinal=count,task_id=task['id'],system=system,plan_sha256=expected_plan_sha256,
                    input_sha256=task['input_sha256'],request_sha256=plan_digest(request),
                    previous_trace_sha256=previous,**result)
                digest=plan_digest(record)
                stream.write(_encoded(dict(record,trace_sha256=digest))+b'\n')
                stream.flush(); os.fsync(stream.fileno())
                previous=digest; count+=1
    summary=dict(schema_version=1,state='collected_unscored',records=count,
        plan_sha256=expected_plan_sha256,last_trace_sha256=previous,
        certification='Unscored source reports; real inference, accounting, resource scope and scoring need independent audit.')
    _write_new(destination/'complete.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True); parser.add_argument('--inputs',required=True)
    parser.add_argument('--plan-sha256',required=True); parser.add_argument('--output',required=True)
    args=parser.parse_args()
    print(json.dumps(asyncio.run(run(_read(args.plan),_read(args.inputs),args.output,
        expected_plan_sha256=args.plan_sha256)),indent=2))


if __name__=='__main__': main()

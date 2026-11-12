"""Measure actual MCP subprocess round trips with synthetic conversation input."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import select
import subprocess
import sys
import time


def percentiles(values):
    values = sorted(values)
    return {'count': len(values), 'mean_ms': sum(values)/len(values),
            **{name:values[max(0,math.ceil(q*len(values))-1)] for name,q in
               (('p50_ms',.5),('p95_ms',.95),('p99_ms',.99))}, 'max_ms':max(values)}


def run(samples=100, timeout=10):
    if type(samples) is not int or not 1 <= samples <= 100000:
        raise ValueError('samples must be in 1..100000')
    env = dict(os.environ)
    # Transport benchmark must not call any real configured provider.
    for key in list(env):
        if key.startswith(('HUMANSENSE_REASONING_', 'HUMANSENSE_COMPANION_')):env.pop(key)
    child = subprocess.Popen([sys.executable,'-m','humansense.mcp'],stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env)
    timings = {'push_text':[], 'get_state':[], 'get_metrics':[]}
    ident=0
    def request(method, params):
        nonlocal ident
        ident += 1
        begin=time.perf_counter_ns()
        child.stdin.write((json.dumps({'jsonrpc':'2.0','id':ident,'method':method,'params':params})+'\n').encode())
        child.stdin.flush()
        ready,_,_=select.select([child.stdout],[],[],timeout)
        if not ready:raise TimeoutError('MCP response timeout')
        raw=child.stdout.readline()
        if not raw:raise RuntimeError('MCP process ended before response')
        response=json.loads(raw)
        elapsed=(time.perf_counter_ns()-begin)/1e6
        if response.get('id')!=ident or response.get('error') or response.get('result',{}).get('isError'):
            raise RuntimeError('Unexpected MCP response: '+str(response)[:300])
        return response,elapsed
    started=time.perf_counter()
    try:
        _,startup=request('initialize',{'protocolVersion':'2025-06-18','clientInfo':{'name':'latency-benchmark','version':'1'}})
        for index in range(samples):
            for name,args in (('push_text',{'text':f'Please clarify synthetic item {index}'}),('get_state',{})):
                _,elapsed=request('tools/call',{'name':name,'arguments':args});timings[name].append(elapsed)
        result,elapsed=request('tools/call',{'name':'get_metrics','arguments':{}})
        timings['get_metrics'].append(elapsed)
        runtime=json.loads(result['result']['content'][0]['text'])
    finally:
        if child.stdin:child.stdin.close()
        try:child.wait(timeout=3)
        except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=3)
        if child.stdout:child.stdout.close()
        if child.stderr:child.stderr.close()
    return {'generated_at':datetime.now(timezone.utc).isoformat(),'workload':'synthetic_text_actual_mcp_stdio',
            'python':platform.python_version(),'platform':platform.platform(),'startup_roundtrip_ms':startup,
            'wall_seconds':time.perf_counter()-started,'roundtrips':{k:percentiles(v) for k,v in timings.items()},
            'runtime_metrics':runtime,'limitations':['No camera/audio/model inference','Includes JSON serialization, pipe transport and SDK processing',
                'Synchronous one-client requests; not concurrent transport throughput','No FPS/ASR/8h validation']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=int,default=100)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.output is not None and args.output.exists():parser.error('Output exists')
    report=run(args.samples)
    rendered=json.dumps(report,ensure_ascii=False,indent=2)
    if args.output:
        with args.output.open('x') as handle:handle.write(rendered+'\n')
    print(rendered)


if __name__=='__main__':main()

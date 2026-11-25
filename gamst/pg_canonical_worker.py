"""Private fixed PG canonical worker. Supervisor owns staging and process reap."""
import argparse
import json
from pathlib import Path
import sys

from .pg_canonical import load_owned
from .pg_canonical_process import REQUEST_LIMIT
from .sam_audio_artifact import _absolute,_file,_json,_keys,_regular_path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',required=True);parser.add_argument('--sha256',required=True)
    args=parser.parse_args();path=_absolute(args.request);stage=path.parent
    _regular_path(stage,directory=True)
    if path.name!='request.json' or not stage.name.startswith('.gamst-pg-') or {p.name for p in stage.iterdir()}!={'request.json'}:
        raise ValueError('exclusive supervisor staging required')
    request=_json(_file(path,args.sha256,REQUEST_LIMIT,retain=True),REQUEST_LIMIT)
    keys=('graph','graph_sha','lock','lock_sha','canonical','canonical_sha')
    _keys(request,keys+(('generation',) if 'generation' in request else ()),'PG worker request')
    generation=None
    if 'generation' in request:
        from .pg_generation import policy
        generation=policy(request['generation'],request['graph']['execution'])
    source=_absolute(request['graph']['bundle_root'])
    if stage==source or stage.is_relative_to(source) or source.is_relative_to(stage):
        raise ValueError('source/stage overlap')
    result=load_owned(request['graph'],request['graph_sha'],request['lock'],request['lock_sha'],
                      request['canonical'],request['canonical_sha'],stage)
    if type(result) is dict:
        print(json.dumps(result,allow_nan=False),flush=True);return
    model,receipt=result
    try:
        if generation is not None:
            from .pg_generation import original_generate
            from .pg_physical_worker import serve
            from .pg_artifact import manifest_digest
            generate=original_generate(request['graph'])
            receipt['generation_sha256']=manifest_digest(generation)
            print(json.dumps(receipt,allow_nan=False),flush=True)
            serve(model,generate,request['graph'],generation,receipt)
            return
        print(json.dumps(receipt,allow_nan=False),flush=True)
        # No inference command is accepted in this materialization-only delivery.
        command=sys.stdin.buffer.readline(64)
        if command not in (b'',b'{"operation":"close"}\n'):
            raise ValueError('unsupported PG worker operation')
    finally:
        del model


if __name__=='__main__':
    try:main()
    except Exception as error:
        print(json.dumps(dict(status='unavailable',reason=type(error).__name__+': '+str(error)[:512])),flush=True)
        raise SystemExit(1)

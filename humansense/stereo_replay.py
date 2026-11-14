"""Replay explicit hash-bound stereo packets through the selected owned worker."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

from .capabilities import SensorMetadata
from .plugins import InputFrame
from .process_provider import ProcessProvider
from .stereo_packet import HEADER_LIMIT, PIXEL_LIMIT, unpack_stereo
from .providers.stereo_context import configuration


def _read(path, limit, digest):
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('absolute local recording path required')
    for parent in (path,*path.parents):
        if stat.S_ISLNK(parent.lstat().st_mode):
            raise ValueError('symlinked recording paths unsupported')
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= limit:
        raise ValueError('bounded regular recording file required')
    def identity(s): return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns
    fd = os.open(path,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0)|getattr(os,'O_NONBLOCK',0))
    with os.fdopen(fd,'rb') as stream:
        opened = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened.st_mode) or identity(opened) != identity(before):
            raise ValueError('recording changed before read')
        data = stream.read(limit+1)
        after = os.fstat(stream.fileno())
    if (len(data) != before.st_size or identity(before) != identity(after)
            or identity(path.lstat()) != identity(before) or hashlib.sha256(data).hexdigest() != digest):
        raise ValueError('recording changed or hash mismatch')
    return data


def _json(data):
    def pairs(items):
        result = {}
        for k,v in items:
            if k in result: raise ValueError('duplicate recording JSON key')
            result[k] = v
        return result
    def reject(value): raise ValueError('nonfinite recording JSON')
    try: return json.loads(data.decode('utf-8'),object_pairs_hook=pairs,parse_constant=reject)
    except (UnicodeError,RecursionError) as exc: raise ValueError('invalid recording JSON') from exc


def _sha(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}',value) is None:
        raise ValueError('lowercase SHA256 required')
    return value


def recording(manifest, sha256):
    path = Path(manifest)
    doc = _json(_read(path,1024*1024,_sha(sha256)))
    if (type(doc) is not dict or set(doc) != {'schema_version','config','packets'}
            or type(doc['schema_version']) is not int or doc['schema_version'] != 1):
        raise ValueError('exact stereo recording schema1 required')
    config = configuration(doc['config'])
    rows = doc['packets']
    if type(rows) is not list or not 1 <= len(rows) <= 1024:
        raise ValueError('recording needs1..1024 packets')
    seen, entries, total = set(), [], 0
    for row in rows:
        if type(row) is not dict or set(row) != {'file','sha256'}:
            raise ValueError('exact stereo recording entry required')
        name = row['file']
        if type(name) is not str or re.fullmatch(r'[a-zA-Z0-9_-]{1,100}\.gstereo',name) is None or name in seen:
            raise ValueError('unique stereo recording basename required')
        seen.add(name)
        target = path.parent/name
        data = _read(target,12+HEADER_LIMIT+2*PIXEL_LIMIT,_sha(row['sha256']))
        unpack_stereo(data)
        total += len(data)
        if total > 128*1024*1024: raise ValueError('recording exceeds128MiB replay budget')
        entries.append((target,row['sha256']))
    return config,entries


def recorded_frame(path, digest):
    data = _read(path,12+HEADER_LIMIT+2*PIXEL_LIMIT,digest)
    header,_,_ = unpack_stereo(data)
    meta = header['left']; sensor = dict(meta['sensor'])
    sensor['channels'] = tuple(sensor['channels'])
    return InputFrame(data,'video',meta['interval_ms'][1],subject_id=meta['subject_id'],
                      sensor=SensorMetadata(**sensor))


async def replay(manifest, sha256, *, python=sys.executable):
    """Yield bounded registered values; historical capture time is preserved."""
    config,entries = recording(manifest,sha256)
    worker = await ProcessProvider.open(python,'opencv.rfdetr.stereo',config)
    try:
        for path,digest in entries:
            frame = recorded_frame(path,digest)
            value = await worker.process(frame)
            yield dict(packet_sha256=digest,capture_timestamp_ms=frame.timestamp_ms,
                       capability_id=worker.spec.id,capability_version=worker.spec.version,
                       value=value,physical_accuracy_verified=False)
    finally:
        await worker.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--sha256',required=True)
    parser.add_argument('--python',default=sys.executable)
    parser.add_argument('--consent',action='store_true',required=True)
    args = parser.parse_args()
    async def run():
        async for row in replay(args.manifest,args.sha256,python=args.python):
            print(json.dumps(row,sort_keys=True,allow_nan=False),flush=True)
    asyncio.run(run())


if __name__ == '__main__': main()

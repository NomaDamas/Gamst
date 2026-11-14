"""Separately consented, exclusive transport artifacts; no invoice verification."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import stat
import time
import uuid
from .native_entry import encoded, validate_owner
from .streams import _finish

RAW=1048576; MANIFEST=65536; TOMBSTONE=4096; REFERENCE=768
RESERVATION=RAW+MANIFEST+TOMBSTONE+REFERENCE


def anchor():
    before=time.monotonic_ns();utc=time.time_ns();after=time.monotonic_ns()
    return dict(monotonic_before_ns=before,utc_ns=utc,monotonic_after_ns=after)


def _directory(path):
    """Open every existing ancestor without following symlinks; create leaf only."""
    path=Path(os.path.abspath(os.fspath(path)))
    fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            next_fd=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
            os.close(fd);fd=next_fd
        os.mkdir(path.name,0o700,dir_fd=fd)
        result=os.open(path.name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
        return result,path
    finally:os.close(fd)


class ResponsesEvidenceSession:
    def __init__(self,directory,*,owner_kind,expected_owner,retain_response_bytes=False,retain_financial_hashes=False,max_bytes=64*1024*1024):
        if retain_response_bytes is not True or retain_financial_hashes is not True:raise PermissionError('both separate retention consents must be exactly True')
        if type(max_bytes) is not int or not RESERVATION+1024<=max_bytes<=64*1024*1024:raise ValueError('bounded evidence capacity required')
        self.owner_kind=owner_kind;self._owner=encoded(validate_owner(owner_kind,expected_owner))
        self._fd,self.directory=_directory(directory)
        self._max=max_bytes;self._used=1024;self._reserved=0;self._leases=set();self._refs=[];self._next=0
        self._closing=False;self._closed=False;self._close_task=None;self._incomplete=False;self._identities={}

    def expected_owner(self):return json.loads(self._owner)

    def reserve(self,lease):
        if self._closing or self._closed or self._incomplete:raise RuntimeError('evidence session admission closed')
        if self._next>=1024 or self._used+self._reserved+RESERVATION>self._max:raise RuntimeError('evidence capacity exhausted before send')
        ordinal=self._next;self._next+=1
        raw_path=f'raw-{ordinal:04d}.bin'
        fd=os.open(raw_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=self._fd)
        self._reserved+=RESERVATION;self._leases.add(lease)
        return ordinal,raw_path,fd

    def _write(self,path,raw):
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=self._fd)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):raise ValueError('regular artifact required')
            _write_fd(fd,raw);os.fsync(fd)
        finally:os.close(fd)

    def finish(self,lease,manifest,raw):
        if lease not in self._leases:raise RuntimeError('lease already terminal')
        manifest_path=f'call-{lease.ordinal:04d}.json';tombstone_path=None
        written=0;manifest_raw=None;tombstone_raw=None;raw_hash=hashlib.sha256(raw).hexdigest()
        try:
            if self._closed:raise RuntimeError('terminal evidence session cannot write')
            for key in ('server_request_id','response_id'):
                identity=manifest[key]
                if identity is not None:
                    identity=(manifest['provider'],manifest['declared_account_scope'],key,identity)
                    if identity in self._identities:raise ValueError('provider identity reused by distinct native entry')
                    self._identities[identity]=lease.ordinal
            _write_fd(lease.raw_fd,raw);os.fsync(lease.raw_fd);written+=len(raw)
            manifest['raw_response'].update(path=lease.raw_path,sha256=raw_hash,bytes=len(raw))
            manifest_raw=encoded(manifest)+b'\n'
            if len(manifest_raw)>MANIFEST:raise ValueError('reserved manifest quota exceeded')
            self._write(manifest_path,manifest_raw);written+=len(manifest_raw)
        except BaseException:
            self._incomplete=True
            tombstone_path=f'failed-{lease.ordinal:04d}.json'
            tombstone_raw=encoded(dict(schema_version=1,kind='transport_evidence_incomplete',ordinal=lease.ordinal,
                opaque_financial_token=lease.token,possible_send=lease.possible_send,actual_cost=None,
                failure_category='artifact_failure',receipt_authenticity_verified=False))+b'\n'
            try:self._write(tombstone_path,tombstone_raw);written+=len(tombstone_raw)
            except BaseException:tombstone_path=None;tombstone_raw=None
            raise
        finally:
            os.close(lease.raw_fd);lease.raw_fd=None
            # Partial failed artifacts consume their full reservation; do not
            # reclaim bytes whose persisted length cannot be established.
            reference=dict(ordinal=lease.ordinal,financial_token=lease.token,
                manifest_path=manifest_path if manifest_raw is not None and not self._incomplete else None,
                manifest_sha256=hashlib.sha256(manifest_raw).hexdigest() if manifest_raw is not None and not self._incomplete else None,
                manifest_bytes=len(manifest_raw) if manifest_raw is not None and not self._incomplete else 0,
                tombstone_path=tombstone_path,tombstone_sha256=hashlib.sha256(tombstone_raw).hexdigest() if tombstone_raw else None,
                tombstone_bytes=len(tombstone_raw) if tombstone_raw else 0,
                raw_path=lease.raw_path,raw_sha256=raw_hash if not self._incomplete else None,raw_bytes=len(raw),state='incomplete' if self._incomplete else 'sealed')
            if len(encoded(reference))>REFERENCE:self._incomplete=True;raise RuntimeError('inventory reference quota exceeded')
            self._refs.append(reference);self._reserved-=RESERVATION
            self._used+=RESERVATION if self._incomplete else written+len(encoded(reference))
            self._leases.remove(lease)

    async def _close(self):
        leases=tuple(self._leases)
        for lease in leases:lease.invalidate()
        for lease in leases:await lease.release()
        result=dict(schema_version=1,kind='openai_responses_inventory',owner_kind=self.owner_kind,
            owner_sha256=hashlib.sha256(self._owner).hexdigest(),entries=sorted(self._refs,key=lambda x:x['ordinal']),
            incomplete=self._incomplete,actual_acceptance='incomplete',receipt_authenticity_verified=False,
            whole_pipeline_cost_verified=False)
        raw=encoded(result)+b'\n'
        try:
            if self._leases or self._reserved or len(raw)>1048576:raise RuntimeError('evidence inventory cannot seal')
            self._write('inventory.json',raw);self._closed=True
            return result
        finally:
            os.close(self._fd);self._fd=-1

    async def close(self):
        self._closing=True
        if any(lease.owns_task(asyncio.current_task()) for lease in tuple(self._leases)):
            for lease in tuple(self._leases):lease.invalidate()
            raise RuntimeError('owned transport cannot join its evidence session')
        if self._close_task is None:self._close_task=asyncio.create_task(self._close())
        try:return await asyncio.shield(self._close_task)
        except asyncio.CancelledError:await _finish(self._close_task);raise


def _write_fd(fd,raw):
    view=memoryview(raw)
    for offset in range(0,len(view),65536):
        block=view[offset:offset+65536]
        while block:
            count=os.write(fd,block)
            if count<=0:raise OSError('artifact write failed')
            block=block[count:]

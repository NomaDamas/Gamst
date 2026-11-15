"""Explicit same-interpreter Smol lane; never downloads or selects model code.

The first owned profile intentionally refuses a different Python environment:
parent installation evidence does not authenticate a separate worker install.
"""
import asyncio
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import stat
import sys

from gamst.smolvlm_call import SmolVLMCalls
from gamst.providers.smolvlm_native import MODEL_ID, RUNTIME, configuration, prompt_text
from gamst.provider_profiles import capability
from gamst.streams import _finish
from .recorded_source import digest, encoded
from .recording_dependencies import VerifiedDependencyInventory
from .recording_runtime_contract import validate_adapter, validate_assets


def implementation_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _identity(info):
    return (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)


class _Pins:
    def __init__(self):self.files=[]
    def read(self,path,expected,size,limit,*,content=False):
        if not Path(path).is_absolute() or type(size) is not int or not 0<size<=limit:
            raise ValueError('bounded absolute original Smol asset required')
        directory=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
        fd=None
        try:
            for part in Path(path).parts[1:-1]:
                child=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=directory)
                os.close(directory);directory=child
            fd=os.open(Path(path).name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=directory)
            before=os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_size!=size:
                raise ValueError('original Smol asset file type/size mismatch')
            hashed=hashlib.sha256();parts=[];count=0
            while True:
                block=os.read(fd,min(65536,size+1-count))
                if not block:break
                count+=len(block)
                if count>size:raise ValueError('Smol asset grew during verification')
                hashed.update(block)
                if content:parts.append(block)
            if count!=size or hashed.hexdigest()!=expected or _identity(before)!=_identity(os.fstat(fd)):
                raise ValueError('original Smol asset bytes/identity mismatch')
            self.files.append((fd,path,_identity(before)));fd=None
            return b''.join(parts) if content else None
        finally:
            if fd is not None:os.close(fd)
            os.close(directory)
    def check(self):
        for fd,path,identity in self.files:
            if _identity(os.fstat(fd))!=identity or _identity(os.stat(path,follow_symlinks=False))!=identity:
                raise ValueError('retained Smol asset changed before activation')
    def close(self):
        files,self.files=self.files,[]
        for fd,_,_ in files:os.close(fd)


def _admit(adapter,assets,documents,dependencies,scheduler):
    adapter=validate_adapter(adapter);assets=validate_assets(assets)
    if adapter['implementation']!='smolvlm_process_v1' or scheduler!='smolvlm_sentinel_v1':
        raise ValueError('fixed Smol sentinel profile required')
    if adapter['model_id']!=MODEL_ID or adapter['implementation_sha256']!=implementation_sha256():
        raise ValueError('actual fixed Smol implementation/model required')
    if type(dependencies) is not VerifiedDependencyInventory or dependencies.sha256!=adapter['dependency_inventory_sha256']:
        raise ValueError('actual issued current-environment dependencies required')
    for name,version in RUNTIME.items():
        if dependencies.require(name)['version'].split('+')[0]!=version:
            raise ValueError('actual pinned Smol runtime version required')
    for name in ('numpy','pillow'):dependencies.require(name)
    def asset(kind,sha):
        rows=[a for a in assets['assets'] if a['kind']==kind and a['sha256']==sha]
        if len(rows)!=1:raise ValueError('one exact original Smol asset required')
        return rows[0]
    worker=asset('worker_config',adapter['config']['worker_config_sha256'])
    if worker['path'] not in documents:raise ValueError('owned original Smol worker document missing')
    config=configuration(documents[worker['path']])
    raw=encoded(config)+b'\n'
    if config!=documents[worker['path']] or len(raw)!=worker['bytes'] or hashlib.sha256(raw).hexdigest()!=worker['sha256']:
        raise ValueError('original effective Smol worker configuration mismatch')
    python=asset('python',adapter['config']['python_asset_sha256'])
    # Only the current interpreter's exact canonical executable is admitted.
    # A different virtualenv spelling can change imports despite sharing an inode.
    actual=Path(sys.executable).absolute()
    if Path(python['path']).absolute()!=actual:
        raise ValueError('separate Python environment has no verified dependency inventory')
    resolved=actual.resolve(strict=True)
    if sys.prefix!=sys.base_prefix and actual!=resolved:
        raise ValueError('symlink-dependent virtual environment cannot use a canonical worker launch')
    if not os.access(actual,os.X_OK):raise ValueError('existing executable Python required')
    pins=_Pins()
    try:
        pins.read(str(resolved),python['sha256'],python['bytes'],256*1024**2)
        for pathkey,hashkey,limit in (('checkpoint_path','checkpoint_sha256',2500*1024**2),
                                     ('tokenizer_path','tokenizer_sha256',32*1024**2)):
            original=asset('checkpoint',config[hashkey])
            if original['path']!=config[pathkey]:raise ValueError('Smol asset path differs from frozen config')
            pins.read(original['path'],original['sha256'],original['bytes'],limit)
        spec=capability('smolvlm2.500m.local',dict(config,window_frames=1))
        if (adapter['provider'],adapter['model_revision'])!=(spec.provider,spec.version):
            raise ValueError('actual Smol provider/revision differs from frozen adapter')
        pins.check()
        # Launch the exact no-follow target that was verified and retained, not
        # the alias whose resolution could change after admission.
        return adapter,config,str(resolved),pins
    except BaseException:
        pins.close();raise


class OwnedSmolLane:
    """One actual SDK worker lifetime; source capture correlation stays upstream."""
    def __init__(self,adapter,pins,owner,binding_index):
        self.adapter=adapter;self.worker=adapter.worker;self.backend=adapter.worker
        self._pins=pins;self._close_task=None
        self._owner=owner;self._binding_index=binding_index
    def entry_for_decision(self,decision_sha256):
        return self._owner.semantic_entry_for_decision(self._binding_index,decision_sha256)
    def release_entry(self,decision_sha256):
        # There is no additional wrapper cache. Original journal entries must
        # remain owned by the recording after the consumer releases a lookup.
        return None
    @staticmethod
    def prompt_sha256(timestamp_ms):
        return hashlib.sha256(prompt_text([timestamp_ms]).encode()).hexdigest()
    async def _close(self):
        try:await self.adapter.close()
        finally:self._pins.close()
    async def close(self):
        if self._close_task is None:self._close_task=asyncio.create_task(self._close())
        try:await asyncio.shield(self._close_task)
        except asyncio.CancelledError:await _finish(self._close_task);raise


async def open_smol_lane(human,controller_id,adapter,assets,documents,owner,binding_index,dependencies,*,scheduler):
    """Pin originals, open the actual owned SDK worker, bind its stream scope."""
    adapter=validate_adapter(adapter);assets=validate_assets(assets)
    selected=[a for a in assets['assets'] if a['kind']=='worker_config' and
              a['sha256']==adapter['config'].get('worker_config_sha256')]
    if len(selected)!=1 or selected[0]['path'] not in documents:
        raise ValueError('one owned original worker configuration required')
    path=selected[0]['path']
    normalized=configuration(documents[path])
    if normalized!=documents[path]:raise ValueError('explicit effective worker configuration required')
    documents={path:normalized}
    documents=deepcopy(documents)
    job=asyncio.create_task(asyncio.to_thread(_admit,adapter,assets,documents,dependencies,scheduler))
    try:checked,config,python,pins=await asyncio.shield(job)
    except asyncio.CancelledError:
        result=await _finish(asyncio.gather(job,return_exceptions=True))
        if result and isinstance(result[0],tuple):result[0][3].close()
        raise
    opened=None
    try:
        declarations=owner.declarations
        collection=declarations['collection']
        invocation=collection['invocations'][owner.context['invocation_ordinal']]
        bindings=next(a['provider_bindings'] for a in collection['arms'] if a['arm']==invocation['arm'])
        if type(binding_index) is not int or not 0<=binding_index<len(bindings):
            raise ValueError('exact original Smol binding index required')
        binding=bindings[binding_index]
        if (binding['role']!='small_vlm' or binding['entry_boundary']!='process_adapter' or
                binding['operation_kinds']!=['semantic'] or
                binding['config_sha256']!=digest(dict(config,window_frames=1))):
            raise ValueError('original small-VLM process/config binding required')
        pins.check()
        opened=await SmolVLMCalls.open(human,controller_id,python,config,
            startup_timeout=checked['config']['startup_timeout_ms']/1000,
            inference_timeout=checked['config']['inference_timeout_ms']/1000,
            evaluation_observer=owner,evaluation_worker_scope=owner.stream_scope(binding_index))
        owner.bind_backend(binding_index,opened.worker,'process')
        return OwnedSmolLane(opened,pins,owner,binding_index)
    except BaseException:
        try:
            if opened is not None:await _finish(asyncio.create_task(opened.close()))
        finally:pins.close()
        raise

"""Complete custom PG constructor-dtype state; no legacy pickle conversion.

Only the fixed original graph worker invokes this loader. Helpers accepting tiny
models are private testable mechanics, not an injected-model product interface.
"""
import hashlib
import os
from pathlib import Path
import shutil
import stat
import struct
import sys
from dataclasses import dataclass

from .pg_artifact import validate_manifest as graph_manifest, _encoded, _json, _file
from .pg_sources import PG_REVISION,PG_SHA256,PG_BYTES
from .sam_audio_artifact import _keys,_digest,_relative,_regular_path,_identity
from .pg_graph import _construct_graph,PROJECTION

FORMAT='pg-reference-constructor-storage-v1'
MAX_MANIFEST=4*1024**2
MAX_SHARDS=64
MAX_SHARD=64*1024**3
MAX_BYTES=128*1024**3
MAX_HEADER=4*1024**2
MAX_HEADERS=16*1024**2
MAX_ROWS=20000
CHUNK_BYTES=1024**2
CHUNK_ELEMENTS=262144
MIN_WORKSPACE=256*1024**2
DTYPES={'F32':('torch.float32',4),'F16':('torch.float16',2),
        'BF16':('torch.bfloat16',2),'I64':('torch.int64',8)}


def _int(value,low,high):
    if type(value) is not int or not low<=value<=high:raise ValueError('integer outside canonical bound')


def _name(value):
    if type(value) is not str or not 1<=len(value.encode('utf-8'))<=512 or any(ord(c)<32 for c in value):
        raise ValueError('invalid canonical tensor name')
    return value


def validate_canonical(value,*,expected_sha256):
    raw=_encoded(value)
    if len(raw)>MAX_MANIFEST or hashlib.sha256(raw).hexdigest()!=_digest(expected_sha256):
        raise ValueError('canonical manifest bounds/digest mismatch')
    value=_json(raw,MAX_MANIFEST)
    _keys(value,('schema_version','format','graph_manifest_sha256','runtime_lock_sha256',
        'graph_schema_sha256','graph_buffer_sha256','source_identity','export_provenance',
        'tensors','shards','budgets'),'canonical manifest')
    _int(value['schema_version'],1,1)
    if value['format']!=FORMAT:raise ValueError('unsupported canonical format')
    for key in ('graph_manifest_sha256','runtime_lock_sha256','graph_schema_sha256'):_digest(value[key])
    buffers=value['graph_buffer_sha256']
    if type(buffers) is not dict or len(buffers)>MAX_ROWS:raise ValueError('buffer digest bound')
    for name,digest in buffers.items():_name(name);_digest(digest)
    if value['source_identity']!=dict(pg_revision=PG_REVISION,pg_sha256=PG_SHA256,pg_bytes=PG_BYTES):
        raise ValueError('canonical publisher declaration mismatch')
    provenance=value['export_provenance']
    _keys(provenance,('tool_revision','source_evidence_sha256','reference_replay_sha256'),'export provenance')
    if type(provenance['tool_revision']) is not str or len(provenance['tool_revision'])!=40 or any(c not in '0123456789abcdef' for c in provenance['tool_revision']):
        raise ValueError('bounded export tool revision required')
    for key in ('source_evidence_sha256','reference_replay_sha256'):_digest(provenance[key])
    shards=value['shards']
    if type(shards) is not list or not 1<=len(shards)<=MAX_SHARDS:raise ValueError('shard count bound')
    paths=set();total=0
    for shard in shards:
        _keys(shard,('path','size','sha256'),'canonical shard')
        path=_relative(shard['path']);_int(shard['size'],10,MAX_SHARD);_digest(shard['sha256'])
        if path in paths:raise ValueError('duplicate canonical shard path')
        paths.add(path);total+=shard['size']
    if total>MAX_BYTES:raise ValueError('canonical aggregate bound')
    tensors=value['tensors']
    if type(tensors) is not dict or not 1<=len(tensors)<=MAX_ROWS:raise ValueError('canonical tensor count')
    logical=set();payloads=set()
    for rep,row in tensors.items():
        _name(rep);_keys(row,('logical_names','shard','tensor_key'),'canonical tensor mapping')
        names=row['logical_names']
        if type(names) is not list or not 1<=len(names)<=MAX_ROWS or names!=sorted(set(names)):
            raise ValueError('canonical logical names must be sorted and unique')
        for name in names:
            _name(name)
            if name in logical:raise ValueError('duplicate logical state path')
            logical.add(name)
        if rep not in names or len(logical)>MAX_ROWS:raise ValueError('representative/logical coverage bound')
        _int(row['shard'],0,len(shards)-1);_name(row['tensor_key'])
        payload=(row['shard'],row['tensor_key'])
        if payload in payloads:raise ValueError('distinct aliases cannot reuse one tensor payload')
        payloads.add(payload)
    budgets=value['budgets']
    _keys(budgets,('disk_bytes','ram_destination_bytes','vram_destination_bytes','ram_workspace_bytes'),'canonical budgets')
    for amount in budgets.values():_int(amount,0,512*1024**3)
    if budgets['disk_bytes']<total or budgets['ram_workspace_bytes']<MIN_WORKSPACE:
        raise ValueError('snapshot disk/workspace planned budget insufficient')
    return value


def schema_digest(schema,graph_sha,lock_sha):
    return hashlib.sha256(_encoded(dict(projection=PROJECTION,graph_manifest_sha256=graph_sha,
        runtime_lock_sha256=lock_sha,rows=schema['rows'],buffer_sha256=schema['buffer_sha256']))).hexdigest()


@dataclass
class _Snapshot:
    path: Path
    fd: int
    identity: tuple

    def read(self,size,offset):
        if _identity(os.fstat(self.fd))!=self.identity:
            raise ValueError('owned snapshot identity changed')
        raw=os.pread(self.fd,size,offset)
        if len(raw)!=size or _identity(os.fstat(self.fd))!=self.identity:
            raise ValueError('owned snapshot changed/truncated')
        return raw

    def read_into(self,size,offset):
        if not hasattr(os,'preadv'):raise RuntimeError('bounded owned FD read requires preadv')
        if _identity(os.fstat(self.fd))!=self.identity:raise ValueError('owned snapshot identity changed')
        value=bytearray(size);view=memoryview(value);count=0
        while count<size:
            amount=os.preadv(self.fd,[view[count:]],offset+count)
            if not amount:raise ValueError('owned snapshot truncated')
            count+=amount
        if _identity(os.fstat(self.fd))!=self.identity:raise ValueError('owned snapshot identity changed')
        return value

    def close(self):
        if self.fd>=0:
            descriptor,self.fd=self.fd,-1
            os.close(descriptor)


def _snapshot_into(root,shards,stage,paths):
    """Stream checked source FDs to exclusive supervisor-owned snapshot names."""
    for index,descriptor in enumerate(shards):
        source=root/descriptor['path'];before=_regular_path(source)
        if before.st_size!=descriptor['size']:raise ValueError('canonical source size mismatch')
        fd=os.open(source,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0)|getattr(os,'O_NONBLOCK',0))
        destination=stage/f'shard-{index:02d}.safetensors'
        with os.fdopen(fd,'rb') as reader:
            if _identity(os.fstat(reader.fileno()))!=_identity(before):raise ValueError('source changed before open')
            out=os.open(destination,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600)
            digest=hashlib.sha256();count=0
            with os.fdopen(out,'wb') as writer:
                while True:
                    block=reader.read(min(CHUNK_BYTES,descriptor['size']-count+1))
                    if not block:break
                    count+=len(block)
                    if count>descriptor['size']:raise ValueError('canonical source grew')
                    digest.update(block);writer.write(block)
                writer.flush();os.fsync(writer.fileno());written=os.fstat(writer.fileno())
            after=os.fstat(reader.fileno())
        if (_identity(after)!=_identity(before) or _identity(source.lstat())!=_identity(before) or
                count!=descriptor['size'] or digest.hexdigest()!=descriptor['sha256']):
            raise ValueError('canonical source changed or digest mismatch')
        os.chmod(destination,0o400)
        fd=os.open(destination,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0))
        info=os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_size!=descriptor['size'] or
                (info.st_dev,info.st_ino,info.st_mtime_ns)!=(written.st_dev,written.st_ino,written.st_mtime_ns)):
            os.close(fd);raise ValueError('owned snapshot changed before open')
        paths.append(_Snapshot(destination,fd,_identity(info)))
    return paths


def _headers(paths,shards):
    """Strict custom safetensors headers; all bytes accounted for, no metadata."""
    if len(paths)!=len(shards):raise ValueError('snapshot/shard count mismatch')
    result=[];header_total=0;count=0
    for path,descriptor in zip(paths,shards):
        size=struct.unpack('<Q',path.read(8,0))[0]
        _int(size,2,MAX_HEADER);header_total+=size
        if header_total>MAX_HEADERS:raise ValueError('aggregate canonical header bound')
        raw=path.read(size,8)
        header=_json(raw,MAX_HEADER)
        if type(header) is not dict or not header:raise ValueError('canonical tensor header object required')
        ranges=[];normalized={}
        for name,row in header.items():
            _name(name);_keys(row,('dtype','shape','data_offsets'),'canonical tensor header')
            dtype=row['dtype'];shape=row['shape'];offsets=row['data_offsets']
            if dtype not in DTYPES:raise ValueError('unsupported canonical tensor dtype')
            if type(shape) is not list or len(shape)>8:raise ValueError('canonical tensor rank bound')
            elements=1
            for dimension in shape:_int(dimension,1,2**35);elements*=dimension
            if elements>2**35:raise ValueError('canonical tensor element bound')
            if type(offsets) is not list or len(offsets)!=2:raise ValueError('canonical offset pair required')
            for offset in offsets:_int(offset,0,MAX_SHARD)
            if offsets[1]-offsets[0]!=elements*DTYPES[dtype][1]:raise ValueError('canonical shape/byte mismatch')
            ranges.append(tuple(offsets));normalized[name]=dict(row,data_start=8+size+offsets[0],elements=elements)
        end=0
        for start,stop in sorted(ranges):
            if start!=end:raise ValueError('canonical payload overlap/gap')
            end=stop
        if end+size+8!=descriptor['size']:raise ValueError('unaccounted canonical bytes')
        count+=len(header)
        if count>MAX_ROWS:raise ValueError('canonical header entry count')
        result.append(normalized)
    return result


def _coverage(schema,canonical,headers):
    rows=schema['rows'];groups={};all_paths=set()
    for row in rows:
        if row['name'] in all_paths:raise ValueError('duplicate actual graph path')
        all_paths.add(row['name'])
        groups.setdefault(row['alias'],[]).append(row)
    needed={}
    for rep,group in groups.items():
        if len({r['kind'] for r in group})!=1:
            raise ValueError('mixed parameter/buffer alias unsupported')
        if len({(tuple(r['shape']),r['constructor_dtype']) for r in group})!=1:
            raise ValueError('unsupported graph view/shape alias')
        flags={r['kind']=='parameter' or r['persistent'] for r in group}
        if len(flags)!=1:raise ValueError('mixed persistent/nonpersistent alias unsupported')
        if True in flags:needed[rep]=group
    if set(needed)!=set(canonical['tensors']):raise ValueError('complete graph representative coverage required')
    actual_payloads={(index,key) for index,header in enumerate(headers) for key in header}
    expected_payloads=set()
    for rep,group in needed.items():
        mapping=canonical['tensors'][rep]
        if mapping['logical_names']!=sorted(row['name'] for row in group):raise ValueError('invented/missing graph identity alias')
        pair=(mapping['shard'],mapping['tensor_key'])
        if pair in expected_payloads:raise ValueError('distinct graph groups cannot reuse a payload')
        expected_payloads.add(pair)
        if pair not in actual_payloads:raise ValueError('canonical payload missing')
        header=headers[pair[0]][pair[1]];row=group[0]
        if header['shape']!=row['shape'] or DTYPES[header['dtype']][0]!=row['constructor_dtype']:
            raise ValueError('canonical shape or reference constructor dtype mismatch')
    if actual_payloads!=expected_payloads:raise ValueError('unreferenced/missing canonical payload')
    return needed


def _planned(schema,canonical,device):
    needed=schema['unique_cpu_target_bytes'] if device=='cpu' else schema['unique_constructor_bytes']
    budget=canonical['budgets']['ram_destination_bytes' if device=='cpu' else 'vram_destination_bytes']
    if budget<needed:raise ValueError('planned destination budget insufficient')
    return needed


def _resolve(model,name):
    parent,_,leaf=name.rpartition('.')
    return (model.get_submodule(parent) if parent else model),leaf


def _materialize(model,schema,canonical,headers,paths,device):
    """Private mechanics; caller must provide the actual authenticated PG graph."""
    import torch
    if sys.byteorder!='little':raise ValueError('canonical little-endian runtime required')
    groups=_coverage(schema,canonical,headers)
    _planned(schema,canonical,device)  # BEFORE the first destination allocation.
    dtype_map={key:getattr(torch,name.split('.')[1]) for key,(name,_) in DTYPES.items()}
    # All headers/coverage are checked before tensor allocation. Only one work
    # unit and one conversion unit may exist; CUDA copy is always synchronous.
    for rep,group in groups.items():
        mapping=canonical['tensors'][rep];index=mapping['shard'];header=headers[index][mapping['tensor_key']]
        constructor=dtype_map[header['dtype']]
        target=torch.float32 if device=='cpu' and constructor.is_floating_point else constructor
        destination=torch.empty(group[0]['shape'],dtype=target,device=device)
        flat=destination.reshape(-1);item_size=DTYPES[header['dtype']][1]
        work_elements=min(CHUNK_ELEMENTS,CHUNK_BYTES//item_size)
        for offset in range(0,header['elements'],work_elements):
            count=min(work_elements,header['elements']-offset)
            raw=paths[index].read_into(count*item_size,header['data_start']+offset*item_size)
            source=torch.frombuffer(raw,dtype=constructor)
            if source.is_floating_point() and not torch.isfinite(source).all().item():
                raise ValueError('nonfinite canonical constructor state')
            flat[offset:offset+count].copy_(source,non_blocking=False)
            if device=='cuda':torch.cuda.synchronize()
            if flat.is_floating_point() and not torch.isfinite(flat[offset:offset+count]).all().item():
                raise ValueError('nonfinite canonical destination state')
            del source,raw
        parameter=group[0]['kind']=='parameter'
        if any((row['kind']=='parameter')!=parameter for row in group):raise ValueError('mixed parameter/buffer alias unsupported')
        owned=torch.nn.Parameter(destination,requires_grad=False) if parameter else destination
        for row in group:
            module,leaf=_resolve(model,row['name'])
            (module._parameters if parameter else module._buffers)[leaf]=owned
        del owned,destination,flat
    # Original deterministic buffers are retained/verified independently, even
    # when persistent buffers were included in the canonical payload.
    rows_by_name={row['name']:row for row in schema['rows']}
    for name,digest in schema['buffer_sha256'].items():
        module,leaf=_resolve(model,name);buffer=module._buffers[leaf]
        constructor=getattr(torch,rows_by_name[name]['constructor_dtype'].split('.')[1])
        raw=buffer.detach().to(device='cpu',dtype=constructor).contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('deterministic graph buffer changed')
    retained={}
    for row in schema['rows']:
        if row['kind']=='buffer' and not row['persistent']:
            module,leaf=_resolve(model,row['name'])
            if row['alias'] not in retained:
                dtype=getattr(torch,(row['cpu_target_dtype'] if device=='cpu' else row['cuda_target_dtype']).split('.')[1])
                retained[row['alias']]=module._buffers[leaf].to(device=device,dtype=dtype)
            module._buffers[leaf]=retained[row['alias']]
    actual=dict(model.named_parameters(remove_duplicate=False));actual.update(model.named_buffers(remove_duplicate=False))
    if set(actual)!={row['name'] for row in schema['rows']}:raise ValueError('materialized graph path mismatch')
    identity={}
    for row in schema['rows']:
        tensor=actual[row['name']]
        expected=row['cpu_target_dtype'] if device=='cpu' else row['cuda_target_dtype']
        if tensor.is_meta or tensor.device.type!=device or list(tensor.shape)!=row['shape'] or str(tensor.dtype)!=expected:
            raise ValueError('missing/meta/wrong materialized state')
        previous=identity.setdefault(row['alias'],tensor)
        if tensor is not previous:raise ValueError('materialized alias identity lost')
    if len({id(tensor) for tensor in identity.values()})!=len(identity):raise ValueError('independent graph groups incorrectly shared')
    model.qformer_text_input=False
    model.eval();model.requires_grad_(False)


def load_owned(graph,graph_sha,lock,lock_sha,canonical,canonical_sha,stage):
    """Fixed original graph only. Worker owns the returned model until exit."""
    value=graph_manifest(graph,expected_sha256=graph_sha)
    if os.name!='posix' or not all(hasattr(os,name) for name in ('pread','preadv','O_NOFOLLOW','O_NONBLOCK')):
        return dict(status='unavailable',reason='owned_fd_platform_unavailable')
    spec=validate_canonical(canonical,expected_sha256=canonical_sha)
    if spec['graph_manifest_sha256']!=graph_sha or spec['runtime_lock_sha256']!=lock_sha:
        raise ValueError('canonical graph/runtime binding mismatch')
    result=_construct_graph(value,expected_sha256=graph_sha,runtime_lock=lock,runtime_lock_sha256=lock_sha)
    if type(result) is dict:return result
    model,schema=result
    if schema_digest(schema,graph_sha,lock_sha)!=spec['graph_schema_sha256'] or schema['buffer_sha256']!=spec['graph_buffer_sha256']:
        raise ValueError('canonical full graph schema/buffer mismatch')
    _planned(schema,spec,value['execution']['device'])
    stage=Path(stage);_regular_path(stage,directory=True)
    if shutil.disk_usage(stage).free<sum(shard['size'] for shard in spec['shards']):
        raise ValueError('insufficient free snapshot disk')
    paths=[]
    try:
        _snapshot_into(Path(value['bundle_root']),spec['shards'],stage,paths)
        headers=_headers(paths,spec['shards'])
        _materialize(model,schema,spec,headers,paths,value['execution']['device'])
    finally:
        for snapshot in paths:snapshot.close()
    return model,dict(status='loaded',canonical_sha256=canonical_sha,
        graph_schema_sha256=spec['graph_schema_sha256'],planned_storage_bytes=_planned(schema,spec,value['execution']['device']),
        trained_identity_verified=False,reference_parity_verified=False,
        runtime_compatibility_verified=False,resource_admission='unavailable',actual_acceptance='incomplete')

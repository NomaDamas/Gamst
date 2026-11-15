"""Owned fixed-path metadata input; never activates a model or executes assets."""
import asyncio
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat

from gamst.streams import _finish
from .recording_native_schema import encoded, digest, keys, sha, integer, validate_collection
from .recording_runtime_contract import validate_runtime_policy, validate_assets


MAX_METADATA_BYTES=16*1024**2
MAX_WORKSPACE_BYTES=256*1024**2
MAX_FILES=1024
MAX_TREE_NODES=1048576
_DIR_FLAGS=os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_NONBLOCK
_FILE_FLAGS=os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK
_ISSUER=object()
_BASE=('runtime/source-bundles.json','runtime/policy.json','runtime/latency.json',
       'runtime/assets.json','runtime/plan.json','runtime/collection.json',
       'runtime/scopes.json','runtime/instrumentation.json')
_NUMERIC=('runtime/accounting.json','runtime/opportunities.json','runtime/exposure.json','runtime/financial-adapters.json','runtime/charge-scenarios.json')
_JSON_PATH=re.compile(r'(?:runtime/(?:source-bundles|policy|latency|assets|plan|collection|scopes|instrumentation|accounting|opportunities|exposure|financial-adapters|charge-scenarios)\.json|recordings/[0-9]{2}/(?:manifest|tasks|profiles)\.json|invocations/[0-9]{2}/execution\.json|(?:configurations|responses|budgets)/[0-9]{3}\.json)')


def _identity(info):
    return (info.st_dev,info.st_ino,info.st_mode,info.st_size,info.st_mtime_ns,info.st_ctime_ns)


def _parts(path, *, absolute=False):
    if type(path) is not str or not path or len(path.encode())>4096 or '\0' in path:
        raise ValueError('bounded explicit input path required')
    if absolute:
        if not path.startswith('/'):raise ValueError('absolute owned input root required')
        path=path[1:]
    elif path.startswith('/'):raise ValueError('relative input path required')
    parts=path.split('/')
    if any(p in ('','.','..') for p in parts):raise ValueError('canonical path components required')
    return parts


def _directory(parent, name):
    fd=os.open(name,_DIR_FLAGS,dir_fd=parent);os.set_inheritable(fd,False)
    return fd


def _absolute_directory(path):
    parts=_parts(path,absolute=True);fd=os.open('/',_DIR_FLAGS);os.set_inheritable(fd,False)
    try:
        for component in parts:
            new=_directory(fd,component);os.close(fd);fd=new
        return fd
    except BaseException:os.close(fd);raise


def _parent_fd(root,path):
    parts=_parts(path);fd=os.dup(root);os.set_inheritable(fd,False)
    try:
        for component in parts[:-1]:
            new=_directory(fd,component);os.close(fd);fd=new
        return fd,parts[-1]
    except BaseException:os.close(fd);raise


def _open_file(root,path):
    parent,name=_parent_fd(root,path)
    try:fd=os.open(name,_FILE_FLAGS,dir_fd=parent)
    finally:os.close(parent)
    os.set_inheritable(fd,False)
    if not stat.S_ISREG(os.fstat(fd).st_mode):os.close(fd);raise ValueError('regular metadata input required')
    return fd


def _file_limit(path):
    if path.startswith('configurations/') or path.startswith('budgets/'):return 65536
    if path.startswith('responses/') or path=='runtime/assets.json':return 1024**2
    if path in ('runtime/source-bundles.json','runtime/financial-adapters.json','runtime/charge-scenarios.json') or path.endswith('/execution.json'):return 65536
    if path in ('runtime/policy.json','runtime/latency.json','runtime/scopes.json','runtime/instrumentation.json'):return 65536
    return MAX_METADATA_BYTES


def _lexical(data):
    """Bound JSON structural work BEFORE constructing Python JSON containers."""
    depth=nodes=0;string=False;escape=False;atom=False
    for byte in data:
        if byte>127:raise ValueError('canonical metadata must be ASCII')
        if string:
            if escape:escape=False
            elif byte==92:escape=True
            elif byte==34:string=False
            elif byte<32:raise ValueError('unescaped JSON control byte')
            continue
        if byte==34:
            string=True;nodes+=1;atom=False
        elif byte in (91,123):
            depth+=1;nodes+=1;atom=False
            if depth>32:raise ValueError('metadata lexical depth bound exceeded')
        elif byte in (93,125):
            depth-=1;atom=False
            if depth<0:raise ValueError('invalid JSON container order')
        elif byte in (32,9,10,13,44,58):atom=False
        elif not atom:nodes+=1;atom=True
        if nodes>MAX_TREE_NODES:raise ValueError('metadata lexical node bound exceeded')
    if string or escape or depth:raise ValueError('incomplete metadata JSON')


def _decode(data):
    _lexical(data)
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('duplicate metadata key')
            result[key]=value
        return result
    def number(token):
        # K2 rational numerators/denominators may use256 bits. Exact field
        # validators still enforce N63 for all ordinary counters and clocks.
        if len(token.lstrip('-'))>78:raise ValueError('metadata integer exceeds256 bits')
        value=int(token)
        if not -(2**256-1)<=value<=2**256-1:raise ValueError('metadata integer exceeds256 bits')
        return value
    def constant(_):raise ValueError('nonfinite metadata value')
    try:value=json.loads(data,object_pairs_hook=pairs,parse_int=number,parse_constant=constant)
    except (UnicodeError,RecursionError) as error:raise ValueError('bounded canonical JSON required') from error
    if encoded(value)+b'\n'!=data:raise ValueError('exact canonical ASCII JSON plus one LF required')
    return value


def _scan(root):
    files={};payload={};directories=0
    def visit(fd,prefix):
        nonlocal directories
        directories+=1
        if directories>512:raise ValueError('bounded fixed input directory inventory required')
        with os.scandir(fd) as scan:
            entries=[]
            for entry in scan:
                if len(entries)>=1024:raise ValueError('bounded fixed input directory inventory required')
                entries.append(entry)
        for entry in entries:
            path=prefix+entry.name;info=entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(info.st_mode):raise ValueError('input symlinks forbidden')
            if stat.S_ISREG(info.st_mode):
                if _JSON_PATH.fullmatch(path) is None:raise ValueError('unlisted input metadata path')
                if len(files)>=MAX_FILES or not 1<=info.st_size<=_file_limit(path):raise ValueError('metadata per-file bound exceeded')
                files[path]=_identity(info)
            elif stat.S_ISDIR(info.st_mode):
                allowed=(path in ('runtime','recordings','invocations','configurations','responses','budgets') or
                    re.fullmatch(r'(?:recordings|invocations)/[0-9]{2}',path) or
                    re.fullmatch(r'recordings/[0-9]{2}/payload',path))
                if not allowed:raise ValueError('unlisted input directory')
                child=_directory(fd,entry.name)
                try:
                    if path.endswith('/payload'):
                        payload[path]=os.dup(child);os.set_inheritable(payload[path],False)
                    else:visit(child,path+'/')
                finally:os.close(child)
            else:raise ValueError('only regular input metadata and directories allowed')
    try:visit(root,'');return files,payload
    except BaseException:
        for fd in payload.values():os.close(fd)
        raise


def _read(root,path,identity):
    fd=_open_file(root,path)
    try:
        if _identity(os.fstat(fd))!=identity:raise ValueError('input changed after metadata admission')
        size=identity[3];data=bytearray()
        while len(data)<=size:
            part=os.read(fd,min(65536,size+1-len(data)))
            if not part:break
            data.extend(part)
        if len(data)!=size or _identity(os.fstat(fd))!=identity:raise ValueError('input changed while reading')
        return bytes(data)
    finally:os.close(fd)


@dataclass(frozen=True,slots=True,init=False)
class RuntimeInputReceipt:
    """Process-local original-input identity; not model or financial authority."""
    _owner: object
    def __init__(self,owner,*,_issuer=None):
        if _issuer is not _ISSUER:raise TypeError('owned loader must issue input receipt')
        object.__setattr__(self,'_owner',owner)
    def documents(self):return self._owner._snapshot()
    def source_bundles(self):return self._owner._bundles()
    def declarations(self):return self._owner._declarations()
    @property
    def identities(self):
        d=self.documents();execution=d[self._owner._execution_path]
        return dict(execution_sha256=digest(execution),runtime_policy_sha256=digest(d['runtime/policy.json']),
            collection_sha256=digest(d['runtime/collection.json']),scope_protocol_sha256=digest(d['runtime/scopes.json']),
            instrumentation_protocol_sha256=digest(d['runtime/instrumentation.json']),
            latency_policy_sha256=digest(d['runtime/latency.json']),asset_inventory_sha256=digest(d['runtime/assets.json']),
            input_index_sha256=digest(d['runtime/source-bundles.json']))


class OwnedRuntimeInputs:
    """Own metadata snapshots, original payload roots and an actual read thread."""
    def __init__(self):raise TypeError('use async open for owned inputs')

    @classmethod
    async def open(cls,root,*,expected_execution_sha256,invocation_ordinal):
        sha(expected_execution_sha256);integer(invocation_ordinal,0,95)
        self=object.__new__(cls);self._root=None;self._payload={};self._raw=None;self._files={}
        self._closed=False;self._close_task=None;self.workspace_bytes=0;self.metadata_bytes=0
        self._execution_path=f'invocations/{invocation_ordinal:02}/execution.json'
        task=asyncio.create_task(asyncio.to_thread(self._load,root,expected_execution_sha256,invocation_ordinal))
        self._read_task=task
        try:
            await asyncio.shield(task)
            self.receipt=RuntimeInputReceipt(self,_issuer=_ISSUER)
            return self
        except BaseException:
            await _finish(asyncio.gather(task,return_exceptions=True));await self.close();raise

    def _load(self,root,expected_execution_sha256,ordinal):
        self._root=_absolute_directory(root);self._root_path=root
        mode=stat.S_IMODE(os.fstat(self._root).st_mode)
        if mode!=0o700:raise ValueError('private runtime input root must be mode0700')
        files,payload=_scan(self._root);self._payload=payload;self._files=files
        total=sum(info[3] for info in files.values());workspace=128*total
        if total>MAX_METADATA_BYTES or workspace>MAX_WORKSPACE_BYTES:raise ValueError('joint metadata/workspace admission exceeded')
        self.metadata_bytes=total;self.workspace_bytes=workspace
        # No JSON conversion occurs before every file's size and the complete
        # joint parse reservation are known. Hold the reservation until close.
        raw={path:_read(self._root,path,identity) for path,identity in files.items()}
        documents={path:_decode(data) for path,data in raw.items()}
        self._validate(documents,raw,ordinal,expected_execution_sha256)
        for path,identity in files.items():
            fd=_open_file(self._root,path)
            try:
                if _identity(os.fstat(fd))!=identity:raise ValueError('input mutated during validation')
            finally:os.close(fd)
        self._raw=raw

    def _validate(self,docs,raw,ordinal,expected_execution_sha256):
        if not set(_BASE)<=set(docs) or self._execution_path not in docs:raise ValueError('complete fixed runtime inputs required')
        policy=docs['runtime/policy.json'];collection=docs['runtime/collection.json'];plan=docs['runtime/plan.json']
        collection,plan=validate_collection(collection,plan)
        index=docs['runtime/source-bundles.json'];keys(index,('schema_version','kind','collection_sha256','runtime_policy_sha256','recordings'))
        if type(index['schema_version']) is not int or index['schema_version']!=1 or index['kind']!='recording_runtime_input_sources':raise ValueError('fixed source input index required')
        if index['collection_sha256']!=digest(collection) or index['runtime_policy_sha256']!=digest(policy):raise ValueError('source index original identity mismatch')
        if type(index['recordings']) is not list or len(index['recordings'])!=len(collection['recordings']):raise ValueError('one original source index row per recording required')
        bundles={};expected=set(_BASE)
        for n,(row,original) in enumerate(zip(index['recordings'],collection['recordings'])):
            keys(row,('recording_id','recording_index','manifest','tasks','profiles'))
            if row['recording_id']!=original['recording_id'] or type(row['recording_index']) is not int or row['recording_index']!=n:raise ValueError('original recording index/order required')
            bundle={}
            for kind in ('manifest','tasks','profiles'):
                path=f'recordings/{n:02}/{kind}.json';expected.add(path);ref=row[kind]
                keys(ref,('path','file_sha256','bytes'))
                if path not in raw or ref!=dict(path=path,file_sha256=hashlib.sha256(raw[path]).hexdigest(),bytes=len(raw[path])):raise ValueError('source input original file hash/size/path mismatch')
                field='source_manifest_sha256' if kind=='manifest' else kind+'_sha256'
                if digest(docs[path])!=original[field]:raise ValueError('source document differs from original collection')
                bundle[kind]=docs[path]
            if f'recordings/{n:02}/payload' not in self._payload:raise ValueError('separately owned original payload root required')
            bundles[row['recording_id']]=bundle
        expected_payload={f'recordings/{n:02}/payload' for n in range(len(collection['recordings']))}
        if set(self._payload)!=expected_payload:raise ValueError('unbound payload root refused')
        validated=validate_runtime_policy(policy,comparison_plan=plan,latency_policy=docs['runtime/latency.json'],source_bundles=bundles)
        if validated.sha256!=collection['runtime_policy_sha256']:raise ValueError('collection runtime hash mismatch')
        validate_assets(docs['runtime/assets.json'])
        namespaces=set();caches=set()
        for inv in collection['invocations']:
            path=f"invocations/{inv['ordinal']:02}/execution.json";expected.add(path)
            if path not in docs:raise ValueError('every original invocation execution required')
            execution=docs[path]
            integer(execution.get('invocation_ordinal'),0,95);integer(execution.get('recording_index'),0,31)
            keys(execution,('schema_version','kind','runtime_policy_sha256','collection_sha256','scope_protocol_sha256','instrumentation_protocol_sha256','invocation_ordinal','recording_index','arm','stream_id','execution_namespace','cache_namespace','asset_inventory_sha256'))
            if (type(execution['schema_version']) is not int or execution['schema_version']!=1 or execution['kind']!='recording_execution' or
                execution['runtime_policy_sha256']!=digest(policy) or execution['collection_sha256']!=digest(collection) or
                execution['scope_protocol_sha256']!=digest(docs['runtime/scopes.json']) or execution['instrumentation_protocol_sha256']!=digest(docs['runtime/instrumentation.json']) or
                execution['asset_inventory_sha256']!=digest(docs['runtime/assets.json']) or any(execution[k]!=inv[k] for k in ('recording_index','arm','stream_id'))):
                raise ValueError('execution differs from original declarations')
            if execution['invocation_ordinal']!=inv['ordinal']:raise ValueError('original invocation ordinal required')
            from .recording_native_schema import identifier
            identifier(execution['execution_namespace']);identifier(execution['cache_namespace'])
            if execution['execution_namespace'] in namespaces or execution['cache_namespace'] in caches:raise ValueError('distinct owned invocation namespaces required')
            namespaces.add(execution['execution_namespace']);caches.add(execution['cache_namespace'])
        if digest(docs[self._execution_path])!=expected_execution_sha256:raise ValueError('coordinator execution hash mismatch')
        if policy['schema_version']==2:
            if not set(_NUMERIC)<=set(docs):raise ValueError('prospective numeric original inputs required')
            expected.update(_NUMERIC)
        elif any(p in docs for p in _NUMERIC):raise ValueError('structural input cannot contain numeric retrofit')
        expected.update(self._metadata_assets(docs,policy))
        if policy['schema_version']==2:
            from .recording_runtime_finance import validate_runtime_charge_scenarios
            validate_runtime_charge_scenarios(docs['runtime/charge-scenarios.json'],runtime_policy=policy,collection=collection,
                responses=[v for p,v in docs.items() if p.startswith('responses/')],
                accounting_policy=docs['runtime/accounting.json'],financial_adapter_inventory=docs['runtime/financial-adapters.json'])
        if set(docs)!=expected:raise ValueError('unreferenced or missing metadata input')
        self._recording_index=collection['invocations'][ordinal]['recording_index']
        for n,original in enumerate(collection['recordings']):
            self._validate_payload_tree(self._payload[f'recordings/{n:02}/payload'],bundles[original['recording_id']]['manifest'])
        from .recording_runtime_native import validate_runtime_scope_protocol
        numeric=None
        if policy['schema_version']==2:
            numeric=dict(opportunities=docs['runtime/opportunities.json'],exposure_policy=docs['runtime/exposure.json'],financial_adapter_inventory=docs['runtime/financial-adapters.json'])
        validate_runtime_scope_protocol(self._make_declarations(docs),latency_policy=docs['runtime/latency.json'],source_bundles=bundles,numeric_inputs=numeric)

    def _metadata_assets(self,docs,policy):
        # Exact finite content parsers are applied by their fixed runtime
        # constructors before activation; this loader first seals byte identity.
        required={c['budget_sha256'] for a in policy['arms'] for c in a['controllers']}
        budgets={p for p in docs if p.startswith('budgets/')}
        if len(budgets)!=len(required) or {digest(docs[p]) for p in budgets}!=required:raise ValueError('one original budget document per unique frozen hash required')
        from gamst.call_budget import CallBudget
        for path in budgets:
            keys(docs[path],('unit','pricing_revision','ceiling','max_calls','max_inflight','receipt_limit'))
            CallBudget(**docs[path])
        allowed=set(budgets);configs={};responses={};worker_hashes={};prompts={};asset_paths=set()
        for asset in docs['runtime/assets.json']['assets']:
            if asset['kind'] not in ('worker_config','provider_config','fixture_responses','system_template'):continue
            # Internal JSON assets use fixed enumerated relative names or exact
            # absolute names under this root; external executable/checkpoints
            # remain a separate asset preflight and cannot smuggle JSON inputs.
            path=asset['path']
            if path.startswith('/'):
                prefix=self._root_path+'/'
                if not path.startswith(prefix):raise ValueError('JSON metadata asset outside owned fixed input root')
                path=path[len(prefix):]
            if path not in docs or _JSON_PATH.fullmatch(path) is None or not path.startswith(('configurations/','responses/')):raise ValueError('fixed JSON asset input required')
            if path in asset_paths:raise ValueError('one original metadata asset per path required')
            asset_paths.add(path)
            data=encoded(docs[path])+b'\n'
            if (asset['bytes'],asset['sha256'])!=(len(data),hashlib.sha256(data).hexdigest()):raise ValueError('original JSON asset file identity mismatch')
            allowed.add(path)
            if asset['kind']=='provider_config':
                if digest(docs[path]) in configs:raise ValueError('duplicate original provider configuration')
                configs[digest(docs[path])]=docs[path]
            elif asset['kind']=='fixture_responses':
                from .recording_runtime_contract import validate_fixture_responses
                validate_fixture_responses(docs[path])
                if digest(docs[path]) in responses:raise ValueError('duplicate original fixture response inventory')
                responses[digest(docs[path])]=docs[path]
            elif asset['kind']=='system_template':
                from .recording_runtime_contract import validate_visual_prompt
                validate_visual_prompt(docs[path],digest(docs[path]));prompts[digest(docs[path])]=docs[path]
            else:
                from gamst.providers.smolvlm_native import configuration
                if configuration(docs[path])!=docs[path]:raise ValueError('explicit effective worker configuration required')
                worker_hashes[asset['sha256']]=docs[path]
        required_configs=set();required_responses=set();required_workers=set();required_prompts=set()
        for adapter in policy['adapters']:
            if adapter['implementation']=='fixture_completion_v1':
                required_responses.add(adapter['config']['responses_sha256'])
            elif adapter['implementation']=='smolvlm_process_v1':
                required_workers.add(adapter['config']['worker_config_sha256'])
        for arm in policy['arms']:
            for recording in arm['recording_policies']:
                if recording['periodic'] is not None:
                    prompt_hash=recording['periodic']['prompt_policy_sha256'];required_prompts.add(prompt_hash)
                    if recording['periodic']['scheduler']=='smolvlm_sentinel_v1':
                        from gamst.providers.smolvlm_native import PROMPT
                        if prompt_hash not in prompts or prompts[prompt_hash]['question']!=PROMPT:
                            raise ValueError('original fixed Smol prompt policy required before worker startup')
                for delivery in recording['deliveries']:
                    if delivery['profile']=='acquire_only':continue
                    h=delivery['config_sha256'];required_configs.add(h)
                    if h not in configs:raise ValueError('missing original finite provider configuration')
                    cfg=configs[h];profile=delivery['profile']
                    if profile=='fixture_bgr':keys(cfg,('delay_ms',));integer(cfg['delay_ms'],0,500)
                    elif profile in ('pcm_level','opencv_lk'):keys(cfg,())
                    elif profile=='opencv_fill':
                        from gamst.providers.fill import _configuration
                        if encoded(_configuration(**cfg))!=encoded(cfg):raise ValueError('explicit effective fill configuration required')
                fill=recording['fill_chain']
                if fill is not None:
                    h=fill['fill_config_sha256'];required_configs.add(h)
                    if h not in configs:raise ValueError('missing direct fill configuration')
                    from gamst.providers.fill import _configuration
                    if encoded(_configuration(**configs[h]))!=encoded(configs[h]):raise ValueError('explicit effective fill configuration required')
        if set(configs)!=required_configs or set(responses)!=required_responses or set(worker_hashes)!=required_workers or set(prompts)!=required_prompts:
            raise ValueError('missing or unreferenced configuration/response input')
        return allowed

    @staticmethod
    def _validate_payload_tree(root,manifest):
        # A trie keeps workspace linear in original manifest path bytes, even
        # when a supplied legal relative path has many nested components.
        trie={};terminal=object()
        for record in manifest['records']:
            node=trie
            for part in _parts(record['payload_path']):node=node.setdefault(part,{})
            node[terminal]=record['payload_bytes']
        stack=[(os.dup(root),trie)]
        try:
            while stack:
                fd,node=stack.pop()
                try:
                    with os.scandir(fd) as scan:
                        names=set()
                        for entry in scan:
                            if entry.name not in node or entry.name in names:raise ValueError('unlisted original payload path')
                            names.add(entry.name);expected=node[entry.name];info=entry.stat(follow_symlinks=False)
                            if terminal in expected:
                                if len(expected)!=1 or not stat.S_ISREG(info.st_mode) or info.st_size!=expected[terminal]:raise ValueError('original payload regular-file/size mismatch')
                            else:
                                if not stat.S_ISDIR(info.st_mode):raise ValueError('original payload directory required')
                                stack.append((_directory(fd,entry.name),expected))
                        if names!=set(node):raise ValueError('missing original payload path')
                finally:os.close(fd)
        finally:
            for fd,_ in stack:os.close(fd)

    def _snapshot(self):
        if self._closed or self._raw is None:raise RuntimeError('owned input receipt is closed')
        return {path:_decode(data) for path,data in self._raw.items()}
    def _bundles(self):
        docs=self._snapshot();return {row['recording_id']:dict(manifest=docs[row['manifest']['path']],tasks=docs[row['tasks']['path']],profiles=docs[row['profiles']['path']]) for row in docs['runtime/source-bundles.json']['recordings']}
    def _declarations(self):
        return self._make_declarations(self._snapshot())
    def _make_declarations(self,docs):
        n=self._recording_index
        result=dict(plan=docs['runtime/plan.json'],collection=docs['runtime/collection.json'],scope_protocol=docs['runtime/scopes.json'],instrumentation_protocol=docs['runtime/instrumentation.json'],runtime_policy=docs['runtime/policy.json'],source_manifest=docs[f'recordings/{n:02}/manifest.json'],tasks=docs[f'recordings/{n:02}/tasks.json'],profiles=docs[f'recordings/{n:02}/profiles.json'])
        if 'runtime/accounting.json' in docs:result['stream_protocol']=docs['runtime/accounting.json']
        return result
    def duplicate_payload_root(self,recording_id):
        if self._closed:raise RuntimeError('input root closed')
        index=self._snapshot()['runtime/source-bundles.json']['recordings']
        row=next((r for r in index if r['recording_id']==recording_id),None)
        if row is None:raise ValueError('unknown original recording')
        fd=os.dup(self._payload[f"recordings/{row['recording_index']:02}/payload"]);os.set_inheritable(fd,False);return fd

    def validate_native_paths(self,paths):
        """Bind channel bootstrap paths to the SAME retained original files."""
        if self._closed:raise RuntimeError('input root closed')
        mapping=dict(plan='runtime/plan.json',collection='runtime/collection.json',scope_protocol='runtime/scopes.json',instrumentation_protocol='runtime/instrumentation.json',runtime_policy='runtime/policy.json',source_manifest=f'recordings/{self._recording_index:02}/manifest.json',tasks=f'recordings/{self._recording_index:02}/tasks.json',profiles=f'recordings/{self._recording_index:02}/profiles.json')
        if 'runtime/accounting.json' in self._raw:mapping['stream_protocol']='runtime/accounting.json'
        if type(paths) is not dict or set(paths)!=set(mapping):raise ValueError('exact native original path inventory required')
        for name,relative in mapping.items():
            if paths[name]!=self._root_path+'/'+relative:raise ValueError('same original fixed native path required')
            parts=_parts(paths[name],absolute=True);parent=_absolute_directory('/'+'/'.join(parts[:-1]))
            try:
                fd=os.open(parts[-1],_FILE_FLAGS,dir_fd=parent);os.set_inheritable(fd,False)
                try:
                    if _identity(os.fstat(fd))!=self._files[relative]:raise ValueError('native input path is not same original file')
                    data=bytearray()
                    while len(data)<=len(self._raw[relative]):
                        part=os.read(fd,min(65536,len(self._raw[relative])+1-len(data)))
                        if not part:break
                        data.extend(part)
                    if bytes(data)!=self._raw[relative] or _identity(os.fstat(fd))!=self._files[relative]:raise ValueError('native input original bytes changed')
                finally:os.close(fd)
            finally:os.close(parent)

    async def _close(self):
        await _finish(asyncio.gather(self._read_task,return_exceptions=True))
        self._closed=True;self._raw=None
        for fd in self._payload.values():os.close(fd)
        self._payload.clear()
        if self._root is not None:os.close(self._root);self._root=None
        self._files.clear();self.workspace_bytes=0;self.metadata_bytes=0
    async def close(self):
        if self._close_task is None:self._close_task=asyncio.create_task(self._close())
        try:await asyncio.shield(self._close_task)
        except asyncio.CancelledError:await _finish(self._close_task);raise
    async def __aenter__(self):return self
    async def __aexit__(self,*_):await self.close()

"""Original-source PG graph/schema worker; never a trained inference provider.

The audited projection retains original graph methods and notices. Supplied
sources keep their Salesforce BSD / modeling_t5 Apache-2.0 licenses. No installed
LAVIS package, weights, conversion, generation or package acquisition is used.
"""
import argparse
import ast
import builtins
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import sys

from .pg_artifact import validate_manifest, read_manifest, _encoded, _json, _file
from .sam_audio_artifact import _keys, _digest, _absolute
from .pg_sources import SOURCE_HASHES

PROJECTION = 'pg-original-graph-v1'
PREFIX = 'lavis/models/'
CONFIGS = {
    'bert_config': ('86b5e0934494bd15c9632b12f734a8a67f723594',
                    '7160e1553ad2ca51d8c1cb066be533db31826e12d173824c1bb0cb1a4f187d20'),
    't5_config': ('ae7c9136adc7555eeccc78cdd960dfd60fb346ce',
                  'dd923bd687781f336e7845b7e011b7bbeca5fdde5c27f8b06071540340707dbd'),
}
# These are currently published config identities, not author-pinned PG assets.
FILES = {
    'eva': PREFIX+'eva_vit.py', 'base': PREFIX+'base_model.py',
    'blip': PREFIX+'blip2_models/blip2.py',
    'qformer': PREFIX+'blip2_models/Qformer.py',
    't5': PREFIX+'blip2_models/modeling_t5.py',
    'model': PREFIX+'blip2_models/blip2_t5_instruct.py',
}
METHODS = {
    'BaseModel': {'__init__', 'device'},
    'Blip2Base': {'init_tokenizer', 'maybe_autocast', 'init_Qformer', 'init_vision_encoder'},
    'Blip2T5Instruct': {'__init__'},
}
CLASSES = {
    'eva': {'DropPath','Mlp','Attention','Block','PatchEmbed','RelativePositionBias','VisionTransformer'},
    'base': {'BaseModel'}, 'blip': {'Blip2Base','LayerNorm'},
    'model': {'Blip2T5Instruct'},
    'qformer': {'BertEmbeddings','BertSelfAttention','BertSelfOutput','BertAttention',
                'BertIntermediate','BertOutput','BertLayer','BertEncoder','BertPooler',
                'BertPredictionHeadTransform','BertLMPredictionHead','BertOnlyMLMHead',
                'BertPreTrainedModel','BertModel','BertLMHeadModel','BertForMaskedLM'},
    't5': {'T5LayerNorm','T5DenseActDense','T5DenseGatedActDense','T5LayerFF','T5Attention',
           'T5LayerSelfAttention','T5LayerCrossAttention','T5Block','T5PreTrainedModel',
           'T5Stack','T5Model','T5ForConditionalGeneration','T5EncoderModel'},
}
FUNCTIONS = {'eva': {'convert_weights_to_fp16','create_eva_vit_g'},
             'blip': {'disabled_train'}, 't5': {'load_tf_weights_in_t5'}}
# Only upstream import statements in these exact modules can execute. Source
# hashes additionally freeze imported symbol lists and aliases.
IMPORTS = {
    'eva': {'math','functools','torch','torch.nn','torch.nn.functional',
            'torch.utils.checkpoint','timm.models.layers'},
    'base': {'torch','torch.nn'},
    'blip': {'contextlib','torch','torch.nn','transformers'},
    'model': {'logging','torch','torch.nn','transformers','transformers.modeling_outputs'},
    'qformer': {'math','os','warnings','dataclasses','typing','torch','torch.nn',
        'torch.utils.checkpoint','torch.nn.functional','transformers.activations',
        'transformers.file_utils','transformers.modeling_outputs',
        'transformers.modeling_utils','transformers.utils',
        'transformers.models.bert.configuration_bert'},
    't5': {'copy','math','os','warnings','typing','torch','torch.nn',
        'torch.nn.functional','torch.utils.checkpoint','transformers.activations',
        'transformers.modeling_outputs','transformers.modeling_utils',
        'transformers.pytorch_utils','transformers.utils',
        'transformers.utils.model_parallel_utils','transformers.models.t5.configuration_t5'},
}
ASSIGNS = {
    'qformer': {'logger'},
    't5': {'logger','_CONFIG_FOR_DOC','_TOKENIZER_FOR_DOC','_CHECKPOINT_FOR_DOC',
        'T5_PRETRAINED_MODEL_ARCHIVE_LIST','PARALLELIZE_DOCSTRING',
        'DEPARALLELIZE_DOCSTRING','T5_START_DOCSTRING','T5_INPUTS_DOCSTRING',
        'T5_ENCODER_INPUTS_DOCSTRING','__HEAD_MASK_WARNING_MSG'},
}


def project_sources(payloads):
    """Authenticate ALL input originals before returning any executable AST."""
    if set(payloads) != set(FILES):
        raise ValueError('exact original graph source set required')
    for key, raw in payloads.items():
        if type(raw) is not bytes or not 0 < len(raw) <= 1024**2:
            raise ValueError('bounded original source bytes required')
        if hashlib.sha256(raw).hexdigest() != SOURCE_HASHES['lavis_candidate'][FILES[key]]:
            raise ValueError('original source hash mismatch: '+key)
    result = {}
    for key, raw in payloads.items():
        tree = ast.parse(raw)
        selected = []
        found = set()
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = {a.name for a in node.names} if isinstance(node, ast.Import) else {node.module}
                if names <= IMPORTS[key]:
                    selected.append(node)
            elif isinstance(node, ast.ClassDef) and node.name in CLASSES[key]:
                found.add(node.name)
                if node.name in METHODS:
                    node.body = [n for n in node.body if
                        (isinstance(n, ast.FunctionDef) and n.name in METHODS[node.name]) or
                        (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value,str)) or
                        (node.name == 'Blip2T5Instruct' and isinstance(n,ast.Assign) and
                         isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'PRETRAINED_MODEL_CONFIG_DICT')]
                    node.decorator_list = []
                selected.append(node)
            elif isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS.get(key,set()):
                if node.name == 'create_eva_vit_g':
                    # Exact pinned architecture assignment + original fp16 cast.
                    if not (isinstance(node.body[0],ast.Assign) and
                            isinstance(node.body[-2],ast.If) and isinstance(node.body[-1],ast.Return)):
                        raise ValueError('unsupported EVA factory projection')
                    node.body = [node.body[0],node.body[-2],node.body[-1]]
                selected.append(node)
            elif isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ASSIGNS.get(key,set()):
                selected.append(node)
            elif isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):
                selected.append(node)  # original license/docstring
        if found != CLASSES[key]:
            raise ValueError('projected class inventory mismatch')
        result[key] = ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[]))
    return result


def _unavailable(reason, **extra):
    return dict(status='unavailable',reason=reason,projection=PROJECTION,
        native_ready=False,trained_state_loaded=False,runtime_compatibility_verified=False,
        reference_parity_verified=False,resource_admission='unavailable',
        actual_acceptance='incomplete',**extra)


def _runtime(manifest):
    mismatch=[]
    if platform.python_version()!=manifest['runtime']['python']:
        mismatch.append('python')
    for name,version in manifest['runtime']['packages'].items():
        try: actual=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: actual=None
        if actual!=version:mismatch.append(name)
    return mismatch


def _verify_runtime_lock(lock, expected_sha256):
    """Freeze the entire worker distribution metadata, including transitives.

    This is stronger than the small reference manifest's direct package list,
    but still is not package binary integrity or tested compatibility.
    """
    raw=_encoded(lock)
    if len(raw)>256*1024 or hashlib.sha256(raw).hexdigest()!=_digest(expected_sha256):
        raise ValueError('runtime lock bounds/digest mismatch')
    value=_json(raw,256*1024)
    _keys(value,('schema_version','python','platform','machine','packages'),'worker runtime lock')
    if type(value['schema_version']) is not int or value['schema_version']!=1:
        raise ValueError('unsupported worker runtime lock version')
    for key in ('python','platform','machine'):
        if type(value[key]) is not str or not 1<=len(value[key])<=128:
            raise ValueError('invalid runtime identity')
    packages=value['packages']
    if type(packages) is not dict or not 1<=len(packages)<=1024:
        raise ValueError('invalid worker distribution count')
    for name,version in packages.items():
        if type(name) is not str or not re.fullmatch('[a-z0-9]+(?:-[a-z0-9]+)*',name) or len(name)>128:
            raise ValueError('canonical distribution name required')
        if type(version) is not str or not 1<=len(version)<=128 or any(ord(c)<33 or ord(c)>126 for c in version):
            raise ValueError('bounded distribution version required')
    actual={}
    for distribution in importlib.metadata.distributions():
        name=distribution.metadata.get('Name')
        if type(name) is not str:raise ValueError('distribution without metadata identity')
        normalized=re.sub('[-_.]+','-',name).lower()
        if normalized in actual:raise ValueError('duplicate installed distribution identity')
        actual[normalized]=distribution.version
        if len(actual)>1024:raise ValueError('installed distribution count exceeds bound')
    return (value['python']==platform.python_version() and value['platform']==sys.platform
            and value['machine']==platform.machine() and packages==actual)


def _read_entry(root,item,limit):
    raw=_file(root/item['path'],item['sha256'],limit,retain=True)
    if len(raw)!=item['size']:raise ValueError('declared source/config size mismatch')
    return raw


def _tokenizer_dir(root,artifact):
    """Only explicitly declared, checked, immutable local tokenizer directories."""
    files=artifact['files']
    parents={str(Path(item['path']).parent) for item in files}
    if len(parents)!=1:raise ValueError('tokenizer files must share one directory')
    directory=root/next(iter(parents))
    declared={Path(item['path']).name for item in files}
    # No hidden config/tokenizer file outside the declared role can be consulted.
    if {p.name for p in directory.iterdir()} != declared:
        raise ValueError('tokenizer directory contains undeclared files')
    for item in files:_read_entry(root,item,32*1024**2)
    return directory


def _execute_projection(trees, bindings):
    namespaces={}
    def unsupported_clip(*args,**kwargs):
        raise ValueError('only the fixed EVA backbone is supported')
    real_import=builtins.__import__
    for key in ('eva','base','qformer','t5','blip','model'):
        allowed=IMPORTS[key]
        def bounded_import(name,globals=None,locals=None,fromlist=(),level=0,allowed=allowed):
            if level or name not in allowed:raise ImportError('outside PG source import closure: '+name)
            return real_import(name,globals,locals,fromlist,level)
        scope={'__name__':'_gamst_pg_'+key,'__builtins__':dict(vars(builtins),__import__=bounded_import)}
        if key=='blip':
            scope.update(BaseModel=namespaces['base']['BaseModel'],
                BertConfig=bindings['bert_config'],BertLMHeadModel=bindings['qformer_factory'](namespaces['qformer']),
                create_eva_vit_g=namespaces['eva']['create_eva_vit_g'],create_clip_vit_L=unsupported_clip)
        if key=='model':
            scope.update(Blip2Base=namespaces['blip']['Blip2Base'],disabled_train=namespaces['blip']['disabled_train'],
                T5Config=bindings['t5_config'],T5ForConditionalGeneration=bindings['t5_factory'](namespaces['t5']))
        exec(compile(trees[key],'verified/'+FILES[key],'exec'),scope)
        # Override original transformers tokenizer import only in this private
        # namespace. No host classmethod or global registry is patched.
        if key=='blip':scope['BertTokenizer']=bindings['bert_tokenizer']
        if key=='model':scope['T5TokenizerFast']=bindings['t5_tokenizer']
        namespaces[key]=scope
    return namespaces


def _inventory(model,torch):
    """Duplicate-name coverage and identity aliases, not deduped state counting."""
    parameters=dict(model.named_parameters(remove_duplicate=False))
    buffers=dict(model.named_buffers(remove_duplicate=False))
    persistent=set(model.state_dict())
    rows=[];groups={};unique=0;cpu=0;buffer_digests={}
    for kind,items in (('parameter',parameters),('buffer',buffers)):
        for name,tensor in items.items():
            if tensor.layout!=torch.strided:raise ValueError('unsupported graph layout')
            if kind=='parameter' and tensor.device.type!='meta':raise ValueError('materialized parameter in schema construction')
            shape=list(tensor.shape)
            if len(shape)>8 or tensor.numel()>2**35:raise ValueError('graph tensor outside schema bound')
            identity=id(tensor)
            alias=groups.get(identity)
            if alias is None:
                alias=name;groups[identity]=name
                unique+=tensor.numel()*tensor.element_size()
                cpu+=tensor.numel()*(4 if tensor.is_floating_point() else tensor.element_size())
            if kind=='buffer' and tensor.device.type!='meta':
                if tensor.numel()*tensor.element_size()>1024**2:raise ValueError('buffer exceeds schema allocation bound')
                if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
                    raise ValueError('nonfinite deterministic graph buffer')
                raw=tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                buffer_digests[name]=hashlib.sha256(raw).hexdigest()
            rows.append(dict(name=name,kind=kind,shape=shape,constructor_dtype=str(tensor.dtype),
                cpu_target_dtype='torch.float32' if tensor.is_floating_point() else str(tensor.dtype),
                cuda_target_dtype=str(tensor.dtype),alias=alias,persistent=name in persistent,
                is_meta=tensor.device.type=='meta'))
    if set(parameters)|set(buffers) != {row['name'] for row in rows}:raise ValueError('duplicate graph path')
    if persistent-set(parameters)-set(buffers):raise ValueError('unaccounted persistent state')
    if len(rows)>20000:raise ValueError('graph row bound exceeded')
    return dict(rows=rows,unique_constructor_bytes=unique,unique_cpu_target_bytes=cpu,
        buffer_sha256=buffer_digests,schema_sha256=hashlib.sha256(_encoded(rows)).hexdigest(),
        peak_memory_bytes=None,transient_construction_bytes=None)


def inspect_graph(manifest, *, expected_sha256, runtime_lock=None, runtime_lock_sha256=None):
    """Explicit worker-only schema construction; never returns a usable model."""
    value=validate_manifest(manifest,expected_sha256=expected_sha256)
    mismatch=_runtime(value)
    if mismatch:return _unavailable('declared_runtime_mismatch',mismatches=mismatch)
    if runtime_lock is None or runtime_lock_sha256 is None:
        return _unavailable('complete_worker_runtime_lock_required')
    if not _verify_runtime_lock(runtime_lock,runtime_lock_sha256):
        return _unavailable('complete_worker_runtime_lock_mismatch')
    root=Path(value['bundle_root'])
    sources=value['sources']['lavis_candidate']['files']
    # Authenticate the license and all graph bytes before any supplied exec.
    _read_entry(root,sources['LICENSE.txt'],1024**2)
    trees=project_sources({k:_read_entry(root,sources[p],1024**2) for k,p in FILES.items()})
    configs={}
    for role,(revision,digest) in CONFIGS.items():
        asset=value['artifacts'][role]
        if asset['revision']!=revision or len(asset['files'])!=1 or asset['files'][0]['sha256']!=digest:
            raise ValueError('unsupported fixed architecture config identity: '+role)
        configs[role]=_json(_read_entry(root,asset['files'][0],16384),16384)
    bert_dir=_tokenizer_dir(root,value['artifacts']['bert_tokenizer'])
    t5_dir=_tokenizer_dir(root,value['artifacts']['t5_tokenizer'])
    # Exact dependency metadata gate above occurs before all heavyweight imports.
    import torch
    from accelerate import init_empty_weights
    from transformers import BertConfig,BertTokenizer,T5TokenizerFast
    from transformers.models.t5.configuration_t5 import T5Config

    def config_factory(cls,role,identifier):
        class LocalConfig:
            @staticmethod
            def from_pretrained(name):
                if name!=identifier:raise ValueError('unexpected pretrained config name')
                return cls.from_dict(copy.deepcopy(configs[role]))
        return LocalConfig
    def tokenizer_factory(cls,directory,identifier):
        class LocalTokenizer:
            @staticmethod
            def from_pretrained(name,**kwargs):
                if name!=identifier:raise ValueError('unexpected pretrained tokenizer name')
                return cls.from_pretrained(str(directory),local_files_only=True,**kwargs)
        return LocalTokenizer
    def factory(symbol,identifier):
        def bind(namespace):
            class ArchitectureOnly:
                @staticmethod
                def from_pretrained(name,*,config):
                    if name!=identifier:raise ValueError('unexpected pretrained graph name')
                    return namespace[symbol](config)
            return ArchitectureOnly
        return bind
    bindings=dict(bert_config=config_factory(BertConfig,'bert_config','bert-base-uncased'),
        t5_config=config_factory(T5Config,'t5_config','google/flan-t5-xxl'),
        bert_tokenizer=tokenizer_factory(BertTokenizer,bert_dir,'bert-base-uncased'),
        t5_tokenizer=tokenizer_factory(T5TokenizerFast,t5_dir,'google/flan-t5-xxl'),
        qformer_factory=factory('BertLMHeadModel','bert-base-uncased'),
        t5_factory=factory('T5ForConditionalGeneration','google/flan-t5-xxl'))
    namespaces=_execute_projection(trees,bindings)
    # Parameter registration only: EVA's small linspace(...).item() stays real.
    # This may still incur transient allocations; no peak/admission claim.
    with init_empty_weights(include_buffers=False):
        model=namespaces['model']['Blip2T5Instruct'](t5_model='google/flan-t5-xxl',qformer_text_input=True)
    # Original graph-defined shared encoder/decoder embeddings; registration
    # interception can break identity. Reapply the original tying routine only.
    model.t5_model.tie_weights()
    shared=model.t5_model.shared.weight
    if model.t5_model.encoder.embed_tokens.weight is not shared or model.t5_model.decoder.embed_tokens.weight is not shared:
        raise ValueError('original T5 shared embeddings not restored')
    if model.t5_model.lm_head.weight is shared:
        raise ValueError('T5 untied language head unexpectedly tied')
    if model.Qformer.bert.embeddings.word_embeddings is None or model.Qformer.bert.embeddings.position_embeddings is None:
        raise ValueError('retained Qformer text graph absent')
    for layer in model.Qformer.bert.encoder.layer:
        if layer.output is None or layer.intermediate is None:raise ValueError('retained Qformer FFN absent')
    if len(model.tokenizer)!=30523:raise ValueError('unexpected BERT tokenizer vocabulary')
    schema=_inventory(model,torch)
    del model
    return _unavailable('schema_only_no_canonical_state',schema=schema,
        original_source_graph_constructed=True,construction_qformer_text_input=True,
        inference_qformer_text_input=False,graph_coverage_verified=False,
        config_provenance='published_configs_not_author_pinned',
        precision_policy='reference_constructor_cast_then_final_device_cast')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True);parser.add_argument('--sha256',required=True)
    parser.add_argument('--runtime-lock');parser.add_argument('--runtime-sha256')
    args=parser.parse_args()
    if bool(args.runtime_lock)!=bool(args.runtime_sha256):parser.error('runtime lock and digest required together')
    lock=_json(_file(_absolute(args.runtime_lock),None,256*1024,retain=True),256*1024) if args.runtime_lock else None
    result=inspect_graph(read_manifest(args.manifest,expected_sha256=args.sha256),expected_sha256=args.sha256,
                         runtime_lock=lock,runtime_lock_sha256=args.runtime_sha256)
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()

"""PG-InstructBLIP local inventory prerequisite, never a model loader/provider."""
import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import re

from .pg_sources import PG_REVISION, LAVIS_CANDIDATE, PG_BYTES, PG_SHA256, SOURCE_HASHES
# Reuse the already audited dependency-light, streaming regular-file reader.
from .sam_audio_artifact import (_absolute, _relative, _digest, _keys, _json, _encoded,
                                 _file, _regular_path, MANIFEST_LIMIT)

MAX_FILES=128
MAX_AGGREGATE=128*1024**3
MAX_WEIGHT=64*1024**3
MAX_SMALL=32*1024**2
ORIGINS={
    'pg_weights':'bidiptas/PG-InstructBLIP',
    'bert_config':'bert-base-uncased', 'bert_tokenizer':'bert-base-uncased', 'bert_weights':'bert-base-uncased',
    't5_config':'google/flan-t5-xxl', 't5_tokenizer':'google/flan-t5-xxl', 't5_weights':'google/flan-t5-xxl',
    'eva_weights':'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth',
    'instructblip_weights':'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_flanxxl_trimmed.pth',
}
PACKAGES=frozenset(('torch','torchvision','transformers','timm','fairscale','omegaconf',
                   'sentencepiece','accelerate','safetensors','numpy','pillow'))
CANDIDATE_PACKAGES={'transformers':'4.33.2','timm':'0.4.12','fairscale':'0.4.4'}


def _integer(value,low,high,label):
    if type(value) is not int or not low<=value<=high:
        raise ValueError('invalid '+label)


def _string(value,limit,label):
    if type(value) is not str or not 1<=len(value)<=limit or any(ord(c)<32 for c in value):
        raise ValueError('invalid '+label)


def manifest_digest(manifest):
    return hashlib.sha256(_encoded(manifest)).hexdigest()


def validate_manifest(manifest, *, expected_sha256):
    payload=_encoded(manifest)
    if hashlib.sha256(payload).hexdigest()!=_digest(expected_sha256):
        raise ValueError('frozen manifest digest mismatch')
    value=_json(payload,MANIFEST_LIMIT)
    _keys(value,('schema_version','bundle_root','sources','artifacts','runtime','execution'),'PG manifest')
    _integer(value['schema_version'],1,1,'schema_version')
    _absolute(value['bundle_root'])
    paths=set();aggregate=0

    def entry(item,limit):
        nonlocal aggregate
        _keys(item,('path','size','sha256'),'file')
        name=_relative(item['path']);_digest(item['sha256'])
        _integer(item['size'],1,limit,'file size')
        if name in paths:raise ValueError('duplicate file path across roles')
        paths.add(name);aggregate+=item['size']
        if len(paths)>MAX_FILES or aggregate>MAX_AGGREGATE:
            raise ValueError('aggregate file/count budget exceeded')

    _keys(value['sources'],SOURCE_HASHES,'source groups')
    for group,expected in SOURCE_HASHES.items():
        source=value['sources'][group]
        _keys(source,('revision','files'),'source group')
        if source['revision']!=({'pg':PG_REVISION,'lavis_candidate':LAVIS_CANDIDATE}[group]):
            raise ValueError('source revision mismatch')
        _keys(source['files'],expected,'known source inventory')
        for name,item in source['files'].items():
            entry(item,1024**2)
            if item['sha256']!=expected[name]:raise ValueError('pinned source digest mismatch')
    _keys(value['artifacts'],ORIGINS,'reference constructor asset roles')
    for role,artifact in value['artifacts'].items():
        _keys(artifact,('origin','revision','license','files'),'artifact role')
        if artifact['origin']!=ORIGINS[role]:raise ValueError('reference asset origin mismatch')
        revision=artifact['revision']
        if type(revision) is not str or (revision!='unresolved' and re.fullmatch('[0-9a-f]{40}',revision) is None):
            raise ValueError('explicit immutable revision or unresolved required')
        _string(artifact['license'],1024,'declared license')
        files=artifact['files']
        if type(files) is not list or not 1<=len(files)<=64:raise ValueError('bounded nonempty artifact inventory required')
        for item in files:entry(item,MAX_WEIGHT if role.endswith('_weights') else MAX_SMALL)
        if role=='pg_weights':
            if revision!=PG_REVISION or len(files)!=1 or files[0]['size']!=PG_BYTES or files[0]['sha256']!=PG_SHA256 or Path(files[0]['path']).name!='pgvlm_weights.bin':
                raise ValueError('exact published PG artifact identity required; no synthetic substitute')
    runtime=value['runtime'];_keys(runtime,('python','packages'),'runtime')
    if type(runtime['python']) is not str or not re.fullmatch('3\\.[0-9]{1,2}\\.[0-9]{1,3}',runtime['python']):
        raise ValueError('exact declared Python patch version required')
    _keys(runtime['packages'],PACKAGES,'runtime packages')
    for name,version in runtime['packages'].items():
        if type(version) is not str or not re.fullmatch('[0-9][a-zA-Z0-9.+_-]{0,63}',version):
            raise ValueError('exact package version required')
        if name in CANDIDATE_PACKAGES and version!=CANDIDATE_PACKAGES[name]:
            raise ValueError('historical candidate package declaration mismatch')
    execute=value['execution']
    _keys(execute,('construction_qformer_text_input','inference_qformer_text_input','device','precision',
        'batch_size','image_size','prompt_tokens','max_new_tokens','num_beams','num_captions',
        'use_nucleus_sampling','length_penalty','repetition_penalty'),'execution policy')
    if execute['construction_qformer_text_input'] is not True or execute['inference_qformer_text_input'] is not False:
        raise ValueError('retain constructed text modules; disable text only during inference')
    if (execute['device'],execute['precision']) not in (('cpu','float32'),('cuda','reference_mixed')):
        raise ValueError('explicit reference precision/device policy required')
    for key,low,high in (('batch_size',1,1),('image_size',224,224),('prompt_tokens',1,128),
                         ('max_new_tokens',1,32),('num_beams',2,5),('num_captions',1,3)):
        _integer(execute[key],low,high,key)
    if execute['num_captions']>execute['num_beams'] or execute['use_nucleus_sampling'] is not False:
        raise ValueError('bounded beam output policy required')
    for key,low,high in (('length_penalty',-5,5),('repetition_penalty',1,5)):
        # The first manifest profile uses explicit integer penalties only.
        _integer(execute[key],low,high,key)
    return value


def _inventory(value):
    result=[]
    for group,source in value['sources'].items():
        for name,item in source['files'].items():
            result.append(dict(item,role='source:'+group,identity_authority='pinned_source',source_name=name))
    for role,artifact in value['artifacts'].items():
        for item in artifact['files']:
            result.append(dict(item,role=role,identity_authority='publisher_metadata' if role=='pg_weights' else 'caller_declared'))
    return result


def inspect_manifest(manifest, *, expected_sha256, inspect_files=False):
    """Only named files; never unpickle tensors, construct a graph or import ML."""
    if type(inspect_files) is not bool:raise ValueError('explicit boolean file inspection switch required')
    value=validate_manifest(manifest,expected_sha256=expected_sha256)
    files=_inventory(value);rows=[]
    root=_absolute(value['bundle_root'])
    if inspect_files:
        try:_regular_path(root,directory=True)
        except FileNotFoundError:pass
    for item in files:
        row=dict(item,status='not_inspected')
        if inspect_files:
            path=root/item['path']
            try:
                size=_file(path,item['sha256'],item['size'])
            except FileNotFoundError:
                row['status']='missing'
            else:
                if size!=item['size']:raise ValueError('declared file size mismatch')
                row['status']='hash_verified_snapshot'
        rows.append(row)
    installed={}
    for name in sorted(PACKAGES):
        try:installed[name]=metadata.version(name)
        except metadata.PackageNotFoundError:installed[name]=None
    runtime=value['runtime']
    mismatches=[name for name in sorted(PACKAGES) if installed[name]!=runtime['packages'][name]]
    if platform.python_version()!=runtime['python']:mismatches.insert(0,'python')
    return dict(schema_version=1,manifest_sha256=expected_sha256,
        inspection='files' if inspect_files else 'manifest_only',
        source_revision=PG_REVISION,lavis_candidate_revision=LAVIS_CANDIDATE,
        lavis_candidate_author_runtime_verified=False,known_source_inventory_complete_for_execution=False,
        files=rows,declared_file_count=len(rows),declared_total_bytes=sum(r['size'] for r in rows),
        missing_paths=[r['path'] for r in rows if r['status']=='missing'],
        unresolved_revisions=[role for role,a in value['artifacts'].items() if a['revision']=='unresolved'],
        installed_runtime=dict(python=platform.python_version(),packages=installed),runtime_mismatches=mismatches,
        runtime_compatibility_verified=False,licenses_verified=False,
        native_ready=False,graph_coverage_verified=False,legacy_checkpoint_schema_verified=False,
        resource_admission='unavailable',actual_acceptance='incomplete',
        inspected_snapshot_only=True,requires_immutable_trusted_bundle=True,
        later_native_load_requires_reverification=True)


def read_manifest(path, *, expected_sha256):
    data=_file(_absolute(path),None,MANIFEST_LIMIT,retain=True)
    return validate_manifest(_json(data,MANIFEST_LIMIT),expected_sha256=expected_sha256)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--sha256',required=True,help='frozen canonical manifest SHA256')
    parser.add_argument('--inspect-files',action='store_true',help='explicitly stream every declared local file; can read over25GB')
    args=parser.parse_args()
    value=read_manifest(args.manifest,expected_sha256=args.sha256)
    print(json.dumps(inspect_manifest(value,expected_sha256=args.sha256,inspect_files=args.inspect_files),
                     sort_keys=True,allow_nan=False))


if __name__=='__main__':main()

"""Offline SAM-Audio file preflight; byte identity is not executed-model proof."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat

SOURCE_PINS = {
    'sam_audio': 'bb4c6999d2677c7402360e426afc01ddfad6dce0',
    'perception_models': 'e72b6810b1133e1c879f2cc965d276eb73803f1f',
    'dacvae': '414c20785fc3a28373073ea8ef7a1316eeeaca6e',
}
T5_REQUIRED = {'config.json', 'tokenizer_config.json', 'tokenizer.json', 'model.safetensors'}
T5_OPTIONAL = {'special_tokens_map.json', 'added_tokens.json', 'spiece.model'}
MANIFEST_LIMIT = 1024*1024
CONFIG_LIMIT = 128*1024


def _keys(value, names, label):
    if type(value) is not dict or set(value) != set(names):
        raise ValueError('exact '+label+' fields required')


def _digest(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('lowercase SHA256 required')
    return value


def _json(data, limit):
    if type(data) is not bytes or not 0 < len(data) <= limit:
        raise ValueError('JSON byte bound exceeded')
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError('duplicate JSON key')
            value[key] = item
        return value
    def reject(value):
        raise ValueError('nonfinite JSON number')
    try:
        value = json.loads(data.decode('utf-8'), object_pairs_hook=pairs, parse_constant=reject)
        encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    except (UnicodeError, RecursionError, OverflowError) as error:
        raise ValueError('invalid bounded JSON') from error
    if len(encoded) > limit:
        raise ValueError('canonical JSON byte bound exceeded')
    return value


def _encoded(value):
    try:
        data = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    except (TypeError, UnicodeError, RecursionError, OverflowError) as error:
        raise ValueError('JSON declaration required') from error
    if len(data) > MANIFEST_LIMIT:
        raise ValueError('manifest exceeds1MiB')
    return data


def manifest_digest(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _absolute(value):
    if type(value) is not str or not 1 <= len(value) <= 4096 or '\x00' in value:
        raise ValueError('bounded absolute local path required')
    path = Path(value)
    if not path.is_absolute() or '..' in path.parts or str(path) != value:
        raise ValueError('normalized absolute local path required')
    return path


def _relative(value):
    if type(value) is not str or not 1 <= len(value) <= 512 or '\\' in value or '\x00' in value:
        raise ValueError('bounded POSIX relative file name required')
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or str(path) != value or value == '.':
        raise ValueError('normalized relative file name required')
    return value


def _identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _regular_path(path, *, directory=False):
    # Reject symlinks in parents too; a resolved target alone hides substitutions.
    for parent in reversed((path, *path.parents)):
        info = parent.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError('symlinked asset paths are unsupported')
    info = path.lstat()
    if not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)):
        raise ValueError('regular local asset required')
    return info


def _file(path, expected, limit, *, retain=False, allow_empty=False):
    if expected is not None:
        _digest(expected)
    before = _regular_path(path)
    if not (0 if allow_empty else 1) <= before.st_size <= limit:
        raise ValueError('asset file size outside declared SDK bound')
    digest, blocks, count = hashlib.sha256(), [], 0
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) |
                         getattr(os, 'O_NONBLOCK', 0))
    with os.fdopen(descriptor, 'rb') as stream:
        opened = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened.st_mode) or _identity(opened) != _identity(before):
            raise ValueError('asset replaced before reading')
        while True:
            block = stream.read(min(1024*1024, limit-count+1))
            if not block:
                break
            count += len(block)
            if count > limit:
                raise ValueError('asset grew beyond its bound')
            digest.update(block)
            if retain:
                blocks.append(block)
        after = os.fstat(stream.fileno())
    if (_identity(after) != _identity(before) or count != before.st_size or
            _identity(path.lstat()) != _identity(before) or
            (expected is not None and digest.hexdigest() != expected)):
        raise ValueError('asset changed or SHA256 mismatch')
    return b''.join(blocks) if retain else count


def validate_manifest(value, *, expected_sha256):
    """Detach and validate declarations without opening assets or importing ML."""
    raw = _encoded(value)
    if hashlib.sha256(raw).hexdigest() != _digest(expected_sha256):
        raise ValueError('manifest differs from frozen digest')
    value = _json(raw, MANIFEST_LIMIT)
    _keys(value, ('schema_version','sources','runtime','sam','t5'), 'manifest')
    if type(value['schema_version']) is not int or value['schema_version'] != 1:
        raise ValueError('unsupported artifact manifest version')
    _keys(value['sources'], SOURCE_PINS, 'source groups')
    for name, source in value['sources'].items():
        _keys(source, ('revision','root','files'), 'source')
        if source['revision'] != SOURCE_PINS[name]:
            raise ValueError('unsupported source revision')
        _absolute(source['root'])
        files = source['files']
        if type(files) is not dict or not 1 <= len(files) <= 1024:
            raise ValueError('declare1..1024 source files per group')
        for path, digest in files.items():
            if not _relative(path).endswith('.py'):
                raise ValueError('source inventory must contain Python files')
            _digest(digest)
    runtime = value['runtime']
    _keys(runtime, ('python','packages'), 'runtime')
    if type(runtime['python']) is not str or re.fullmatch(r'3\.(?:1[1-9]|[2-9][0-9])\.[0-9]{1,3}', runtime['python']) is None:
        raise ValueError('exact Python>=3.11 patch version required')
    if type(runtime['packages']) is not dict or not 1 <= len(runtime['packages']) <= 128:
        raise ValueError('declare bounded exact runtime packages')
    for name, version in runtime['packages'].items():
        if (type(name) is not str or re.fullmatch('[a-z0-9][a-z0-9-]{0,63}', name) is None or
                type(version) is not str or re.fullmatch('[0-9][a-zA-Z0-9.+_-]{0,63}', version) is None):
            raise ValueError('normalized distribution name and exact version required')
    sam = value['sam']
    _keys(sam, ('config_path','config_sha256','checkpoint_path','checkpoint_sha256'), 'SAM assets')
    for kind in ('config','checkpoint'):
        _absolute(sam[kind+'_path']); _digest(sam[kind+'_sha256'])
    t5 = value['t5']
    _keys(t5, ('root','files'), 'T5 assets'); _absolute(t5['root'])
    if type(t5['files']) is not dict or not T5_REQUIRED <= set(t5['files']) <= T5_REQUIRED | T5_OPTIONAL:
        raise ValueError('supported complete local unsharded T5 bundle required')
    for digest in t5['files'].values():
        _digest(digest)
    return value


def read_manifest(path, *, expected_sha256):
    """Read a strict bounded JSON file and verify its frozen canonical digest."""
    data = _file(_absolute(path), None, MANIFEST_LIMIT, retain=True)
    return validate_manifest(_json(data, MANIFEST_LIMIT), expected_sha256=expected_sha256)


def verify_manifest(value, *, expected_sha256):
    """Hash explicit files; preserve config snapshots, not huge weight snapshots.

    Native loading must reverify the files it actually consumes, imported code
    origins/closure and tensor coverage. Declared source hashes are not proof
    of official origin or executed code. No runtime/model compatibility verdict.
    """
    value = validate_manifest(value, expected_sha256=expected_sha256)
    for source in value['sources'].values():
        root = _absolute(source['root']); _regular_path(root, directory=True)
        total = 0
        for name, digest in source['files'].items():
            total += _file(root/name, digest, 4*1024*1024, allow_empty=True)
            if total > 64*1024*1024:
                raise ValueError('source group exceeds64MiB')
    sam = value['sam']
    config_bytes = _file(_absolute(sam['config_path']), sam['config_sha256'], CONFIG_LIMIT, retain=True)
    original = _json(config_bytes, CONFIG_LIMIT)
    if type(original) is not dict:
        raise ValueError('SAM config must be an object')
    _file(_absolute(sam['checkpoint_path']), sam['checkpoint_sha256'], 16*1024**3)
    t5 = value['t5']; root = _absolute(t5['root']); _regular_path(root, directory=True)
    with os.scandir(root) as entries:
        actual = set()
        for entry in entries:
            if entry.name not in t5['files'] or len(actual) >= len(t5['files']):
                raise ValueError('undeclared T5 bundle file or directory')
            actual.add(entry.name)
    if actual != set(t5['files']):
        raise ValueError('missing T5 bundle files')
    snapshots = {}
    for name, digest in t5['files'].items():
        retain = name in ('config.json','tokenizer_config.json')
        limit = 8*1024**3 if name == 'model.safetensors' else 32*1024*1024
        if retain: limit = CONFIG_LIMIT
        data = _file(root/name, digest, limit, retain=retain)
        if retain: snapshots[name] = _json(data, CONFIG_LIMIT)
    if (type(snapshots['config.json']) is not dict or snapshots['config.json'].get('model_type') != 't5' or
            'auto_map' in snapshots['config.json']):
        raise ValueError('builtin T5 configuration without remote code required')
    tokenizer = snapshots['tokenizer_config.json']
    if (type(tokenizer) is not dict or 'auto_map' in tokenizer or
            tokenizer.get('tokenizer_class') not in ('T5Tokenizer','T5TokenizerFast')):
        raise ValueError('builtin local T5 tokenizer required')
    effective = deepcopy(original)
    if type(effective.get('audio_codec')) is not dict or type(effective.get('text_encoder')) is not dict:
        raise ValueError('explicit codec and text encoder configuration required')
    codec = effective['audio_codec']; rates = codec.get('encoder_rates')
    if (type(codec.get('sample_rate')) is not int or codec['sample_rate'] != 48000 or
            type(rates) is not list or not 1 <= len(rates) <= 16 or
            any(type(rate) is not int or not 1 <= rate <= 48000 for rate in rates)):
        raise ValueError('explicit bounded48k codec configuration required')
    hop = 1
    for rate in rates: hop *= rate
    maximum = effective['text_encoder'].get('max_length')
    if hop > 48000 or type(maximum) is not int or not 1 <= maximum <= 512:
        raise ValueError('unsupported codec hop or text token bound')
    effective.update(span_predictor=None, text_ranker=None, visual_ranker=None)
    effective['text_encoder']['name'] = str(root)
    return dict(manifest=value, manifest_sha256=expected_sha256, original_config=original,
                original_config_bytes=config_bytes, original_config_sha256=sam['config_sha256'],
                effective_config=effective, effective_config_sha256=manifest_digest(effective),
                runtime_manifest_sha256=manifest_digest(value['runtime']),
                t5_manifest_sha256=manifest_digest(t5), hop_length=hop,
                declared_file_bytes_verified=True, imported_source_verified=False,
                tensor_coverage_verified=False, runtime_compatibility_verified=False)


def verify_runtime(value, *, expected_sha256):
    """Compare only explicitly declared versions in the selected worker.

    No ML modules are imported. Distribution metadata does not attest installed
    binaries, transitive dependency coverage or compatibility with the model.
    """
    from importlib import metadata
    import platform
    value = validate_manifest(value, expected_sha256=expected_sha256)
    runtime = value['runtime']
    if platform.python_version() != runtime['python']:
        raise RuntimeError('selected worker Python differs from frozen runtime')
    for name, expected in runtime['packages'].items():
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError as error:
            raise RuntimeError('required declared runtime package missing: '+name) from error
        if actual != expected:
            raise RuntimeError('declared runtime package version mismatch: '+name)
    return dict(runtime_manifest_sha256=manifest_digest(runtime),
                declared_runtime_versions_verified=True, runtime_compatibility_verified=False)

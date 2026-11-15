"""Local-only Qwen single-safetensors bundle validation; no optional imports."""
import hashlib
import json
import os
from pathlib import Path
import re
import stat

REQUIRED = frozenset(('config.json', 'generation_config.json', 'preprocessor_config.json',
    'tokenizer_config.json', 'chat_template.json', 'merges.txt', 'vocab.json', 'model.safetensors'))
OPTIONAL = frozenset(('README.md', 'LICENSE', '.gitattributes'))


def artifact_version(files):
    if type(files) is not dict or not REQUIRED <= set(files) <= REQUIRED | OPTIONAL:
        raise ValueError('declare hashes for every required Qwen bundle file; only documented optional files are allowed')
    if any(type(digest) is not str or not re.fullmatch('[0-9a-f]{64}', digest) for digest in files.values()):
        raise ValueError('bundle file hashes must be lowercase SHA256')
    canonical = json.dumps(files, sort_keys=True, separators=(',', ':')).encode()
    return 'qwen-asr0.0.6.manifest.'+hashlib.sha256(canonical).hexdigest()


def verify_bundle(path, files):
    version = artifact_version(files)
    if type(path) is not str or not Path(path).expanduser().is_absolute():
        raise ValueError('supply an absolute local bundle directory, not a model ID or URL')
    root = Path(path).expanduser().resolve(strict=True)
    if not root.is_dir(): raise ValueError('model bundle must be a local directory')
    names = set()
    with os.scandir(root) as entries:
        for entry in entries:
            if len(names) >= 11 or entry.name not in files:
                raise ValueError('bundle contains undeclared files or directories')
            names.add(entry.name)
    if names != set(files): raise ValueError('model bundle files are missing')
    for name, expected in files.items():
        target = root/name
        info = target.lstat()
        limit = 8*1024**3 if name == 'model.safetensors' else 32*1024**2
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= limit:
            raise ValueError('bundle files must be nonempty regular files within size limits; symlinks are unsupported')
        digest = hashlib.sha256()
        with target.open('rb') as source:
            for block in iter(lambda: source.read(1024*1024), b''): digest.update(block)
        after = target.stat()
        identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
        if digest.hexdigest() != expected or identity(after) != identity(info):
            raise ValueError('bundle file changed or SHA256 mismatch')
    config_path = root/'config.json'
    if config_path.stat().st_size > 1024*1024: raise ValueError('model config exceeds 1 MiB')
    config = json.loads(config_path.read_text())
    if type(config) is not dict or config.get('model_type') != 'qwen3_asr' or config.get('architectures') != ['Qwen3ASRForConditionalGeneration']:
        raise ValueError('expected a Qwen3-ASR configuration')
    return str(root), version

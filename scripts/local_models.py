"""Project-owned model downloads. No global model cache or device access."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import ssl
import subprocess
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / '.local-models'
WHISPER = '536b0662742c02347bc0e980a01041f333bce120'
VOICE = '441a7599c3b22107314dcbd9166621c5c83f2cc5'
VOICE_FILES = {f'voice-emotion/{f}': f'https://huggingface.co/superb/wav2vec2-base-superb-er/resolve/{VOICE}/{f}' for f in ('config.json', 'preprocessor_config.json', 'pytorch_model.bin', 'README.md')}
POSE_FILES = {'pose/pose_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'}
QWEN = 'ea6a11058182570be6436b9a2e4ee7f7b49f908d'
FILES = {
    **POSE_FILES,
    'face_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
    'silero_vad.jit': 'https://raw.githubusercontent.com/snakers4/silero-vad/066409f3d9db45a30d21dfcbf9147826ffdd7052/src/silero_vad/data/silero_vad.jit',
    **{f'whisper-small/{f}': f'https://huggingface.co/Systran/faster-whisper-small/resolve/{WHISPER}/{f}' for f in ('config.json', 'model.bin', 'tokenizer.json', 'vocabulary.txt')},
    **{f'qwen3-vl/{f}': f'https://huggingface.co/ggml-org/Qwen3-VL-2B-Instruct-GGUF/resolve/{QWEN}/{f}' for f in ('Qwen3-VL-2B-Instruct-Q8_0.gguf', 'mmproj-Qwen3-VL-2B-Instruct-Q8_0.gguf')},
}
LLAMA_URL = 'https://github.com/ggml-org/llama.cpp/releases/download/b12318/llama-b12318-bin-macos-arm64.tar.gz'
# Expected (bytes, sha256). Cross-checked against publisher metadata: Hugging Face
# LFS oids / git blob ids, GitHub release digest and blob id, GCS md5 headers.
EXPECTED = {
    'face_landmarker.task': (3758596, '64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff'),
    'pose/pose_landmarker.task': (5777746, '59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a'),
    'silero_vad.jit': (2272526, 'e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720'),
    'whisper-small/config.json': (2370, 'b55496ac7940a7ae47d2c01eab40edfd8701feec1229d9cce3b40014383fb828'),
    'whisper-small/model.bin': (483546902, '3e305921506d8872816023e4c273e75d2419fb89b24da97b4fe7bce14170d671'),
    'whisper-small/tokenizer.json': (2203239, 'fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab'),
    'whisper-small/vocabulary.txt': (459861, '34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913'),
    'qwen3-vl/Qwen3-VL-2B-Instruct-Q8_0.gguf': (1834427296, 'b7802e29f71a9e5b5e3f83f613df898a2204342dcea71a231ea501d481813c39'),
    'qwen3-vl/mmproj-Qwen3-VL-2B-Instruct-Q8_0.gguf': (445053056, '69066c8f279ec85ff48ab4059f6ebba0d2932ca57667f2bbdac7d9805bca9e7b'),
    'llama.tar.gz': (16896103, 'c8de54b88616af7f76911608905ab4fd2e2343aefcdf997b4823f73ea7341f53'),
    'voice-emotion/config.json': (2153, '631bd282ad58492a69cf870c2917bae90ef963079ed2d6e0611eee9e40a19894'),
    'voice-emotion/preprocessor_config.json': (215, '99272fe8ccfab114b68b478681ea47ee3a1ce62bb788cb92dd6e4f69fb1f1da2'),
    'voice-emotion/pytorch_model.bin': (378361105, 'dac0f46128deec048751ce33eb1a9caecb570904d3d6cf732d6ec345cd6c2e1b'),
    'voice-emotion/README.md': (3499, '99985cae3f746d8d11e758cd2c5d8d2918b457f6e1117191003d27656ba4d29b'),
}

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def verify(name, path):
    """Raise unless the file has the pinned size and digest."""
    size, sha = EXPECTED[name]
    actual_size = path.stat().st_size
    if actual_size != size:
        raise SystemExit(f'{name}: expected {size} bytes, got {actual_size}')
    actual_sha = digest(path)
    if actual_sha != sha:
        raise SystemExit(f'{name}: sha256 mismatch (expected {sha}, got {actual_sha})')
    return actual_sha

def download(item):
    name, url = item
    path = BASE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f'Downloading {name}', flush=True)
        temporary = path.with_suffix(path.suffix + '.partial')
        try:
            import certifi
            context = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            context = ssl.create_default_context()
        try:
            with urllib.request.urlopen(url, timeout=120, context=context) as response, temporary.open('wb') as output:
                shutil.copyfileobj(response, output)
            # A truncated or tampered artifact must never be promoted into place.
            verify(name, temporary)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        temporary.replace(path)
    return name, {'url': url, 'bytes': path.stat().st_size, 'sha256': verify(name, path)}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['install', 'install-voice', 'install-pose', 'status', 'remove', 'remove-voice', 'remove-pose'])
    parser.add_argument('--yes', action='store_true', help='Confirm removal of this installer-owned model directory')
    args = parser.parse_args()
    if BASE.is_symlink() or BASE.resolve().parent != ROOT:
        raise SystemExit('Refusing a model directory outside this checkout')
    marker = BASE / 'owner.json'
    if args.action in ('install', 'install-voice', 'install-pose'):
        if BASE.exists() and not marker.exists():
            raise SystemExit('Existing model directory has no ownership marker; will not overwrite it')
        BASE.mkdir(exist_ok=True)
        marker.write_text(json.dumps({'project': str(ROOT), 'installer': 'scripts/local_models.py'}))
        files = dict(VOICE_FILES if args.action == "install-voice" else POSE_FILES if args.action == "install-pose" else FILES)
        if args.action == 'install' and platform.system() == 'Darwin' and platform.machine() == 'arm64':
            files['llama.tar.gz'] = LLAMA_URL
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            manifest = dict(pool.map(download, files.items()))
        if args.action == 'install' and (BASE / 'llama.tar.gz').exists():
            target = BASE / 'llama'
            target.mkdir(exist_ok=True)
            with tarfile.open(BASE / 'llama.tar.gz') as archive:
                # Python 3.10-compatible validation; symlinks must remain internal.
                for member in archive.getmembers():
                    destination = (target / member.name).resolve()
                    if not destination.is_relative_to(target.resolve()) or member.isdev():
                        raise ValueError('Unsafe archive member')
                    if member.issym() or member.islnk():
                        link = (destination.parent / member.linkname).resolve()
                        if not link.is_relative_to(target.resolve()):
                            raise ValueError('Unsafe archive link')
                archive.extractall(target)
        manifest_path = BASE / 'manifest.json'
        previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        previous.update(manifest)
        manifest_path.write_text(json.dumps(previous, indent=2))
        if args.action == 'install-voice':
            config = [{'provider':'huggingface','capacity':4,'options':{
                'path':str(BASE / 'voice-emotion'),'modality':'audio',
                'name':'classification.voice_emotion','revision':VOICE,
                'exclusive':True,'window_seconds':3,'hop_seconds':1}}]
            (BASE / 'voice-emotion/plugins.json').write_text(json.dumps(config,indent=2))
        print(f'Installed {len(manifest)} artifacts in {BASE}')
    elif args.action in ('remove', 'remove-voice', 'remove-pose'):
        if not marker.exists() or json.loads(marker.read_text()).get('project') != str(ROOT):
            raise SystemExit('No matching ownership marker')
        folder = {'remove-voice': 'voice-emotion', 'remove-pose': 'pose'}.get(args.action)
        removal = BASE / folder if folder else BASE
        if removal.is_symlink() or not removal.resolve().is_relative_to(BASE.resolve()):
            raise SystemExit('Refusing removal outside owned model directory')
        if not removal.exists():
            raise SystemExit('Selected models are not installed')
        # Source archives have no .git; still inspect status, then enforce the
        # same canonical-path and installer-ownership checks above.
        subprocess.run(['git', 'status', '--short'], cwd=ROOT, check=False)
        handles = subprocess.run(['lsof', '+D', str(removal)], capture_output=True, text=True)
        if handles.returncode == 0 or handles.stdout.strip():
            raise SystemExit('Model files are open. Stop the local stack before removal.')
        if handles.stderr.strip():
            raise SystemExit('Unable to verify open handles: ' + handles.stderr)
        if not args.yes:
            raise SystemExit(f'Would remove only {removal}. Re-run with --yes.')
        shutil.rmtree(removal)
        if folder:
            manifest_path = BASE / 'manifest.json'
            if manifest_path.exists():
                entries = json.loads(manifest_path.read_text())
                manifest_path.write_text(json.dumps({k:v for k,v in entries.items() if not k.startswith(folder + '/')},indent=2))
        print('Removed project model artifacts. Source, .venv and other projects are untouched.')
    else:
        manifest = BASE / 'manifest.json'
        if not manifest.exists():
            raise SystemExit('Models not installed')
        entries = json.loads(manifest.read_text())
        print(json.dumps({'directory': str(BASE), 'files': entries}, indent=2))

if __name__ == '__main__':
    main()

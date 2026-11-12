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

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

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
        with urllib.request.urlopen(url, timeout=120, context=context) as response, temporary.open('wb') as output:
            shutil.copyfileobj(response, output)
        temporary.replace(path)
    return name, {'url': url, 'bytes': path.stat().st_size, 'sha256': digest(path)}

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
            files['llama.tar.gz'] = 'https://github.com/ggml-org/llama.cpp/releases/download/b12318/llama-b12318-bin-macos-arm64.tar.gz'
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

#!/usr/bin/env python3
"""Explicitly install/remove only this checkout's isolated official SAM3 runtime."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / '.local-models'
TARGET = BASE / 'sam3'
OWNER = {'project': str(ROOT), 'installer': 'scripts/setup_sam3.py'}
MODEL = 'facebook/sam3'


def checked_target():
    if BASE.is_symlink() or TARGET.is_symlink() or not TARGET.resolve().is_relative_to(ROOT):
        raise RuntimeError('SAM3 installation must stay inside this checkout')
    global_owner = BASE / 'owner.json'
    if global_owner.exists() and json.loads(global_owner.read_text()).get('project') != str(ROOT):
        raise RuntimeError('Model directory belongs to another checkout')
    if TARGET.exists():
        marker = TARGET / 'owner.json'
        if not marker.is_file() or json.loads(marker.read_text()) != OWNER:
            raise RuntimeError('Existing SAM3 directory is not owned by this installer')
    return TARGET


def status():
    target = checked_target()
    marker = target / 'ready.json'
    if not marker.exists():
        return {'state': 'not_installed', 'model': MODEL, 'path': str(target)}
    config = json.loads(marker.read_text())
    complete = Path(config['python_executable']).is_file() and (Path(config['model_path']) / 'config.json').is_file()
    return {'state': 'download_ready' if complete else 'incomplete', 'model': MODEL,
            'inference_verified': False, 'revision': config['revision'], 'path': str(target)}


def install(device='cpu'):
    target = checked_target()
    # Check official gated access before creating directories or installing dependencies.
    from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url, snapshot_download, get_token
    token = get_token()
    try:
        revision = HfApi(token=token).model_info(MODEL).sha
        get_hf_file_metadata(hf_hub_url(MODEL, 'config.json', revision=revision), token=token)
    except Exception as exc:
        response = getattr(exc, 'response', None)
        reason = response.headers.get('x-error-message', '') if response is not None else ''
        if 'awaiting a review' in reason:
            raise RuntimeError('SAM3 account access request is pending review by the model authors. Local login is present; wait for approval at https://huggingface.co/facebook/sam3.') from None
        raise RuntimeError('Official SAM3 access check failed (' + type(exc).__name__ + '). '
            'Request access at https://huggingface.co/facebook/sam3 and run hf auth login locally, then retry.') from None
    target.mkdir(parents=True, exist_ok=True)
    (target / 'owner.json').write_text(json.dumps(OWNER, indent=2)+'\n')
    runtime = target / 'runtime'
    subprocess.run([sys.executable, '-m', 'venv', str(runtime)], check=True)
    python = runtime / 'bin' / 'python'
    subprocess.run([str(python), '-m', 'pip', 'install', '--no-cache-dir',
                    'transformers==5.20.2', 'torch==2.14.1', 'Pillow>=10,<13',
                    'opencv-python>=4.10,<5', 'numpy>=1.26,<3'], check=True)
    weights = target / 'weights'
    snapshot_download(MODEL, revision=revision, token=token, local_dir=str(weights),
                      allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.tiktoken'])
    subprocess.run([str(python), '-c',
                    'from transformers import Sam3Model, Sam3Processor; import torch, cv2, PIL'], check=True)
    config = dict(model_path=str(weights), python_executable=str(python), revision=revision,
                  prompt='person', interval=2.0, threshold=.5, device=device)
    pending = target / 'ready.json.tmp'
    pending.write_text(json.dumps(config, indent=2)+'\n')
    pending.replace(target / 'ready.json')
    return status()


def remove(confirmed=False):
    target = checked_target()
    if not target.exists():
        return {'state': 'not_installed'}
    if not confirmed:
        raise RuntimeError('Use remove --yes to remove this checkout\'s SAM3 weights and runtime')
    subprocess.run(['git', 'status', '--short'], cwd=ROOT, check=True)
    lsof = shutil.which('lsof')
    if not lsof:
        raise RuntimeError('Cannot verify open handles: lsof is required')
    handles = subprocess.run([lsof, '+D', str(target)], capture_output=True, text=True)
    if handles.returncode != 1 or handles.stdout.strip():
        raise RuntimeError('SAM3 directory may be in use. Stop its dashboard/worker before removal.')
    checked_target()
    shutil.rmtree(target)
    return {'state': 'removed', 'path': str(target)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['install', 'status', 'remove'])
    parser.add_argument('--yes', action='store_true')
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    args = parser.parse_args()
    try:
        result = install(args.device) if args.command == 'install' else remove(args.yes) if args.command == 'remove' else status()
        print(json.dumps(result, indent=2))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())

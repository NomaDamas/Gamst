"""Read-only host readiness report. No keys, transcripts or device capture."""
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys


def command(args, cwd):
    try:
        result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=15)
        # Only fixed diagnostics are called; avoid listing remotes or configs that
        # could contain credential-bearing URLs or unrelated personal settings.
        return {'exit_code': result.returncode, 'stdout': result.stdout.strip()[:2000],
                'stderr': result.stderr.strip()[:2000]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'error': type(exc).__name__}


def main():
    root = Path(__file__).resolve().parents[1]
    result = {'python': platform.python_version(), 'platform': platform.system(),
              'project': str(root), 'git_available': shutil.which('git') is not None,
              'optional_dependencies': {name: importlib.util.find_spec(name) is not None for name in
                  ('numpy','cv2','sounddevice','mediapipe','faster_whisper','torch','websockets')},
              'linux_video_devices': sorted(str(p) for p in Path('/dev').glob('video*')),
              'linux_audio_device_directory': Path('/dev/snd').exists(),
              'notice': 'No media captured, no key values read, no system configuration changed.'}
    probe = None
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.bind(('127.0.0.1', 0))
        probe.listen(1)
        result['loopback_bind'] = {'available': True}
    except OSError as exc:
        result['loopback_bind'] = {'available': False, 'error': type(exc).__name__, 'errno': exc.errno}
    finally:
        if probe is not None: probe.close()
    if shutil.which('git'):
        result['git_worktree'] = command(['git','rev-parse','--is-inside-work-tree'], root)
        result['github_access'] = command(['git','-c','credential.interactive=never','ls-remote',
            'https://github.com/NomaDamas/Gamst.git','HEAD'], root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

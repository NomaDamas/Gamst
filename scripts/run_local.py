"""Run isolated local perception, VLM, dashboard and agent stream. Ctrl-C stops all children."""
import os
import argparse
import json
import signal
import shlex
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--language', help='Pin speech recognition language, e.g. ko; default auto')
    parser.add_argument('--stop', action='store_true', help='Stop only the verified launcher belonging to this checkout')
    parser.add_argument('--plugins-config', help='Explicit JSON file for additional perception plugins')
    parser.add_argument('--without-voice-emotion', action='store_true', help='Skip an installed voice classifier')
    args = parser.parse_args()
    marker = ROOT / '.local-run/launcher.json'
    if args.stop:
        if not marker.exists():
            raise SystemExit('No launcher recorded for this checkout.')
        owner = json.loads(marker.read_text())
        if owner.get('root') != str(ROOT) or not owner.get('running'):
            raise SystemExit('No active launcher owned by this checkout.')
        pid = int(owner['pid'])
        command = subprocess.run(['ps', '-p', str(pid), '-o', 'command='], capture_output=True, text=True).stdout.strip()
        started = subprocess.run(['ps', '-p', str(pid), '-o', 'lstart='], capture_output=True, text=True).stdout.strip()
        executable = subprocess.run(['ps', '-p', str(pid), '-o', 'comm='], capture_output=True, text=True).stdout.strip()
        cwd = subprocess.run(['lsof', '-a', '-p', str(pid), '-d', 'cwd', '-Fn'], capture_output=True, text=True)
        if (command != owner.get('command') or started != owner.get('started')
                or len(shlex.split(command)) < 2
                or shlex.split(command)[1] not in ('scripts/run_local.py', str(Path(__file__).resolve()))
                or not Path(executable).name.lower().startswith('python')
                or 'n' + str(ROOT) not in cwd.stdout.splitlines()):
            raise SystemExit('Process identity no longer matches; refusing to signal it.')
        os.kill(pid, signal.SIGINT)
        print('Stop requested for this checkout. Wait for model handles to close before removal.')
        return
    models = ROOT / '.local-models'
    binaries = list((models / 'llama').glob('*/llama-server'))
    if not binaries:
        raise SystemExit('Run python scripts/local_models.py install first (bundled VLM server: Apple Silicon).')
    voice_config = models / 'voice-emotion/plugins.json'
    if not args.plugins_config and voice_config.exists() and not args.without_voice_emotion:
        args.plugins_config = str(voice_config)
    logs = ROOT / '.local-run'
    logs.mkdir(exist_ok=True)
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous.get('running'):
            running = subprocess.run(['ps', '-p', str(previous.get('pid', 0)), '-o', 'command='], capture_output=True, text=True).stdout.strip()
            if running and running == previous.get('command'):
                raise SystemExit('This checkout already has a running launcher. Use --stop first.')
    owner = {'root': str(ROOT), 'pid': os.getpid(), 'running': True,
             'command': subprocess.check_output(['ps', '-p', str(os.getpid()), '-o', 'command='], text=True).strip(),
             'started': subprocess.check_output(['ps', '-p', str(os.getpid()), '-o', 'lstart='], text=True).strip()}
    marker.write_text(json.dumps(owner))
    children, handles = [], []
    def start(args, name, env=None):
        handle = (logs / (name + '.log')).open('w')
        handles.append(handle)
        process = subprocess.Popen([str(x) for x in args], cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        children.append(process)
        return process
    try:
        vlm = start([binaries[0], '--model', models/'qwen3-vl/Qwen3-VL-2B-Instruct-Q8_0.gguf',
            '--mmproj', models/'qwen3-vl/mmproj-Qwen3-VL-2B-Instruct-Q8_0.gguf',
            '--host', '127.0.0.1', '--port', '18767', '--ctx-size', '16384', '--parallel', '2', '--alias', 'local-vlm'], 'vlm')
        for _ in range(120):
            if vlm.poll() is not None:
                raise RuntimeError('VLM exited; see .local-run/vlm.log')
            try:
                urllib.request.urlopen('http://127.0.0.1:18767/health', timeout=1).close()
                break
            except OSError:
                time.sleep(1)
        else:
            raise RuntimeError('VLM startup timed out')
        env = dict(os.environ, HUMANSENSE_COMPANION_URL='http://127.0.0.1:18767/v1/chat/completions',
                   HUMANSENSE_COMPANION_MODEL='local-vlm', MPLCONFIGDIR=str(logs/'matplotlib'))
        # Companion remains optional. No automatic reasoning/emotion classification.
        app = start([sys.executable, '-m', 'humansense.dashboard', '--face-model', models/'face_landmarker.task',
            '--asr-model', models/'whisper-small', '--vad-model', models/'silero_vad.jit',
            '--scene-url', 'http://127.0.0.1:18767/v1/chat/completions', '--scene-model', 'local-vlm',
            '--websocket-port', '8765', *(['--asr-language', args.language] if args.language else []),
            *(['--pose-model', str(models/'pose/pose_landmarker.task')] if (models/'pose/pose_landmarker.task').exists() else []),
            *(['--sam3-config', str(models/'sam3/ready.json')] if (models/'sam3/ready.json').is_file() else []),
            *(['--plugins-config', str(Path(args.plugins_config).resolve())] if args.plugins_config else [])], 'dashboard', env)
        print('Dashboard: http://127.0.0.1:8766\nLive context: ws://127.0.0.1:8765\nLocal VLM: http://127.0.0.1:18767\nDevices remain OFF until consent. Ctrl-C stops this stack.', flush=True)
        while all(p.poll() is None for p in children):
            time.sleep(1)
        raise RuntimeError('A service exited; see .local-run logs')
    except KeyboardInterrupt:
        print('\nStopping project services.')
    finally:
        for process in reversed(children):
            if process.poll() is None:
                process.terminate()
        for process in children:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for handle in handles:
            handle.close()
        owner['running'] = False
        marker.write_text(json.dumps(owner))

if __name__ == '__main__':
    main()

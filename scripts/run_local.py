"""Run isolated local perception, VLM, dashboard and agent stream. Ctrl-C stops all children."""
import os
import argparse
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--language', help='Pin speech recognition language, e.g. ko; default auto')
    args = parser.parse_args()
    models = ROOT / '.local-models'
    binaries = list((models / 'llama').glob('*/llama-server'))
    if not binaries:
        raise SystemExit('Run python scripts/local_models.py install first (bundled VLM server: Apple Silicon).')
    logs = ROOT / '.local-run'
    logs.mkdir(exist_ok=True)
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
            '--host', '127.0.0.1', '--port', '18767', '--ctx-size', '8192', '--parallel', '2', '--alias', 'local-vlm'], 'vlm')
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
            '--websocket-port', '8765', *(['--asr-language', args.language] if args.language else [])], 'dashboard', env)
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

if __name__ == '__main__':
    main()

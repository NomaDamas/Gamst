#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
python3 -m venv .venv
.venv/bin/python -m pip install --no-cache-dir -e '.[vision,audio,websocket,mediapipe,asr,vad]'
.venv/bin/python scripts/local_models.py install
printf '\nReady: .venv/bin/python scripts/run_local.py\nRemove models: .venv/bin/python scripts/local_models.py remove --yes\n'

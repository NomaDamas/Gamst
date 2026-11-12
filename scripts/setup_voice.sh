#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
if [ ! -x .venv/bin/python ]; then
  printf 'Run sh scripts/setup_local.sh first.\n' >&2
  exit 1
fi
.venv/bin/python -m pip install --no-cache-dir -e '.[huggingface]'
.venv/bin/python scripts/local_models.py install-voice
printf '\nVoice emotion is ready. Restart scripts/run_local.py when capture is stopped.\nSkip: --without-voice-emotion\nRemove: .venv/bin/python scripts/local_models.py remove-voice --yes\n'

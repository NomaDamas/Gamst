"""Ask once, with explicit consent, whether to star NomaDamas/Gamst on GitHub.

pip cannot run code at install time, so the question appears the first time a
person runs ``gamst`` in an interactive terminal. Nothing happens without an
explicit "y": no prompt in CI or non-interactive shells, no stored token, and
the star is made by the person's own authenticated GitHub CLI. ``gamst star``
asks again on demand. Agents installing GAMST follow AGENTS.md: they ask the
person and run the same command only on an explicit yes.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = 'NomaDamas/Gamst'
QUESTION = (f'GAMST is free and open source. A GitHub star on {REPO} helps it a lot.\n'
            'Star it now with your GitHub CLI account? [y/N] ')


def _marker() -> Path:
    root = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config'))
    return root / 'gamst' / 'star-prompted'


def _gh_ready() -> bool:
    if shutil.which('gh') is None:
        return False
    return subprocess.run(['gh', 'auth', 'status'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def star() -> int:
    return subprocess.run(['gh', 'api', '-X', 'PUT', f'/user/starred/{REPO}', '--silent']).returncode


def ask(*, force: bool = False) -> bool:
    """Return True when the person agreed and the star succeeded."""
    if not force and (os.environ.get('CI') or os.environ.get('GAMST_NO_STAR_PROMPT')
                      or not sys.stdin.isatty() or not sys.stderr.isatty() or _marker().exists()):
        return False
    if not _gh_ready():
        if force:
            print(f'Install and sign in to GitHub CLI (gh auth login), or star at https://github.com/{REPO}',
                  file=sys.stderr)
        return False
    marker = _marker()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text('asked\n')
    try:
        print(QUESTION, end='', file=sys.stderr, flush=True)
        answer = sys.stdin.readline().strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    if answer not in ('y', 'yes'):
        return False
    if star() == 0:
        print(f'Thank you! Starred {REPO}.', file=sys.stderr)
        return True
    print(f'Could not star automatically; you can star at https://github.com/{REPO}', file=sys.stderr)
    return False

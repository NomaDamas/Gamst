"""The message of an error, fit to be shown to the owner or an agent: one line, bounded, without secrets."""
from __future__ import annotations

import os
from pathlib import Path
import re

LIMIT = 300
_NAMES = re.compile(r'KEY|TOKEN|SECRET|PASSW|CREDENTIAL|AUTH', re.IGNORECASE)
_PATTERNS = (
    (re.compile(r'(://)[^/\s@]+@'), r'\1***@'),                                        # user:password of an address
    (re.compile(r'(://[^\s?\'"]+\?)[^\s\'"]+'), r'\1***'),                             # its query
    (re.compile(r'(?i)\b(bearer|basic)\s+[\w.~+/=-]{8,}'), r'\1 ***'),
    (re.compile(r'(?i)\b([\w-]*(?:api[_-]?key|token|password|passwd|secret|authorization)[\w-]*["\']?\s*[:=]\s*)'
                r'(?!bearer |basic )\S+'), r'\1***'),
    (re.compile(r'\b(?:sk|hf|ghp|gho|github_pat|xox[a-z]|AIza|AKIA)[-_A-Za-z0-9]{12,}'), '***'),
)


def public_text(text, *, limit=LIMIT, environ=None):
    """``text`` on one line: values of secret-named environment variables and usual forms of credentials become
    ``***``, the home directory ``~``, and what is beyond ``limit`` characters is cut."""
    text = ' '.join(''.join(char if char.isprintable() else ' ' for char in str(text)[:20_000]).split())
    for name, value in sorted((os.environ if environ is None else environ).items(), key=lambda item: -len(item[1])):
        if len(value) >= 6 and _NAMES.search(name):
            text = text.replace(value, '***')
    for pattern, shown in _PATTERNS:
        text = pattern.sub(shown, text)
    home = str(Path.home())
    if len(home) > 1:
        text = text.replace(home, '~')
    return text if len(text) <= limit else text[:limit - 3] + '...'


def public_error(error, *, limit=LIMIT):
    """``Type: message``, and the cause it was raised from."""
    parts = []
    while error is not None and len(parts) < 3:
        parts.append(f'{type(error).__name__}: {error}' if str(error) else type(error).__name__)
        error = error.__cause__
    return public_text(' (from ' .join(parts) + ')' * (len(parts) - 1), limit=limit)

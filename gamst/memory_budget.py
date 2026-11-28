"""How much system memory is free, for a run's memory budget (L2 replicas, video memory).

macOS: ``memory_pressure -Q`` ("System-wide memory free percentage", the number the local run guard watches) and
the machine's memory; Linux: ``MemAvailable`` of ``/proc/meminfo``. Anything else, or a failure: unknown (None), and
a caller that needs memory does not start what would take it.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys


def free_and_total():
    """(free bytes, total bytes), or (None, None) when unknown."""
    try:
        if sys.platform == 'darwin':
            text = subprocess.run(['memory_pressure', '-Q'], capture_output=True, text=True, timeout=5).stdout
            total = re.search(r'The system has (\d+)', text)
            share = re.search(r'free percentage: (\d+)%', text)
            if total and share:
                total = int(total.group(1))
                return total * int(share.group(1)) // 100, total
            return None, None
        if os.path.exists('/proc/meminfo'):
            values = {}
            with open('/proc/meminfo') as info:
                for line in info:
                    key, _, rest = line.partition(':')
                    values[key] = int(rest.split()[0]) * 1024
            if 'MemAvailable' in values and 'MemTotal' in values:
                return values['MemAvailable'], values['MemTotal']
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None, None


def free_pct():
    free, total = free_and_total()
    return None if free is None or not total else round(100 * free / total, 1)


def budget(*, replica_mb, min_free_pct, most):
    """How many replicas of ``replica_mb`` each can start now and leave ``min_free_pct`` of memory free (0..most)."""
    free, total = free_and_total()
    if free is None or not total:
        return 0
    spare = free - total * min_free_pct / 100
    return max(0, min(int(most), int(spare // (replica_mb * 1024 * 1024)) if replica_mb else int(most)))

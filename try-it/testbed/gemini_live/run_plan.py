"""Run every Gemini Live condition of the plan, twice, one at a time, under a USD cap on the measured spend.

    python try-it/testbed/gemini_live/run_plan.py --project P [--cap 20] [--dry]

Order: cheapest first. A condition already done (``runs/<name>/result.json`` without errors) is skipped. Before each
run: measured spend so far (``runs/ledger.jsonl``, from the API's usage metadata) + 2 x this run's estimate must stay
<= ``--cap``; otherwise it stops and says so.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness  # noqa: E402
import live_eval as le  # noqa: E402


def spent():
    ledger = harness.RUNS / 'ledger.jsonl'
    if not ledger.exists():
        return 0.0
    return sum(json.loads(line).get('usd') or 0 for line in ledger.read_text().splitlines() if line.strip())


def done(name):
    path = harness.RUNS / name / 'result.json'
    if not path.exists():
        return False
    return not json.loads(path.read_text()).get('errors')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project', required=True)
    ap.add_argument('--cap', type=float, default=20.0)
    ap.add_argument('--dry', action='store_true')
    args = ap.parse_args(argv)
    plan = sorted(harness.PLAN, key=lambda c: le.gemini_estimate(c[0], c[1])['usd'])
    for run in range(1, harness.RUNS_EACH + 1):
        for clip, mode, lang in plan:
            name = f'{clip}.{mode}.{lang}.r{run}'
            if done(name):
                print(f'skip {name} (done)', flush=True)
                continue
            est = le.gemini_estimate(clip, mode)['usd']
            total = spent()
            if total + 2 * est > args.cap:
                print(f'STOP before {name}: spent ${total:.3f} + 2 x estimate ${est:.3f} > cap ${args.cap}', flush=True)
                return 3
            print(f'run {name}: spent so far ${total:.3f}, estimate ${est:.3f}', flush=True)
            if args.dry:
                continue
            cmd = [sys.executable, str(Path(__file__).with_name('harness.py')), '--clip', clip, '--mode', mode,
                   '--lang', lang, '--run', str(run), '--project', args.project, '--go']
            proc = subprocess.run(cmd, capture_output=True, text=True)
            tail = proc.stdout[proc.stdout.find('{'):] if '{' in proc.stdout else proc.stdout
            try:
                summary = json.loads(tail)
                print(json.dumps(dict(name=name, wall_s=summary['wall_s'], spent=summary['spent']['usd'],
                                      errors=summary['errors']), ensure_ascii=False), flush=True)
            except ValueError:
                print(f'{name}: rc={proc.returncode} {proc.stdout[-500:]} {proc.stderr[-1500:]}', flush=True)
    print(f'all done; measured spend ${spent():.4f}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())

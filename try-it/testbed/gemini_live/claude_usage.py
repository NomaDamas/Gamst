"""Tokens, wall time and files read of one Claude Code subagent run, from its JSONL transcript.

    python try-it/testbed/gemini_live/claude_usage.py TRANSCRIPT.jsonl [--allowed DIR]

Sums the API usage of every assistant message (each message id once: input, cache write, cache read, output), takes
the wall time from the first to the last timestamp, lists the files the agent read, and flags any read outside
``--allowed`` (the arm's own folder) — a run that looked elsewhere is not a clean run. Prints one JSON object and the
final text (the answer) under ``answer``.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime


def _ts(value):
    try:
        return datetime.fromisoformat(str(value).replace('Z', '+00:00')).timestamp()
    except ValueError:
        return None


def summarize(path, allowed=None):
    usage = dict(input_tokens=0, cache_write_tokens=0, cache_read_tokens=0, output_tokens=0)
    seen, reads, other_tools, stamps, answer, model = {}, [], [], [], '', None
    tool_ids = set()
    for line in open(path, encoding='utf-8'):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        stamp = _ts(row.get('timestamp')) if row.get('timestamp') else None
        if stamp:
            stamps.append(stamp)
        message = row.get('message') if isinstance(row.get('message'), dict) else None
        if not message or message.get('role') != 'assistant':
            continue
        model = message.get('model') or model
        u = message.get('usage') or {}
        key = message.get('id') or id(message)
        seen[key] = dict(input_tokens=u.get('input_tokens') or 0,
                         cache_write_tokens=u.get('cache_creation_input_tokens') or 0,
                         cache_read_tokens=u.get('cache_read_input_tokens') or 0,
                         output_tokens=u.get('output_tokens') or 0)
        for block in message.get('content') or ():
            if not isinstance(block, dict):
                continue
            if block.get('type') == 'tool_use':
                if block.get('id') in tool_ids:
                    continue
                tool_ids.add(block.get('id'))
                args = block.get('input') or {}
                if block.get('name') == 'SubagentHandback':
                    answer = args.get('message') or answer
                elif block.get('name') == 'Read':
                    reads.append(args.get('file_path'))
                else:
                    other_tools.append(dict(name=block.get('name'), input=json.dumps(args, ensure_ascii=False)[:200]))
            elif block.get('type') == 'text' and block.get('text', '').strip():
                answer = block['text']
    for row in seen.values():
        for k in usage:
            usage[k] += row[k]
    outside = [p for p in reads if allowed and not str(p).startswith(allowed)]
    sheets = sorted({str(p).rsplit('/', 1)[-1] for p in reads if '/sheets/' in str(p)})
    return dict(model=model, api_calls=len(seen), **usage, sheets_read=len(sheets),
                wall_s=round(max(stamps) - min(stamps), 1) if stamps else None,
                reads=len(reads), reads_outside=outside, other_tools=other_tools, answer=answer)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('transcript')
    ap.add_argument('--allowed')
    args = ap.parse_args(argv)
    print(json.dumps(summarize(args.transcript, args.allowed), ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())

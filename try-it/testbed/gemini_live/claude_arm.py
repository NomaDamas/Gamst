"""The Claude arm: the same task for a Claude Code subagent, on time-stamped contact sheets, without any GAMST output.

    python try-it/testbed/gemini_live/claude_arm.py build --clip man --fps 4      # sheets + prompt, prints the estimate
    python try-it/testbed/gemini_live/claude_arm.py score --clip man --fps 4 --run 1 --answer A.txt \
        [--input-tokens N --output-tokens N --cache-read-tokens N --cache-write-tokens N --wall-s S]

``build`` writes, under ``<try-it>/testbed/gemini_live/claude/<clip>@<fps>/`` (outside the repository):
- ``photos/1..5.jpg``: the reference photos of 감스트 (the same five as the Gemini arm);
- ``sheets/sheet_NNN.jpg``: 4 x 4 frames, 384 px wide each, every tile labelled with its clip time;
- ``prompt.md``: the task (action text, answer format, the files to read). It holds no truth and no GAMST output.

A subagent (Agent tool, or ``claude -p``) is given ``prompt.md`` and answers with JSON lines; ``score`` parses that
answer with the same parser and scores it with the same scorer as the Gemini arm, and prices the reported tokens at
Anthropic's list price for the model.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_eval as le  # noqa: E402

OUT = le.TRYIT / 'testbed' / 'gemini_live' / 'claude'
TILE_W, COLS, ROWS = 384, 4, 4


def arm_dir(clip, fps):
    return OUT / f'{clip}@{fps:g}'


def extract(clip, fps, folder):
    info = le.CLIPS[clip]
    frames = folder / 'frames'
    if (frames / 'done').exists():
        return sorted(frames.glob('*.jpg'))
    shutil.rmtree(frames, ignore_errors=True)
    frames.mkdir(parents=True)
    vf = f'scale={TILE_W}:-2' if fps >= info['fps'] else f'fps={fps},scale={TILE_W}:-2'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(le.clip_path(clip)), '-vf', vf, '-q:v', '3',
                    str(frames / '%06d.jpg')], check=True)
    (frames / 'done').write_text('ok')
    return sorted(frames.glob('*.jpg'))


def build(clip, fps):
    from PIL import Image, ImageDraw
    info = le.CLIPS[clip]
    folder = arm_dir(clip, fps)
    files = extract(clip, fps, folder)
    rate = info['fps'] if fps >= info['fps'] else fps
    sheets = folder / 'sheets'
    shutil.rmtree(sheets, ignore_errors=True)
    sheets.mkdir(parents=True)
    per = COLS * ROWS
    first = Image.open(files[0])
    tw, th = first.size
    label_h = 18
    for s in range(math.ceil(len(files) / per)):
        sheet = Image.new('RGB', (COLS * tw, ROWS * (th + label_h)), 'black')
        draw = ImageDraw.Draw(sheet)
        for k, path in enumerate(files[s * per:(s + 1) * per]):
            i = s * per + k
            x, y = (k % COLS) * tw, (k // COLS) * (th + label_h)
            sheet.paste(Image.open(path), (x, y + label_h))
            draw.text((x + 4, y + 2), f't={i / rate:.2f}s', fill='yellow')
        sheet.save(sheets / f'sheet_{s:03d}.jpg', quality=85)
    photos = folder / 'photos'
    photos.mkdir(exist_ok=True)
    for k, src in enumerate(le.PHOTOS, 1):
        image = Image.open(src)
        image.thumbnail((768, 768))
        image.save(photos / f'{k}.jpg', quality=90)
    n_sheets = len(list(sheets.glob('*.jpg')))
    (folder / 'prompt.md').write_text(prompt(clip, fps, rate, folder, n_sheets))
    return dict(folder=str(folder), frames=len(files), sheets=n_sheets, sheet_size=sheet.size)


def prompt(clip, fps, rate, folder, n_sheets, lang='ko'):
    info = le.CLIPS[clip]
    action = info['action'][lang]
    point = info['kind'] == 'point'
    fmt = ('{"who":"감스트" 또는 "다른 사람","t":초}' if point else '{"who":"감스트" 또는 "다른 사람","start":초,"end":초}')
    return f"""# 과제: 영상에서 동작 찾기

영상 하나({info['duration_s']:.1f}초)에서 뽑은 프레임을 봅니다. 초당 {rate:g}장, 4x4 장표 {n_sheets}장입니다.
각 칸 위의 노란 글씨가 영상 시각(초)입니다. 시각은 이 표시를 기준으로 말합니다.

1. 먼저 `{folder}/photos/` 의 사진 5장을 봅니다. 이 사람이 '감스트'입니다. 영상에는 다른 사람도 나옵니다.
2. `{folder}/sheets/sheet_000.jpg` 부터 `sheet_{n_sheets - 1:03d}.jpg` 까지 모두 순서대로 봅니다(Read 도구).
3. 찾을 동작 — {action}
   {'동작의 시각은 부딪히는 순간(t)입니다.' if point else '각 동작의 시작(start)과 끝(end)을 초로 말합니다.'}

다른 파일, 다른 폴더, 저장소, 인터넷은 보지 않습니다. 정답 파일이나 다른 도구의 결과를 찾지 않습니다.

마지막 답은 동작 하나당 한 줄, JSON만 씁니다(감스트와 다른 사람 모두, 시간 순서): {fmt}
해당하는 것이 없으면 정확히 '없음'이라고만 씁니다. 답 앞뒤에 설명을 쓰지 않습니다.
"""


def score_answer(clip, fps, run, answer_text, usage):
    found, rejected = le.parse_answer(answer_text, source=f'claude.{clip}@{fps:g}.r{run}')
    found = le.dedupe_exact(found)
    p = le.CLAUDE_PRICE
    usd = None
    if usage.get('input_tokens') is not None or usage.get('total_tokens') is not None:
        usd = ((usage.get('input_tokens') or 0) * p['input'] + (usage.get('output_tokens') or 0) * p['output'] +
               (usage.get('cache_read_tokens') or 0) * p['cache_read'] +
               (usage.get('cache_write_tokens') or 0) * p['cache_write']) / 1e6
    info = le.CLIPS[clip]
    rate = info['fps'] if fps >= info['fps'] else fps
    return dict(arm='claude', model=le.CLAUDE_MODEL, clip=clip, fps=rate, run=run, items=found, rejected=rejected,
                score=le.score(clip, found), usage=usage, usd=None if usd is None else round(usd, 4),
                estimate=le.claude_estimate(clip, fps))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('action', choices=('build', 'score', 'estimate'))
    ap.add_argument('--clip', choices=sorted(le.CLIPS), required=True)
    ap.add_argument('--fps', type=float, required=True)
    ap.add_argument('--run', type=int, default=1)
    ap.add_argument('--answer')
    for name in ('input-tokens', 'output-tokens', 'cache-read-tokens', 'cache-write-tokens', 'total-tokens'):
        ap.add_argument(f'--{name}', type=int)
    ap.add_argument('--wall-s', type=float)
    args = ap.parse_args(argv)
    if args.action in ('build', 'estimate'):
        if args.action == 'build':
            print(json.dumps(build(args.clip, args.fps), ensure_ascii=False))
        print(json.dumps(le.claude_estimate(args.clip, args.fps), ensure_ascii=False))
        return 0
    usage = {k: getattr(args, k) for k in ('input_tokens', 'output_tokens', 'cache_read_tokens', 'cache_write_tokens',
                                           'total_tokens')}
    usage['wall_s'] = args.wall_s
    result = score_answer(args.clip, args.fps, args.run, Path(args.answer).read_text(), usage)
    out = arm_dir(args.clip, args.fps) / f'result.r{args.run}.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(json.dumps({k: result[k] for k in ('clip', 'fps', 'run', 'score', 'usd')}, ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())

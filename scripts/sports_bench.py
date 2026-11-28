"""Score a sports recipe run against a dataset's own labels (docs/sports.md, section 9).

    python scripts/sports_bench.py openttgames --video test_2.mp4 --markup test_2/ --start-s 10 --seconds 60 \\
        --stack sota --timing post --out runs/ott-sota-post [--fps 30]

What it does, for one stack and timing:

1. cuts ``--seconds`` from ``--start-s`` out of ``--video`` with ffmpeg (``--fps``: also resampled to that rate);
2. runs ``gamst recipe run racket-rally --sport table-tennis`` on the cut, with a Store file (``--max-records`` large
   enough to keep every ball record) and ``--report``;
3. reads the ball states (``sports.ball``) and contacts (``sports.contacts``) from the Store and scores them against the
   labels: ball positions per frame (TrackNet's measure: a prediction within 4 px of the label at 512 x 288 is a hit,
   scaled to the video's width), and bounce events (matched one to one within ``--tolerance-ms``);
4. writes ``metrics.json`` (and ``points.json``, the ball trail for ``python -m gamst.count_video --points``).

OpenTTGames (lab.osai.ai, CC BY-NC-SA 4.0): ``ball_markup.json`` {frame: {x, y}} (-1 where the ball is not visible) and
``events_markup.json`` {frame: bounce | net | empty_event}, frames of the original 120 fps video. Ball positions are
labelled only on frames near events; only those are scored. The test videos have no sound track (``--without onsets``). Labels and videos are
used for local evaluation only and never redistributed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time


def probe(path):
    """(fps, width, height, seconds) of a video, from ffprobe."""
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                          'stream=r_frame_rate,width,height:format=duration', '-of', 'json', str(path)],
                         check=True, capture_output=True, text=True).stdout
    data = json.loads(out)
    stream = data['streams'][0]
    num, den = (int(v) for v in stream['r_frame_rate'].split('/'))
    return num / den, int(stream['width']), int(stream['height']), float(data['format']['duration'])


def cut(video, first, count, step, fps, out):
    """Frames ``first``, ``first + step``, ... (``count`` of them) of ``video`` by their decode index, at ``fps``.

    Selected by frame number, not by time: ``-ss`` with ``-r`` picked frame 2k - 3 instead of 2k on OpenTTGames
    (measured by pixel comparison), which shifted every position 25 ms against the labels."""
    chain = f"select='between(n\\,{first}\\,{first + step * (count - 1)})*not(mod(n-{first}\\,{step}))',setpts=N/({fps}*TB)"
    # -vsync passthrough (ffmpeg 4.4; -fps_mode in 5+): no output frame-rate filter that would pick frames again
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(video), '-vf', chain, '-vsync', 'passthrough', '-an',
                    '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', str(out)], check=True)


def verify_cut(video, clip, first, step, probes=(0, 10, 40)):
    """Each probed clip frame must be closest, pixel by pixel, to the original frame it was cut from."""
    import cv2
    import numpy as np

    def read(path, indices):
        cap, found, index = cv2.VideoCapture(str(path)), {}, 0
        while len(found) < len(indices):
            ok, image = cap.read()
            if not ok:
                break
            if index in indices:
                found[index] = image.astype(np.int16)
            index += 1
        return found
    probes = [k for k in probes]
    clip_frames = read(clip, set(probes))
    wanted = {first + step * k + d for k in probes for d in (-2, -1, 0, 1, 2)}
    source = read(video, wanted)
    for k in probes:
        if k not in clip_frames:
            continue
        best = min((float(np.abs(clip_frames[k] - source[j]).mean()), j) for j in sorted(wanted) if j in source
                   and abs(j - (first + step * k)) <= 2)[1]
        if best != first + step * k:
            raise SystemExit(f'cut is misaligned: clip frame {k} is original frame {best}, not {first + step * k}')


def run_recipe(clip, out, *, stack, timing, sport, court=None, extra=()):
    argv = ['run', 'racket-rally', '--sport', sport, '--stack', stack, '--timing', timing, '--video', str(clip),
            '--headless', '--store', str(out / 'run.db'), '--max-records', '400000', '--report', str(out / 'report.json'),
            *(['--court', str(court)] if court else []), *extra]
    started = time.monotonic()
    with open(out / 'run.log', 'w') as log:
        code = subprocess.call([sys.executable, '-c', 'import sys; from gamst.recipes import main; main(sys.argv[1:])',
                                *argv], stdout=log, stderr=subprocess.STDOUT)
    return code, time.monotonic() - started


def stored(db, capability, events_only):
    from gamst.storage import Store
    store = Store.open_read_only(str(db))
    records, position = [], 0
    try:
        while True:
            page, _ = store.observation_page_rows(subject='local-user', after=position, limit=2000,
                                                  capability_id=capability, events_only=events_only)
            if not page:
                return records
            records += [record for _, record in page]
            position = page[-1][0]
    finally:
        store.close()


def ball_states(db):
    found = []
    for record in stored(db, 'sports.ball', False):
        value = record['observation']['value']
        found.append(value if isinstance(value, dict) else {})
    return found


def contact_events(db):
    return [event for record in stored(db, 'sports.contacts', True)
            for event in record['observation']['value'].get('events') or ()]


def match(truth_ms, found_ms, tolerance_ms):
    """One-to-one matching in time order: (matched pairs, unmatched truth, unmatched found)."""
    truth, found, pairs = sorted(truth_ms), sorted(found_ms), []
    used = set()
    for t in truth:
        best = min((abs(f - t), i) for i, f in enumerate(found) if i not in used) if len(used) < len(found) else None
        if best is not None and best[0] <= tolerance_ms:
            used.add(best[1])
            pairs.append((t, found[best[1]]))
    return pairs, len(truth) - len(pairs), len(found) - len(pairs)


def prf(tp, fn, fp):
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    return dict(tp=tp, fn=fn, fp=fp, precision=precision and round(precision, 4), recall=recall and round(recall, 4),
                f1=f1 and round(f1, 4))


def openttgames(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source_fps, width, height, _ = probe(args.video)
    fps = args.fps or source_fps
    step = max(1, round(source_fps / fps))
    first_frame, count = round(args.start_s * source_fps), round(args.seconds * source_fps / step)
    clip = out / 'clip.mp4'
    if not clip.exists():
        cut(args.video, first_frame, count, step, source_fps / step, clip)
        verify_cut(args.video, clip, first_frame, step)
    markup = Path(args.markup)
    balls = {int(k): v for k, v in json.loads((markup / 'ball_markup.json').read_text()).items()}
    events = {int(k): v for k, v in json.loads((markup / 'events_markup.json').read_text()).items()}
    first, last = round(args.start_s * source_fps), round((args.start_s + args.seconds) * source_fps)

    code, wall_s = run_recipe(clip, out, stack=args.stack, timing=args.timing, sport='table-tennis', court=args.court,
                              extra=args.extra or ())
    report = json.loads((out / 'report.json').read_text()) if (out / 'report.json').exists() else {}
    start_ms = report.get('start_ms', 0)

    def frame_of(ms):          # a GAMST capture time -> a frame of the original video
        return first + round((ms - start_ms) * source_fps / 1000)

    # ball positions, per labelled frame of the cut (at the cut's own rate)
    radius = 4 * width / 512
    states = ball_states(out / 'run.db')
    predicted = {}
    for state in states:
        if state.get('state') == 'tracking' and 'x' in state:
            predicted[frame_of(state['at_ms'])] = (state['x'] * width, state['y'] * height)
    tp = fn = fp = 0
    errors = []
    labelled = [frame for frame in range(first, last, step) if frame in balls]
    for frame in labelled:      # only labelled frames: elsewhere the truth is unknown (OpenTTGames labels near events)
        label = balls[frame]
        visible = label.get('x', -1) >= 0 and label.get('y', -1) >= 0
        guess = predicted.get(frame)
        if visible and guess is not None:
            distance = math.hypot(guess[0] - label['x'], guess[1] - label['y'])
            errors.append(distance)
            if distance <= radius:
                tp += 1
            else:
                fp += 1
                fn += 1
        elif visible:
            fn += 1
        elif guess is not None:
            fp += 1
    ball = dict(prf(tp, fn, fp), radius_px=round(radius, 1),
                median_error_px=round(sorted(errors)[len(errors) // 2], 1) if errors else None,
                labelled_frames=len(labelled), frames=len(range(first, last, step)))

    # bounces: the labelled bounce frames inside the cut against contact.bounce events
    truth_ms = [(frame - first) * 1000 / source_fps for frame, name in events.items() if name == 'bounce' and first <= frame < last]
    found = contact_events(out / 'run.db')
    bounce_ms = [event['timestamp_ms'] - start_ms for event in found if event.get('kind') == 'bounce']
    pairs, missed, extra = match(truth_ms, bounce_ms, args.tolerance_ms)
    bounces = dict(prf(len(pairs), missed, extra), tolerance_ms=args.tolerance_ms,
                   mean_abs_error_ms=round(sum(abs(f - t) for t, f in pairs) / len(pairs), 1) if pairs else None,
                   truth=len(truth_ms), found=len(bounce_ms))
    kinds = {}
    for event in found:
        kinds[event.get('kind')] = kinds.get(event.get('kind'), 0) + 1

    metrics = dict(dataset='openttgames', video=Path(args.video).name, start_s=args.start_s, seconds=args.seconds,
                   source_fps=round(source_fps, 3), fps=round(fps, 3), size=[width, height], stack=args.stack,
                   timing=args.timing, exit_code=code, wall_s=round(wall_s, 1), run_s=report.get('run_s'),
                   real_time_factor=round(report['run_s'] / args.seconds, 3) if report.get('run_s') else None,
                   frames=report.get('frames'), complete=report.get('complete'), lost=report.get('lost'),
                   model_choice=report.get('model_choice'), variant=report.get('variant'), ball=ball, bounces=bounces,
                   contacts_by_kind=kinds, errors=report.get('errors'))
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=1))
    # the ball trail per frame of the cut, for gamst.count_video --points
    trail = [[list(predicted[frame])] if frame in predicted else None for frame in range(first, last, step)]   # [[x, y]]
    (out / 'points.json').write_text(json.dumps(trail))
    print(json.dumps({key: metrics[key] for key in ('stack', 'timing', 'exit_code', 'real_time_factor', 'ball', 'bounces')}))
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', choices=['openttgames'])
    parser.add_argument('--video', required=True)
    parser.add_argument('--markup', required=True, help='folder with ball_markup.json and events_markup.json')
    parser.add_argument('--start-s', type=float, default=0.)
    parser.add_argument('--seconds', type=float, default=60.)
    parser.add_argument('--fps', type=float, help='resample the cut to this rate (default: the video\'s own)')
    parser.add_argument('--stack', default='sota')
    parser.add_argument('--timing', choices=['realtime', 'post'], default='post')
    parser.add_argument('--court', help='court reference JSON (gamst/sports/court.py)')
    parser.add_argument('--tolerance-ms', type=float, default=50.)
    parser.add_argument('--out', required=True)
    args, extra = parser.parse_known_args(argv)     # anything else (after --) goes to gamst recipe run
    args.extra = [item for item in extra if item != '--']
    return openttgames(args)


if __name__ == '__main__':
    main()

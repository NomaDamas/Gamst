"""Score the player tracks of ``team-field`` against a MOT-format ground truth (SportsMOT), docs/sports.md section 9.

    python scripts/sports_mot_bench.py --seq SportsMOT/dataset/train/v_xxx --sport football --stack sota \\
        --timing post --out runs/mot-xxx-sota-post [-- extra gamst recipe run options]

1. the sequence's ``img1/*.jpg`` become ``clip.mp4`` at its ``frameRate`` (``seqinfo.ini``), frame n = image n;
2. ``gamst recipe run team-field --sport S --stack ... --timing ...`` on it, with a Store file;
3. the ``sports.tracks`` records (the players module: track IDs of the stack's tracker) are scored against
   ``gt/gt.txt`` (frame, id, x, y, w, h in pixels) with py-motmetrics at IoU 0.5: MOTA, IDF1, ID switches, precision,
   recall. HOTA (SportsMOT's headline metric) needs TrackEval and is not computed here.

SportsMOT is CC BY-NC 4.0: local evaluation only, never redistributed.
"""
from __future__ import annotations

import argparse
import configparser
import json
from pathlib import Path
import subprocess
import sys
import time


def seqinfo(seq):
    parser = configparser.ConfigParser()
    parser.read(Path(seq) / 'seqinfo.ini')
    info = parser['Sequence']
    return float(info['frameRate']), int(info['imWidth']), int(info['imHeight']), int(info['seqLength']), info['imExt']


def make_clip(seq, fps, ext, out):
    """img1/000001.jpg ... as frames 1..n of a video at the sequence's rate (image n is frame n)."""
    images = sorted((Path(seq) / 'img1').glob(f'*{ext}'))
    pattern = str((Path(seq) / 'img1').resolve() / f'%0{len(images[0].stem)}d{ext}')
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-framerate', f'{fps}', '-start_number', str(int(images[0].stem)),
                    '-i', pattern, '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p', str(out)],
                   check=True)
    return len(images)


def tracks(db):
    from gamst.storage import Store
    store = Store.open_read_only(str(db))
    found, position = [], 0
    try:
        while True:
            page, _ = store.observation_page_rows(subject='local-user', after=position, limit=2000,
                                                  capability_id='sports.tracks', events_only=False)
            if not page:
                return found
            found += [record for _, record in page]
            position = page[-1][0]
    finally:
        store.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seq', required=True)
    parser.add_argument('--sport', required=True)
    parser.add_argument('--stack', default='sota')
    parser.add_argument('--timing', choices=['realtime', 'post'], default='post')
    parser.add_argument('--out', required=True)
    parser.add_argument('--score-only', action='store_true', help='score a finished run (its run.db, report.json) again')
    args, extra = parser.parse_known_args(argv)
    extra = [item for item in extra if item != '--']
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fps, width, height, length, ext = seqinfo(args.seq)
    clip = out / 'clip.mp4'
    if not clip.exists():
        count = make_clip(args.seq, fps, ext, clip)
        if count != length:
            raise SystemExit(f'{count} images, seqinfo says {length}')
    code, wall = 0, None
    if args.score_only:
        argv_run = None
    else:
        argv_run = ['run', 'team-field', '--sport', args.sport, '--stack', args.stack, '--timing', args.timing,
                    '--video', str(clip), '--headless', '--store', str(out / 'run.db'), '--max-records', '400000',
                    '--report', str(out / 'report.json'), *extra]
        started = time.monotonic()
        with open(out / 'run.log', 'w') as log:
            code = subprocess.call([sys.executable, '-c', 'import sys; from gamst.recipes import main; main(sys.argv[1:])',
                                    *argv_run], stdout=log, stderr=subprocess.STDOUT)
        wall = time.monotonic() - started
    report = json.loads((out / 'report.json').read_text()) if (out / 'report.json').exists() else {}
    start_ms = report.get('start_ms', 0)

    import numpy as np
    if not hasattr(np, 'asfarray'):     # py-motmetrics 1.4 calls np.asfarray, removed in NumPy 2
        np.asfarray = lambda value, dtype=float: np.asarray(value, dtype=dtype)
    import motmetrics as mm
    truth = {}
    for line in (Path(args.seq) / 'gt' / 'gt.txt').read_text().splitlines():
        frame, track, x, y, w, h = (float(v) for v in line.split(',')[:6])
        truth.setdefault(int(frame), []).append((int(track), [x, y, w, h]))
    predicted, numbers = {}, {}     # py-motmetrics wants numeric IDs: each GAMST track ID gets one
    for record in tracks(out / 'run.db'):
        frame = round((record['timestamp_ms'] - start_ms) * fps / 1000) + 1
        value = record['observation']['value']
        boxes = []
        for item in value.get('objects') or ():
            if item.get('label') != 'person':
                continue
            box = item['location']['bbox']
            boxes.append((numbers.setdefault(item['track_id'], len(numbers) + 1), [box['x'] * width, box['y'] * height, box['width'] * width,
                                             box['height'] * height]))
        predicted[frame] = boxes
    accumulator = mm.MOTAccumulator(auto_id=False)
    for frame in range(1, length + 1):
        gt, pr = truth.get(frame, []), predicted.get(frame, [])
        distances = mm.distances.iou_matrix([b for _, b in gt], [b for _, b in pr], max_iou=.5) if gt and pr else \
            np.empty((len(gt), len(pr)))
        accumulator.update([t for t, _ in gt], [t for t, _ in pr], distances, frameid=frame)
    summary = mm.metrics.create().compute(accumulator, metrics=['mota', 'idf1', 'num_switches', 'precision', 'recall',
                                                                'num_frames', 'mostly_tracked', 'mostly_lost'],
                                          name='run')
    row = {key: (None if value != value else round(float(value), 4)) for key, value in summary.loc['run'].items()}
    metrics = dict(dataset='sportsmot', seq=Path(args.seq).name, sport=args.sport, stack=args.stack, timing=args.timing,
                   frames=length, fps=fps, frames_with_tracks=len(predicted), exit_code=code,
                   wall_s=None if wall is None else round(wall, 1),
                   run_s=report.get('run_s'),
                   real_time_factor=round(report['run_s'] / (length / fps), 3) if report.get('run_s') else None,
                   complete=report.get('complete'), model_choice=report.get('model_choice'), mot=row,
                   counts=report.get('counts'), errors=report.get('errors'))
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=1))
    print(json.dumps({key: metrics[key] for key in ('seq', 'stack', 'timing', 'exit_code', 'real_time_factor', 'mot')}))


if __name__ == '__main__':
    main()

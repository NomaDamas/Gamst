"""Measure an ``audio.diarization`` candidate on a recording fed as a live stream, and score its slots.

  python scripts/qa_diarization.py run OUT.json MODULE:FACTORY '{"kwargs"}' --wav a.wav [--pace]
  python scripts/qa_diarization.py score OUT.json [...]

``run`` feeds 100 ms blocks (paced at real time with ``--pace``) to the provider in this interpreter (the candidate's
own environment) and saves every slot record with wall times, compute time and peak RSS. ``score`` uses the owner's
word labels of the shop clip (``try-it/testbed/labels_bbo.md``) as a proxy for diarization truth:
docs/measurements/diarization-2026-11-26.md says how.
"""
import argparse
import importlib
import json
import resource
import sys
import time
import wave

import numpy as np

# Person's words and other voices' words: Whisper-turbo word spans whose ends are the owner's labels
# (14.0 s was never transcribed: 13.6-14.0 is assumed); 49-52 s is another person.
PERSON = [(11.64, 12.04), (13.60, 14.00), (19.38, 19.74), (19.74, 20.14), (21.06, 21.46), (21.94, 22.30),
          (27.98, 28.10), (37.66, 38.20), (45.92, 46.64), (47.28, 47.74)]
OTHER = [(12.52, 12.64), (20.14, 20.34), (20.34, 20.74), (27.06, 27.44), (49.44, 49.86)]
OTHER_REGION = (49.0, 52.0)
EXCHANGE = (19.4, 20.8)
# The owner's overlapped cases (person's word / another voice's word said over or right at it): 12.04/12.64,
# 19.74-20.74, 27.44/28.10. Each region spans those words (Whisper-turbo spans with the owner's ends).
OVERLAP_REGIONS = [(11.64, 12.64), (19.38, 20.74), (27.06, 28.10)]
OVERLAP_WORDS = [('person', 11.64, 12.04), ('other', 12.52, 12.64), ('person', 19.38, 19.74), ('person', 19.74, 20.14),
                 ('other', 20.14, 20.34), ('other', 20.34, 20.74), ('other', 27.06, 27.44), ('person', 27.98, 28.10)]


def mb(value):
    return value / 2 ** 20 if sys.platform == 'darwin' else value / 1024      # ru_maxrss: bytes on macOS


def run(args):
    from gamst.capabilities import SensorMetadata
    from gamst.plugins import InputFrame
    with wave.open(args.wav) as source:
        if source.getframerate() != 16000 or source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise SystemExit('16 kHz mono 16-bit WAV required')
        pcm = np.frombuffer(source.readframes(source.getnframes()), '<i2')
    pcm = pcm[:len(pcm) // 160 * 160]
    module, _, attribute = args.factory.partition(':')
    factory = importlib.import_module(module)
    for part in attribute.split('.'):
        factory = getattr(factory, part)
    started = time.perf_counter()
    provider = factory(**json.loads(args.kwargs))
    load_s, loaded = time.perf_counter() - started, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    step, records, compute = args.block_ms * 16, [], []
    start = time.perf_counter()
    for sequence, position in enumerate(range(0, len(pcm), step)):
        block = pcm[position:position + step]
        end_ms = (position + len(block)) // 16
        if args.pace:
            delay = start + end_ms / 1000 - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        frame = InputFrame(block.copy(), 'audio', end_ms, 'local-user', 16000,
                           SensorMetadata(source_id='capture', sensor_id='mic', sequence=sequence, channels=('mono',)))
        began = time.perf_counter()
        value = provider.process(frame)
        now = time.perf_counter()
        compute.append(now - began)
        if value:
            records.append(dict(wall=now - start, value=value['diarization']))
    began = time.perf_counter()
    value = provider.finish()
    now = time.perf_counter()
    if value:
        records.append(dict(wall=now - start, value=value['diarization'], finish=True))
    getattr(provider, 'close', lambda: None)()
    result = dict(factory=args.factory, kwargs=json.loads(args.kwargs), pace=args.pace, block_ms=args.block_ms,
                  audio_s=len(pcm) / 16000, load_s=load_s, wall_s=now - start, finish_s=now - began,
                  compute_s=sum(compute) + now - began, compute_ms_p95=float(np.percentile(compute, 95) * 1000),
                  rss_loaded_mb=mb(loaded), rss_peak_mb=mb(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                  records=records)
    with open(args.out, 'w') as handle:
        json.dump(result, handle)
    print(f"{args.out}: {len(records)} records, compute {result['compute_s']:.1f} s for {result['audio_s']:.1f} s, "
          f"wall {result['wall_s']:.1f} s, peak RSS {result['rss_peak_mb']:.0f} MB")


def grid(result):
    count = int(round(result['audio_s'] * 100)) + 1
    active = np.zeros((8, count), bool)
    for record in result['records']:
        for chunk in record['value']['chunks']:
            for item in chunk['activity']:
                for begin, end in item['intervals_ms']:
                    active[item['slot'], max(0, begin // 10):min(count, end // 10)] = True
    return active


def cover(active, begin, end):
    return active[:, int(round(begin * 100)):int(round(end * 100))].mean(axis=1)


def score(path):
    result = json.load(open(path))
    active = grid(result)
    person = [cover(active, *span) for span in PERSON]
    other = [cover(active, *span) for span in OTHER]
    region = cover(active, *OTHER_REGION)
    on = lambda shares: {slot for slot, share in enumerate(shares) if share >= .5}
    counts = {}
    for shares in person:
        for slot in on(shares):
            counts[slot] = counts.get(slot, 0) + 1
    best = max(counts, key=lambda slot: (counts[slot], -slot)) if counts else None
    others = set().union(*(on(shares) for shares in other)) | {slot for slot, share in enumerate(region) if share >= .3}
    label = lambda shares: ' '.join(f'{slot}:{share:.0%}' for slot, share in
                                    sorted(enumerate(shares), key=lambda item: -item[1]) if share >= .3) or '-'
    runs = []
    for index in range(int(EXCHANGE[0] * 100), int(EXCHANGE[1] * 100)):
        slots = '+'.join(map(str, np.flatnonzero(active[:, index]))) or 'none'
        if runs and runs[-1][0] == slots:
            runs[-1][2] = (index + 1) / 100
        else:
            runs.append([slots, index / 100, (index + 1) / 100])
    lag = [record['wall'] - chunk['end_ms'] / 1000 for record in result['records'] if not record.get('finish')
           for chunk in record['value']['chunks']] if result['pace'] else []
    return dict(
        person=[label(shares) for shares in person], other=[label(shares) for shares in other], region=label(region),
        best=best, person_in=sum(best in on(shares) for shares in person),
        other_in=sum(best in on(shares) for shares in other), other_only=sum(on(shares) == {best} for shares in other),
        region_best=round(float(region[best]), 2) if best is not None else None,
        strict_purity=sum(bool(on(shares) - others) for shares in person),
        exchange=' '.join(f'{slots}[{begin:.2f}-{end:.2f}]' for slots, begin, end in runs if end - begin >= .02),
        lag_median_s=round(float(np.median(lag)), 2) if lag else None,
        lag_p95_s=round(float(np.percentile(lag, 95)), 2) if lag else None,
        compute_per_audio=round(result['compute_s'] / result['audio_s'], 2), peak_rss_mb=round(result['rss_peak_mb']))


def overlap_intervals(active, min_ms=0):
    """[(start_s, end_s)] where two or more slots are active at once, runs shorter than ``min_ms`` left out."""
    both = active.sum(axis=0) >= 2
    runs, start = [], None
    for index, flag in enumerate(list(both) + [False]):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if (index - start) * 10 >= min_ms:
                runs.append((start / 100, index / 100))
            start = None
    return runs


def overlap(path, margin_s=0.):
    """Recall of the owner's overlapped cases by a candidate's overlap flags (two slots at once), and how much else it flags."""
    result = json.load(open(path))
    runs = overlap_intervals(grid(result))
    hits = lambda begin, end: any(low - margin_s < end and high + margin_s > begin for low, high in runs)
    person_alone = [span for span in PERSON if not any(span[0] < high and span[1] > low for low, high in OVERLAP_REGIONS)]
    return dict(
        regions=[hits(*region) for region in OVERLAP_REGIONS],
        words=[f'{who}@{end:.2f}:{"yes" if hits(begin, end) else "no"}' for who, begin, end in OVERLAP_WORDS],
        person_alone_flagged=[f'{end:.2f}' for begin, end in person_alone if hits(begin, end)],
        flagged_s=round(sum(high - low for low, high in runs), 2), intervals=len(runs),
        runs=' '.join(f'{low:.2f}-{high:.2f}' for low, high in runs))


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    measure = commands.add_parser('run')
    measure.add_argument('out')
    measure.add_argument('factory')
    measure.add_argument('kwargs')
    measure.add_argument('--wav', required=True)
    measure.add_argument('--pace', action='store_true')
    measure.add_argument('--block-ms', type=int, default=100)
    scoring = commands.add_parser('score')
    scoring.add_argument('paths', nargs='+')
    flags = commands.add_parser('overlap')
    flags.add_argument('paths', nargs='+')
    flags.add_argument('--margin-s', type=float, default=0.)
    args = parser.parse_args()
    if args.command == 'run':
        run(args)
    elif args.command == 'overlap':
        for path in args.paths:
            print(path, json.dumps(overlap(path, args.margin_s), ensure_ascii=False))
    else:
        for path in args.paths:
            print(path, json.dumps(score(path), ensure_ascii=False))


if __name__ == '__main__':
    main()

"""View generic GAMST rotation observations and rotation.ended Watch alarms.

Examples:
  gamst demo rotation --video spin.mp4 --targets coin,wheel,fan
  gamst demo rotation --camera 0 --expected-max-rev-s 8
  gamst demo rotation --video motion.mp4 --detector motion
  gamst demo rotation --take gamst-takes/take-20261125-101500 --symmetry 3 --roi .3,.3,.4,.4

The application only captures and pushes frames. Tracking, measurement,
records and events belong to GAMST; this module renders committed records only.
"""
from __future__ import annotations

import asyncio
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ..crew.registry import weight_assets
from ..providers.rotation import RotationPlugin


# Pinned weights come from the one GAMST role registry (gamst.crew.registry); these are views of it.
DETECTOR = ('vision.detect', 'yoloe.26s')
REFINER = ('vision.video_segmentation', 'efficienttam.ti_512')
MODEL_ASSETS = weight_assets(*DETECTOR)
# Take refinement (--refine): promptable video segmenter, Hugging Face revision pinned.
REFINE_ASSETS = weight_assets(*REFINER)
DEFAULT_MODELS = Path(os.environ.get('GAMST_MODELS', Path.home() / '.cache' / 'gamst' / 'models')) / 'yoloe'


def fetch_models(directory, names=None):
    """Download the pinned weights (only ``names`` when given); setup's downloader verifies SHA256."""
    from ..crew.registry import candidate
    from ..crew.setup import SetupError, download_weight
    directory = Path(directory)
    for role, model in (DETECTOR, REFINER):
        for weight in candidate(role, model).weights:
            if names is not None and weight.name not in names:
                continue
            try:
                download_weight(weight.as_dict(), directory / weight.name)
            except SetupError as error:
                raise SystemExit(str(error)) from error
            print(f'{weight.name}: sha256 verified')


def detector_config(models, labels, device, threshold=.25):
    missing = [name for name in MODEL_ASSETS if not (models / name).is_file()]
    if missing:
        raise SystemExit(f'missing detector weights {missing} in {models}; run: gamst demo rotation --fetch-models')
    runtime = models.parent / 'runtime'
    runtime.mkdir(parents=True, exist_ok=True)
    return dict(checkpoint_path=str(models / 'yoloe-26s-seg.pt'),
        checkpoint_sha256=MODEL_ASSETS['yoloe-26s-seg.pt'][1], runtime_root=str(runtime),
        mode='text', labels=labels, encoder_path=str(models / 'mobileclip2_b.ts'),
        encoder_sha256=MODEL_ASSETS['mobileclip2_b.ts'][1], encoder_family='mobileclip2:b',
        device=device, imgsz=640, threshold=threshold, max_objects=16)


def _detector_image(frame):
    import cv2
    height, width = frame.shape[:2]
    scale = min(1., (1_048_576 / (height * width)) ** .5, 2048 / max(height, width))
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame


def _check_worker_dependencies(python):
    probe = ('import json\nfrom importlib.metadata import distribution,version\n'
             'print(json.dumps({"ultralytics":version("ultralytics"),"clip":json.loads(distribution("clip").read_text("direct_url.json") or "{}").get("vcs_info",{}).get("commit_id")}))')
    result = subprocess.run([python, '-c', probe], capture_output=True, text=True)
    found = json.loads(result.stdout) if result.returncode == 0 else {}
    from ..providers.yoloe_native import CLIP_REVISION
    if found.get('clip') != CLIP_REVISION or not found.get('ultralytics'):
        raise SystemExit('the YOLOE detector worker needs:\n'
            f'  {python} -m pip install "gamst-sdk[rotation,yoloe]" '
            f'"git+https://github.com/ultralytics/CLIP.git@{CLIP_REVISION}"\n'
            'or use --detector motion')


class Panel:
    def draw(self, frame, record, alarms, fps):
        import cv2
        import numpy as np
        import textwrap
        value = record['observation']['value'] if record else {'objects': []}
        for item in value['objects']:
            x, y, size, _ = item['box']
            color = (60, 220, 60) if item['observed'] else (0, 160, 255)
            cv2.rectangle(frame, (x, y), (x + size, y + size), color, 2)
            label = item['label'] or 'unknown'
            text = f"{label}: {item['status']}; plane {item['in_plane_turns']:.2f}; flip {item['flip_turns']}"
            cv2.putText(frame, text, (max(0, x), max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
        panel = np.full((frame.shape[0], 480, 3), 24, np.uint8)
        lines = ['GAMST vision.rotation', f'capture {fps:.0f} fps']
        if record:
            lines.append(f'record {record["id"][:8]}')
        lines.append('')
        for item in value['objects']:
            lines.append(f"{item['track_id']}: {item['status']} | {item['capture_fps']} fps")
            lines.append(f"  in-plane {item['in_plane_rev_s'] or 0:.2f} rev/s")
            lines.append(f"  flip {item['flip_rev_s'] or 0:.2f} rev/s")
        lines += ['', 'Watch alarms (rotation.ended):']
        for event in alarms[-5:]:
            lines += textwrap.wrap(json.dumps(event, ensure_ascii=False), 56,
                                   initial_indent='- ', subsequent_indent='  ')
        for index, line in enumerate(lines[:max(1, frame.shape[0] // 22 - 1)]):
            cv2.putText(panel, line, (10, 26 + 22 * index), cv2.FONT_HERSHEY_SIMPLEX, .48,
                        (255, 255, 255) if index else (80, 220, 255), 1)
        return np.hstack([frame, panel])


async def _detector_lane(args, human):
    if args.detector == 'motion' or args.roi is not None:
        return None
    models = Path(args.models)
    if args.detector == 'auto' and any(not (models / name).is_file() for name in MODEL_ASSETS):
        return None
    from .. import ProcessProvider, SensorStreams
    _check_worker_dependencies(args.worker_python)
    config = detector_config(models, args.targets, args.device)
    spec = ProcessProvider.capability('yoloe.local', config)
    worker = await ProcessProvider.open(args.worker_python, 'yoloe.local', config)
    streams = SensorStreams(human)
    try:
        streams.register(spec, lambda: worker)
        await streams.open('rotation-demo', 'detector', [spec.id])
        return streams
    except BaseException:
        await worker.close()
        raise


class _TakeCapture:
    """A GAMST 240 take read like a video file; positions are the phone's capture times."""

    def __init__(self, path):
        from .rotation_live import read_recording
        path = Path(path)
        self.frames = read_recording(path / 'raw.gmst' if path.is_dir() else path)
        self.index, self.position_ms = 0, 0.

    def isOpened(self):
        return bool(self.frames)

    def read(self):
        from .rotation_live import decode
        while self.index < len(self.frames):
            info, _, payload = self.frames[self.index]
            self.index += 1
            frame = decode(info, payload)
            if frame is not None:
                self.position_ms = 1000 * (info.seconds - self.frames[0][0].seconds)
                return True, frame
        return False, None

    def get(self, prop):
        import cv2
        span = self.frames[-1][0].seconds - self.frames[0][0].seconds
        return {cv2.CAP_PROP_FPS: (len(self.frames) - 1) / span if span else 0.,
                cv2.CAP_PROP_FRAME_COUNT: len(self.frames), cv2.CAP_PROP_POS_MSEC: self.position_ms}[prop]

    def release(self):
        self.frames = []


async def run(args):
    import cv2
    from .. import Gamst
    from ..capabilities import SensorMetadata
    from ..plugins import InputFrame

    recorded = args.video or args.take
    source = recorded or args.camera
    cap = _TakeCapture(args.take) if args.take else cv2.VideoCapture(
        source, cv2.CAP_ANY if args.video else cv2.CAP_AVFOUNDATION if sys.platform == 'darwin' else cv2.CAP_ANY)
    if not cap.isOpened():
        raise SystemExit(f'cannot open input: {source}')
    detector = None
    plugin = None
    report = dict(source=str(source), frames=0, observed_frames=0, statuses={}, events=[], alarms=0,
                  detector='pending', region_source='roi' if args.roi is not None else 'auto')
    pending, alarm_events = [], []
    panel = Panel()
    try:
        async with Gamst(consent=True, store=args.store) as human:
            detector = await _detector_lane(args, human)
            report['detector'] = 'yoloe' if detector is not None else 'motion'
            targets = args.targets if detector is not None else None
            config = dict(expected_max_rev_s=args.expected_max_rev_s, targets=targets,
                          region_source='auto', symmetry=args.symmetry, in_plane=args.in_plane)
            if args.roi is not None:
                x, y, width, height = args.roi
                config.update(region_source='roi', roi=dict(x=x, y=y, width=width, height=height))
            plugin = RotationPlugin(human, **config)
            human.register(plugin)
            watch = human.watches.create(owner='user',
                description='Notify me when a measured object rotation ends.',
                rule=dict(conditions=[dict(capability_id='vision.rotation', path='events', op='contains',
                                           value='rotation.ended', within_ms=1000)],
                          cooldown_ms=0, max_per_minute=60, context_window_ms=3000))
            human.watches.add_sink(pending.append)
            fps_video = cap.get(cv2.CAP_PROP_FPS) if recorded else 0
            start_wall = time.time_ns() // 1_000_000
            start = time.monotonic()
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) if recorded else 0
            base = start_wall - (round(1000 * frame_count / (args.capture_fps or fps_video)) + 1000 if recorded else 0)
            seq = detector_seq = 0
            last_ts = last_detection = None
            intervals = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    if recorded and seq:
                        break
                    raise SystemExit('no camera/video frame available')
                if recorded:
                    offset = seq * 1000 / args.capture_fps if args.capture_fps else cap.get(cv2.CAP_PROP_POS_MSEC)
                    stamp = base + round(offset)
                else:
                    stamp = base + round(1000 * (time.monotonic() - start))
                if last_ts is not None and stamp <= last_ts:
                    stamp = last_ts + 1
                if last_ts is not None:
                    intervals.append(stamp - last_ts)
                last_ts = stamp
                if detector is not None and (last_detection is None or stamp - last_detection >= args.detect_every_ms):
                    sensor = SensorMetadata(source_id='rotation-demo', sensor_id='detector', sequence=detector_seq,
                                            coordinate_frame='image')
                    await detector.submit(InputFrame(_detector_image(frame), 'video', stamp, sensor=sensor))
                    if recorded:
                        await detector.drain()
                    detector_seq += 1
                    last_detection = stamp
                await human.push_video(frame, timestamp_ms=stamp)
                records = await human.get_observations(capability_id='vision.rotation', window='1h', limit=1)
                record = records['items'][0] if records['items'] else None
                if record is not None and record['timestamp_ms'] == stamp:
                    value = record['observation']['value']
                    if any(item['observed'] for item in value['objects']):
                        report['observed_frames'] += 1
                    for item in value['objects']:
                        key = item['status']
                        report['statuses'][key] = report['statuses'].get(key, 0) + 1
                    report['events'].extend(value['events'])
                while pending:
                    alarm = pending.pop(0)
                    report['alarms'] += 1
                    trigger = alarm['trigger_record_id']
                    ended = [event for item in alarm['context']['records'] if item['id'] == trigger
                             for event in item['observation']['value']['events']
                             if event['type'] == 'rotation.ended']
                    alarm_events.extend(ended)
                    print(json.dumps({'watch': watch['id'], 'alarm': alarm['id'], 'events': ended}), flush=True)
                fps = 1000 / (sum(intervals[-30:]) / len(intervals[-30:])) if intervals else 0
                if not args.headless:
                    cv2.imshow('GAMST rotation', panel.draw(frame, record, alarm_events, fps))
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord('q')):
                        break
                    if key == ord('r'):
                        plugin.reset()
                seq += 1
                report['frames'] = seq
                if args.frames and seq >= args.frames:
                    break
                await asyncio.sleep(0)
            if detector is not None:
                await detector.drain()
            # A file or take can end mid-spin (a fan never stops): commit that count too.
            final = await plugin.finish()
            report['events'].extend(final['events'])
            for event in final['events']:
                print(json.dumps({'ended_with_input': event}), flush=True)
            report['capture_fps'] = round(1000 / (sum(intervals) / len(intervals)), 1) if intervals else None
            report['observed_frame_coverage'] = round(report['observed_frames'] / seq, 4) if seq else 0.
            plugin.close()
    finally:
        cap.release()
        if detector is not None:
            await detector.close()
        if plugin is not None:
            plugin.close()
        if not args.headless:
            cv2.destroyAllWindows()
    if args.report:
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({key: report[key] for key in
                      ('frames', 'observed_frames', 'observed_frame_coverage', 'capture_fps', 'statuses', 'alarms')} |
                     {'events': len(report['events'])}), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog='gamst demo rotation', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--camera', type=int, default=0)
    source.add_argument('--video', help='recorded file; capture timestamps come from the file')
    source.add_argument('--iphone', type=int, metavar='PORT',
                        help='live HUD for the GAMST 240 iPhone app; Start/Stop on the phone records takes')
    source.add_argument('--iphone-replay', metavar='FILE', help='run a GAMST 240 recording (.gmst) through the HUD')
    source.add_argument('--take', metavar='DIR',
                        help='measure a GAMST 240 take (folder or raw.gmst) with vision.rotation at its capture times')
    parser.add_argument('--record-dir', default='gamst-takes',
                        help='with --iphone/--iphone-replay: one folder per take (raw.gmst, hud.mp4, refined.json/mp4)')
    parser.add_argument('--targets', type=lambda value: [item.strip() for item in value.split(',') if item.strip()],
                        default=['coin', 'wheel', 'fan'], help='detector labels and optional rotation label filter')
    parser.add_argument('--expected-max-rev-s', type=float, default=40.)
    parser.add_argument('--symmetry', type=int, default=1,
                        help='times the object looks the same per in-plane turn (a 3-blade fan: 3); '
                             'in-plane turns count only below capture fps / (2 * symmetry) rev/s')
    parser.add_argument('--in-plane', choices=['points', 'polar'], default='points',
                        help='in-plane estimator: feature points, or polar registration of what moves '
                             '(textured or repetitive objects such as fan blades)')
    parser.add_argument('--capture-fps', type=float, help='override file timing for high-fps captures')
    parser.add_argument('--detector', choices=['auto', 'yoloe', 'motion'], default='auto')
    parser.add_argument('--roi', type=lambda value: [float(item) for item in value.split(',')],
                        help='normalized x,y,width,height; bypasses detector/motion regions')
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    parser.add_argument('--models', default=str(DEFAULT_MODELS))
    parser.add_argument('--fetch-models', action='store_true', help='download pinned detector weights and exit')
    parser.add_argument('--worker-python', default=sys.executable)
    parser.add_argument('--detect-every-ms', type=int, default=100)
    parser.add_argument('--store', default=':memory:')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--frames', type=int)
    parser.add_argument('--report', help='write observed coverage, records-derived events and alarms as JSON')
    parser.add_argument('--refine', metavar='LABEL',
                        help='with --iphone/--iphone-replay: after each take, find LABEL (open-vocabulary text) '
                             'and follow it with EfficientTAM to re-count its throws (needs --efficienttam-source)')
    parser.add_argument('--efficienttam-source', metavar='DIR',
                        help='checkout of yformer/EfficientTAM at the pinned revision, for --refine')
    parser.add_argument('--refine-on', choices=['stop', 'button'], default='stop',
                        help='count each take on Stop, or only when ANALYZE (key a) is pressed')
    parser.add_argument('--refine-fast', action='store_true',
                        help='detector only where the picture moves; stop at the first complete count')
    parser.add_argument('--refine-show', action='store_true',
                        help='open refined.mp4 (8x slow motion with the running count) when it is ready')
    parser.add_argument('--refine-cached', metavar='TAKE_DIR',
                        help='stage backup: ANALYZE shows this already counted take (refined.json, refined.mp4) at once')
    args = parser.parse_args(argv)
    if args.fetch_models:
        fetch_models(Path(args.models))
        return
    if args.roi is not None and len(args.roi) != 4:
        parser.error('--roi requires x,y,width,height')
    if args.refine and not args.efficienttam_source:
        parser.error('--refine requires --efficienttam-source')
    if args.take and args.refine:
        parser.error('--refine counts flips through the HUD: use --iphone-replay TAKE/raw.gmst')
    if args.iphone or args.iphone_replay:
        from .rotation_live import run as run_live
        asyncio.run(run_live(args))
        return
    asyncio.run(run(args))


if __name__ == '__main__':
    main()

"""View generic GAMST rotation observations and rotation.ended Watch alarms.

Examples:
  gamst demo rotation --video spin.mp4 --targets coin,wheel,fan
  gamst demo rotation --camera 0 --expected-max-rev-s 8
  gamst demo rotation --video motion.mp4 --detector motion

The application only captures and pushes frames. Tracking, measurement,
records and events belong to GAMST; this module renders committed records only.
"""
from __future__ import annotations

import asyncio
import argparse
import hashlib
import json
import os
from pathlib import Path
import ssl
import subprocess
import sys
import time
from urllib.request import urlopen

from ..providers.rotation import RotationPlugin


MODEL_ASSETS = {
    'yoloe-26s-seg.pt': ('https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26s-seg.pt',
                         '48f24206bc8680d60cbbfa296b0140da849669b9515058b72f5a945142df0654'),
    'mobileclip2_b.ts': ('https://github.com/ultralytics/assets/releases/download/v8.4.0/mobileclip2_b.ts',
                         '35d7f213e4d75f38514e4656ad3cb91158bd33e3805d8ac349f23b186f66982f'),
}
DEFAULT_MODELS = Path(os.environ.get('GAMST_MODELS', Path.home() / '.cache' / 'gamst' / 'models')) / 'yoloe'


def _tls():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        while chunk := source.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_models(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in MODEL_ASSETS.items():
        target = directory / name
        temporary = target.with_suffix('.download')
        if target.exists() and _sha256(target) == expected:
            print(f'{name}: sha256 verified')
            continue
        with urlopen(url, timeout=120, context=_tls()) as response, temporary.open('wb') as output:
            while chunk := response.read(1 << 20):
                output.write(chunk)
        if _sha256(temporary) != expected:
            temporary.unlink(missing_ok=True)
            raise SystemExit(f'{name}: SHA256 mismatch; download discarded')
        temporary.replace(target)
        print(f'{name}: sha256 verified')


def detector_config(models, labels, device):
    missing = [name for name in MODEL_ASSETS if not (models / name).is_file()]
    if missing:
        raise SystemExit(f'missing detector weights {missing} in {models}; run: gamst demo rotation --fetch-models')
    runtime = models.parent / 'runtime'
    runtime.mkdir(parents=True, exist_ok=True)
    return dict(checkpoint_path=str(models / 'yoloe-26s-seg.pt'),
        checkpoint_sha256=MODEL_ASSETS['yoloe-26s-seg.pt'][1], runtime_root=str(runtime),
        mode='text', labels=labels, encoder_path=str(models / 'mobileclip2_b.ts'),
        encoder_sha256=MODEL_ASSETS['mobileclip2_b.ts'][1], encoder_family='mobileclip2:b',
        device=device, imgsz=640, threshold=.25, max_objects=16)


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


async def run(args):
    import cv2
    from .. import Gamst
    from ..capabilities import SensorMetadata
    from ..plugins import InputFrame

    source = args.video if args.video else args.camera
    cap = cv2.VideoCapture(source, cv2.CAP_ANY if args.video else
                           cv2.CAP_AVFOUNDATION if sys.platform == 'darwin' else cv2.CAP_ANY)
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
                          region_source='auto')
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
            fps_video = cap.get(cv2.CAP_PROP_FPS) if args.video else 0
            start_wall = time.time_ns() // 1_000_000
            start = time.monotonic()
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) if args.video else 0
            base = start_wall - (round(1000 * frame_count / (args.capture_fps or fps_video)) + 1000 if args.video else 0)
            seq = detector_seq = 0
            last_ts = last_detection = None
            intervals = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    if args.video and seq:
                        break
                    raise SystemExit('no camera/video frame available')
                if args.video:
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
                    if args.video:
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


IPHONE_HEADER = 24  # b"GMST" | u32 sequence | f64 capture seconds | u16 width | u16 height | u32 jpeg bytes


async def run_iphone(args):
    """Receive 240 fps frames from the GAMST 240 iPhone app and feed them to GAMST.

    The phone is only a camera: it sends downscaled luma JPEGs with their sensor
    presentation timestamps. Capture timing comes from those timestamps, so the
    rotation counter sees the true 240 fps cadence even when frames arrive in bursts.
    ``--record-dir`` keeps the raw stream; ``--iphone-replay`` feeds a recording back
    through the same path with no real-time pressure.
    """
    import struct
    import cv2
    import numpy as np
    from .. import Gamst

    # Tosses are short bursts: a 20 s backlog adds latency between tosses but keeps every frame,
    # and a counter fed a gapped sequence would see a lower capture rate than the phone delivered.
    queue = asyncio.Queue(maxsize=4800)
    stats = dict(received=0, dropped=0, processed=0)

    async def handle(reader, writer):
        peer = writer.get_extra_info('peername')
        sink = None
        if args.record_dir:
            Path(args.record_dir).mkdir(parents=True, exist_ok=True)
            sink = open(Path(args.record_dir) / f'iphone-{time.strftime("%Y%m%d-%H%M%S")}.gmst', 'wb')
        print(json.dumps({'iphone': 'connected', 'peer': str(peer), 'recording': sink and sink.name}), flush=True)
        try:
            while True:
                header = await reader.readexactly(IPHONE_HEADER)
                if header[:4] != b'GMST':
                    raise ValueError('not a GAMST 240 frame stream')
                sequence, seconds, _, _, size = struct.unpack('<IdHHI', header[4:])
                payload = await reader.readexactly(size)
                if sink is not None:
                    sink.write(header + payload)
                if queue.full():
                    queue.get_nowait()
                    stats['dropped'] += 1
                queue.put_nowait((sequence, seconds, payload))
                stats['received'] += 1
        except (asyncio.IncompleteReadError, ConnectionError):
            print(json.dumps({'iphone': 'disconnected'}), flush=True)
        finally:
            if sink is not None:
                sink.close()
            writer.close()

    async def replay(path):
        data = Path(path).read_bytes()
        frames, offset = [], 0
        while offset + IPHONE_HEADER <= len(data):
            if data[offset:offset + 4] != b'GMST':
                raise SystemExit(f'{path}: not a GAMST 240 recording')
            sequence, seconds, _, _, size = struct.unpack('<IdHHI', data[offset + 4:offset + IPHONE_HEADER])
            offset += IPHONE_HEADER
            frames.append((sequence, seconds, data[offset:offset + size]))
            offset += size
        # Replay runs faster than real time; anchor the recording so its last frame is not in the future.
        stats['span_ms'] = round(1000 * (frames[-1][1] - frames[0][1])) if frames else 0
        for frame in frames:
            await queue.put(frame)
            stats['received'] += 1
        await queue.put(None)


    server = feeder = None
    if args.iphone_replay:
        feeder = asyncio.create_task(replay(args.iphone_replay))
        source = args.iphone_replay
    else:
        server = await asyncio.start_server(handle, '0.0.0.0', args.iphone)
        print(json.dumps({'listening': args.iphone, 'note': 'open GAMST 240 on the iPhone and tap Start'}), flush=True)
        source = f'iphone:{args.iphone}'
    pending, alarm_events, report = [], [], dict(source=source, frames=0, alarms=0, events=[])
    panel = Panel()
    plugin = None
    try:
        async with Gamst(consent=True, store=args.store) as human:
            config = dict(expected_max_rev_s=args.expected_max_rev_s, region_source='motion', record_every_ms=33)
            if args.roi is not None:
                x, y, width, height = args.roi
                config.update(region_source='roi', roi=dict(x=x, y=y, width=width, height=height))
            plugin = RotationPlugin(human, **config)
            human.register(plugin)
            watch = human.watches.create(owner='user', description='Tell me how many turns each rotation made.',
                rule=dict(conditions=[dict(capability_id='vision.rotation', path='events', op='contains',
                                           value='rotation.ended', within_ms=1000)],
                          cooldown_ms=0, max_per_minute=600, context_window_ms=3000))
            human.watches.add_sink(pending.append)
            first = base = last = None
            capture_times, last_print, printed = [], time.monotonic(), 0
            title = 'GAMST rotation - iPhone 240 fps'
            if not args.headless:
                cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
                cv2.moveWindow(title, 40, 60)
                waiting = np.full((540, 1680, 3), 24, np.uint8)
                cv2.putText(waiting, f'waiting for the iPhone on port {args.iphone}: open GAMST 240 and tap Start',
                            (30, 270), cv2.FONT_HERSHEY_SIMPLEX, 1., (80, 220, 255), 2)
                cv2.imshow(title, waiting)
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), 0.05)
                except asyncio.TimeoutError:
                    if not args.headless and (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
                        break
                    continue
                if item is None:
                    break
                sequence, seconds, payload = item
                frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                if first is None:
                    first, base = seconds, time.time_ns() // 1_000_000 - 1000 - stats.get('span_ms', 0)
                stamp = base + round((seconds - first) * 1000)
                if last is not None and stamp <= last:
                    stamp = last + 1
                last = stamp
                capture_times.append(seconds)
                await human.push_video(frame, timestamp_ms=stamp)
                stats['processed'] += 1
                report['frames'] += 1
                while pending:
                    alarm = pending.pop(0)
                    report['alarms'] += 1
                    ended = [event for item in alarm['context']['records'] if item['id'] == alarm['trigger_record_id']
                             for event in item['observation']['value']['events'] if event['type'] == 'rotation.ended']
                    alarm_events.extend(ended)
                    report['events'].extend(ended)
                    for event in ended:
                        print(json.dumps({'watch': watch['id'], 'rotation.ended': event}, ensure_ascii=False), flush=True)
                window = capture_times[-241:]
                capture_fps = (len(window) - 1) / (window[-1] - window[0]) if len(window) > 1 and window[-1] > window[0] else 0
                if time.monotonic() - last_print >= 2:
                    elapsed = time.monotonic() - last_print
                    print(json.dumps({'capture_fps': round(capture_fps, 1), 'received': stats['received'],
                                      'processed': stats['processed'],
                                      'processed_fps': round((stats['processed'] - printed) / elapsed, 1),
                                      'queue': queue.qsize(), 'dropped_on_mac': stats['dropped']}), flush=True)
                    last_print, printed = time.monotonic(), stats['processed']
                if not args.headless and stats['processed'] % 8 == 0:
                    records = await human.get_observations(capability_id='vision.rotation', window='1h', limit=1)
                    record = records['items'][0] if records['items'] else None
                    view = panel.draw(frame, record, alarm_events, capture_fps)
                    cv2.imshow(title, cv2.resize(view, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_NEAREST))
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord('q')):
                        break
                    if key == ord('r'):
                        plugin.reset()
                if args.frames and report['frames'] >= args.frames:
                    break
    finally:
        if server is not None:
            server.close()
        if feeder is not None:
            feeder.cancel()
        if plugin is not None:
            plugin.close()
        if not args.headless:
            cv2.destroyAllWindows()
    report.update(received=stats['received'], dropped_on_mac=stats['dropped'])
    if args.report:
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog='gamst demo rotation', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--camera', type=int, default=0)
    source.add_argument('--video', help='recorded file; capture timestamps come from the file')
    source.add_argument('--iphone', type=int, metavar='PORT',
                        help='listen for the GAMST 240 iPhone app (240 fps frames with capture timestamps)')
    source.add_argument('--iphone-replay', metavar='FILE', help='replay a GAMST 240 recording (.gmst)')
    parser.add_argument('--record-dir', help='with --iphone: keep each raw phone stream as DIR/iphone-*.gmst')
    parser.add_argument('--targets', type=lambda value: [item.strip() for item in value.split(',') if item.strip()],
                        default=['coin', 'wheel', 'fan'], help='detector labels and optional rotation label filter')
    parser.add_argument('--expected-max-rev-s', type=float, default=40.)
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
    args = parser.parse_args(argv)
    if args.fetch_models:
        fetch_models(Path(args.models))
        return
    if args.roi is not None and len(args.roi) != 4:
        parser.error('--roi requires x,y,width,height')
    asyncio.run(run_iphone(args) if args.iphone or args.iphone_replay else run(args))


if __name__ == '__main__':
    main()

"""Coin Halo: count how many times a coin turns, with evidence, using GAMST.

    gamst demo coin --fetch-models                 # once: pinned YOLOE + MobileCLIP2 weights
    gamst demo coin --camera 0                     # live camera
    gamst demo coin --video toss_240fps.mov        # e.g. iPhone slow-motion capture

What runs where:

* YOLOE-26s (prompt "coin") names objects in its own ProcessProvider worker at
  about 10 Hz. A label only ever comes from this model.
* GAMST re-finds each object in every capture, fits its ellipse and counts
  edge-on half turns (``vision.object_spin``) and in-plane turns
  (``vision.object_rotations``). It withholds a count as ``undersampled`` when
  the capture rate cannot resolve the spin; a hand-tossed coin turns ~36-40
  times per second and needs at least ~240 fps.
* A user-owned Watch fires when a spin episode ends. Only then is the optional
  VLM consulted, once, with the final crop and the measured episode.

No raw frame is stored. ``--store`` keeps the derived records in SQLite.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from urllib.request import Request, urlopen

MODEL_ASSETS = {
    'yoloe-26s-seg.pt': ('https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26s-seg.pt',
                         '48f24206bc8680d60cbbfa296b0140da849669b9515058b72f5a945142df0654'),
    'mobileclip2_b.ts': ('https://github.com/ultralytics/assets/releases/download/v8.4.0/mobileclip2_b.ts',
                         '35d7f213e4d75f38514e4656ad3cb91158bd33e3805d8ac349f23b186f66982f'),
}
DEFAULT_MODELS = Path(os.environ.get('GAMST_MODELS', Path.home() / '.cache' / 'gamst' / 'models')) / 'yoloe'
MAX_DETECTOR_PIXELS = 1_048_576  # YOLOEContext input bound


def _tls():
    """Prefer certifi's CA bundle (installed with the coin extra); python.org macOS builds ship none."""
    import ssl
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def fetch_models(directory: Path) -> None:
    """Explicit download of the pinned detector assets; every byte is sha256-checked."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, (url, digest) in MODEL_ASSETS.items():
        target = directory / name
        if target.is_file() and _sha256(target) == digest:
            print(f'{name}: present, sha256 verified')
            continue
        print(f'{name}: downloading {url}')
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as handle, urlopen(url, timeout=60, context=_tls()) as response:
            hasher = hashlib.sha256()
            while chunk := response.read(1 << 20):
                hasher.update(chunk)
                handle.write(chunk)
        if hasher.hexdigest() != digest:
            os.unlink(handle.name)
            raise SystemExit(f'{name}: sha256 mismatch, download discarded')
        os.replace(handle.name, target)
        print(f'{name}: sha256 verified')


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, 'rb') as handle:
        while chunk := handle.read(1 << 20):
            hasher.update(chunk)
    return hasher.hexdigest()


def detector_config(models: Path, labels: list[str]) -> dict:
    missing = [name for name in MODEL_ASSETS if not (models / name).is_file()]
    if missing:
        raise SystemExit(f'missing detector weights {missing} in {models}; run: gamst demo coin --fetch-models')
    runtime = models.parent / 'runtime'
    runtime.mkdir(parents=True, exist_ok=True)
    return dict(checkpoint_path=str(models / 'yoloe-26s-seg.pt'), checkpoint_sha256=MODEL_ASSETS['yoloe-26s-seg.pt'][1],
                runtime_root=str(runtime), mode='text', labels=labels,
                encoder_path=str(models / 'mobileclip2_b.ts'), encoder_sha256=MODEL_ASSETS['mobileclip2_b.ts'][1],
                encoder_family='mobileclip2:b', device='cpu', imgsz=640, threshold=.25, max_objects=16)


def _check_worker_dependencies(python: str) -> None:
    """Name the exact missing install instead of the worker's generic failure."""
    import subprocess
    probe = ('import json\nfrom importlib.metadata import distribution,version\n'
             'print(json.dumps({"ultralytics":version("ultralytics"),"clip":json.loads(distribution("clip").read_text("direct_url.json") or "{}").get("vcs_info",{}).get("commit_id")}))')
    result = subprocess.run([python, '-c', probe], capture_output=True, text=True)
    found = json.loads(result.stdout) if result.returncode == 0 else {}
    from ..providers.yoloe_native import CLIP_REVISION
    if found.get('clip') != CLIP_REVISION or not found.get('ultralytics'):
        raise SystemExit('the YOLOE detector worker needs:\n'
                         f'  {python} -m pip install "gamst-sdk[coin]" '
                         f'"git+https://github.com/ultralytics/CLIP.git@{CLIP_REVISION}"\n'
                         'or run without labels: gamst demo coin --detector round')


def _detector_image(frame):
    import cv2
    height, width = frame.shape[:2]
    scale = min(1., (MAX_DETECTOR_PIXELS / (height * width)) ** .5, 2048 / max(height, width))
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame


def ask_vlm(endpoint, model, key_env, crop, episode):
    """One OpenAI-compatible chat call with the final crop and the measured episode."""
    import cv2
    ok, jpeg = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError('could not encode crop')
    prompt = ('GAMST measured this spin episode (JSON, untrusted sensor data): ' + json.dumps(episode) +
              '\nLook at the image of the object after it stopped. In one short sentence: what is it, '
              'and which face is up (heads/tails/unclear)? Do not estimate the turn count yourself.')
    body = dict(model=model, max_tokens=120, messages=[dict(role='user', content=[
        dict(type='text', text=prompt),
        dict(type='image_url', image_url=dict(url='data:image/jpeg;base64,' + base64.b64encode(jpeg.tobytes()).decode()))])])
    headers = {'Content-Type': 'application/json'}
    if key_env:
        headers['Authorization'] = 'Bearer ' + os.environ[key_env]
    started = time.perf_counter()
    with urlopen(Request(endpoint, json.dumps(body).encode(), headers), timeout=60) as response:
        reply = json.loads(response.read())
    return reply['choices'][0]['message']['content'].strip(), round((time.perf_counter() - started) * 1000)


class Panel:
    """OpenCV presentation only; every number shown comes from a committed GAMST record."""

    def __init__(self):
        self.alarms: list[dict] = []
        self.notes: list[str] = []

    def draw(self, frame, result, fps):
        import cv2
        import numpy as np
        spin = {o['track_id']: o for o in result['spin']['objects']}
        rotation = {o['track_id']: o for o in result['rotation']['objects']}
        for obj in result['objects']:
            x, y, s, _ = obj['box']
            color = (60, 220, 60) if obj['observed'] else (0, 160, 255)
            cv2.rectangle(frame, (x, y), (x + s, y + s), color, 2)
            state = spin.get(obj['track_id'], {})
            if state.get('status') in ('counting', 'spinning') and state.get('turns') is not None:
                text = f"{state['turns']:.1f} turns ({state['rev_s'] or 0:.1f} rev/s)"
            elif state.get('status') == 'undersampled':
                text = f"spinning; camera counts only <= {state['max_countable_rev_s']} rev/s"
            else:
                turns = rotation.get(obj['track_id'], {}).get('rotation_count', 0)
                text = f'{turns:.2f} in-plane turns' if turns >= .25 else 'watching'
            cv2.putText(frame, f"{obj['label']}: {text}", (max(0, x), max(18, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
        import textwrap
        height = frame.shape[0]
        panel = np.full((height, 460, 3), 24, np.uint8)
        lines = ['GAMST Coin Halo', f'capture {fps:.0f} fps | record {result["spin_record"]["id"][:8]}', '']
        for state in spin.values():
            lines.append(f"{state['track_id']}: {state['status']}"
                         + (f" {state['turns']} turns" if state['turns'] is not None else ''))
        lines += ['', 'Watch alarms (spin ended):']
        for alarm in self.alarms[-5:]:
            lines += textwrap.wrap(alarm, 54, initial_indent='- ', subsequent_indent='  ')
        lines.append('')
        for note in self.notes[-3:]:
            lines += textwrap.wrap(note, 54)
        for index, line in enumerate(lines[:max(1, height // 22 - 1)]):
            cv2.putText(panel, line, (10, 26 + 22 * index), cv2.FONT_HERSHEY_SIMPLEX, .5,
                        (255, 255, 255) if index else (80, 220, 255), 1)
        return np.hstack([frame, panel])


def episode_text(event):
    seconds = event['duration_ms'] / 1000
    if event['status'] == 'counting':
        text = f"{event['turns']:.1f} turns in {seconds:.2f}s at {event['capture_fps']} fps"
    elif event['status'] == 'undersampled':
        text = (f"spin too fast for {event['capture_fps']} fps (countable up to "
                f"{event['max_countable_rev_s']} rev/s)")
    else:
        text = 'spin too short to count'
    return text + f"; a 1 fps VLM would see {int(seconds)} frame(s) of it"


async def run(args):
    import cv2
    from .. import Gamst, ProcessProvider
    from ..capabilities import SensorMetadata
    from ..plugins import InputFrame
    from ..providers.shape_tracks import round_candidates
    from ..usecases import CoinWowUseCase

    source = args.video if args.video else args.camera
    cap = cv2.VideoCapture(source, cv2.CAP_ANY if args.video else cv2.CAP_AVFOUNDATION if sys.platform == 'darwin' else cv2.CAP_ANY)
    if not cap.isOpened():
        raise SystemExit(f'cannot open input: {source}')
    detector = None
    if args.detector == 'yoloe':
        _check_worker_dependencies(args.worker_python)
        detector = await ProcessProvider.open(args.worker_python, 'yoloe.local',
                                              detector_config(Path(args.models), args.labels))
    report = dict(source=str(source), detector=args.detector, frames=0, episodes=[], alarms=0, vlm=[])
    panel = Panel()
    try:
        async with Gamst(consent=True, store=args.store) as human:
            usecase = CoinWowUseCase(human)
            watch = human.watches.create(owner='user', description='When a coin stops spinning, tell me how many turns it made.',
                rule=dict(conditions=[dict(capability_id='vision.object_spin', path='events', op='contains',
                                           value='object.spin.ended', within_ms=1000)],
                          cooldown_ms=0, max_per_minute=60, context_window_ms=3000))
            pending: list[dict] = []
            human.watches.add_sink(pending.append)
            fps_video = cap.get(cv2.CAP_PROP_FPS) if args.video else 0
            start_wall = time.time_ns() // 1_000_000
            start = time.monotonic()
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) if args.video else 0
            base = start_wall - (round(1000 * frame_count / (args.capture_fps or fps_video)) + 1000 if args.video else 0)
            in_flight = None
            last_detect_ts = None
            seq, last_ts, frame_times = 0, None, []
            while True:
                ok, frame = cap.read()
                if not ok:
                    if args.video and seq:
                        break
                    raise SystemExit('no camera/video frame available')
                if args.video:
                    offset = (seq * 1000 / args.capture_fps) if args.capture_fps else cap.get(cv2.CAP_PROP_POS_MSEC)
                    stamp = base + round(offset)
                else:
                    stamp = base + round(1000 * (time.monotonic() - start))
                if last_ts is not None and stamp <= last_ts:
                    stamp = last_ts + 1
                if last_ts is not None:
                    frame_times.append(stamp - last_ts)
                last_ts = stamp
                sensor = SensorMetadata(source_id='video-file' if args.video else 'camera', sensor_id='coin-halo',
                                        sequence=seq, stream_epoch=str(base), coordinate_frame='image')
                due = last_detect_ts is None or stamp - last_detect_ts >= args.detect_every_ms
                if detector is None and due:
                    usecase.detections(round_candidates(frame), image_shape=frame.shape, timestamp_ms=stamp)
                    last_detect_ts = stamp
                elif detector is not None:
                    if in_flight is not None and in_flight[0].done():
                        task, shot_ts, shape = in_flight
                        usecase.detections(task.result()['objects'], image_shape=shape, timestamp_ms=shot_ts)
                        in_flight = None
                    if due and in_flight is None:
                        shot = InputFrame(_detector_image(frame), 'video', stamp, subject_id='local-user', sensor=sensor)
                        task = asyncio.ensure_future(detector.process(shot))
                        in_flight = (task, stamp, frame.shape)
                        last_detect_ts = stamp
                        if args.video:  # files are processed deterministically, not against a wall clock
                            await task
                result = await usecase.process_frame(frame, subject_id='local-user', timestamp_ms=stamp, sensor=sensor)
                for event in result['spin']['events']:
                    if event['type'] == 'object.spin.ended':
                        report['episodes'].append(event)
                while pending:
                    alarm = pending.pop(0)
                    report['alarms'] += 1
                    ended = [e for r in alarm['context']['records'] if r['id'] == alarm['trigger_record_id']
                             for e in r['observation']['value']['events'] if e['type'] == 'object.spin.ended']
                    for event in ended:
                        panel.alarms.append(episode_text(event))
                        print(json.dumps({'watch': watch['id'], 'alarm': alarm['id'], 'episode': event}), flush=True)
                        if args.vlm_endpoint:
                            obj = next((o for o in result['objects'] if o['track_id'] == event['track_id']), None)
                            if obj is not None:
                                x, y, s, _ = obj['box']
                                crop = frame[max(0, y - s // 2):y + s + s // 2, max(0, x - s // 2):x + s + s // 2]
                                try:
                                    answer, ms = await asyncio.to_thread(ask_vlm, args.vlm_endpoint, args.vlm_model,
                                                                         args.vlm_key_env, crop, event)
                                    panel.notes.append(f'VLM ({ms} ms): {answer}')
                                    report['vlm'].append(dict(episode=event, answer=answer, latency_ms=ms))
                                except Exception as exc:  # report, never invent an answer
                                    panel.notes.append(f'VLM call failed: {type(exc).__name__}')
                                    report['vlm'].append(dict(episode=event, error=str(exc)[:200]))
                fps = 1000 / (sum(frame_times[-30:]) / len(frame_times[-30:])) if frame_times else 0
                if not args.headless:
                    cv2.imshow('GAMST Coin Halo', panel.draw(frame, result, fps))
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord('q')):
                        break
                    if key == ord('r'):
                        usecase.reset_capture()
                seq += 1
                report['frames'] = seq
                if args.frames and seq >= args.frames:
                    break
                await asyncio.sleep(0)
            report['capture_fps'] = round(1000 / (sum(frame_times) / len(frame_times)), 1) if frame_times else None
            usecase.close()
    finally:
        cap.release()
        if detector is not None:
            await detector.close()
        if not args.headless:
            cv2.destroyAllWindows()
    if args.report:
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: report[k] for k in ('frames', 'capture_fps', 'alarms')} |
                     {'episodes': len(report['episodes'])}), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog='gamst demo coin', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--camera', type=int, default=0)
    source.add_argument('--video', help='recorded file; capture timestamps come from the file')
    parser.add_argument('--capture-fps', type=float, help='override file timing (e.g. 240 for a slowed-down slow-motion export)')
    parser.add_argument('--detector', choices=['yoloe', 'round'], default='yoloe',
                        help="yoloe: semantic labels (default); round: detector-free 'round candidate' geometry")
    parser.add_argument('--labels', type=lambda s: [x.strip() for x in s.split(',') if x.strip()], default=['coin'])
    parser.add_argument('--models', default=str(DEFAULT_MODELS))
    parser.add_argument('--fetch-models', action='store_true', help='download the pinned detector weights and exit')
    parser.add_argument('--worker-python', default=sys.executable, help='Python with gamst-sdk[yoloe] for the detector worker')
    parser.add_argument('--detect-every-ms', type=int, default=100)
    parser.add_argument('--store', default=':memory:', help='SQLite path for derived records only')
    parser.add_argument('--vlm-endpoint', help='OpenAI-compatible chat/completions URL, called once per finished spin')
    parser.add_argument('--vlm-model', default='gemini-3.8-flash')
    parser.add_argument('--vlm-key-env', help='environment variable holding the bearer key')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--frames', type=int)
    parser.add_argument('--report', help='write a JSON summary of episodes and alarms')
    args = parser.parse_args(argv)
    if args.fetch_models:
        fetch_models(Path(args.models))
        return
    asyncio.run(run(args))


if __name__ == '__main__':
    main()

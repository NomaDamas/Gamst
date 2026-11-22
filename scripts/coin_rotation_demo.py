"""Local camera or recorded video → GAMST perception → stored evidence overlay.

    PYTHONPATH=. python scripts/coin_rotation_demo.py --camera 0 --no-semantic
    PYTHONPATH=. python scripts/coin_rotation_demo.py --video spin.mp4 \
        --reference coin=coin-face.jpg --reference wheel=wheel-face.jpg --store proof.sqlite

References are tightly cropped photos of the actual props. Without them the
SDK detects round candidates only. Counts are net image-plane turns within a
continuous visible track, not the number of 3-D flips of an airborne coin.
"""
from __future__ import annotations

import argparse
import asyncio
import time

import cv2

from gamst import Gamst
from gamst.capabilities import SensorMetadata
from gamst.usecases import CoinWowUseCase


def references(entries):
    images = {}
    for entry in entries:
        label, sep, path = entry.partition('=')
        if not sep or not label or label in images:
            raise ValueError('--reference requires a unique label=path')
        photo = cv2.imread(path)
        if photo is None:
            raise ValueError(f'cannot read reference: {path}')
        images[label] = photo
    return images


def draw(frame, result):
    record = result['rotation_record']
    regions = {r['track_id']: r for r in result['tracks_record']['observation']['value']['regions']}
    for obj in record['observation']['value']['objects']:
        region = regions[obj['track_id']]
        x, y, w, h = region['box']
        cv2.rectangle(frame, (x,y), (x+w,y+h), (30,200,200), 2)
        turns = f"{obj['rotation_count']:.2f} turns" if obj['confidence'] >= .65 else 'insufficient evidence'
        text = f"{region['label']} {obj['track_id']}: {turns}"
        event_record = result['event_records'].get(obj['track_id'])
        if event_record:
            event = event_record['observation']['value']
            text += f" | {event['face']} | {event['flip_count']} face changes"
        cv2.putText(frame, text, (max(0,x),max(40,y-4)), cv2.FONT_HERSHEY_SIMPLEX, .48, (30,200,200), 1)
    caption = f"GAMST SDK | {record['observation']['capability']['id']} | {record['id'][:12]} | {len(regions)} tracks"
    cv2.putText(frame, caption, (8,22), cv2.FONT_HERSHEY_SIMPLEX, .58, (80,240,80), 2)


async def run(args):
    photos = references(args.reference)
    source = args.video if args.video else args.camera
    cap = cv2.VideoCapture(source, cv2.CAP_ANY if args.video else cv2.CAP_AVFOUNDATION)
    try:
        if not cap.isOpened():
            raise RuntimeError(f'cannot open input: {source}')
        if not args.video and args.width and args.height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        async with Gamst(consent=True, store=args.store, sensor_inference_interval_ms=500) as human:
            usecase = CoinWowUseCase(human, references=photos, rotation_options={'min_quality': .65})
            if args.vlm_endpoint:
                from gamst.realtime import LocalScenePlugin, QueuedPlugin
                human.register(QueuedPlugin(human, LocalScenePlugin(args.vlm_endpoint, args.vlm_model, human=human), capacity=1))
            try:
                seq = 0
                base = time.time_ns() // 1_000_000
                start = time.monotonic()
                # File timestamps retain source cadence, anchored in the past for Store queries.
                fps = cap.get(cv2.CAP_PROP_FPS)
                if args.video and not 0 < fps < 1000:
                    raise ValueError('video has no usable frame rate')
                if args.video:
                    base -= round(1000 * cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        if seq and args.video:
                            break
                        raise RuntimeError('no camera/video frame available')
                    stamp = base + round(1000*(seq/fps if args.video else time.monotonic()-start))
                    sensor = SensorMetadata(source_id='video-file' if args.video else 'mac-camera',
                                            sensor_id='coin-wow', sequence=seq, stream_epoch=str(base))
                    if args.vlm_endpoint:
                        await human.push_video(frame, timestamp_ms=stamp)
                    result = await usecase.process_frame(frame, subject_id='local-user', timestamp_ms=stamp, sensor=sensor)
                    draw(frame, result)
                    cv2.imshow('GAMST Coin + Wheel | image-plane rotation', frame)
                    if seq == 0:
                        print(f"SDK record {result['rotation_record']['id']} committed; preview ready", flush=True)
                    seq += 1
                    if args.frames and seq >= args.frames:
                        cv2.waitKey(1)
                        break
                    key = cv2.waitKey(1) & 0xff
                    if key in (27, ord('q')):
                        break
                    if key == ord('r'):
                        usecase.reset_capture()
                    await asyncio.sleep(0)
            finally:
                usecase.close()
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument('--camera', type=int, default=0)
    sources.add_argument('--video')
    parser.add_argument('--frames', type=int, help='stop preview after this many frames')
    parser.add_argument('--width', type=int, help='requested camera width (use with --height)')
    parser.add_argument('--height', type=int, help='requested camera height (use with --width)')
    parser.add_argument('--reference', action='append', default=[], metavar='LABEL=PHOTO')
    parser.add_argument('--store', default=':memory:', help='optional SQLite path for derived SDK evidence only')
    semantic = parser.add_mutually_exclusive_group()
    semantic.add_argument('--no-semantic', action='store_true', help='local only (default)')
    semantic.add_argument('--vlm-endpoint', help='optional loopback scene-description endpoint')
    parser.add_argument('--vlm-model', default='qwen3.5-9b')
    asyncio.run(run(parser.parse_args()))

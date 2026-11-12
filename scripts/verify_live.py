"""Consented hardware smoke test. Reports measurements; does not save raw media."""
import argparse
import asyncio
import json
import time
from collections import Counter

from humansense import HumanSense
from humansense.capture import CaptureSession
from humansense.plugins import AudioLevelPlugin, OpenCVPersonPlugin, MediaPipeFacePlugin, FasterWhisperPlugin, SileroVADPlugin


async def verify(args):
    async with HumanSense(consent=args.consent) as human:
        if args.camera is not None:
            human.register(MediaPipeFacePlugin(args.face_model) if args.face_model else OpenCVPersonPlugin())
        if args.audio:
            human.register(AudioLevelPlugin())
            if args.vad_model:
                human.register(SileroVADPlugin(args.vad_model))
            if args.asr_model:
                human.register(FasterWhisperPlugin(args.asr_model))
        counts = Counter()
        async def collect():
            async for record in human.signals():
                counts[record['observation']['name']] += 1
        consumer = asyncio.create_task(collect())
        await asyncio.sleep(0)
        session = CaptureSession(human, consent=args.consent, video=args.camera, audio=args.audio)
        started = time.monotonic()
        try:
            await session.start()
            await asyncio.sleep(args.seconds)
            active_status = session.status()
            state = await human.get_state()
        finally:
            await session.stop()
            consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)
        elapsed = time.monotonic() - started
        required = []
        if args.camera is not None:
            required.append('scene.faces' if args.face_model else 'scene.person_count')
        if args.audio:
            required.append('audio.rms')
        if args.vad_model:
            required.append('speech.activity')
        # A silent session may correctly produce no ASR transcript.
        passed = not active_status['errors'] and all(counts[name] > 0 for name in required)
        report = {'passed': passed, 'elapsed_s': round(elapsed, 3),
                  'signal_counts': dict(counts), 'capture': active_status,
                  'modalities': state['modalities'], 'plugin_errors': state['plugin_errors'],
                  'raw_media_saved': False,
                  'notes': ['Speech transcript absence can reflect silence.',
                            'Detection count zero is a measurement, not proof that no person is present.',
                            'This smoke test does not establish emotional/intent accuracy or identity.']}
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if passed else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--consent', action='store_true', required=True)
    parser.add_argument('--camera', type=int)
    parser.add_argument('--audio', action='store_true')
    parser.add_argument('--seconds', type=float, default=30)
    parser.add_argument('--face-model')
    parser.add_argument('--asr-model')
    parser.add_argument('--vad-model')
    args = parser.parse_args()
    if not (args.camera is not None or args.audio):
        parser.error('Select --camera and/or --audio')
    if not 1 <= args.seconds <= 28800:
        parser.error('--seconds must be between 1 and 28800')
    try:
        raise SystemExit(asyncio.run(verify(args)))
    except KeyboardInterrupt:
        raise SystemExit(130)

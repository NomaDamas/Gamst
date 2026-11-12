"""Consented hardware smoke test. Reports measurements; does not save raw media."""
import argparse
import asyncio
import json
import time
import platform
from pathlib import Path
from collections import Counter

from humansense import HumanSense
from humansense.capture import CaptureSession
from humansense.plugins import AudioLevelPlugin, OpenCVPersonPlugin, MediaPipeFacePlugin, FasterWhisperPlugin, SileroVADPlugin


async def verify(args):
    async with HumanSense(consent=args.consent) as human:
        if args.output and Path(args.output).exists():
            raise ValueError("Output exists; choose a new report filename")
        if args.camera is not None:
            if args.yolo_model:
                from humansense.yolo import UltralyticsYOLOPlugin
                human.register(UltralyticsYOLOPlugin(args.yolo_model))
            elif args.onnx_model:
                from humansense.onnx import ONNXPersonPlugin
                human.register(ONNXPersonPlugin(args.onnx_model))
            else:
                human.register(OpenCVPersonPlugin())
            if args.face_model:
                human.register(MediaPipeFacePlugin(args.face_model))
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
            active_elapsed = time.monotonic()-started
            active_status = session.status()
            state = await human.get_state()
            runtime_metrics = await human.get_metrics()
        finally:
            await session.stop()
            consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)
        elapsed = time.monotonic() - started
        required = []
        if args.camera is not None:
            required.append('scene.person_count')
            if args.face_model:required.append('scene.faces')
        if args.audio:
            required.append('audio.rms')
        if args.vad_model:
            required.append('speech.activity')
        # A silent session may correctly produce no ASR transcript.
        passed = not active_status['errors'] and all(counts[name] > 0 for name in required)
        report = {'passed': passed, 'elapsed_s': round(elapsed, 3),
                  'signal_counts': dict(counts), 'capture': active_status,
                  'runtime_metrics': runtime_metrics,
                  'effective_video_fps': active_status['video_frames']/active_elapsed,
                  'active_capture_seconds':active_elapsed,
                  'environment': {'python': platform.python_version(), 'platform': platform.platform()},
                  'models': [{'name':getattr(p,'name',type(p).__name__),
                              'asset_sha256':getattr(p,'model_hash',None)} for p in human.plugins],
                  'modalities': state['modalities'], 'plugin_errors': state['plugin_errors'],
                  'raw_media_saved': False,
                  'notes': ['Speech transcript absence can reflect silence.',
                            'Detection count zero is a measurement, not proof that no person is present.',
                            'This smoke test does not establish emotional/intent accuracy or identity.']}
        stages = runtime_metrics['latency']['stages']
        report['latency_checks'] = {}
        for stage, target in (('sdk.get_state',50), ('sdk.push_video',200), ('sdk.push_audio',200)):
            value = stages.get(stage)
            report['latency_checks'][stage] = {'target_p95_ms':target,
                'status':'not_measured' if not value else 'within_target' if value['p95_ms'] <= target else 'exceeds_target',
                'observed_p95_ms':value['p95_ms'] if value else None,
                'sample_count':value['count'] if value else 0}
        report['asr_end_of_speech_latency'] = {'status':'not_measured',
            'reason':'Windowed ASR processing time is not end-of-speech-to-transcript latency'}
        if args.output:
            with Path(args.output).open('x',encoding='utf-8') as handle:
                json.dump(report,handle,ensure_ascii=False,indent=2)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if passed else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--consent', action='store_true', required=True)
    parser.add_argument('--camera', type=int)
    parser.add_argument('--audio', action='store_true')
    parser.add_argument('--seconds', type=float, default=30)
    parser.add_argument('--face-model')
    parser.add_argument('--onnx-model')
    parser.add_argument('--yolo-model')
    parser.add_argument('--asr-model')
    parser.add_argument('--output',help='Write a new JSON report; refuses to overwrite')
    parser.add_argument('--vad-model')
    args = parser.parse_args()
    if args.onnx_model and args.yolo_model:parser.error('Select ONNX or YOLO, not both')
    if not (args.camera is not None or args.audio):
        parser.error('Select --camera and/or --audio')
    if not 1 <= args.seconds <= 28800:
        parser.error('--seconds must be between 1 and 28800')
    try:
        raise SystemExit(asyncio.run(verify(args)))
    except KeyboardInterrupt:
        raise SystemExit(130)

"""Local camera/mic event stream; models and dependencies installed separately.

python examples/realtime_capture.py --consent --camera 0 --audio
"""
import argparse
import asyncio
from humansense import HumanSense
from humansense.capture import CaptureSession
from humansense.plugins import AudioLevelPlugin, OpenCVPersonPlugin, SileroVADPlugin, FasterWhisperPlugin, MediaPipeFacePlugin


async def run(args):
    async with HumanSense(consent=args.consent) as human:
        if args.camera is not None:
            human.register(MediaPipeFacePlugin(args.face_model) if args.face_model else OpenCVPersonPlugin())
        if args.audio:
            human.register(AudioLevelPlugin())
            if args.vad_model:
                human.register(SileroVADPlugin(args.vad_model))
            if args.asr_model:
                human.register(FasterWhisperPlugin(args.asr_model))
        async def watch():
            async for event in human.events():
                print(event["event"], event["observation"], flush=True)
        consumer = asyncio.create_task(watch())
        await asyncio.sleep(0)  # Subscribe before starting capture.
        try:
            async with CaptureSession(human, video=args.camera, audio=args.audio, consent=args.consent) as capture:
                print("Capture active. Ctrl-C to stop. Raw media is not saved.", flush=True)
                await capture.wait()
        finally:
            consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--consent", action="store_true", required=True)
    parser.add_argument("--camera", type=int)
    parser.add_argument("--audio", action="store_true")
    parser.add_argument("--face-model")
    parser.add_argument("--vad-model")
    parser.add_argument("--asr-model")
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass

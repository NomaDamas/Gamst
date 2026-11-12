"""Real model fixture QA; no device access. Place official fixtures in .local-run/qa first."""
import asyncio
import json
from pathlib import Path
import time

async def main():
    import cv2
    from faster_whisper.audio import decode_audio
    from humansense.plugins import MediaPipeFacePlugin, FasterWhisperPlugin, SileroVADPlugin, InputFrame
    from humansense.realtime import LocalScenePlugin
    from humansense import HumanSense
    from humansense.companion import Companion, LocalChatBackend
    root = Path(__file__).resolve().parents[1]
    assets = root / '.local-models'
    qa = root / '.local-run/qa'
    report = {}
    image = cv2.imread(str(qa / 'portrait.jpg'))
    if image is None:
        raise ValueError('Missing official MediaPipe portrait.jpg fixture')
    frame = InputFrame(image, 'video', time.time_ns() // 1000000)
    face = MediaPipeFacePlugin(str(assets / 'face_landmarker.task'))
    t = time.monotonic()
    faces = await face.process(frame)
    report['face'] = {'count': len(faces[0]['value']), 'latency_ms': round((time.monotonic()-t)*1000),
                      'blendshapes': len(faces[0]['value'][0]['blendshapes']) if faces[0]['value'] else 0}
    await face.close()
    audio = decode_audio(str(qa / 'jfk.flac'), sampling_rate=16000)
    vad = SileroVADPlugin(str(assets / 'silero_vad.jit'))
    t = time.monotonic()
    signals = await vad.process(InputFrame(audio, 'audio', time.time_ns() // 1000000, sample_rate=16000))
    report['vad'] = {'signals': len(signals), 'latency_ms': round((time.monotonic()-t)*1000)}
    asr = FasterWhisperPlugin(str(assets / 'whisper-small'), window_seconds=3)
    for name in ('jfk.flac', 'korean.aiff'):
        if not (qa / name).exists():
            continue
        audio = decode_audio(str(qa / name), sampling_rate=16000)
        # Flush a final partial window with silence; this is an offline fixture test.
        import numpy as np
        pad = (-len(audio)) % 48000
        audio = np.pad(audio, (0, pad))
        asr.reset()
        asr.language = 'ko' if name == 'korean.aiff' else 'en'
        t = time.monotonic()
        signals = await asr.process(InputFrame(audio, 'audio', time.time_ns() // 1000000, sample_rate=16000))
        texts = [x['value'] for x in signals if x['name'] == 'text.transcript']
        report['asr_' + name] = {'transcripts': texts, 'audio_seconds': len(audio)/16000,
                                 'latency_ms': round((time.monotonic()-t)*1000)}
    scene = LocalScenePlugin('http://127.0.0.1:18767/v1/chat/completions', 'local-vlm')
    t = time.monotonic()
    result = await scene.process(frame)
    report['vlm'] = {'description': result[0]['value'], 'latency_ms': round((time.monotonic()-t)*1000)}
    async with HumanSense(consent=True) as human:
        await human.push_signal('scene.description', result[0]['value'], 'video')
        companion = Companion(human, LocalChatBackend('http://127.0.0.1:18767/v1/chat/completions', 'local-vlm'))
        t = time.monotonic()
        out = await companion.reply('한국어로 짧게 인사해 줘.')
        report['companion'] = {'reply': out['assistant_text'], 'latency_ms': round((time.monotonic()-t)*1000)}
    report['passed'] = report['face']['count'] > 0 and bool(report['asr_jfk.flac']['transcripts']) and bool(report['vlm']['description']) and bool(report['companion']['reply'])
    (qa / 'model-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report['passed']:
        raise SystemExit(1)

if __name__ == '__main__':
    asyncio.run(main())

"""Measure real installed model adapters on an explicit image and/or WAV asset.

No model or input is downloaded. Supplied media requires participant consent.
This measures input-block processing, not end-of-speech-to-transcript delay.
"""
import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
import wave

from humansense import HumanSense


def file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


async def run(args):
    plugins=[]
    if args.yolo_model:
        from humansense.yolo import UltralyticsYOLOPlugin
        plugins.append(UltralyticsYOLOPlugin(args.yolo_model))
    if args.onnx_model:
        from humansense.onnx import ONNXPersonPlugin
        plugins.append(ONNXPersonPlugin(args.onnx_model))
    if args.face_model:
        from humansense.plugins import MediaPipeFacePlugin
        plugins.append(MediaPipeFacePlugin(args.face_model))
    if args.vad_model:
        from humansense.plugins import SileroVADPlugin
        plugins.append(SileroVADPlugin(args.vad_model))
    if args.asr_model:
        from humansense.plugins import FasterWhisperPlugin
        plugins.append(FasterWhisperPlugin(args.asr_model))
    if not plugins:raise ValueError('Select at least one actual local model')
    if any('video' in p.modalities for p in plugins) and not args.image:
        raise ValueError('Vision models require --image')
    if any('audio' in p.modalities for p in plugins) and not args.wav:
        raise ValueError('Audio models require --wav')
    image=None
    if args.image:
        import cv2
        image=cv2.imread(args.image)
        if image is None:raise ValueError('Image decode failed')
    audio=[]
    if args.wav:
        with wave.open(args.wav,'rb') as handle:
            if (handle.getframerate(),handle.getnchannels(),handle.getsampwidth())!=(16000,1,2):
                raise ValueError('WAV must be 16kHz mono signed PCM16')
            if handle.getnframes() > 16000*600:raise ValueError('Use a WAV of at most 10 minutes')
            while block:=handle.readframes(1600):audio.append(block)
        if not audio:raise ValueError('WAV has no samples')
    counts=Counter()
    begin=time.perf_counter()
    async with HumanSense(consent=True) as human:
        for plugin in plugins:human.register(plugin)
        for _ in range(args.repeats):
            if image is not None:
                records=await human.push_video(image)
                for record in records:counts[record.get('observation',{}).get('name',record['type'])]+=1
            for block in audio:
                records=await human.push_audio(block,sample_rate=16000)
                for record in records:counts[record.get('observation',{}).get('name',record['type'])]+=1
                await asyncio.sleep(0)
        await human.get_state()
        metrics=await human.get_metrics()
        errors=list(human.plugin_errors)
        models=[{'name':p.name,'asset_sha256':getattr(p,'model_hash',None)} for p in plugins]
    return {'generated_at':datetime.now(timezone.utc).isoformat(), 'workload':'actual_model_explicit_media_replay',
            'python':platform.python_version(),'platform':platform.platform(),
            'elapsed_seconds':time.perf_counter()-begin,'repeats':args.repeats,
            'inputs':{key:{'sha256':file_hash(value)} for key,value in (('image',args.image),('wav',args.wav)) if value},
            'models':models,'signal_counts':dict(counts),'runtime_metrics':metrics,'errors':errors,
            'limitations':['Repeated image is not live tracking accuracy','No raw media or transcript contents included in this report',
                'Audio block completion time is not end-of-speech latency','No camera, microphone, transport or eight-hour soak validation']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--consent',action='store_true',required=True)
    for name in ('image','wav','yolo-model','onnx-model','face-model','vad-model','asr-model'):parser.add_argument('--'+name)
    parser.add_argument('--repeats',type=int,default=10)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not 1<=args.repeats<=10000:parser.error('repeats must be in 1..10000')
    if args.output.exists():parser.error('Output exists')
    try:report=asyncio.run(run(args))
    except (ImportError,RuntimeError,ValueError,OSError) as exc:parser.exit(2,f'Model benchmark not completed: {exc}\n')
    rendered=json.dumps(report,indent=2,ensure_ascii=False)
    with args.output.open('x') as handle:handle.write(rendered+'\n')
    print(rendered)


if __name__=='__main__':main()

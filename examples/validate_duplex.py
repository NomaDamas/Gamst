"""Explicit paired PCM16 WAV AEC smoke; no devices/downloads or acoustic acceptance."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import time
import wave

from humansense import SensorMetadata
from humansense.plugins import InputFrame
from humansense.providers.duplex import WebRTCDuplex


def read_pcm(path):
    import numpy as np
    with Path(path).open('rb') as source:payload=source.read(16*1024*1024+1)
    if not 0<len(payload)<=16*1024*1024:raise ValueError('WAV must be1..16MiB')
    with wave.open(io.BytesIO(payload),'rb') as source:
        rate=source.getframerate();count=source.getnframes()
        if (source.getnchannels()!=1 or source.getsampwidth()!=2 or source.getcomptype()!='NONE' or
            rate not in (16000,32000,48000) or not 0<count<=rate*30 or count%(rate//100)):
            raise ValueError('mono PCM16 WAV,16/32/48kHz, at most30s, exact10ms blocks required')
        data=source.readframes(count)
        if len(data)!=count*2:raise ValueError('truncated WAV')
    return rate,np.frombuffer(data,dtype='<i2').astype(np.int16),hashlib.sha256(payload).hexdigest()


def run(args):
    import numpy as np
    from importlib.metadata import version
    rate,near,near_hash=read_pcm(args.microphone);far_rate,far,far_hash=read_pcm(args.render)
    if rate!=far_rate or len(near)!=len(far):raise ValueError('paired captures must have identical rate/length')
    start=time.perf_counter();provider=WebRTCDuplex.from_installed(sample_rate=rate,stream_delay_ms=args.delay_ms)
    report=dict(kind='paired_pcm_smoke_not_acoustic_acceptance',microphone_sha256=near_hash,render_sha256=far_hash,
        runtime=dict(python=platform.python_version(),platform=platform.platform(),binding=version('pywebrtc-audio'),numpy=version('numpy')),
        initialization_ms=(time.perf_counter()-start)*1000,sample_rate=rate,configured_delay_ms=args.delay_ms,
        timestamp_origin='synthetic_contiguous_replay_end_times',pairing_verified=False)
    timings=[];ambient_hash=hashlib.sha256();speech_hash=hashlib.sha256();clipped=0;preserved=True
    try:
        hop=rate//100
        for seq,start in enumerate(range(0,len(near),hop)):
            data=np.column_stack((near[start:start+hop],far[start:start+hop]))
            frame=InputFrame(data,'audio',(seq+1)*10,sample_rate=rate,sensor=SensorMetadata(
                'aec-replay','paired',seq,channels=('microphone','render'),calibration_id='declared-replay-pair'))
            before=time.perf_counter();out=provider.transform(frame);timings.append((time.perf_counter()-before)*1000)
            preserved &= bool(np.array_equal(out.ambient.data,near[start:start+hop].astype(np.float32)/32768))
            ambient_hash.update(out.ambient.data.tobytes());speech_hash.update(out.speech.data.tobytes())
            clipped+=out.context['processing']['output_samples_clipped']
            report['last_context']=out.context
    finally:provider.close()
    report.update(blocks=len(timings),audio_ms=len(timings)*10,first_block_ms=timings[0],
        processing_p50_ms=float(np.percentile(timings,50)),processing_p95_ms=float(np.percentile(timings,95)),
        ambient_preserved=preserved,ambient_float32_sha256=ambient_hash.hexdigest(),
        speech_float32_sha256=speech_hash.hexdigest(),output_samples_clipped=clipped,
        echo_reduction_verified=False,recognition_improvement_verified=False)
    with Path(args.output).open('x') as target:json.dump(report,target,indent=2,allow_nan=False);target.write('\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('microphone','render','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--delay-ms',type=int,default=0)
    run(parser.parse_args())


if __name__=='__main__':main()

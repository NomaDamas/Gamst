"""Voice emotion fixture QA. Uses only public WAV fixtures, never devices.

Requires .local-run/qa/emotion-happy.wav and emotion-neutral.wav from the
SUPERB model card, plus `sh scripts/setup_voice.sh` installation.
"""
import asyncio
import json
import statistics
import time
from pathlib import Path


async def main():
    import torch
    from faster_whisper.audio import decode_audio
    from humansense import HumanSense
    from humansense.extensions import register_config
    from humansense.mcp import MCPServer
    torch.set_num_threads(2)
    config=json.loads(Path('.local-models/voice-emotion/plugins.json').read_text())
    report={'model':'superb/wav2vec2-base-superb-er','revision':config[0]['options']['revision'],'fixtures':[]}
    async with HumanSense(consent=True,sensor_inference_interval_ms=250) as human:
        register_config(human,config)
        worker=human.plugins[0]
        for name in ('emotion-happy.wav','emotion-neutral.wav'):
            await human.delete_data()
            samples=decode_audio('.local-run/qa/'+name,sampling_rate=16000)
            latencies=[]
            started=time.monotonic()
            for i in range(0,len(samples),8000):
                t=time.monotonic()
                await human.push_audio(samples[i:i+8000],sample_rate=16000)
                if worker.task:await worker.task
                if worker.last_latency_ms is not None:latencies.append(worker.last_latency_ms)
            context=await MCPServer(human).call_tool('get_context')
            obs=context['observations']['classification.voice_emotion']
            assert set(obs['value']['labels'])=={'neu','hap','ang','sad'}
            assert all(0 <= v <= 1 for v in obs['value']['labels'].values())
            assert 'samples' not in json.dumps(context)
            report['fixtures'].append(dict(file=name,audio_seconds=round(len(samples)/16000,2),
                labels=obs['value']['labels'],model_latency_ms=round(statistics.median(latencies)),
                elapsed_ms=round((time.monotonic()-started)*1000),evidence_id_present=bool(obs['evidence_id'])))
    report['passed']=True
    Path('.local-run/qa/voice-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':asyncio.run(main())

"""Real temporal VLM QA using the official portrait fixture; no device access. Run from the repository root with PYTHONPATH=. and the local launcher running."""
import asyncio,json,time
from pathlib import Path
import cv2
from humansense import HumanSense
from humansense.plugins import InputFrame,MediaPipeFacePlugin
from humansense.realtime import LocalScenePlugin,QueuedPlugin
from humansense.mcp import MCPServer
async def main():
    async with HumanSense(consent=True) as h:
        image=cv2.imread('.local-run/qa/portrait.jpg')
        p=LocalScenePlugin('http://127.0.0.1:18767/v1/chat/completions','local-vlm',human=h)
        p.interval=60
        h.register(QueuedPlugin(h,p,1))
        face=MediaPipeFacePlugin('.local-models/face_landmarker.task')
        base=int(time.time()*1000)-5000
        for i in range(8):
            f=InputFrame(image,'video',base+i*500)
            await h.ingest_plugin_output(await face.process(f),f,face)
            for name,value in [('audio.rms',.03),('speech.activity',True),('audio.peak',.4)]:
                await h.push_signal(name,value,'audio',timestamp_ms=f.timestamp_ms)
            await p.prepare(f)
        await face.close()
        job=InputFrame({'frames':list(p.frames),'settings':p.settings()},'video',base+7*500)
        start=time.monotonic()
        result=await p.process(job)
        await h.ingest_plugin_output(result,job,p)
        ctx=await MCPServer(h).call_tool('get_context')
        window=ctx['observations']['scene.window']['value']
        assert window['frame_count']==8,window
        assert 'jpeg' not in json.dumps(ctx)
        report={'passed':True,'frame_count':window['frame_count'],'seconds':window['actual_seconds'],'fps':window['actual_fps'],'latency_ms':round((time.monotonic()-start)*1000),'description':ctx['observations']['scene.description']['value'],'numeric_samples':len(window['measurements_series'])}
        Path('.local-run/qa/temporal-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(json.dumps(report,ensure_ascii=False,indent=2))
asyncio.run(main())

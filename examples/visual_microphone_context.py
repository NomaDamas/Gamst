"""Explicit local GVMC001 window -> owned offline worker -> periodic MCP context."""
import argparse
import asyncio
import json
from pathlib import Path

from humansense import HumanSense, ProcessProvider, SensorMetadata, SensorStreams
from humansense.plugins import InputFrame
from humansense.mcp import MCPServer
from humansense.streams import _finish
from humansense.providers.visual_microphone_context import decode,LIMIT


async def deliver(packet,python,interval_ms=100):
    if type(interval_ms) is not int or not 10<=interval_ms<=3600000:
        raise ValueError('context interval must be10..3600000milliseconds')
    meta,_,_,_,end=decode(packet)
    sensor=SensorMetadata(meta['source_id'],meta['sensor_id'],meta['window_sequence'],
        coordinate_frame=meta['coordinate_frame'],calibration_id=meta['config']['calibration_id'],
        capture_sequence=meta['window_sequence'])
    frame=InputFrame(packet,'video',end,meta['subject_id'],sensor=sensor)
    async with HumanSense(consent=True) as human:
        streams=SensorStreams(human)
        spec=ProcessProvider.capability('visual_microphone.phase',meta['config'])
        async def factory():return await ProcessProvider.open(python,'visual_microphone.phase',meta['config'])
        streams.register(spec,factory)
        try:
            await streams.open(sensor.source_id,sensor.sensor_id,[spec.id],subject_id=frame.subject_id)
            server=MCPServer(human)
            query=dict(capability_id=spec.id,subject_id=frame.subject_id,source_id=sensor.source_id,sensor_id=sensor.sensor_id)
            async def infer():
                await streams.submit(frame);await streams.drain()
                lanes=streams.status()['streams'][0]['providers']
                if len(lanes)!=1 or lanes[0]['errors']:raise RuntimeError('visual microphone context worker failed')
            async def publish():
                loop=asyncio.get_running_loop();deadline=loop.time();interval=interval_ms/1000
                while True:
                    print(json.dumps({'context':await server.call_tool('get_observations',query)}))
                    deadline+=interval
                    if deadline<=loop.time():deadline=loop.time()+interval
                    await asyncio.sleep(max(0,deadline-loop.time()))
            tasks=[asyncio.create_task(infer()),asyncio.create_task(publish())]
            try:
                done,_=await asyncio.wait(tasks,return_when=asyncio.FIRST_COMPLETED)
                for task in done:task.result()
                result=await server.call_tool('get_observations',query)
                print(json.dumps({'final_context':result,'offline':True,'capture_end_ms':end}))
                return result
            finally:
                for task in tasks:task.cancel()
                await _finish(asyncio.gather(*tasks,return_exceptions=True))
        finally:await streams.clear()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',required=True)
    parser.add_argument('--python',required=True)
    parser.add_argument('--context-interval-ms',type=int,default=100)
    args=parser.parse_args()
    path=Path(args.packet)
    if not path.is_absolute() or not path.is_file():parser.error('existing absolute packet path required')
    with path.open('rb') as source:packet=source.read(LIMIT+1)
    asyncio.run(deliver(packet,args.python,args.context_interval_ms))


if __name__=='__main__':main()

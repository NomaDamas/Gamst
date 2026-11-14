"""Replay one explicitly supplied PG RGB packet into sensory Context; no LLM call."""
import argparse
import asyncio
import json
from pathlib import Path

from humansense import HumanSense,ProcessProvider,SensorStreams,SensorMetadata
from humansense.plugins import InputFrame
from humansense.mcp import MCPServer
from humansense.pg_generation import PACKET_LIMIT,unpack_capture
from humansense.sam_audio_artifact import _json,_keys


async def deliver(python,config,packet,metadata):
    unpack_capture(packet)
    _keys(metadata,('timestamp_ms','subject_id','sensor'),'PG recorded capture')
    sensor=dict(metadata['sensor']);sensor['channels']=tuple(sensor['channels'])
    frame=InputFrame(packet,'video',metadata['timestamp_ms'],metadata['subject_id'],sensor=SensorMetadata(**sensor))
    async with HumanSense(consent=True) as human:
        streams=SensorStreams(human,capacity=1,max_packet_bytes=32*1024**2,max_buffer_bytes=64*1024**2)
        spec=ProcessProvider.capability('pg.physical.local',config)
        async def factory():return await ProcessProvider.open(python,'pg.physical.local',config)
        streams.register(spec,factory)
        try:
            await streams.open(frame.sensor.source_id,frame.sensor.sensor_id,[spec.id],subject_id=frame.subject_id)
            await streams.submit(frame);await streams.drain()
            if streams.status()['streams'][0]['providers'][0]['errors']:raise RuntimeError('physical provider failed; inspect local runtime/assets')
            result=await MCPServer(human).call_tool('get_observations',dict(capability_id=spec.id,subject_id=frame.subject_id,
                source_id=frame.sensor.source_id,sensor_id=frame.sensor.sensor_id))
            print(json.dumps(result));return result
        finally:await streams.clear()


def read(path,limit):
    with Path(path).open('rb') as stream:value=stream.read(limit+1)
    if len(value)>limit:raise ValueError('bounded replay input required')
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('python','config','packet','metadata'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    asyncio.run(deliver(args.python,_json(read(args.config,8*1024**2),8*1024**2),
        read(args.packet,PACKET_LIMIT),_json(read(args.metadata,65536),65536)))


if __name__=='__main__':main()

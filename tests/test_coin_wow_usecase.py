import asyncio
import math
import time

from gamst.capabilities import SensorMetadata
from gamst.engine import Gamst
from gamst.usecases import CoinWowUseCase


def objects(angle, support=True, velocity=0, face="unknown", face_confidence=0):
    c=[100.,100.]
    points=[[c[0]+30*math.cos(angle+i*math.pi/4),c[1]+30*math.sin(angle+i*math.pi/4)] for i in range(8)]
    return [{"track_id":"coin-1","label":"coin","center":c,"points":points,
             "support_contact":support,"vertical_velocity_px_s":velocity,
             "face":face,"face_confidence":face_confidence}]


def test_coin_wow_commits_rotation_and_event_context_end_to_end():
    async def run():
        async with Gamst(consent=True) as human:
            usecase=CoinWowUseCase(human,event_options={"stable_face_frames":1})
            base = int(time.time() * 1000) - 1000
            for seq in range(10):
                sensor=SensorMetadata(source_id="camera",sensor_id="wow",sequence=seq,stream_epoch="demo")
                await usecase.process(subject_id="local-user",timestamp_ms=base + seq*100,sensor=sensor,
                                       semantic_objects=objects(seq*math.pi/4, support=seq in (0,9),
                                                               velocity=100 if seq==1 else 0,
                                                               face="heads" if seq<5 else "tails", face_confidence=.9))
            rotation=await human.get_observations(capability_id="vision.object_rotations",window="1h",limit=20)
            events=await human.get_observations(capability_id="vision.object_events",window="1h",limit=20)
            return rotation, events
    rotation, events = asyncio.run(run())
    assert rotation["returned_count"] == 10
    assert events["returned_count"] >= 1
    value=rotation["items"][0]["observation"]["value"]
    assert value["objects"][0]["rotation_count"] > .9

from gamst.capabilities import SensorMetadata
from gamst.plugins import InputFrame
from gamst.providers.object_events import ObjectEventReducer

def f(seq, ts, support, velocity, face="unknown", confidence=0):
    sensor=SensorMetadata(source_id="camera",sensor_id="coin",sequence=seq,stream_epoch="test")
    return InputFrame({"track_id":"coin-1","support_contact":support,"vertical_velocity_px_s":velocity,
                       "face":face,"face_confidence":confidence},"video",ts,sensor=sensor)

def test_reduces_release_airborne_catch_and_face_flips():
    p=ObjectEventReducer(min_release_speed=40,min_airborne_ms=80,stable_face_frames=1)
    p.process(f(0,0,True,0,"heads",.9));p.process(f(1,100,False,100,"heads",.9));p.process(f(2,200,False,50,"tails",.9));value=p.process(f(3,300,True,0,"tails",.9))
    assert value["toss_count"]==1;assert value["flip_count"]==1;assert {event["event"] for event in value["events"]}=={"object.toss.completed"}
    value=p.process(f(4,400,True,0,"heads",.9));assert value["flip_count"]==2;assert value["events"][0]["event"]=="object.face_changed"

def test_missing_observation_does_not_invent_toss():
    p=ObjectEventReducer();value=p.process(f(0,0,None,None,"unknown",0));assert value["state"]=="held";assert value["toss_count"]==0

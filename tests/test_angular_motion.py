import math
from gamst.capabilities import SensorMetadata
from gamst.plugins import InputFrame
from gamst.providers.angular_motion import AngularMotionCounter

def frame(sequence, timestamp_ms, angle):
    center=[100.,100.]
    points=[[center[0]+30*math.cos(angle+i*math.pi/4),center[1]+30*math.sin(angle+i*math.pi/4)] for i in range(8)]
    sensor=SensorMetadata(source_id='camera',sensor_id='rotation',sequence=sequence,stream_epoch='test')
    return InputFrame({'track_id':'object-1','center':center,'points':points},'video',timestamp_ms,sensor=sensor)

def test_counts_object_agnostic_full_turns_and_velocity():
    provider=AngularMotionCounter(); values=[provider.process(frame(i,i*100,i*math.pi/4)) for i in range(10)]; final=values[-1]
    assert final['track_id']=='object-1'; assert abs(final['rotation_count']-1.)<.01; assert final['direction']=='clockwise'; assert final['angular_velocity_deg_s']>400; assert final['confidence']>=.99

def test_track_change_resets_without_false_rotation():
    provider=AngularMotionCounter(); provider.process(frame(0,0,0)); provider.process(frame(1,100,math.pi)); changed=frame(2,200,math.pi);changed.data['track_id']='object-2';value=provider.process(changed)
    assert value['track_id']=='object-2';assert value['rotation_count']==0;assert value['confidence']==0

import math
from gamst.capabilities import SensorMetadata
from gamst.plugins import InputFrame
from gamst.providers.object_rotation_bank import ObjectRotationBank

def obj(track, angle):
    c=[100,100];p=[[c[0]+30*math.cos(angle+i*math.pi/4),c[1]+30*math.sin(angle+i*math.pi/4)] for i in range(8)]
    return {'track_id':track,'center':c,'points':p}

def test_fans_out_by_track_id():
    bank=ObjectRotationBank()
    for seq in range(10):
        sensor=SensorMetadata(source_id='camera',sensor_id='multi',sequence=seq,stream_epoch='test')
        v=bank.process(InputFrame({'objects':[obj('coin',seq*math.pi/4),obj('wheel',-seq*math.pi/4)]},'video',seq*100,sensor=sensor))
    result={x['track_id']:x for x in v['objects']}
    assert set(result)=={'coin','wheel'}
    assert result['coin']['rotation_count']>.9
    assert result['wheel']['rotation_count']>.9

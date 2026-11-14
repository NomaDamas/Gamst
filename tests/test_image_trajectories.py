import dataclasses
import json
import time
import unittest

from humansense import HumanSense, SensorMetadata
from humansense.mcp import MCPServer
from humansense.plugins import InputFrame
from humansense.providers.trajectories import ImageTrajectories


def frame(seq=0, ts=0, x=0):
    return InputFrame(dict(units='normalized_image', image_size=[1920,1080],
        tracker_id='tracker', tracker_version='v1', camera_motion='stationary',
        objects=[dict(track_id='ball', centroid=[x,.5], association_confidence=None,
                      evidence_sha256='a'*64)]), 'video', ts,
        sensor=SensorMetadata('camera','rgb',seq,'image',stream_epoch='e1'))


class Trajectories(unittest.TestCase):
    def test_unequal_intervals_and_bounded_owned_history(self):
        p=ImageTrajectories(max_samples=3)
        initial=frame(); first=p.process(initial)['objects'][0]
        initial.data['objects'][0]['centroid'][0]=1
        first['observations'][0]['centroid'][0]=1
        for i,ts in enumerate((100,350,750,1000),1):
            obj=p.process(frame(i,ts,ts/1000))['objects'][0]
            self.assertAlmostEqual(obj['velocity_image_s'][0],1)
            self.assertEqual(obj['segment_id'],first['segment_id'])
        self.assertEqual([x['timestamp_ms'] for x in obj['observations']],[350,750,1000])
        self.assertEqual(obj['velocity_interval_ms'],[750,1000])
        self.assertEqual(obj['predictions'],[])

    def test_boundaries_and_missing_track(self):
        changes=[{'stream_epoch':'e2'}, {'source_id':'other'}, {'sensor_id':'other'},
                 {'coordinate_frame':'other'}, {'calibration_id':'new'}, {'channels':('red',)},
                 {'capture_sequence':1}]
        for change in changes:
            p=ImageTrajectories();a=p.process(frame())['objects'][0]
            f=frame(1,100,.1); f=dataclasses.replace(f,sensor=dataclasses.replace(f.sensor,**change))
            b=p.process(f)['objects'][0]
            self.assertNotEqual(a['segment_id'],b['segment_id']);self.assertIsNone(b['velocity_image_s'])
        for key,value in [('camera_motion','moving'),('camera_motion','cut'),('camera_motion','unknown'),
                          ('image_size',[1280,720]),('tracker_version','v2')]:
            p=ImageTrajectories();p.process(frame());f=frame(1,100,.1);f.data[key]=value
            self.assertIsNone(p.process(f)['objects'][0]['velocity_image_s'])
            self.assertIsNone(p.process(frame(2,200,.2))['objects'][0]['velocity_image_s'])
        for f in (frame(2,100),frame(1,1001)):
            p=ImageTrajectories();p.process(frame())
            self.assertIsNone(p.process(f)['objects'][0]['velocity_image_s'])
        p=ImageTrajectories();a=p.process(frame())['objects'][0]
        f=frame(1,100);f.data['objects']=[];self.assertEqual(p.process(f)['objects'],[])
        self.assertNotEqual(p.process(frame(2,200))['objects'][0]['segment_id'],a['segment_id'])

    def test_validation_clears_and_lifecycle(self):
        bad=[]
        for key,value in [('centroid',[True,0]),('centroid',[float('nan'),0]),('centroid',[-.1,0]),
                          ('association_confidence',True),('evidence_sha256','no')]:
            f=frame(1,100);f.data['objects'][0][key]=value;bad.append(f)
        bad.extend([frame(0,100),frame(1,0),dataclasses.replace(frame(1,100),timestamp_ms=True)])
        for f in bad:
            p=ImageTrajectories();p.process(frame())
            with self.assertRaises(ValueError):p.process(f)
            self.assertIsNone(p.process(frame(2,200))['objects'][0]['velocity_image_s'])
        p.reset();self.assertIsNone(p.process(frame())['objects'][0]['velocity_image_s'])
        p.close();p.close()
        with self.assertRaises(RuntimeError):p.process(frame())
        for config in ({'max_samples':9},{'max_samples':True},{'max_gap_ms':0}):
            with self.assertRaises(ValueError):ImageTrajectories(**config)

    def test_maximum_output_and_capture_sequence(self):
        p=ImageTrajectories()
        for i in range(10):
            f=frame(i,i*100,.1)
            item=f.data['objects'][0]
            f.data['objects']=[dict(item,track_id=str(n)) for n in range(8)]
            result=p.process(f)
        self.assertEqual(len(result['objects']),8)
        self.assertEqual(len(result['objects'][0]['observations']),8)
        self.assertLess(len(json.dumps(result)),65536)
        p.reset()
        f=frame();p.process(dataclasses.replace(f,sensor=dataclasses.replace(f.sensor,capture_sequence=10)))
        f=frame(1,100)
        result=p.process(dataclasses.replace(f,sensor=dataclasses.replace(f.sensor,capture_sequence=12)))
        self.assertIsNone(result['objects'][0]['velocity_image_s'])


class Context(unittest.IsolatedAsyncioTestCase):
    async def test_real_store_mcp_and_original_evidence(self):
        p=ImageTrajectories();now=int(time.time()*1000)
        p.process(frame(0,now-100));f=frame(1,now,.1)
        async with HumanSense(consent=True) as human:
            human.register_capability(p.spec)
            await human.push_observation(p.spec.id,p.process(f),sensor=f.sensor,timestamp_ms=now)
            ctx=(await human.get_context())['observations']
            value=next(iter(ctx.values()))['value']['objects'][0]
            self.assertEqual(value['velocity_image_s'],[1,0])
            self.assertEqual(value['observations'][0]['timestamp_ms'],now-100)
            result=await MCPServer(human).call_tool('get_observations',{'capability_id':p.spec.id})
            self.assertIn('a'*64,json.dumps(result))

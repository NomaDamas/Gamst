import dataclasses
import json
import unittest

from humansense import HumanSense, SensorMetadata
from humansense.mcp import MCPServer
from humansense.plugins import InputFrame
from humansense.providers.kinematics import MetricKinematics


def provider(**kwargs):
    return MetricKinematics(**dict(dict(calibration_id='cal1', input_frame='world', output_frame='world'), **kwargs))


def frame(seq=0, ts=0, position=(0, 0, 0), **kwargs):
    sensor = SensorMetadata('camera', 'depth', seq, 'world', calibration_id='cal1', stream_epoch='e1')
    return InputFrame(dict(units='m', tracker_id='tracker', tracker_version='v1',
        objects=[dict(track_id='a', position_m=list(position))]), 'video', ts,
        **dict(dict(sensor=sensor), **kwargs))


class KinematicsTests(unittest.TestCase):
    def test_unequal_intervals_constant_acceleration(self):
        p = provider()
        # x=t^2: v at midpoint is 2t, acceleration exactly 2.
        results = [p.process(frame(i, ts, ((ts/1000)**2, 0, 0)))['objects'][0]
                   for i, ts in enumerate((0, 200, 700, 1000))]
        self.assertIsNone(results[0]['velocity_m_s'])
        self.assertIsNone(results[1]['acceleration_m_s2'])
        self.assertAlmostEqual(results[2]['velocity_m_s'][0], .9)
        self.assertAlmostEqual(results[2]['acceleration_m_s2'][0], 2)
        self.assertAlmostEqual(results[3]['acceleration_m_s2'][0], 2)
        self.assertAlmostEqual(results[2]['approach_to_output_origin_m_s'], -.9)
        self.assertEqual(results[2]['velocity_interval_ms'], [200, 700])

    def test_rigid_transform_and_origin_semantics(self):
        p = provider(input_frame='camera', rotation=((0,-1,0),(1,0,0),(0,0,1)), translation_m=(10,0,0))
        for seq, pos in enumerate(((1,0,0),(2,0,0))):
            f=frame(seq,seq*1000,pos)
            result=p.process(dataclasses.replace(f,sensor=dataclasses.replace(f.sensor,coordinate_frame='camera')))
        obj=result['objects'][0]
        self.assertEqual(obj['position_m'],[10,2,0])
        self.assertEqual(obj['velocity_m_s'],[0,1,0])
        self.assertAlmostEqual(obj['approach_to_output_origin_m_s'],101**.5-104**.5)
        self.assertFalse(result['geometry']['calibration_verified'])

    def test_gap_owner_and_capture_sequence_reset(self):
        p=provider()
        p.process(frame())
        self.assertEqual(p.process(frame(2,100))['objects'][0]['status'],'warmup')
        self.assertEqual(p.process(frame(3,1101))['objects'][0]['status'],'warmup')
        f=frame(4,1200,subject_id='other')
        self.assertEqual(p.process(f)['objects'][0]['status'],'warmup')
        for change in ({'stream_epoch':'e2'}, {'source_id':'other'}, {'sensor_id':'other'}):
            p.process(frame())
            f=frame(1,100)
            self.assertEqual(p.process(dataclasses.replace(f,sensor=dataclasses.replace(f.sensor,**change)))['objects'][0]['status'],'warmup')
        p.reset()
        a=frame(0,0);b=frame(1,100)
        p.process(dataclasses.replace(a,sensor=dataclasses.replace(a.sensor,capture_sequence=10)))
        result=p.process(dataclasses.replace(b,sensor=dataclasses.replace(b.sensor,capture_sequence=12)))
        self.assertEqual(result['objects'][0]['status'],'warmup')

    def test_missing_tracks_are_not_reconnected_and_owned_copies(self):
        p=provider();f=frame();p.process(f)
        f.data['objects'][0]['position_m'][0]=999
        self.assertEqual(p.process(frame(1,100,(1,0,0)))['objects'][0]['velocity_m_s'],[10,0,0])
        empty=frame(2,200);empty.data['objects']=[]
        self.assertEqual(p.process(empty)['objects'],[])
        self.assertEqual(p.process(frame(3,300))['objects'][0]['status'],'warmup')
        p.reset();self.assertEqual(p.process(frame())['objects'][0]['status'],'warmup')
        p.close();p.close()
        with self.assertRaises(RuntimeError):p.process(frame(1,1))

    def test_invalid_input_clears_derivative_history(self):
        invalid=[]
        for data in ({}, {'units':'relative'}, {'units':'m','tracker_id':'x','tracker_version':'v','objects':[dict(track_id='a',position_m=[float('nan'),0,0])]},
                     dict(units='m',tracker_id='x',tracker_version='v',objects=[dict(track_id='a',position_m=[0,0,0])]*2)):
            invalid.append(dataclasses.replace(frame(1,100),data=data))
        invalid += [dataclasses.replace(frame(1,100),timestamp_ms=True), frame(0,100),frame(1,0),
                    dataclasses.replace(frame(1,100),sensor=None)]
        p=provider()
        for bad in invalid:
            p.reset();p.process(frame())
            with self.assertRaises(ValueError):p.process(bad)
            self.assertEqual(p.process(frame(2,200))['objects'][0]['status'],'warmup')

    def test_config_and_limits(self):
        for config in ({'max_gap_ms':True},{'rotation':((1,0,0),(0,1,0),(0,0,-1))},
                       {'translation_m':(1,0,0)}, {'rotation':((2,0,0),(0,1,0),(0,0,1))}):
            with self.assertRaises(ValueError):provider(**config)
        p=provider();f=frame()
        f.data['objects']=[dict(track_id=str(i),position_m=[10000,-10000,10000]) for i in range(32)]
        result=p.process(f)
        self.assertEqual(len(result['objects']),32)
        self.assertLess(len(json.dumps(result)),65536)
        f.data['objects'].append(dict(track_id='33',position_m=[0,0,0]))
        with self.assertRaises(ValueError):p.process(dataclasses.replace(f,timestamp_ms=100))
        self.assertNotEqual(provider().spec.version,provider(max_gap_ms=999).spec.version)


class ContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_context_and_mcp_preserve_metric_provenance(self):
        import time
        now=int(time.time()*1000)
        p=provider();p.process(frame(0,now-100))
        f=frame(1,now,(1,0,0));value=p.process(f)
        async with HumanSense(consent=True) as human:
            human.register_capability(p.spec)
            await human.push_observation(p.spec.id,value,sensor=f.sensor,timestamp_ms=f.timestamp_ms)
            result=await MCPServer(human).call_tool('get_observations',{'capability_id':p.spec.id})
            self.assertIn('velocity_m_s',json.dumps(result))
            self.assertIn('cal1',json.dumps(result))
            observations=(await human.get_context())['observations']
            observation=next(iter(observations.values()))
            self.assertEqual(observation['value']['objects'][0]['velocity_m_s'],[10,0,0])
            self.assertEqual(observation['interpretation'],'estimate')

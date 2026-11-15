import copy
import json
import sys
import unittest
from dataclasses import replace

from gamst import Gamst, ProcessProvider, SensorMetadata, SensorStreams
from gamst.plugins import InputFrame
from gamst.mcp import MCPServer
from gamst.providers.ground_plane import GroundPlane
from gamst.schema import timestamp

CONFIG=dict(id='court-cal',reference_id='view-1',source_id='cam',sensor_id='points',
    input_frame='camera.image',output_frame='court.xy',image_size=[640,480],
    valid_roi=[.1,.1,.9,.9],image_to_plane=[[10,0,0],[0,20,0],[.5,0,1]])


def frame():
    return InputFrame(dict(reference_id='view-1',image_size=[640,480],points=[
        dict(id='ground-contact',xy=[.5,.5],on_plane=True),
        dict(id='airborne-ball',xy=[.5,.5],on_plane=False)]),'video',timestamp(),
        sensor=SensorMetadata('cam','points',1,'camera.image',calibration_id='court-cal',stream_epoch='e'))


class Geometry(unittest.TestCase):
    def test_projective_scale_equivalence_preserves_original_configuration(self):
        for scale in (1, 1e-300, 1e-320, 5e-324, -5e-324):
            with self.subTest(scale=scale):
                config = dict(CONFIG, image_to_plane=[[scale,0,0],[0,scale,0],[0,0,scale]])
                provider = GroundPlane(config)
                self.assertEqual(provider.process(frame())['points'][0]['plane_xy_m'], [.5,.5])
                self.assertEqual(provider.config, config)
                self.assertEqual(provider.spec, GroundPlane.capability(config))

    def test_projective_division_plane_eligibility_and_region_abstention(self):
        p=GroundPlane(CONFIG);f=frame();value=p.process(f)
        self.assertEqual(value['points'][0]['plane_xy_m'],[4,8])
        self.assertIsNone(value['points'][1]['plane_xy_m'])
        self.assertFalse(value['geometry']['calibration_verified'])
        f.data['points'][0]['xy']=[0,0]
        self.assertEqual(p.process(f)['points'][0]['reason'],'outside_calibrated_region')
        self.assertEqual(value['points'][0]['image_xy'],[.5,.5])
        p.close()
        with self.assertRaises(RuntimeError):p.process(f)

    def test_bad_calibration_horizon_and_reference_scope(self):
        for matrix in ([[0,0,0]]*3, [[1,0,0],[0,1,0],[1,0,-.5]],
                       [[1,0,0],[0,True,0],[0,0,1]]):
            with self.assertRaises(ValueError):GroundPlane(dict(CONFIG,image_to_plane=matrix))
        p=GroundPlane(CONFIG)
        for change in ({'calibration_id':'other'},{'coordinate_frame':'other'},{'source_id':'other'}):
            f=frame()
            with self.assertRaises(ValueError):p.process(replace(f,sensor=replace(f.sensor,**change)))
        for field,value in (('reference_id','other'),('image_size',[640,481])):
            f=frame();f.data[field]=value
            with self.assertRaises(ValueError):p.process(f)
        f=frame();f.data['points']*=5
        with self.assertRaises(ValueError):p.process(f)


class Runtime(unittest.IsolatedAsyncioTestCase):
    async def test_real_dependency_free_worker_context_mcp(self):
        worker=await ProcessProvider.open(sys.executable,'geometry.ground_plane',CONFIG)
        try:
            async with Gamst(consent=True) as human:
                streams=SensorStreams(human);streams.register(worker.spec,lambda:worker)
                await streams.open('cam','points',[worker.spec.id])
                f=frame()
                await streams.submit(replace(f,sensor=replace(f.sensor,stream_epoch=None)))
                await streams.drain()
                response=await MCPServer(human).call_tool('get_observations',{'capability_id':worker.spec.id})
                self.assertIn('off_plane_or_unknown',json.dumps(response))
                self.assertIn('court.xy',json.dumps(response))
                await human.pause();self.assertTrue(worker.closed)
        finally:await worker.close()

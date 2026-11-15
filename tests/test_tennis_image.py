import asyncio
import copy
from dataclasses import replace
import json
import subprocess
import sys
import unittest
from unittest.mock import patch

from gamst import Gamst, ProcessProvider, SensorStreams
from gamst.mcp import MCPServer
from gamst.providers.tennis_image import TennisImageTask, configuration
from gamst.providers.detection_trajectories import RFDETRTrajectories
from gamst.schema import timestamp
import test_detection_provider as detection
from test_detection_trajectories import frame, WORKER

np = detection.np
CONFIG = dict(path='/explicit/local/checkpoint.pth', sha256='a'*64, reference=dict(
    id='fixed-tennis-camera-1', source_id='cam', sensor_id='sensor',
    coordinate_frame='camera.normalized', calibration_id=None, image_size=[100, 100],
    line=dict(start=[.5, .1], end=[.5, .9])))


class Discovery(unittest.TestCase):
    def test_exact_config_offline_discovery_and_detached_reference(self):
        code = ('from gamst import ProcessProvider;import sys;'
            'ProcessProvider.capability("rfdetr.tennis_image",'+repr(CONFIG)+');'
            'assert not {"numpy","torch","rfdetr"}&sys.modules.keys()')
        result = subprocess.run([sys.executable, '-c', code], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for bad in (dict(CONFIG, camera_motion='stationary'), dict(CONFIG, reference={}),
                    dict(CONFIG, reference=dict(CONFIG['reference'], image_size=[True, 100]))):
            with self.assertRaises(ValueError):
                TennisImageTask.capability(bad)
        changed = copy.deepcopy(CONFIG)
        changed['reference']['id'] = 'another-reference'
        self.assertNotEqual(TennisImageTask.capability(CONFIG).version, TennisImageTask.capability(changed).version)


@unittest.skipIf(np is None, 'optional numpy absent')
class Pack(unittest.TestCase):
    def setUp(self):
        self.backend = detection.Backend()
        config, args = configuration(CONFIG)
        paths = RFDETRTrajectories(self.backend, version='fixture',
            **{k: v for k, v in args.items() if k not in ('path', 'sha256')})
        self.pack = TennisImageTask(paths, config)

    def ball(self, x, seq, ts=None, extra=()):
        self.backend.rows = ((x, 35, x+30, 65, .9, 32, 'sports ball'),)+extra
        return self.pack.process(frame(seq, ts))

    def test_crossing_retains_original_pair_and_only_ball_events(self):
        self.assertEqual(self.ball(34, 0)['task']['status'], 'collecting')
        value = self.ball(36, 1, 1075)
        self.assertEqual(value['task']['status'], 'observed_endpoints')
        self.assertEqual(len(value['events']), 1)
        event = value['events'][0]
        obj = value['objects'][event['object_index']]
        self.assertEqual([obj['observations'][i]['timestamp_ms'] for i in event['observation_indices']], [1000, 1075])
        self.assertEqual(obj['velocity_interval_ms'], [1000, 1075])
        self.assertFalse(value['task']['quality_verified'])
        self.assertIn('not_bounce', value['task']['event_meaning'])
        # Reference edits after configuration/output cannot change frozen state.
        value['task']['reference']['line']['start'][0] = 0
        self.assertEqual(self.pack.config['reference']['line']['start'][0], .5)

    def test_absent_lookalike_partial_and_unsupported_population_break_continuity(self):
        cases = [(), ((34, 35, 64, 65, .9, 32, 'sports ball'),)*2,
            ((10, 10, 40, 40, .9, 1, 'cup'),),
            ((10, 10, 40, 40, .9, 0, 'person'),)*3,
            ((10, 10, 40, 40, .9, 0, 'person'),)*9]
        for rows in cases:
            with self.subTest(rows=len(rows)):
                self.pack.reset();self.ball(34, 0)
                self.backend.rows = rows
                bad = self.pack.process(frame(1))
                self.assertEqual(bad['task']['status'], 'abstained')
                self.assertEqual(bad['events'], [])
                self.assertEqual(bad['objects'], [])
                self.assertEqual(self.ball(36, 2)['task']['status'], 'collecting')

    def test_reference_geometry_and_epoch_gap_reset(self):
        for change in ({'source_id':'different'}, {'coordinate_frame':'other'}, {'calibration_id':'new'}):
            self.pack.reset();self.ball(34, 0)
            f = frame(1)
            value = self.pack.process(replace(f, sensor=replace(f.sensor, **change)))
            self.assertEqual(value['task']['reason'], 'reference_invalidated_explicit_reset_required')
            self.assertEqual(self.ball(36, 2)['task']['status'], 'abstained')
            self.pack.reset()
            self.assertEqual(self.ball(36, 3)['task']['status'], 'collecting')
        self.pack.reset();self.ball(34, 0)
        changed = replace(frame(1), data=np.zeros((120, 100, 3), np.uint8))
        self.assertEqual(self.pack.process(changed)['task']['reason'], 'reference_invalidated_explicit_reset_required')
        self.pack.reset();self.ball(34,0)
        changed=replace(frame(1),sensor=replace(frame(1).sensor,stream_epoch='new'))
        self.assertEqual(self.pack.process(changed)['task']['reason'], 'reference_invalidated_explicit_reset_required')
        for f in (frame(3), frame(1, 1200)):
            self.pack.reset();self.ball(34, 0)
            self.backend.rows=((36,35,66,65,.9,32,'sports ball'),)
            value=self.pack.process(f)
            self.assertEqual(value['events'], [])
            self.assertEqual(value['task']['status'], 'collecting')

    def test_small_ball_fast_displacement_and_nonball_crossing_abstain(self):
        self.backend.rows=((30,35,33,65,.9,32,'sports ball'),)
        self.assertEqual(self.pack.process(frame())['task']['reason'], 'ball_below_four_pixels')
        self.pack.reset();self.ball(15,0)
        # Overlapping large boxes preserve IoU association while exceeding .1 displacement.
        self.assertEqual(self.ball(27,1)['task']['reason'], 'ball_displacement_outside_envelope')
        self.assertEqual(self.ball(34,2)['task']['status'], 'collecting')
        self.pack.reset()
        self.ball(10,0,extra=((34,5,64,30,.9,0,'person'),))
        value=self.ball(10,1,extra=((36,5,66,30,.9,0,'person'),))
        self.assertEqual(value['events'], [])
        self.pack.close()
        with self.assertRaises(RuntimeError):self.ball(10,2)


@unittest.skipIf(np is None, 'optional numpy absent')
class Runtime(unittest.IsolatedAsyncioTestCase):
    async def test_owned_worker_context_mcp_and_clear(self):
        worker_code = WORKER.replace('x=10+self.index', 'x=34+2*self.index').replace("['cup']", "['sports ball']")
        spawn_original = asyncio.create_subprocess_exec
        children=[]
        async def spawn(*args, **kwargs):
            child=await spawn_original(sys.executable,'-u','-c',worker_code,**kwargs)
            children.append(child)
            return child
        with patch('asyncio.create_subprocess_exec',spawn):
            worker=await ProcessProvider.open(sys.executable,'rfdetr.tennis_image',CONFIG)
            try:
                async with Gamst(consent=True) as human:
                    streams=SensorStreams(human);streams.register(worker.spec,lambda:worker)
                    await streams.open('cam','sensor',[worker.spec.id])
                    for i in range(2):
                        f=frame(i,timestamp()-100+i*50)
                        await streams.submit(replace(f,sensor=replace(f.sensor,stream_epoch=None)))
                        await streams.drain()
                    reply=await MCPServer(human).call_tool('get_observations',{'capability_id':worker.spec.id})
                    self.assertIn('unverified_ball_line_candidate',json.dumps(reply))
                    self.assertIn('line_positive_to_negative',json.dumps(reply))
                    await human.pause()
                    self.assertTrue(worker.closed)
                self.assertIsNotNone(children[0].returncode)
            finally:
                await worker.close()

import copy
import hashlib
import struct
import asyncio
import json
import sys
import unittest
from dataclasses import replace
from unittest.mock import patch

import test_camera_similarity as images
import test_detection_provider as detection
from gamst import SensorMetadata
from gamst.plugins import InputFrame
from gamst.providers.compensated_trajectories import CompensatedTrajectories, ASSUMPTION
from gamst.providers.tennis_image import TennisImageTask, configuration
from gamst import Gamst, ProcessProvider, SensorStreams
from gamst.mcp import MCPServer
from gamst.schema import timestamp
from test_compensated_runtime import WORKER


@unittest.skipIf(images.cv is None, 'optional OpenCV/NumPy absent')
class Anchors(unittest.TestCase):
    def setUp(self):
        source=images.Similarity();source.setUp();self.image=source.texture()
        self.backend=detection.Backend()
        evidence=hashlib.sha256(b'gamst.compensated.trajectory.uint8.HWC.BGR.v1\0'+
            struct.pack('>III',*self.image.shape)+self.image.tobytes()).hexdigest()
        self.reference=dict(id='tennis-court-image',source_id='cam',sensor_id='rgb',
            coordinate_frame='image',calibration_id='cal',image_size=[480,320],
            timestamp_ms=1000,capture_sequence=0,evidence_sha256=evidence,
            rules=[dict(id='line',kind='line',start=[85.5/480,.1],end=[85.5/480,.9])])
        self.config=dict(path='/fixture/checkpoint',sha256='a'*64,background_roi=[0,0,1,1],
            operating_assumption=ASSUMPTION,max_gap_ms=100,event_reference=self.reference)

    def frame(self,seq,shift=0,extra=0):
        image=images.cv.warpAffine(self.image,images.np.array([[1,0,shift],[0,1,0]],float),(480,320))
        self.backend.rows=((80+shift+extra,120,90+shift+extra,140,.9,32,'sports ball'),)
        return InputFrame(image,'video',1000+seq*50,'subject',sensor=SensorMetadata(
            'cam','rgb',seq,'image',stream_epoch='epoch',calibration_id='cal'))

    def test_anchor_rules_original_indices_camera_motion_and_frozen_rebind(self):
        provider=CompensatedTrajectories(self.backend,version='fixture',config=self.config)
        self.addCleanup(provider.close)
        first=provider.process(self.frame(0))
        self.assertEqual(first['event_summary']['status'],'collecting_anchor')
        result=provider.process(self.frame(1,8,1))
        self.assertEqual(result['event_summary']['status'],'bound_estimated_anchor')
        self.assertEqual(len(result['events']),1)
        event=result['events'][0]
        obj=result['objects'][event['object_index']]
        self.assertEqual([obj['observations'][i]['timestamp_ms'] for i in event['observation_indices']],[1000,1050])
        self.assertEqual(obj['anchor_id'],result['event_summary']['anchor']['id'])
        frozen=copy.deepcopy(result)
        reset=provider.process(self.frame(3,16,2))
        self.assertEqual(reset['events'],[])
        self.assertEqual(reset['event_summary']['status'],'reference_unbound')
        self.assertEqual(result,frozen)
        # An explicit reopen with a new immutable anchor reference binds the new view.
        new=copy.deepcopy(self.config)
        new['event_reference'].update({k:reset['geometry']['anchor'][k] for k in
            ('timestamp_ms','capture_sequence','evidence_sha256')})
        new['event_reference']['id']='rebound-court-image'
        rebound=CompensatedTrajectories(self.backend,version='fixture',config=new)
        self.addCleanup(rebound.close)
        self.assertNotEqual(rebound.spec.version,provider.spec.version)
        self.assertEqual(rebound.process(self.frame(3,16,2))['event_summary']['status'],'collecting_anchor')
        self.assertEqual(rebound.process(self.frame(4,20,3))['event_summary']['status'],'bound_estimated_anchor')

    def test_wrong_anchor_source_and_changed_geometry_do_not_evaluate_rules(self):
        for change in ({'evidence_sha256':'b'*64},{'source_id':'other'},{'image_size':[480,321]}):
            config=copy.deepcopy(self.config);config['event_reference'].update(change)
            provider=CompensatedTrajectories(self.backend,version='fixture',config=config)
            try:
                provider.process(self.frame(0))
                result=provider.process(self.frame(1,8,1))
                self.assertEqual(result['events'],[])
                self.assertEqual(result['event_summary']['status'],'reference_unbound')
            finally:provider.close()

    def test_event_profile_has_joint_bounded_history_rules_and_output(self):
        config=copy.deepcopy(self.config)
        config['event_reference']['rules']=[dict(id='r'+str(i),kind='line',
            start=[.2+i*.1,.1],end=[.2+i*.1,.9]) for i in range(4)]
        provider=CompensatedTrajectories(self.backend,version='v'*128,config=config)
        self.addCleanup(provider.close)
        for seq in range(6):
            f=self.frame(seq)
            self.backend.rows=tuple((20+i*50,120,30+i*50,140,.9,i,chr(0x10000+i)*256) for i in range(8))
            result=provider.process(f)
            self.assertLessEqual(len(json.dumps(result,ensure_ascii=True).encode()),65536)
        self.assertEqual([len(obj['observations']) for obj in result['objects']],[3]*8)
        self.assertEqual(len(result['geometry']['projection_records']),3)

    def test_tennis_compensated_population_occlusion_and_capture_support(self):
        reference={k:v for k,v in self.reference.items() if k not in
            ('timestamp_ms','capture_sequence','evidence_sha256','rules')}
        reference['line']={k:self.reference['rules'][0][k] for k in ('start','end')}
        config=dict(path='/fixture/checkpoint',sha256='a'*64,reference=reference,
            compensation=dict(background_roi=[0,0,1,1],operating_assumption=ASSUMPTION,
                anchor={k:self.reference[k] for k in ('timestamp_ms','capture_sequence','evidence_sha256')}),
            envelope=dict(max_people=4,max_rackets=3,min_ball_side_pixels=4,max_capture_gap_ms=75,
                max_endpoint_displacement_normalized=.05))
        checked,args=configuration(config)
        provider=CompensatedTrajectories(self.backend,version='fixture',config=args)
        pack=TennisImageTask(provider,checked);self.addCleanup(pack.close)
        pack.process(self.frame(0));result=pack.process(self.frame(1,8,1))
        self.assertEqual(len(result['events']),1)
        self.assertEqual(result['task']['envelope']['max_people'],4)
        anchor=result['geometry']['anchor']['id']
        absent=self.frame(2,12);self.backend.rows=()
        self.assertEqual(pack.process(absent)['task']['status'],'abstained')
        returned=pack.process(self.frame(3,16,1))
        self.assertEqual(returned['events'],[])
        self.assertEqual(returned['task']['status'],'collecting')
        self.assertEqual(returned['geometry']['anchor']['id'],anchor)
        next_frame=self.frame(4,20,2)
        self.assertEqual(pack.process(next_frame)['task']['status'],'observed_endpoints')
        gap=pack.process(self.frame(7,24,3))
        self.assertEqual(gap['task']['reason'],'anchor_reference_unbound')


@unittest.skipIf(images.cv is None, 'optional OpenCV/NumPy absent')
class Runtime(unittest.IsolatedAsyncioTestCase):
    async def test_compensated_reference_worker_context_mcp(self):
        fixture=Anchors();fixture.setUp()
        now=timestamp()-100
        fixture.reference['timestamp_ms']=now
        reference={k:v for k,v in fixture.reference.items() if k not in
            ('timestamp_ms','capture_sequence','evidence_sha256','rules')}
        reference['line']={k:fixture.reference['rules'][0][k] for k in ('start','end')}
        config=dict(path='/fixture/checkpoint',sha256='a'*64,reference=reference,
            compensation=dict(background_roi=[0,0,1,1],operating_assumption=ASSUMPTION,
                anchor={k:fixture.reference[k] for k in ('timestamp_ms','capture_sequence','evidence_sha256')}))
        original=asyncio.create_subprocess_exec;children=[]
        code=WORKER.replace("['ball']", "['sports ball']")
        async def spawn(*args,**kwargs):
            child=await original(sys.executable,'-u','-c',code,**kwargs)
            children.append(child);return child
        with patch('asyncio.create_subprocess_exec',spawn):
            worker=await ProcessProvider.open(sys.executable,'rfdetr.tennis_image',config)
        try:
            async with Gamst(consent=True) as human:
                streams=SensorStreams(human);streams.register(worker.spec,lambda:worker)
                await streams.open('cam','rgb',[worker.spec.id],subject_id='subject')
                for i in range(2):
                    f=replace(fixture.frame(i,8*i,i),timestamp_ms=now+i*50)
                    await streams.submit(replace(f,sensor=replace(f.sensor,stream_epoch=None)))
                    await streams.drain()
                reply=await MCPServer(human).call_tool('get_observations',
                    {'capability_id':worker.spec.id,'subject_id':'subject'})
                self.assertIn('bound_estimated_anchor',json.dumps(reply))
                self.assertIn('line_positive_to_negative',json.dumps(reply))
                self.assertNotIn('"pixels":',json.dumps(reply))
                await human.pause();self.assertTrue(worker.closed)
            self.assertTrue(all(child.returncode is not None for child in children))
        finally:await worker.close()

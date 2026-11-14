import asyncio
from dataclasses import replace
import json
import sys
import unittest
from unittest.mock import patch

from humansense import HumanSense,ProcessProvider,SensorStreams
from humansense.mcp import MCPServer
from humansense.providers.trajectories import ImageTrajectories
from humansense.providers.detection_trajectories import RFDETRTrajectories
from humansense.schema import timestamp
import test_image_trajectories as trajectory
import test_detection_trajectories as detection

RULES=[dict(id='zone',kind='roi',bounds=[.4,.3,.8,.7]),
       dict(id='line',kind='line',start=[.4,.2],end=[.4,.8]),
       dict(id='turn',kind='reversal',axis='x',min_displacement=.01)]


def points(xs, rules=RULES, ys=None):
    p=ImageTrajectories(event_rules=rules)
    results=[]
    for i,x in enumerate(xs):
        f=trajectory.frame(i,i*100,x)
        if ys is not None:f.data['objects'][0]['centroid'][1]=ys[i]
        results.append(p.process(f))
    return results


class Geometry(unittest.TestCase):
    def test_entry_exit_line_and_reversal_evidence_references(self):
        values=points([.2,.6,.3])
        self.assertEqual(values[0]['events'],[])
        self.assertEqual([e['kind'] for e in values[1]['events']],['roi_entry','line_positive_to_negative'])
        self.assertEqual([e['kind'] for e in values[2]['events']],
                         ['roi_exit','line_negative_to_positive','axis_positive_to_negative'])
        e=values[2]['events'][2]
        self.assertEqual(e['observation_indices'],[0,1,2])
        obj=values[2]['objects'][e['object_index']]
        self.assertEqual([obj['observations'][i]['timestamp_ms'] for i in e['observation_indices']],[0,100,200])
        self.assertTrue(all(obj['observations'][i]['evidence_sha256']=='a'*64 for i in e['observation_indices']))
        self.assertFalse(values[2]['event_summary']['exact_event_time_known'])

    def test_boundary_deadband_finite_line_and_negligible_reversal(self):
        self.assertEqual(points([.2,.4,.6])[1]['events'],[])
        self.assertEqual(points([.2,.4,.6])[2]['events'],[])
        self.assertEqual(points([.2,.6],ys=[.9,.9])[-1]['events'],[])
        line=[RULES[1]]
        self.assertEqual(points([.2,.6],line,[.2,.2])[-1]['events'],[])
        self.assertEqual(points([.2,.6],line,[.8,.8])[-1]['events'],[])
        self.assertEqual(points([.2,.209,.2],[RULES[2]])[-1]['events'],[])
        self.assertEqual(points([.1,.9],[RULES[0]])[-1]['events'],[]) # outside->outside not observed entry
        self.assertEqual(points([.4-1e-7,.6],line)[-1]['events'],[])

    def test_gap_owner_unknown_missing_and_immutable_rules(self):
        rules=json.loads(json.dumps(RULES));p=ImageTrajectories(event_rules=rules)
        rules[0]['bounds']=[0,0,1,1]
        p.process(trajectory.frame(0,0,.2))
        result=p.process(trajectory.frame(1,100,.6))
        self.assertEqual(result['events'][0]['kind'],'roi_entry')
        result['event_summary']['rules'][0]['bounds']=[0,0,1,1]
        self.assertEqual(p.process(trajectory.frame(2,200,.2))['events'][0]['kind'],'roi_exit')
        for change in ('gap','owner','unknown','missing','invalid'):
            p.reset();p.process(trajectory.frame(0,0,.2));f=trajectory.frame(1,100,.6)
            if change=='gap':f=trajectory.frame(2,100,.6)
            elif change=='owner':f=replace(f,sensor=replace(f.sensor,stream_epoch='e2'))
            elif change=='unknown':f.data['camera_motion']='unknown'
            elif change=='missing':
                empty=trajectory.frame(1,50);empty.data['objects']=[];p.process(empty);f=trajectory.frame(2,100,.6)
            else:
                with self.assertRaises(ValueError):p.process(trajectory.frame(0,0,.6))
            self.assertEqual(p.process(f)['events'],[])

    def test_rule_validation_versions_and_default_compatibility(self):
        bad=[[],RULES*2,[dict(RULES[0],bounds=[True,0,1,1])],
             [dict(RULES[1],end=[.4,.2])],[dict(RULES[2],min_displacement=0)],
             [dict(RULES[0],extra=1)],[dict(RULES[2],axis='z')],[RULES[0],RULES[0]]]
        for rules in bad:
            with self.assertRaises(ValueError):ImageTrajectories(event_rules=rules)
        with self.assertRaises(ValueError):ImageTrajectories(max_samples=2,event_rules=[RULES[2]])
        self.assertNotIn('events',ImageTrajectories().process(trajectory.frame()))
        self.assertNotEqual(ImageTrajectories().spec.version,ImageTrajectories(event_rules=RULES).spec.version)
        config=dict(detection.CONFIG,event_rules=RULES)
        self.assertNotEqual(ProcessProvider.capability('rfdetr.trajectories',config).version,
                            ProcessProvider.capability('rfdetr.trajectories',detection.CONFIG).version)


@unittest.skipIf(detection.np is None,'optional numpy absent')
class Pipeline(unittest.TestCase):
    def test_worst_case_detector_labels_history_and_all_events_bounded(self):
        rules=[dict(id=str(i)*128,kind='reversal',axis='x',min_displacement=.000001) for i in range(4)]
        b=detection.detection.Backend();p=RFDETRTrajectories(b,version='v'*128,camera_motion='stationary',event_rules=rules)
        for i in range(8):
            offset=1 if i%2 else 0
            b.rows=tuple((10.123+j+offset,10.456,100.789+j+offset,100.123,.987654321,j,chr(0x10000+j)*256) for j in range(8))
            f=detection.frame(i,2**53-8+i)
            f=replace(f,data=detection.np.zeros((511,1023,3),detection.np.uint8),sensor=replace(f.sensor,
                source_id='s'*128,sensor_id='i'*128,coordinate_frame='f'*128,calibration_id='c'*128,
                stream_epoch='e'*128,channels=tuple(str(n)+'c'*125 for n in range(64))))
            value=p.process(f)
            self.assertLess(len(json.dumps(value,ensure_ascii=True).encode()),65536)
        self.assertEqual(len(value['events']),32)
        self.assertTrue(all(e['observation_indices']==[5,6,7] for e in value['events']))
        b.rows=tuple((10,10,40,40,.9,i,'item') for i in range(9))
        f=replace(f,timestamp_ms=2**53-1,sensor=replace(f.sensor,sequence=8))
        # Last timestamp already max; reset into a new capture epoch for truncation check.
        p.reset();value=p.process(f)
        self.assertEqual(value['events'],[])
        self.assertFalse(value['event_summary']['input_snapshot_complete'])


@unittest.skipIf(detection.np is None,'optional numpy absent')
class Runtime(unittest.IsolatedAsyncioTestCase):
    async def test_owned_pixel_worker_to_context_and_mcp_events(self):
        # Actual owned worker protocol and production constructor; model weights remain deferred.
        spawn=asyncio.create_subprocess_exec;children=[]
        async def fixture(*args,**kwargs):
            p=await spawn(sys.executable,'-u','-c',detection.WORKER,**kwargs);children.append(p);return p
        rules=[dict(id='line',kind='line',start=[.255,.1],end=[.255,.9])]
        with patch('asyncio.create_subprocess_exec',fixture):
            worker=await ProcessProvider.open(sys.executable,'rfdetr.trajectories',dict(detection.CONFIG,event_rules=rules))
            try:
                async with HumanSense(consent=True) as human:
                    streams=SensorStreams(human);streams.register(worker.spec,lambda:worker)
                    await streams.open('cam','sensor',[worker.spec.id]);now=timestamp()-200
                    for i in range(2):
                        f=detection.frame(i,now+i*100);f=replace(f,sensor=replace(f.sensor,stream_epoch=None))
                        await streams.submit(f);await streams.drain()
                    value=next(iter((await human.get_context())['observations'].values()))['value']
                    self.assertEqual(value['events'][0]['kind'],'line_positive_to_negative')
                    self.assertEqual(value['events'][0]['observation_indices'],[0,1])
                    reply=await MCPServer(human).call_tool('get_observations',{'capability_id':worker.spec.id})
                    self.assertIn('line_positive_to_negative',json.dumps(reply))
                self.assertTrue(worker.closed);self.assertIsNotNone(children[0].returncode)
            finally:await worker.close()

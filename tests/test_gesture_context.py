"""Official GestureRecognizer boundary doubles; no trained model claims."""
import asyncio
from dataclasses import replace
import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from gamst import Gamst, ProcessProvider, SensorMetadata, SensorStreams
from gamst.plugins import InputFrame
from gamst.provider_profiles import capability
from gamst.providers.landmarker_video import MediaPipeVideoContext
from gamst.schema import timestamp
try:
    import numpy as np
except ImportError:
    np = None


def result(label='Victory', score=.9):
    return NS(hand_landmarks=[[NS(x=.2+(i%5)*.1,y=.2+(i//5)*.1,z=0.) for i in range(21)]],
        handedness=[[NS(category_name='Left',score=.9)]], gestures=[[NS(category_name=label,score=score)]])


class Detector:
    def __init__(self): self.calls=[]; self.closed=0; self.result=result()
    def recognize_for_video(self, image, stamp):
        self.calls.append((image.copy(),stamp)); return self.result
    def close(self): self.closed+=1


def provider():
    made=[]
    def factory():
        detector=Detector(); made.append(detector); return detector
    api=NS(Image=lambda **kw:kw['data'],ImageFormat=NS(SRGB=1))
    return MediaPipeVideoContext('gesture',factory,api,sha256='0'*64,max_entities=2,max_gap_ms=100),made


def frame(sequence=0,stamp=1000,**kw):
    return InputFrame(np.zeros((80,100,3),dtype=np.uint8),'video',stamp,
        sensor=SensorMetadata('cam','sensor',sequence,**kw))


def native_modules(made, options):
    def create(value):
        options.append(value); d=Detector(); made.append(d); return d
    class Base:
        Delegate=NS(CPU=0)
        def __init__(self,**kw): self.__dict__.update(kw)
    classifier_options=NS(ClassifierOptions=lambda **kw:NS(**kw))
    vision=NS(GestureRecognizer=NS(create_from_options=create),GestureRecognizerOptions=lambda **kw:NS(**kw),
        RunningMode=NS(VIDEO='VIDEO'))
    python=NS(BaseOptions=Base,vision=vision)
    mp=NS(tasks=NS(python=python),Image=lambda **kw:kw['data'],ImageFormat=NS(SRGB=1))
    return {'mediapipe':mp,'mediapipe.tasks':mp.tasks,'mediapipe.tasks.python':python,
        'mediapipe.tasks.python.components':NS(processors=NS(classifier_options=classifier_options)),
        'mediapipe.tasks.python.components.processors':NS(classifier_options=classifier_options)}


class Discovery(unittest.TestCase):
    def test_strict_offline_discovery_and_hash_before_native_import(self):
        config=dict(path='/not-installed.task',sha256='0'*64)
        with patch('importlib.metadata.version',side_effect=AssertionError('native import')):
            spec=capability('mediapipe.gesture.video',config)
        self.assertEqual(spec.output_name,'vision.hand_gestures')
        self.assertTrue(spec.stateful and spec.reset_on_gap)
        for update in ({'max_entities':True},{'max_entities':5},{'max_gap_ms':0},{'unknown':1}):
            with self.assertRaises(ValueError): capability('mediapipe.gesture.video',dict(config,**update))
        with tempfile.TemporaryDirectory(dir='.') as temporary:
            path=Path(temporary).resolve()/'gesture.task'; path.write_bytes(b'not-a-model')
            with patch('importlib.metadata.version',side_effect=AssertionError('native import')):
                with self.assertRaises(ValueError): MediaPipeVideoContext.from_local('gesture',str(path),'0'*64)


@unittest.skipIf(np is None,'optional numpy not installed')
class Gestures(unittest.TestCase):
    def test_native_loader_exact_bytes_options_and_reset(self):
        made=[]; options=[]
        with tempfile.TemporaryDirectory(dir='.') as temporary:
            path=Path(temporary).resolve()/'gesture.task'; payload=b'not-a-model'; path.write_bytes(payload)
            with patch.dict(sys.modules,native_modules(made,options)),patch('importlib.metadata.version',return_value='0.10.35'):
                from gamst.provider_worker import _construct
                p,spec=_construct('mediapipe.gesture.video',dict(path=str(path),sha256=hashlib.sha256(payload).hexdigest(),max_entities=2))
            self.assertFalse(made)
            first=p.process(frame()); second=p.process(frame(1,1033))
            self.assertEqual(first['hands'][0]['gesture_candidates'],[dict(label='Victory',score=.9)])
            self.assertEqual(first['hands'][0]['gesture_status'],'candidate')
            self.assertFalse(first['summary']['command_authority'])
            self.assertEqual(first['summary']['state_segment'],second['summary']['state_segment'])
            self.assertEqual([c[1] for c in made[0].calls],[0,33])
            p.reset(); path.write_bytes(b'changed'); p.process(frame(2,1066)); p.close()
            self.assertEqual(len(made),2); self.assertTrue(all(d.closed==1 for d in made))
            for call in options:
                self.assertEqual(call.base_options.model_asset_buffer,payload)
                self.assertEqual(call.base_options.delegate,0)
                self.assertEqual(call.running_mode,'VIDEO'); self.assertEqual(call.num_hands,2)
                self.assertEqual(call.canned_gesture_classifier_options.max_results,8)
                self.assertEqual(call.custom_gesture_classifier_options.score_threshold,.5)

    def test_unknown_empty_bounds_and_native_error_clear_continuity(self):
        p,made=provider(); p.process(frame())
        for label,score in [('None',.9),('Victory',.49)]:
            made[-1].result=result(label,score)
            value=p.process(frame(len(made[-1].calls),1000+20*len(made[-1].calls)))
            self.assertEqual(value['hands'][0]['gesture_status'],'unknown')
        p.reset(); p.process(frame())
        variants=[result('do_something',.9),result('Victory',float('nan')),result('Victory',True),result('Victory',1.1)]
        invalid=result(); invalid.gestures=[]; variants.append(invalid)
        invalid=result(); invalid.gestures[0]*=2; variants.append(invalid)
        invalid=result(); invalid.gestures[0]*=9; variants.append(invalid)
        for invalid in variants:
            current=made[-1]; current.result=invalid
            with self.assertRaises(ValueError): p.process(frame(1,1033))
            self.assertEqual(current.closed,1)
            self.assertIsNone(p.formatter.result)
            p.process(frame())
        made[-1].result=NS(hand_landmarks=[],handedness=[],gestures=[])
        self.assertEqual(p.process(frame(1,1033))['hands'],[])
        p.close(); p.close()

    def test_owner_gaps_geometry_and_bad_clock_start_new_native_segments(self):
        for next_frame in (frame(2,1033),frame(1,1101),frame(1,1033,stream_epoch='new'),
                replace(frame(1,1033),data=np.zeros((40,100,3),dtype=np.uint8))):
            p,made=provider(); first=p.process(frame()); second=p.process(next_frame)
            self.assertNotEqual(first['summary']['state_segment'],second['summary']['state_segment'])
            self.assertEqual(made[0].closed,1); self.assertEqual(made[1].calls[0][1],0); p.close()
        for invalid in (frame(0,1033),frame(1,1000)):
            p,made=provider(); p.process(frame())
            with self.assertRaises(ValueError): p.process(invalid)
            self.assertEqual(made[0].closed,1); p.close()


@unittest.skipIf(np is None,'optional numpy not installed')
class WorkerPublication(unittest.IsolatedAsyncioTestCase):
    async def test_actual_worker_profile_through_context_mcp_and_cleanup(self):
        # Only the native MediaPipe boundary is substituted inside a real owned
        # worker. Production _construct, wire protocol, provider and SDK run.
        code='''
import importlib.metadata
import sys
from test_gesture_context import native_modules
sys.modules.update(native_modules([],[]))
importlib.metadata.version=lambda name:'0.10.35'
from gamst.provider_worker import main
main()
'''
        code = 'import sys; sys.path.insert(0, '+repr(str(Path(__file__).resolve().parent))+')\n'+code
        real_spawn=asyncio.create_subprocess_exec; children=[]
        async def spawn(*args,**kwargs):
            child=await real_spawn(args[0],'-u','-c',code,**kwargs); children.append(child); return child
        with tempfile.TemporaryDirectory(dir='.') as temporary:
            path=Path(temporary).resolve()/'gesture.task'; payload=b'not-a-model'; path.write_bytes(payload)
            config=dict(path=str(path),sha256=hashlib.sha256(payload).hexdigest(),max_entities=2,max_gap_ms=100)
            with patch('asyncio.create_subprocess_exec',spawn):
                async with Gamst(consent=True) as human:
                    spec=capability('mediapipe.gesture.video',config); streams=SensorStreams(human)
                    async def factory(): return await ProcessProvider.open(sys.executable,'mediapipe.gesture.video',config)
                    streams.register(spec,factory); await streams.open('cam','sensor',[spec.id])
                    now=timestamp()-50
                    await streams.submit(frame(0,now)); await streams.drain()
                    await streams.submit(frame(2,now+30)); await streams.drain()
                    from gamst.mcp import MCPServer
                    answer=await MCPServer(human).call_tool('get_observations',{'capability_id':spec.id})
                    self.assertEqual(answer['returned_count'],2)
                    items=answer['items']
                    self.assertTrue(all(item['observation']['value']['hands'][0]['gesture_status']=='candidate' for item in items))
                    self.assertNotEqual(items[0]['observation']['value']['summary']['state_segment'],items[1]['observation']['value']['summary']['state_segment'])
                    await streams.clear()
            self.assertTrue(children)
            self.assertTrue(all(child.returncode is not None for child in children))


if __name__=='__main__': unittest.main()

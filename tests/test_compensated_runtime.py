"""Real owned process and OpenCV; only trained detector loading is a fixture."""
import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import test_camera_similarity as images
from test_compensated_trajectories import CONFIG
from humansense import HumanSense,ProcessProvider,SensorStreams,SensorMetadata
from humansense.plugins import InputFrame
from humansense.mcp import MCPServer
from humansense.schema import timestamp

WORKER='''
import os,time
from pathlib import Path
import numpy as np
from types import SimpleNamespace
from humansense.providers.detection import RFDETRObjects
from humansense import provider_worker as w
class Backend:
    def __init__(self):self.index=0
    def predict(self,data,**kwargs):
        ready=os.environ.get('GAMST_COMPENSATED_TEST_READY')
        if ready:
            Path(ready).write_text('inference-entered');time.sleep(30)
        x=80+9*self.index;self.index+=1
        return SimpleNamespace(xyxy=np.array([[x,120,x+10,140]]),confidence=np.array([.9]),class_id=np.array([1]),data={'class_name':['ball']})
def local(cls,path,sha256,device,**kwargs):
    return cls(Backend(),version='1.12.1.eager.sha256.'+sha256,**kwargs)
RFDETRObjects.from_local=classmethod(local)
w.main()
'''


@unittest.skipIf(images.cv is None,'optional OpenCV/NumPy missing')
class Runtime(unittest.IsolatedAsyncioTestCase):
    def frame(self,index,end=None):
        fixture=images.Similarity();fixture.setUp();image=fixture.texture()
        transform=images.np.array([[1,0,8*index],[0,1,0]],float)
        return InputFrame(images.cv.warpAffine(image,transform,(480,320)),'video',
            timestamp()-100 if end is None else end,'subject',sensor=SensorMetadata(
                'camera','rgb',index,'image',stream_epoch='capture',calibration_id='cal'))

    async def open(self,ready=None):
        real=asyncio.create_subprocess_exec;children=[]
        async def spawn(*args,**kwargs):
            env=dict(kwargs.get('env',{}))
            if ready is not None:env['GAMST_COMPENSATED_TEST_READY']=str(ready)
            kwargs['env']=env
            child=await real(sys.executable,'-u','-c',WORKER,**kwargs)
            children.append(child);return child
        with patch('asyncio.create_subprocess_exec',spawn):
            worker=await ProcessProvider.open(sys.executable,'rfdetr.compensated_trajectories',CONFIG)
        return worker,children

    async def test_actual_worker_sensorstreams_context_mcp_pause(self):
        worker,children=await self.open()
        try:
            async with HumanSense(consent=True) as human:
                streams=SensorStreams(human);streams.register(worker.spec,lambda:worker)
                await streams.open('camera','rgb',[worker.spec.id],subject_id='subject')
                now=timestamp()-200
                for i in range(2):
                    frame=self.frame(i,now+100*i)
                    await streams.submit(replace(frame,sensor=replace(frame.sensor,stream_epoch=None)))
                    await streams.drain()
                context=await human.get_context(subject_id='subject')
                value=next(iter(context['observations'].values()))['value']
                self.assertEqual(value['geometry']['status'],'estimated')
                obj=value['objects'][0]
                self.assertEqual(len(obj['observations']),2)
                self.assertAlmostEqual(obj['velocity_reference_image_s'][0],1/480/.1,delta=.003)
                self.assertEqual(obj['anchor_id'],value['geometry']['anchor']['id'])
                response=await MCPServer(human).call_tool('get_observations',{'capability_id':worker.spec.id,'subject_id':'subject'})
                text=json.dumps(response)
                self.assertIn('anchor_projection',text);self.assertIn('evidence_sha256',text)
                self.assertNotIn('"pixels":',text)
                await human.pause();self.assertTrue(worker.closed)
            self.assertTrue(all(c.returncode is not None for c in children))
        finally:await worker.close()

    async def test_cancel_and_clear_reap_without_late_observation(self):
        for action in ('cancel','clear'):
            with self.subTest(action=action),tempfile.TemporaryDirectory(dir='.') as root:
                ready=Path(root).absolute()/'ready'
                worker,children=await self.open(ready)
                task=None;streams=None
                try:
                    async with HumanSense(consent=True) as human:
                        if action=='cancel':task=asyncio.create_task(worker.process(self.frame(0)))
                        else:
                            streams=SensorStreams(human);streams.register(worker.spec,lambda:worker)
                            await streams.open('camera','rgb',[worker.spec.id],subject_id='subject')
                            frame=self.frame(0)
                            await streams.submit(replace(frame,sensor=replace(frame.sensor,stream_epoch=None)))
                        async def entered():
                            while not ready.exists():await asyncio.sleep(.01)
                        await asyncio.wait_for(entered(),5)
                        if task is not None:
                            task.cancel()
                            with self.assertRaises(asyncio.CancelledError):await asyncio.wait_for(task,5)
                        else:await asyncio.wait_for(streams.clear(),5)
                        self.assertTrue(all(c.returncode is not None for c in children))
                        self.assertFalse((await human.get_context(subject_id='subject'))['observations'])
                finally:
                    if streams is not None:await streams.close()
                    await worker.close()
                    if task is not None and not task.done():
                        task.cancel();await asyncio.gather(task,return_exceptions=True)

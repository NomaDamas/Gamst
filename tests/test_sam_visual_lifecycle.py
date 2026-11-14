"""Real owned process cancellation with a declared blocking fixture backend."""
import asyncio
import os
from pathlib import Path
import sys
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch

import test_sam_audio_context as audio
import test_sam_visual_window as fixtures
from humansense import HumanSense,ProcessProvider,SensorStreams
from humansense.sam_visual_packet import unpack_visual_packet,visual_policy
from humansense.schema import timestamp
from humansense.separation_media import PROFILE


@unittest.skipIf(fixtures.fixtures.np is None,'optional numpy missing')
class Lifecycle(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_and_stream_clear_reap_blocked_visual_worker(self):
        code='''
import os,time
from pathlib import Path
from humansense import provider_worker as w
from humansense.providers.sam_audio_native import SAMAudioNative
from test_sam_visual_context import Backend
class Blocking(Backend):
    def predict(self,prepared,*,visual=None):
        Path(os.environ['GAMST_VISUAL_TEST_READY']).write_text('entered visual inference')
        time.sleep(30)
        return super().predict(prepared,visual=visual)
SAMAudioNative.from_local=classmethod(lambda cls,config:Blocking())
w.main()
'''
        real_spawn=asyncio.create_subprocess_exec
        for action in ('cancel','clear'):
            with self.subTest(action=action),tempfile.TemporaryDirectory(dir='.') as directory:
                ready=Path(directory).absolute()/'ready';children=[]
                fixture=fixtures.Window();fixture.setUp();f=fixture.fixture
                now=timestamp();shift=(now-f.audio.timestamp_ms)*1000
                f.audio=replace(f.audio,timestamp_ms=now)
                f.kw['intervals_us']=[[a+shift,b+shift] for a,b in f.kw['intervals_us']]
                packet=f.packet();policy=visual_policy(unpack_visual_packet(packet.data)[0])
                config=dict(audio.CONFIG,prompt='',visual_policy=policy)
                async def spawn(*args,**kwargs):
                    env=dict(kwargs.get('env',{}))
                    env['PYTHONPATH']=os.path.abspath('tests')+os.pathsep+os.path.abspath('.')
                    env['GAMST_VISUAL_TEST_READY']=str(ready);kwargs['env']=env
                    child=await real_spawn(sys.executable,'-u','-c',code,**kwargs)
                    children.append(child);return child
                async with HumanSense(consent=True) as human:
                    owner=human.enable_separation_media(subject_id='subject',source_id='microphone',sensor_id='mic0',
                        visual_source={'source_id':'camera','sensor_id':'cam0'})
                    with patch('asyncio.create_subprocess_exec',spawn):
                        provider=await ProcessProvider.open(sys.executable,PROFILE,config,separation_owner=owner)
                    streams=None;task=None
                    try:
                        if action=='cancel':task=asyncio.create_task(provider.process(packet))
                        else:
                            streams=SensorStreams(human);streams.register(provider.spec,lambda:provider)
                            await streams.open('microphone','mic0',[provider.spec.id],subject_id='subject')
                            await streams.submit(replace(packet,sensor=replace(packet.sensor,stream_epoch=None)))
                        for _ in range(300):
                            if ready.exists():break
                            await asyncio.sleep(.01)
                        self.assertTrue(ready.exists(),'worker did not enter visual inference')
                        self.assertTrue(owner._store.reservations)
                        if action=='cancel':
                            task.cancel()
                            with self.assertRaises(asyncio.CancelledError):await asyncio.wait_for(task,5)
                        else:await asyncio.wait_for(streams.clear(),5)
                        self.assertEqual(owner._store.bytes,0)
                        self.assertFalse(owner._store.entries)
                        self.assertFalse(owner._store.reservations)
                        self.assertTrue(all(child.returncode is not None for child in children))
                        context=await human.get_context(subject_id='subject')
                        self.assertFalse(context['observations'])
                    finally:
                        if streams is not None:await streams.close()
                        else:await provider.close()
                        if task is not None and not task.done():
                            task.cancel();await asyncio.gather(task,return_exceptions=True)

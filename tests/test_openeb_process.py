import asyncio
from dataclasses import replace
import struct
import sys
import unittest
from unittest.mock import patch

from humansense import OpenEBProcess,HumanSense,SensorStreams,ProcessProvider
from humansense.openeb_process import configuration,header,prefix

try:import numpy as np
except ImportError:np=None

CONFIG=dict(kind='camera',source='FIXTURE_ONLY',source_id='cam',sensor_id='events',
    sensor_origin_us='0',capture_origin_ms=1000,calibration_id='clock1',coordinate_frame='pixels')

FAKE='''
import os,time,struct,zlib
from humansense import openeb_capture_worker as w
from humansense.plugins import InputFrame
from humansense.capabilities import SensorMetadata
class Source:
    def __init__(self,config):
        self.selected=config;self.sequence=0
        self.config={k:config[k] for k in ('sensor_origin_us','capture_origin_ms','calibration_id')}
        self.config.update(width=16,height=8)
        # START_HOOK
    def next_frame(self):
        # READ_HOOK
        if self.sequence==3:raise StopIteration
        i=self.sequence;self.sequence+=1;start=i*10000;end=start+10000
        packet=struct.pack('<8sHHQQI',b'GECD001\\n',16,8,start,end,1)+struct.pack('<HHBQ',1,2,1,end)
        packet+=struct.pack('<I',zlib.crc32(packet))
        c=self.selected
        sensor=SensorMetadata(c['source_id'],c['sensor_id'],i,coordinate_frame=c['coordinate_frame'],calibration_id=c['calibration_id'])
        return InputFrame(packet,'events',c['capture_origin_ms']+end//1000,c.get('subject_id','local-user'),sensor=sensor)
    def close(self):
        # CLOSE_HOOK
        pass
w._construct=lambda config:Source(w.configuration(config))
# PROTOCOL_HOOK
w.main()
'''


class Contracts(unittest.TestCase):
    def test_discovery_and_wire_limits(self):
        import subprocess
        result=subprocess.run([sys.executable,'-c',
            'from humansense.openeb_process import configuration;import sys;configuration('+repr(CONFIG)+');assert not {"numpy","metavision_core","torch"}&sys.modules.keys()'],capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        for change in ({'source':''},{'kind':'auto'},{'sensor_origin_us':'1'},{'capture_origin_ms':True},{'extra':1}):
            with self.assertRaises(ValueError):configuration(dict(CONFIG,**change))
        for data in (b'{"id":1,"id":2}',b'{"a":NaN}',b'x',b' '*(16385)):
            with self.assertRaises(ValueError):header(data)
        for values in ((0,0),(16385,0),(100,106533)):
            with self.assertRaises(ValueError):prefix(struct.pack('!II',*values))


@unittest.skipIf(np is None,'optional NumPy absent')
class Owned(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.real_spawn=asyncio.create_subprocess_exec;self.children=[]

    async def asyncTearDown(self):
        leaked=[p for p in self.children if p.returncode is None]
        for p in leaked:p.kill();await p.wait()
        self.assertEqual(leaked,[],'capture/context child leaked')

    def fixture(self,code=FAKE,context_code=None):
        async def spawn(*args,**kwargs):
            if 'humansense.openeb_capture_worker' in args:
                child=await self.real_spawn(sys.executable,'-u','-c',code,**kwargs)
            elif context_code is not None and 'humansense.provider_worker' in args:
                child=await self.real_spawn(sys.executable,'-u','-c',context_code,**kwargs)
            else:child=await self.real_spawn(*args,**kwargs)
            self.children.append(child);return child
        return patch('asyncio.create_subprocess_exec',spawn)

    async def test_roundtrip_epoch_eof_and_explicit_reopen(self):
        with self.fixture():
            source=await OpenEBProcess.open(sys.executable,CONFIG)
            self.assertIsNone(source._transport._separation_owner)
            epoch=source.epoch
            for i in range(3):
                f=await source.next_frame()
                self.assertEqual(f.sensor.sequence,i);self.assertEqual(f.sensor.capture_sequence,i)
                self.assertEqual(f.sensor.stream_epoch,epoch);self.assertEqual(f.timestamp_ms,1010+10*i)
            with self.assertRaises(StopAsyncIteration):await source.next_frame()
            self.assertTrue(source.diagnostics()['reaped']);self.assertEqual(source.stop_reason,'eof')
            await source.close();await source.close()
            with self.assertRaises(RuntimeError):await source.next_frame()
            second=await OpenEBProcess.open(sys.executable,CONFIG)
            self.assertNotEqual(second.epoch,epoch)
            self.assertIsNone((await second.next_frame(managed=True)).sensor.stream_epoch)
            await second.close();self.assertTrue(second.diagnostics()['graceful_acknowledged'])
            self.assertFalse(second.diagnostics()['native_cleanup_verified'])

    async def test_startup_timeout_and_cancel_reap_spawned_child(self):
        code=FAKE.replace('# START_HOOK','time.sleep(60)')
        with self.fixture(code):
            with self.assertRaises(asyncio.TimeoutError):
                await OpenEBProcess.open(sys.executable,CONFIG,startup_timeout=.1)
            self.assertIsNotNone(self.children[-1].returncode)
            task=asyncio.create_task(OpenEBProcess.open(sys.executable,CONFIG))
            count=len(self.children)
            for _ in range(100):
                if len(self.children)>count:break
                await asyncio.sleep(.01)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):await task
            self.assertIsNotNone(self.children[-1].returncode)

    async def test_spawn_completion_and_caller_cancel_preserves_cancellation(self):
        loop=asyncio.get_running_loop()
        async def spawn(*args,**kwargs):
            child=await self.real_spawn(sys.executable,'-u','-c',
                FAKE.replace('# START_HOOK','time.sleep(60)'),**kwargs)
            self.children.append(child)
            # Let the shield see spawn completion, then cancel its waiter in
            # that same turn. wait_for on Python 3.10 used to swallow this.
            loop.call_soon(loop.call_soon,task.cancel)
            return child
        with patch('asyncio.create_subprocess_exec',spawn):
            task=asyncio.create_task(OpenEBProcess.open(sys.executable,CONFIG,startup_timeout=.5))
            with self.assertRaises(asyncio.CancelledError):await task
        self.assertEqual(len(self.children),1)
        self.assertIsNotNone(self.children[0].returncode)

    async def test_read_timeout_allows_unrelated_async_progress(self):
        code=FAKE.replace('# READ_HOOK','time.sleep(60)')
        with self.fixture(code):
            source=await OpenEBProcess.open(sys.executable,CONFIG,read_timeout=.15)
            ticks=0
            async def heartbeat():
                nonlocal ticks
                for _ in range(15):ticks+=1;await asyncio.sleep(.01)
            beat=asyncio.create_task(heartbeat())
            with self.assertRaises(asyncio.TimeoutError):await source.next_frame()
            await beat
            self.assertEqual(ticks,15);self.assertTrue(source.diagnostics()['reaped'])
            self.assertEqual(source.stop_reason,'read_timeout')
            await source.close()

    async def test_close_interrupts_blocked_read_without_waiting_read_timeout(self):
        code=FAKE.replace('# READ_HOOK','time.sleep(60)')
        with self.fixture(code):
            source=await OpenEBProcess.open(sys.executable,CONFIG,read_timeout=60)
            task=asyncio.create_task(source.next_frame())
            await asyncio.sleep(.03)
            with self.assertRaises(RuntimeError):await source.next_frame()
            await asyncio.wait_for(source.close(),4)
            with self.assertRaises((asyncio.CancelledError,RuntimeError)):await task
            self.assertTrue(source.diagnostics()['reaped'])
            self.assertEqual(source.frames_received,0)

    async def test_cancel_read_and_repeated_cancel_close_reap(self):
        code=FAKE.replace('# READ_HOOK','time.sleep(60)')
        with self.fixture(code):
            source=await OpenEBProcess.open(sys.executable,CONFIG)
            task=asyncio.create_task(source.next_frame());await asyncio.sleep(.03);task.cancel()
            with self.assertRaises(asyncio.CancelledError):await task
            self.assertTrue(source.diagnostics()['reaped'])
            await source.close()
        if sys.platform=='win32':return
        code=FAKE.replace('import os,time,struct,zlib','import os,time,struct,zlib,signal\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)')
        code=code.replace('# CLOSE_HOOK','time.sleep(60)')
        with self.fixture(code):
            source=await OpenEBProcess.open(sys.executable,CONFIG,close_timeout=.05)
            child=self.children[-1]
            task=asyncio.create_task(source.close());await asyncio.sleep(.1);task.cancel()
            await asyncio.sleep(.03);task.cancel()
            with self.assertRaises(asyncio.CancelledError):await asyncio.wait_for(task,4)
            self.assertEqual(child.returncode,-9)
            self.assertTrue(source.diagnostics()['reaped'])
            self.assertTrue(child._transport.is_closing())

    async def test_protocol_corruption_never_publishes_and_reaps(self):
        mutations=(
            "h['id']+=1",
            "h['result']['frame']['metadata']['sensor']['source_id']='other'",
            "h['result']['frame']['metadata']['timestamp_ms']+=1",
            "p=p[:-1]+bytes([p[-1]^1])",
            "return struct.pack('!II',16385,0)",
            "return struct.pack('!II',10,0)+b'{'", # partial response then worker awaits another request
        )
        for change in mutations:
            hook="""
original=w.envelope
def corrupt(h,p=b''):
    if h.get('ok') and isinstance(h.get('result'),dict) and 'frame' in h['result']:
        CHANGE
    return original(h,p)
w.envelope=corrupt
""".replace('CHANGE',change)
            with self.fixture(FAKE.replace('# PROTOCOL_HOOK',hook)):
                source=await OpenEBProcess.open(sys.executable,CONFIG,read_timeout=.1)
                with self.assertRaises((ValueError,RuntimeError,asyncio.TimeoutError)):await source.next_frame()
                self.assertEqual(source.frames_received,0);self.assertTrue(source.diagnostics()['reaped'])
                await source.close()

    async def test_abrupt_exit_and_native_output_flood(self):
        with self.fixture(FAKE.replace('# READ_HOOK','os._exit(17)')):
            source=await OpenEBProcess.open(sys.executable,CONFIG)
            with self.assertRaises(asyncio.IncompleteReadError):await source.next_frame()
            self.assertEqual(source.diagnostics()['child_returncode'],17)
        code=FAKE.replace('# READ_HOOK',"os.write(1,b'x'*1000000);os.write(2,b'y'*1000000)")
        with self.fixture(code):
            source=await OpenEBProcess.open(sys.executable,CONFIG)
            frame=await source.next_frame();self.assertEqual(frame.sensor.sequence,0)
            await source.close();self.assertEqual(source.diagnostics()['stderr_retained_bytes'],0)

    async def test_owned_source_to_owned_context_sensorstreams_mcp(self):
        from humansense.schema import timestamp
        from humansense.mcp import MCPServer
        with self.fixture():
            source=await OpenEBProcess.open(sys.executable,dict(CONFIG,capture_origin_ms=timestamp()-100))
            async with HumanSense(consent=True) as human:
                streams=SensorStreams(human);spec=ProcessProvider.capability('openeb.eventcd',source.config)
                async def factory():return await ProcessProvider.open(sys.executable,'openeb.eventcd',source.config)
                streams.register(spec,factory);await streams.open('cam','events',[spec.id])
                for _ in range(3):
                    await streams.submit(await source.next_frame(managed=True));await streams.drain()
                record=human.store.latest_signals('local-user',0,timestamp())[0]
                self.assertEqual(record['observation']['value']['activity']['event_count'],1)
                self.assertEqual(record['observation']['sensor']['capture_sequence'],2)
                result=await MCPServer(human).call_tool('get_observations',{'capability_id':spec.id})
                self.assertIn('polarity_grid',str(result))
                await streams.clear()
            await source.close();self.assertEqual(len(self.children),2)

    async def test_bounded_example_emits_context_and_reaps(self):
        import argparse
        import contextlib
        import io
        import json
        from examples.openeb_stream import run
        from humansense.schema import timestamp
        args=argparse.Namespace(**dict(CONFIG,capture_origin_ms=timestamp()-100,
            capture_python=sys.executable,context_python=sys.executable,
            read_timeout=5,pacing='none',max_frames=3,context_interval_ms=10))
        output=io.StringIO()
        with self.fixture(),contextlib.redirect_stdout(output):
            await run(args)
        records=[json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(records[0]['selection']['source'],'FIXTURE_ONLY')
        self.assertTrue(any('context' in value for value in records))
        self.assertTrue(records[-1]['capture_diagnostics']['reaped'])
        self.assertEqual(len(self.children),2)

    async def test_example_cadence_continues_during_stalled_read_and_stops(self):
        import argparse
        import contextlib
        import io
        from examples.openeb_stream import run
        from humansense.schema import timestamp
        from humansense.mcp import MCPServer
        args=argparse.Namespace(**dict(CONFIG,capture_origin_ms=timestamp()-100,
            capture_python=sys.executable,context_python=sys.executable,
            read_timeout=.3,pacing='none',max_frames=3,context_interval_ms=20))
        calls=[];original=MCPServer.call_tool
        async def observe(server,*a,**kw):
            calls.append(asyncio.get_running_loop().time())
            return await original(server,*a,**kw)
        before=set(asyncio.all_tasks())
        with self.fixture(FAKE.replace('# READ_HOOK','time.sleep(60)')), \
                patch.object(MCPServer,'call_tool',observe),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(asyncio.TimeoutError):await run(args)
        self.assertGreaterEqual(len(calls),5)
        self.assertGreater(calls[-1]-calls[0],.15)
        self.assertTrue(all(p.returncode is not None for p in self.children))
        count=len(calls);await asyncio.sleep(.05)
        self.assertEqual(len(calls),count)
        self.assertFalse([t for t in asyncio.all_tasks()-before if not t.done()])

    async def test_example_context_child_failure_propagates_and_reaps(self):
        import argparse
        import contextlib
        import io
        import json
        from examples.openeb_stream import run
        from humansense.schema import timestamp
        args=argparse.Namespace(**dict(CONFIG,capture_origin_ms=timestamp()-100,
            capture_python=sys.executable,context_python=sys.executable,
            read_timeout=5,pacing='none',max_frames=3,context_interval_ms=20))
        code="import os;from humansense.providers.openeb import OpenEBContext;OpenEBContext.process=lambda *a:os._exit(17);from humansense.provider_worker import main;main()"
        output=io.StringIO();before=set(asyncio.all_tasks())
        with self.fixture(context_code=code),contextlib.redirect_stdout(output):
            with self.assertRaisesRegex(RuntimeError,'OpenEB context worker failed'):
                await run(args)
        records=[json.loads(line) for line in output.getvalue().splitlines()]
        self.assertFalse(any('final_context' in record for record in records))
        self.assertEqual(records[-1]['capture_diagnostics']['frames_received'],1)
        self.assertEqual([p.returncode for p in self.children],[0,17])
        self.assertFalse([t for t in asyncio.all_tasks()-before if not t.done()])

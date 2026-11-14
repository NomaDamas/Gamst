import asyncio
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from humansense import HumanSense, SensorStreams
from humansense.mcp import MCPServer
from humansense.providers.tactile import (TactileSource, TactileInputContext, TactileWindow,
    tactile_frame, decode_tactile, recording_frames, MAGIC, HEADER_LIMIT, PACKET_LIMIT)
from humansense.providers.tactile_capture import TactileCapture
from humansense.schema import timestamp


def source(**changes):
    values = dict(source_id='touch', sensor_id='finger', sensor_model='gelsight_mini', serial='declared-serial',
        source_epoch='capture-1', calibration_id='reference-1', illumination_id='lighting-1',
        coordinate_frame='native-bgr', clock_id='host-unix', width=2, height=2, reference=bytes(12))
    values.update(changes)
    return TactileSource(**values)


def frame(seq=0, *, binding=None, ts=None, raw=bytes(range(12))):
    return tactile_frame(raw, binding or source(), sequence=seq, timestamp_ms=1000+seq*30 if ts is None else ts,
        read_start_ms=990+seq*30 if ts is None else ts, read_end_ms=1000+seq*30 if ts is None else ts,
        acquisition='recorded_capture', backend='fixture', backend_version='1', backend_revision='fixture')


def mutate(packet, **changes):
    meta, raw = decode_tactile(packet)
    meta.update(changes)
    header = json.dumps(meta).encode()
    return MAGIC + struct.pack('!I', len(header)) + header + raw


class TactilePackets(unittest.TestCase):
    def test_raw_binding_and_context_never_infer_touch_or_force(self):
        original = frame()
        meta, raw = decode_tactile(original.data)
        self.assertEqual(raw, bytes(range(12)))
        self.assertEqual(meta['reference_sha256'], hashlib.sha256(bytes(12)).hexdigest())
        result = TactileInputContext().process(original)['capture']
        self.assertEqual(result['native_shape'], [2, 2, 3])
        self.assertEqual(result['preprocessing'], 'none')
        self.assertFalse(result['task_inference_performed'])
        self.assertFalse(result['physical_calibration_verified'])
        self.assertFalse(result['reference_verified_as_no_contact'])
        for key in ('contact', 'slip', 'force', 'pixels', 'raw', 'reference'):
            self.assertNotIn(key, result)
        TactileInputContext.capability().validate({'capture': result})

    def test_exact_stride_pair_order_and_bounded_history(self):
        window = TactileWindow()
        for i in range(30):
            meta, pair, reason = window.push(frame(i))
            self.assertLessEqual(len(window.frames), 6)
            if i < 5: self.assertIsNone(pair)
            else:
                self.assertEqual([f.sensor.sequence for f in pair], [i, i-5])
                self.assertEqual([f.timestamp_ms for f in pair], [1000+i*30, 1000+(i-5)*30])

    def test_identity_reference_clock_geometry_and_gap_resets(self):
        for change in (dict(source_epoch='next'), dict(reference=b'x'*12), dict(illumination_id='new'),
                       dict(clock_id='other'), dict(calibration_id='new'), dict(color_order='RGB'),
                       dict(width=1, height=4)):
            window = TactileWindow()
            for i in range(6): window.push(frame(i))
            _, pair, reason = window.push(frame(6, binding=source(**change)))
            self.assertIsNone(pair); self.assertEqual(reason, 'source_or_reference_change')
        for seq, ts in ((7, 1210), (6, 1500)):
            window = TactileWindow()
            for i in range(6): window.push(frame(i))
            _, pair, reason = window.push(frame(seq, ts=ts))
            self.assertIsNone(pair); self.assertEqual(reason, 'gap')
        window = TactileWindow(); window.push(frame(1))
        with self.assertRaises(ValueError): window.push(frame(0))
        self.assertEqual(len(window.frames), 0)

    def test_bad_packets_and_bound_provenance_fail_closed(self):
        original = frame()
        bad = [b'', original.data[:-1], original.data+b'x', b'x'*(PACKET_LIMIT+1),
               MAGIC+struct.pack('!I', HEADER_LIMIT+1)+b'{}',
               mutate(original.data, sequence=True), mutate(original.data, timestamp_ms=-1),
               mutate(original.data, width=641), mutate(original.data, pixels_sha256='0'*64),
               mutate(original.data, extra='bad'), mutate(original.data, acquisition='sensor-exposure'),
               mutate(original.data, color_order='gray'), mutate(original.data, configured_fps=True)]
        for packet in bad:
            with self.subTest(length=len(packet)), self.assertRaises(ValueError): decode_tactile(packet)
        provider = TactileInputContext()
        for changed in (replace(original, modality='video'), replace(original, timestamp_ms=42),
                        replace(original, subject_id='other'),
                        replace(original, sensor=replace(original.sensor, calibration_id='other'))):
            with self.assertRaises(ValueError): provider.process(changed)
        provider.close()
        with self.assertRaises(RuntimeError): provider.process(original)
        with self.assertRaises(ValueError): source(reference=b'')
        with self.assertRaises(ValueError): source(sensor_model='webcam')

    def test_recording_reader_and_actual_replay_command(self):
        with tempfile.TemporaryDirectory(dir='.') as temp:
            root = Path(temp).resolve(); entries = []
            for i in range(6):
                payload = frame(i).data; name = f'touch{i}.gtact'
                (root/name).write_bytes(payload)
                entries.append(dict(file=name, sha256=hashlib.sha256(payload).hexdigest()))
            manifest = root/'recording.json'
            manifest.write_text(json.dumps(dict(schema_version=1, packets=entries)))
            self.assertEqual(len(list(recording_frames(str(manifest)))), 6)
            result = subprocess.run([sys.executable, '-m', 'humansense.tactile_replay', '--manifest', str(manifest)],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = [json.loads(line)['capture'] for line in result.stdout.splitlines()]
            self.assertEqual(len(rows), 6); self.assertTrue(rows[-1]['window_ready'])
            self.assertEqual(rows[-1]['pair_timestamps_ms'], [1150, 1000])
            (root/'touch0.gtact').write_bytes(b'bad')
            with self.assertRaises(ValueError): list(recording_frames(str(manifest)))
            for name in ('../outside.gtact', 'https://x', 'a/b.gtact'):
                manifest.write_text(json.dumps(dict(schema_version=1, packets=[dict(file=name, sha256='0'*64)])))
                with self.assertRaises(ValueError): list(recording_frames(str(manifest)))
            manifest.write_text('{"schema_version":1,"schema_version":1,"packets":[]}')
            with self.assertRaises(ValueError): list(recording_frames(str(manifest)))


class Gate:
    def __init__(self, value=None, error=None):
        self.entered = asyncio.Event(); self.allow = threading.Event()
        self.loop = asyncio.get_running_loop(); self.value = value; self.error = error
    def __call__(self, *args, **kwargs):
        self.loop.call_soon_threadsafe(self.entered.set)
        if not self.allow.wait(5): raise RuntimeError('fixture gate not opened')
        if self.error: raise self.error
        return self.value


class Camera:
    def __init__(self, pixels=None): self.pixels=pixels; self.calls=[]
    def isOpened(self): return True
    def set(self, prop, value): self.calls.append(('set', prop, value))
    def read(self): return True, self.pixels
    def release(self): self.calls.append('release')


def cv(factory):
    return SimpleNamespace(VideoCapture=factory, CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4, __version__='fixture-1')


class TactileNative(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.errors=[]; self.loop=asyncio.get_running_loop(); self.old=self.loop.get_exception_handler()
        self.loop.set_exception_handler(lambda loop, context: self.errors.append(context))
    async def asyncTearDown(self):
        await asyncio.sleep(0); self.loop.set_exception_handler(self.old)
        self.assertEqual(self.errors, [])

    async def test_cancel_or_concurrent_close_drains_late_constructor(self):
        for cancel in (True, False):
            camera=Camera(); gate=Gate(camera)
            capture=TactileCapture(source(), consent=True)
            with patch('humansense.providers.tactile_capture._opencv', return_value=cv(gate)):
                opening=asyncio.create_task(capture.open_gelsight(device=3)); closing=None
                try:
                    await asyncio.wait_for(gate.entered.wait(), 2)
                    if cancel:
                        opening.cancel(); await asyncio.sleep(0); opening.cancel()
                    closing=asyncio.create_task(capture.close()); await asyncio.sleep(0)
                    self.assertFalse(opening.done()); self.assertFalse(closing.done())
                    self.assertNotIn('release', camera.calls)
                    gate.allow.set()
                    with self.assertRaises(asyncio.CancelledError if cancel else RuntimeError): await opening
                    await closing; await capture.close()
                    self.assertEqual(camera.calls.count('release'), 1)
                    self.assertFalse(capture.status()['opened'])
                finally:
                    gate.allow.set()
                    await asyncio.gather(*(t for t in (opening,closing) if t is not None), return_exceptions=True)
                    await capture.close()

    async def test_cancel_read_serializes_release_and_returns_no_late_frame(self):
        camera=Camera(); gate=Gate((True,None)); camera.read=gate
        capture=TactileCapture(source(), consent=True)
        with patch('humansense.providers.tactile_capture._opencv', return_value=cv(lambda _:camera)):
            await capture.open_gelsight(device=0)
            reading=asyncio.create_task(capture.next_frame()); closing=None
            try:
                await asyncio.wait_for(gate.entered.wait(), 2)
                reading.cancel(); await asyncio.sleep(0)
                closing=asyncio.create_task(capture.close()); await asyncio.sleep(0); closing.cancel()
                self.assertNotIn('release', camera.calls)
                gate.allow.set()
                with self.assertRaises(asyncio.CancelledError): await reading
                with self.assertRaises(asyncio.CancelledError): await closing
                self.assertEqual(capture.sequence, 0); self.assertEqual(camera.calls.count('release'), 1)
            finally:
                gate.allow.set()
                await asyncio.gather(*(t for t in (reading,closing) if t is not None), return_exceptions=True)
                await capture.close()

    async def test_digit_uses_exact_raw_api_and_cleans_partial_start_failure(self):
        calls=[]
        class Digit:
            STREAMS={'QVGA': {'resolution': {'width':320,'height':240}}}
            def __init__(self, serial, name): calls.append(('constructor',serial,name))
            def connect(self): calls.append('connect')
            def set_resolution(self, value): calls.append(('resolution',value))
            def set_fps(self, value): calls.append(('fps',value))
            def set_intensity(self, value): raise RuntimeError('lighting fixture failure')
            def disconnect(self): calls.append('disconnect')
        binding=source(sensor_model='digit', width=320,height=240,reference=bytes(320*240*3),
                       configured_fps=30,configured_led_intensity=15)
        capture=TactileCapture(binding,consent=True)
        with patch('humansense.providers.tactile_capture._digit_class',return_value=Digit), \
                patch('humansense.providers.tactile_capture._opencv',return_value=cv(None)):
            with self.assertRaisesRegex(RuntimeError,'lighting fixture failure'): await capture.open_digit()
            await capture.close()
            self.assertEqual(calls.count('disconnect'),1)
            self.assertIn(('fps',30),calls)
            self.assertIn('lighting fixture failure',capture.status()['errors'][0])

    async def test_digit_cancelled_connect_retains_handle_until_disconnect(self):
        calls=[]; gate=Gate(error=ValueError('late connect failure'))
        class Digit:
            STREAMS={}
            def __init__(self, serial, name): calls.append('constructor')
            def connect(self): return gate()
            def disconnect(self): calls.append('disconnect')
        binding=source(sensor_model='digit',width=320,height=240,reference=bytes(320*240*3),
                       configured_fps=30,configured_led_intensity=15)
        capture=TactileCapture(binding,consent=True)
        with patch('humansense.providers.tactile_capture._digit_class',return_value=Digit), \
                patch('humansense.providers.tactile_capture._opencv',return_value=cv(None)):
            opening=asyncio.create_task(capture.open_digit())
            try:
                await asyncio.wait_for(gate.entered.wait(),2)
                opening.cancel(); await asyncio.sleep(0); opening.cancel(); await asyncio.sleep(0)
                self.assertEqual(calls,['constructor']); self.assertFalse(opening.done())
                gate.allow.set()
                with self.assertRaises(asyncio.CancelledError): await opening
                self.assertEqual(calls,['constructor','disconnect'])
                self.assertEqual(capture.status()['errors'],['open: late connect failure'])
            finally:
                gate.allow.set(); await asyncio.gather(opening,return_exceptions=True); await capture.close()

    async def test_digit_raw_read_and_release_error_are_observable(self):
        calls=[]
        class Digit:
            STREAMS={'QVGA': {'resolution': {'width':320,'height':240}}}
            def __init__(self, serial, name): pass
            def connect(self): pass
            def set_resolution(self, value): pass
            def set_fps(self, value): pass
            def set_intensity(self, value): pass
            def get_frame(self, *, transpose=False):
                calls.append(('transpose',transpose))
                raise ValueError('fixture read failure')
            def disconnect(self):
                calls.append('disconnect')
                raise RuntimeError('fixture release failure')
        binding=source(sensor_model='digit',width=320,height=240,reference=bytes(320*240*3),
                       configured_fps=30,configured_led_intensity=15)
        capture=TactileCapture(binding,consent=True)
        with patch('humansense.providers.tactile_capture._digit_class',return_value=Digit), \
                patch('humansense.providers.tactile_capture._opencv',return_value=cv(None)):
            await capture.open_digit()
            with self.assertRaisesRegex(ValueError,'fixture read failure'): await capture.next_frame()
            self.assertIn(('transpose',True),calls)
            self.assertEqual(calls.count('disconnect'),1)
            self.assertEqual(capture.status()['errors'],['read: fixture read failure','release: fixture release failure'])
            await capture.close()
            self.assertEqual(calls.count('disconnect'),1)

    async def test_close_cancellation_drains_release_once_and_records_failure(self):
        camera=Camera(); gate=Gate(error=RuntimeError('delayed release failure')); camera.release=gate
        capture=TactileCapture(source(),consent=True)
        with patch('humansense.providers.tactile_capture._opencv',return_value=cv(lambda _:camera)):
            await capture.open_gelsight(device=0)
            closing=asyncio.create_task(capture.close())
            try:
                await asyncio.wait_for(gate.entered.wait(),2)
                closing.cancel(); await asyncio.sleep(0); closing.cancel(); await asyncio.sleep(0)
                self.assertFalse(closing.done())
                gate.allow.set()
                with self.assertRaises(asyncio.CancelledError): await closing
                with self.assertRaisesRegex(RuntimeError,'delayed release failure'): await capture.close()
                self.assertEqual(capture.status()['errors'],['release: delayed release failure'])
            finally:
                gate.allow.set(); await asyncio.gather(closing,return_exceptions=True)

    async def test_native_pixels_exact_and_shape_mismatch_closes(self):
        try: import numpy as np
        except ImportError: self.skipTest('optional NumPy unavailable')
        pixels=np.arange(12,dtype=np.uint8).reshape(2,2,3)
        camera=Camera(pixels); capture=TactileCapture(source(),consent=True)
        with patch('humansense.providers.tactile_capture._opencv',return_value=cv(lambda _:camera)):
            await capture.open_gelsight(device=4)
            result=await capture.next_frame(); meta,raw=decode_tactile(result.data)
            self.assertEqual(raw,pixels.tobytes()); self.assertEqual(meta['acquisition'],'host_read_end')
            self.assertEqual(meta['timestamp_ms'],meta['read_end_ms'])
            self.assertEqual(meta['backend_revision'],'321d6a22da64529138ff10237335038fd8c5189f')
            camera.pixels=np.zeros((3,2,3),dtype=np.uint8)
            with self.assertRaises(ValueError): await capture.next_frame()
            self.assertEqual(camera.calls.count('release'),1)
            await capture.close()

    async def test_opt_in_and_explicit_selection_before_native_access(self):
        with self.assertRaises(PermissionError): TactileCapture(source())
        capture=TactileCapture(source(),consent=True)
        with patch('humansense.providers.tactile_capture._opencv') as backend:
            for device in (None, True, -1, 'https://camera', '/tmp/recording.mp4'):
                with self.assertRaises(ValueError): await capture.open_gelsight(device=device)
            backend.assert_not_called()
        await capture.close()


class TactileRuntime(unittest.IsolatedAsyncioTestCase):
    async def test_typed_input_metadata_reaches_context(self):
        async with HumanSense(consent=True) as human:
            streams=SensorStreams(human); spec=TactileInputContext.capability()
            streams.register(spec,TactileInputContext)
            await streams.open('touch','finger',[spec.id])
            await streams.submit(frame(ts=timestamp()))
            await streams.drain()
            observation=next(iter((await human.get_context())['observations'].values()))
            capture=observation['value']['capture']
            self.assertEqual(capture['source_epoch'],'capture-1')
            self.assertFalse(capture['task_inference_performed'])
            self.assertEqual(observation['sensor']['calibration_id'],'reference-1')
            server=MCPServer(human)
            response=await server.handle({'jsonrpc':'2.0','id':1,'method':'tools/call',
                'params':{'name':'get_context','arguments':{}}})
            self.assertFalse(response['result']['isError'])
            context=json.loads(response['result']['content'][0]['text'])
            via_mcp=next(iter(context['observations'].values()))
            self.assertEqual(via_mcp['value']['capture'],capture)
            await server.close()

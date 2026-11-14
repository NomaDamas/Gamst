import asyncio
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

from humansense import HumanSense, ProcessProvider, SensorStreams
from humansense.mcp import MCPServer
from humansense.schema import timestamp
from humansense.providers.sparsh_context import SparshSlipContext, preprocess_pair, profile_value
from humansense.providers.tactile import TactileSource, tactile_frame

AVAILABLE=all(importlib.util.find_spec(x) for x in ('numpy','PIL'))
REFERENCE=bytes([100,120,140]*12)


def profile(**changes):
    value=dict(profile_id='test-training',source_id='touch',sensor_id='finger',sensor_model='digit',
        serial='serial-1',width=4,height=3,color_order='BGR',calibration_id='cal-1',illumination_id='light-1',
        coordinate_frame='native',clock_id='recording',reference_sha256=hashlib.sha256(REFERENCE).hexdigest(),
        orientation='landscape_cw',training_config_sha256='a'*64,training_domain_id='declared-fixture',
        min_interval_ms=29,max_interval_ms=31,min_pair_span_ms=145,max_pair_span_ms=155,reference_consent=True)
    value.update(changes);return value


def config(**changes):
    value=dict(source_root='/supplied/source',encoder_path='/supplied/encoder',head_path='/supplied/head',
        encoder_sha256='b'*64,head_sha256='c'*64,reference_path='/supplied/reference',profile=profile())
    value.update(changes);return value


def frame(seq, *, ts=None, source_epoch='capture-1', runtime_epoch=None, **changes):
    p=profile()
    values={key:p[key] for key in ('source_id','sensor_id','sensor_model','serial','calibration_id',
        'illumination_id','coordinate_frame','clock_id','width','height','color_order')}
    values.update(source_epoch=source_epoch,reference=REFERENCE);values.update(changes)
    source=TactileSource(**values)
    now=1000+seq*30 if ts is None else ts
    result=tactile_frame(bytes([110,120,130]*12),source,sequence=seq,timestamp_ms=now,
        read_start_ms=now,read_end_ms=now,acquisition='recorded_capture',backend='fixture',
        backend_version='1',backend_revision='fixture')
    return replace(result,sensor=replace(result.sensor,stream_epoch=runtime_epoch))


class Native:
    def __init__(self):self.inputs=[];self.closed=0;self.output=None
    def predict(self,pair):
        import numpy as np
        self.inputs.append(pair.copy())
        return self.output if self.output is not None else dict(slip_logits=np.array([0,2],np.float32),normalized_delta_force=np.array([-.3,0,.5],np.float32))
    def close(self):self.closed+=1


def context(native=None, **changes):
    return SparshSlipContext(native or Native(),REFERENCE,encoder_sha256='b'*64,head_sha256='c'*64,profile=profile(**changes))


class Discovery(unittest.TestCase):
    def test_dependency_free_profile_and_explicit_reference_consent(self):
        code='from humansense import ProcessProvider;import sys;ProcessProvider.capability("sparsh.slip.local",'+repr(config())+');assert not {"torch","numpy","PIL","cv2"}&sys.modules.keys()'
        result=subprocess.run([sys.executable,'-c',code],capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        for change in ({'reference_consent':False},{'min_interval_ms':True},{'min_pair_span_ms':144},
                       {'orientation':'unverified'},{'width':3},{'clock_id':''},{'reference_sha256':'bad'}):
            with self.assertRaises(ValueError):profile_value(profile(**change))
        with self.assertRaises(ValueError):ProcessProvider.capability('sparsh.slip.local',config(extra=True))
        with self.assertRaises(ValueError):ProcessProvider.capability('sparsh.slip.local',config(reference_path='relative'))
        with patch('humansense.providers.sparsh_context.SparshSlipNative.from_local') as native:
            with self.assertRaises(ValueError):SparshSlipContext.from_local(**config(profile=profile(reference_consent=False)))
            native.assert_not_called()


@unittest.skipUnless(AVAILABLE,'optional NumPy/Pillow absent')
class Context(unittest.TestCase):
    def test_preprocessing_color_order_signed_diff_quantization_and_pair_order(self):
        import numpy as np
        p=profile()
        same=preprocess_pair(REFERENCE,REFERENCE,REFERENCE,p)
        self.assertEqual(same.shape,(1,6,320,240));self.assertEqual(same.dtype,np.float32)
        np.testing.assert_array_equal(same,np.float32(127/255))
        current=bytes([0,255,140]*12);previous=bytes([255,0,140]*12)
        pair=preprocess_pair(current,previous,REFERENCE,p)
        # BGR→RGB: channel0 unchanged, signed/clipped green and blue differ.
        np.testing.assert_array_equal(pair[0,0],np.float32(127/255))
        np.testing.assert_array_equal(pair[0,1],1)
        np.testing.assert_array_equal(pair[0,2],np.float32(27/255))
        np.testing.assert_array_equal(pair[0,4],np.float32(7/255))
        np.testing.assert_array_equal(pair[0,5],1)
        rgb=lambda b:np.frombuffer(b,np.uint8).reshape(3,4,3)[:,:,::-1].tobytes()
        rgbp=profile(color_order='RGB',reference_sha256=hashlib.sha256(rgb(REFERENCE)).hexdigest())
        np.testing.assert_array_equal(preprocess_pair(rgb(current),rgb(previous),rgb(REFERENCE),rgbp),pair)

    def test_stride_warmup_policy_and_evidence(self):
        native=Native();provider=context(native)
        for seq in range(8):
            result=provider.process(frame(seq))
            c,e=result['classification'],result['evidence']
            if seq<5:
                self.assertEqual(c['status'],'warming');self.assertIsNone(c['debounced_slip'])
            else:
                self.assertEqual(e['pair_capture_sequences'],[seq,seq-5])
                self.assertEqual(e['pair_timestamps_ms'],[1000+seq*30,1000+(seq-5)*30])
                self.assertEqual(e['capture_pair_span_ms'],150)
                self.assertTrue(c['candidate_slip']);self.assertEqual(c['debounced_slip'],seq==7)
                self.assertEqual(c['contact_state'],'not_assessed')
                self.assertFalse(e['trained_quality_verified']);self.assertFalse(e['native_artifacts_verified'])
                self.assertFalse(e['reference_no_contact_verified'])
                self.assertNotIn('force_newtons',c)
        self.assertEqual(len(native.inputs),3)
        self.assertEqual([x['pair_capture_sequences'] for x in e['debounce_evidence']],[[5,0],[6,1],[7,2]])
        provider.reset()
        self.assertEqual(provider.process(frame(8))['classification']['status'],'warming')
        provider.close();provider.close();self.assertEqual(native.closed,1)
        with self.assertRaises(RuntimeError):provider.process(frame(9))

    def test_misalignment_bad_output_and_epoch_invalidate_history(self):
        import numpy as np
        for changes in ({'source_epoch':'capture-2'},{'runtime_epoch':'runtime-2'}):
            provider=context()
            for i in range(8):provider.process(frame(i))
            result=provider.process(frame(8,**changes))
            self.assertEqual(result['classification']['status'],'warming')
            self.assertEqual(result['evidence']['buffered_captures'],1)
        for change in ({'clock_id':'different'},{'serial':'other'},{'illumination_id':'other'},
                       {'calibration_id':'other'},{'reference':bytes(36)}):
            provider=context()
            for i in range(5):provider.process(frame(i))
            with self.assertRaises(ValueError):provider.process(frame(5,**change))
            self.assertEqual(len(provider.window.frames),0)
        native=Native();provider=context(native)
        for i in range(5):provider.process(frame(i))
        native.output={'slip_logits':np.array([0,np.nan],np.float32),'normalized_delta_force':np.zeros(3,np.float32)}
        with self.assertRaises(ValueError):provider.process(frame(5))
        self.assertEqual(len(provider.window.frames),0)

    def test_no_resampling_on_gap_cadence_pair_span_and_reorder(self):
        for seq,ts in ((7,1210),(6,1160),(6,1300)):
            provider=context()
            for i in range(6):provider.process(frame(i))
            result=provider.process(frame(seq,ts=ts))
            self.assertEqual(result['classification']['status'],'warming')
            self.assertEqual(result['evidence']['buffered_captures'],1)
        provider=context(min_pair_span_ms=151,max_pair_span_ms=155)
        for i in range(6):result=provider.process(frame(i))
        self.assertEqual(result['evidence']['reset_reason'],'pair_span_outside_profile')
        self.assertEqual(result['classification']['status'],'warming')
        with self.assertRaises(ValueError):provider.process(frame(4))
        self.assertEqual(len(provider.window.frames),0)

    def test_close_and_reset_wait_for_owned_inference(self):
        for operation in ('reset','close'):
            native=Native();provider=context(native)
            for i in range(5):provider.process(frame(i))
            entered=threading.Event();release=threading.Event();attempted=threading.Event();done=threading.Event()
            original=native.predict;errors=[]
            def predict(data):
                entered.set()
                if not release.wait(5):raise AssertionError('fixture watchdog')
                return original(data)
            native.predict=predict
            def run():
                try:provider.process(frame(5))
                except BaseException as exc:errors.append(exc)
            def lifecycle():
                attempted.set();getattr(provider,operation)();done.set()
            worker=threading.Thread(target=run);worker.start()
            self.assertTrue(entered.wait(5))
            ending=threading.Thread(target=lifecycle);ending.start()
            self.assertTrue(attempted.wait(5));self.assertFalse(done.is_set())
            release.set();worker.join(5);ending.join(5)
            self.assertFalse(worker.is_alive());self.assertFalse(ending.is_alive())
            self.assertEqual(errors,[]);self.assertTrue(done.is_set())
            self.assertEqual(len(provider.window.frames),0)
            self.assertEqual(len(provider._decision_evidence),0)
            if operation=='close':self.assertEqual(native.closed,1)
            else:
                self.assertEqual(provider.process(frame(6))['classification']['status'],'warming')
                provider.close()

    def test_reference_before_native_and_worker_factory(self):
        from humansense.provider_worker import _construct
        with tempfile.TemporaryDirectory(dir='.') as tmp:
            path=Path(tmp).resolve()/'reference';path.write_bytes(REFERENCE)
            cfg=config(reference_path=str(path));native=Native()
            with patch('humansense.providers.sparsh_context.SparshSlipNative.from_local',return_value=native):
                provider,spec=_construct('sparsh.slip.local',cfg)
                self.assertEqual(spec,provider.spec);self.assertTrue(provider._verified)
                provider.close();self.assertEqual(native.closed,1)
            path.write_bytes(b'bad')
            with patch('humansense.providers.sparsh_context.SparshSlipNative.from_local') as load:
                with self.assertRaises(ValueError):SparshSlipContext.from_local(**cfg)
                load.assert_not_called()

    def test_recorded_replay_cli_path_with_fixture_native_only(self):
        with tempfile.TemporaryDirectory(dir='.') as tmp:
            root=Path(tmp).resolve();(root/'reference').write_bytes(REFERENCE)
            entries=[]
            for i in range(8):
                payload=frame(i).data;name=f'{i}.gtact';(root/name).write_bytes(payload)
                entries.append(dict(file=name,sha256=hashlib.sha256(payload).hexdigest()))
            manifest=root/'recording.json';manifest.write_text(json.dumps(dict(schema_version=1,packets=entries)))
            cfg=root/'config.json';cfg.write_text(json.dumps(config(reference_path=str(root/'reference'))))
            code='''
import numpy as np
from humansense.providers.sparsh_native import SparshSlipNative
class Fixture:
    def predict(self,x):return {'slip_logits':np.array([0,2],np.float32),'normalized_delta_force':np.zeros(3,np.float32)}
    def close(self):pass
SparshSlipNative.from_local=lambda *a,**kw:Fixture()
from humansense.sparsh_replay import main
main()
'''
            run=subprocess.run([sys.executable,'-c',code,'--manifest',str(manifest),'--config',str(cfg)],capture_output=True,text=True,timeout=15)
            self.assertEqual(run.returncode,0,run.stderr)
            rows=[json.loads(line) for line in run.stdout.splitlines()]
            self.assertEqual(len(rows),8)
            self.assertTrue(rows[-1]['classification']['debounced_slip'])
            self.assertFalse(rows[-1]['evidence']['trained_quality_verified'])


@unittest.skipUnless(AVAILABLE,'optional NumPy/Pillow absent')
class Managed(unittest.IsolatedAsyncioTestCase):
    async def test_real_owned_worker_dispatch_replay_and_close(self):
        code="""
import numpy as np
from humansense.providers.sparsh_native import SparshSlipNative
class Fixture:
    def predict(self,x):return {'slip_logits':np.array([0,2],np.float32),'normalized_delta_force':np.zeros(3,np.float32)}
    def close(self):pass
SparshSlipNative.from_local=lambda *a,**kw:Fixture()
from humansense.provider_worker import main
main()
"""
        children=[];real_spawn=asyncio.create_subprocess_exec
        async def spawn(*args,**kwargs):
            child=await real_spawn(sys.executable,'-u','-c',code,**kwargs)
            children.append(child);return child
        with tempfile.TemporaryDirectory(dir='.') as tmp:
            ref=Path(tmp).resolve()/'reference';ref.write_bytes(REFERENCE)
            cfg=config(reference_path=str(ref))
            try:
                async with HumanSense(consent=True) as human:
                    streams=SensorStreams(human)
                    spec=ProcessProvider.capability('sparsh.slip.local',cfg)
                    with patch('asyncio.create_subprocess_exec',side_effect=spawn):
                        worker=await ProcessProvider.open(sys.executable,'sparsh.slip.local',cfg,startup_timeout=10)
                    streams.register(spec,lambda:worker)
                    await streams.open('touch','finger',[spec.id])
                    now=timestamp()-300
                    for i in range(8):
                        await streams.submit(frame(i,ts=now+i*30));await streams.drain()
                    result=await human.get_context()
                    observed=next(iter(result['observations'].values()))['value']
                    self.assertTrue(observed['classification']['debounced_slip'])
                    self.assertEqual(observed['evidence']['pair_capture_sequences'],[7,2])
                    await streams.close()
                    self.assertTrue(worker.closed)
                self.assertEqual(len(children),1)
                self.assertIsNotNone(children[0].returncode)
            finally:
                for child in children:
                    if child.returncode is None:child.kill();await child.wait()

    async def test_actual_context_stream_and_mcp(self):
        async with HumanSense(consent=True) as human:
            native=Native();provider=context(native);streams=SensorStreams(human)
            streams.register(provider.spec,lambda:provider)
            await streams.open('touch','finger',[provider.spec.id])
            now=timestamp()-300
            for i in range(8):
                await streams.submit(frame(i,ts=now+i*30));await streams.drain()
            result=await human.get_context();observation=next(iter(result['observations'].values()))
            self.assertTrue(observation['value']['classification']['debounced_slip'])
            self.assertIsNotNone(observation['value']['evidence']['runtime_epoch'])
            server=MCPServer(human)
            response=await server.handle({'jsonrpc':'2.0','id':1,'method':'tools/call',
                'params':{'name':'get_context','arguments':{}}})
            via_mcp=json.loads(response['result']['content'][0]['text'])
            self.assertEqual(next(iter(via_mcp['observations'].values()))['value'],observation['value'])
            await streams.close();self.assertEqual(native.closed,1)


@unittest.skipUnless(AVAILABLE and os.environ.get('GAMST_TEST_SPARSH_PREPROCESS_SOURCE'),
                     'explicit local official preprocessing source required; never downloads')
class OfficialPreprocessing(unittest.TestCase):
    def test_actual_official_helper_pixel_and_tensor_parity(self):
        import numpy as np
        path=Path(os.environ['GAMST_TEST_SPARSH_PREPROCESS_SOURCE'])
        source=path.read_bytes()
        self.assertEqual(hashlib.sha256(source).hexdigest(),
            '8bfc5e3c255ec7617236e41ad6112e001d59badf7da94b2fe81dbf20d6b5103c')
        upstream=types.ModuleType('_verified_sparsh_preprocessing')
        exec(compile(source,'verified/digit/utils.py','exec'),upstream.__dict__)
        rng=np.random.default_rng(236)
        for h,w in ((3,4),(320,240),(480,640)):
            reference=rng.integers(0,256,(h,w,3),dtype=np.uint8)
            images=[rng.integers(0,256,(h,w,3),dtype=np.uint8) for _ in range(2)]
            p=profile(width=w,height=h,reference_sha256=hashlib.sha256(reference.tobytes()).hexdigest(),
                      orientation='landscape_cw' if w>h else 'portrait')
            actual=preprocess_pair(*(item.tobytes() for item in images),reference.tobytes(),p)
            resize=upstream.get_resize_transform((320,240))
            expected=np.concatenate([resize(upstream.load_sample_from_buf(item[:,:,::-1],reference[:,:,::-1])).numpy()
                                     for item in images],axis=0)[None]
            np.testing.assert_array_equal(actual,expected)


if __name__=='__main__':unittest.main()

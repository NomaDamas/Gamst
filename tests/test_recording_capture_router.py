"""Actual original replay -> bounded router -> observed PCM SDK provider."""
import os
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace
from benchmarks.recording_capture_router import RecordingCaptureRouter
from benchmarks.recording_runtime_profiles import builtin_profile
from benchmarks.owned_capture import OwnedCaptureSource,CaptureCapacity
from benchmarks.recorded_source import digest
from benchmarks.recording_native_validation import validate_recording_native_inventory
from gamst import Gamst,SensorStreams
from recording_native_fixtures import declarations,rehash
from benchmarks.recording_evaluation import RecordingEvaluationSession
from benchmarks.recording_native_writer import RecordingNativeWriter
from benchmarks.recording_native_schema import run_context
from benchmarks.replay_clock import ReplayClock
from test_owned_capture import Journal


class RouterTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_replay_sdk_delivery_context_and_independent_cleanup(self):
        with tempfile.TemporaryDirectory(dir='.') as temporary:
            root=Path(temporary).resolve()
            d=declarations(root,provider='gamst.pcm_level',revision='1',role='signal',boundary='in_process_backend')
            for arm in d['collection']['arms']:
                arm['provider_bindings'][0].update(operation_kinds=['process'],semantic_lane=None)
            rehash(d)
            writer=await RecordingNativeWriter.open(root/'native',run_context(d,'a'*64,0))
            owner=RecordingEvaluationSession.from_frozen(d,writer,run_nonce='a'*64,execution_ordinal=0,retain_evaluation_hashes=True)
            human=Gamst(consent=True);streams=SensorStreams(human)
            delivery=dict(profile='pcm_level',config_sha256=digest({}),source_id='mic',binding_id='semantic')
            spec,factory,config=builtin_profile(delivery,{})
            streams.register(spec,factory)
            await streams.open('mic','mic-1',[spec.id],evaluation_bindings={spec.id:owner.stream_scope(0)},evaluation_configurations={spec.id:config})
            clock=ReplayClock(d['source_manifest']);owner.bind_mapping(clock.mapping,clock.sha256)
            router=RecordingCaptureRouter(d['source_manifest'],streams=streams,source_bindings=[dict(source_id='mic',coordinate_frame=None,channels=None)],subject_id='local-user',delivery_sources=['mic'])
            fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY)
            try:source=await OwnedCaptureSource.preflight(d['source_manifest'],fd,CaptureCapacity(3200,1000000000),journal=Journal())
            finally:os.close(fd)
            try:
                await source.start(clock,router);await router.close();await streams.drain()
                context=await human.get_context()
                level=next(o for o in context['observations'].values() if o['capability']['id']=='audio.replay.level')
                self.assertEqual(level['value']['level']['audio.rms'],.125)
                self.assertEqual(source.counters()[0]['captured'],5)
                self.assertEqual(router.counters()[0]['submitted'],5)
                self.assertEqual(router.watermarks()[0]['last_submitted_sequence'],4)
                self.assertFalse(router.watermarks()[0]['processed_known'])
                await source.close();self.assertFalse(human.closed)
            finally:
                await source.close();await router.close();await human.close();await owner.close()
            index=validate_recording_native_inventory(d,writer.output,source_inventory=d['source_manifest'])
            self.assertEqual(index.count,5)

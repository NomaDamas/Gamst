"""Actual accepted store records are exported with their entered operation."""
import threading
import unittest
from gamst.evaluation_publication import AcceptedObservationSink
from gamst.native_operation import NativeOperationRef
import test_recording_stream_scopes as stream_tests


class Sink(AcceptedObservationSink):
    def __init__(self, fail=False):self.rows=[];self.fail=fail
    def record(self, record, reference):
        self.rows.append((record, reference, threading.get_ident()))
        if self.fail:raise RuntimeError('fixture exporter failure')


class PublicationTests(unittest.IsolatedAsyncioTestCase):
    setup=stream_tests.Streams.setup
    open=stream_tests.Streams.open
    capture=stream_tests.Streams.capture
    rows=stream_tests.Streams.rows
    close_native=stream_tests.Streams.close_native

    async def test_process_endpoint_finish_actual_records_and_original_entry_are_joined(self):
        sink=Sink();provider=stream_tests.Provider();await self.setup(lambda:provider,publication_sink=sink);await self.open()
        frame, ref=self.capture();await self.runtime.submit(frame,evaluation_capture_ref=ref);await self.runtime.drain()
        epoch=self.runtime.status()['streams'][0]['providers'][0]['stream_epoch']
        await self.runtime.endpoint('mic','mic-1',self.spec.id,stream_epoch=epoch,after_sequence=0)
        await self.runtime.finish('mic','mic-1');await self.close_native()
        self.assertEqual(len(sink.rows),3)
        for ordinal,(record,reference,thread) in enumerate(sink.rows):
            self.assertIs(type(reference),NativeOperationRef)
            self.assertEqual(reference.snapshot()['entry_ordinal'],ordinal)
            self.assertEqual(reference.snapshot()['evidence_sha256'],ref['record_sha256'])
            self.assertEqual(record['timestamp_ms'],frame.timestamp_ms)
            self.assertEqual(record['observation']['sensor']['capture_sequence'],0)
            self.assertEqual(thread,threading.get_ident())
        self.assertEqual([r['fields']['outcome'] for r in self.rows() if r['event']=='published'],['accepted']*3)

    async def test_failed_export_does_not_relabel_already_accepted_observation(self):
        sink=Sink(fail=True);provider=stream_tests.Provider();await self.setup(lambda:provider,publication_sink=sink);await self.open()
        frame, ref=self.capture();await self.runtime.submit(frame,evaluation_capture_ref=ref);await self.runtime.drain()
        self.assertEqual(len(sink.rows),1)
        self.assertEqual(self.runtime.status()['streams'][0]['providers'][0]['errors'],1)
        await self.close_native()
        self.assertEqual([r['fields']['outcome'] for r in self.rows() if r['event']=='published'],['accepted'])
        self.assertEqual(sink.rows[0][0]['observation']['capability']['id'],self.spec.id)

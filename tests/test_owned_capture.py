import asyncio
from dataclasses import FrozenInstanceError
import hashlib
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from benchmarks.owned_capture import (OwnedCaptureSource, CaptureCapacity, CaptureSink,
                                      DeliveryAdmission, _read_owned)
from benchmarks.recorded_source import digest
from benchmarks.replay_clock import ReplayClock
from gamst import Gamst


class Journal:
    def __init__(self): self.events = []
    def emit(self, kind, value): self.events.append((kind, value))


class Sink(CaptureSink):
    def __init__(self, accept=True): self.packets = []; self.accept = accept
    def offer(self, packet):
        if not self.accept: return DeliveryAdmission(False, 'capacity', None)
        self.packets.append(packet)
        return DeliveryAdmission(True, 'accepted', len(self.packets)-1)


def fixture(root):
    payload = b'\x00\x10'*80; (root/'pcm.bin').write_bytes(payload)
    source = dict(source_id='mic', sensor_id='microphone', stream_epoch='original-epoch',
                  format='pcm_s16le_mono', modality='audio', calibration_id=None)
    rows = [dict(source_id='mic', sensor_id='microphone', stream_epoch='original-epoch',
        sequence=i, capture_ns=1_000_000_000+offset, offset_ns=offset,
        format='pcm_s16le_mono', sample_rate=8000, sample_count=80,
        payload_path='pcm.bin', payload_bytes=len(payload),
        payload_sha256=hashlib.sha256(payload).hexdigest())
        for i, offset in enumerate((10_000_000, 30_000_000, 50_000_000))]
    return dict(schema_version=1, recording_id='original', origin_capture_ns=1_000_000_000,
        duration_ns=80_000_000, sources=[source], records=rows,
        availability=[dict(source_id='mic', intervals=[dict(start_ns=0,
            end_ns=80_000_000, state='available')])])


class OwnedCaptureTests(unittest.IsolatedAsyncioTestCase):
    async def prepare(self, root, manifest, journal=None, capacity=640):
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            return await OwnedCaptureSource.preflight(manifest, fd,
                CaptureCapacity(capacity, 1_000_000_000), journal=journal or Journal())
        finally:
            os.close(fd)

    async def test_owned_packets_preserve_originals_and_do_not_close_shared_sdk(self):
        with tempfile.TemporaryDirectory(dir='.') as directory:
            root = Path(directory).resolve(); manifest = fixture(root)
            source = await self.prepare(root, manifest); sink = Sink()
            human = Gamst(consent=True)
            try:
                clock = ReplayClock(manifest)
                await source.start(clock, sink)
                self.assertEqual(len(sink.packets), 3)
                for ordinal, packet in enumerate(sink.packets):
                    record = manifest['records'][ordinal]
                    self.assertEqual(packet.record_ordinal, ordinal)
                    self.assertEqual(packet.record_sha256, digest(record))
                    self.assertEqual(packet.original_stream_epoch, 'original-epoch')
                    self.assertEqual(packet.original_capture_ns, record['capture_ns'])
                    self.assertEqual(packet.sdk_timestamp_ms, clock.capture(record['offset_ns'])['sdk_timestamp_ms'])
                    self.assertGreaterEqual(packet.acquired_monotonic_ns, packet.scheduled_monotonic_ns)
                    with self.assertRaises(FrozenInstanceError): packet.payload = b'changed'
                await source.close()
                self.assertFalse(human.closed)
                self.assertEqual(source.counters()[0]['captured'], 3)
                self.assertEqual(source.counters()[0]['uninspected_suffix'], 0)
                self.assertIsNotNone(source.terminal_reached_ns)
                with self.assertRaises(RuntimeError): source.start(clock, sink)
            finally:
                await source.close(); await human.close()

    async def test_delivery_capacity_refusal_is_not_source_loss(self):
        with tempfile.TemporaryDirectory(dir='.') as directory:
            root = Path(directory).resolve(); manifest = fixture(root); journal = Journal()
            source = await self.prepare(root, manifest, journal)
            try:
                await source.start(ReplayClock(manifest), Sink(False))
                counters = source.counters()[0]
                self.assertEqual((counters['attempted'], counters['captured'], counters['dropped']), (3, 3, 0))
                self.assertEqual((counters['admitted'], counters['refused']), (0, 3))
                self.assertEqual(sum(k == 'delivery_refused' for k, _ in journal.events), 3)
            finally:
                await source.close()

    async def test_mutation_after_preflight_keeps_actual_drop_and_suffix(self):
        with tempfile.TemporaryDirectory(dir='.') as directory:
            root = Path(directory).resolve(); manifest = fixture(root); journal = Journal()
            source = await self.prepare(root, manifest, journal)
            (root/'pcm.bin').write_bytes(b'\x01'*160)
            try:
                with self.assertRaisesRegex(ValueError, 'changed or unavailable'):
                    await source.start(ReplayClock(manifest), Sink())
                counters = source.counters()[0]
                self.assertEqual((counters['attempted'], counters['captured'], counters['dropped']), (1, 0, 1))
                self.assertEqual(counters['uninspected_suffix'], 2)
                self.assertIsNone(source.terminal_reached_ns)
            finally:
                await source.close()

    async def test_repeated_cancel_joins_actual_read_before_fd_close(self):
        with tempfile.TemporaryDirectory(dir='.') as directory:
            root = Path(directory).resolve(); manifest = fixture(root)
            source = await self.prepare(root, manifest)
            entered = threading.Event(); release = threading.Event(); finished = threading.Event()
            def slow(record, fd):
                entered.set(); release.wait(2)
                try: return _read_owned(record, fd)
                finally: finished.set()
            with patch('benchmarks.owned_capture._read_owned', slow):
                running = source.start(ReplayClock(manifest), Sink())
                await asyncio.to_thread(entered.wait, 1)
                running.cancel(); running.cancel()
                closing = asyncio.create_task(source.close())
                await asyncio.sleep(0); closing.cancel()
                self.assertFalse(source._closed)
                self.assertFalse(finished.is_set())
                release.set()
                with self.assertRaises(asyncio.CancelledError): await running
                with self.assertRaises(asyncio.CancelledError): await closing
            self.assertTrue(finished.is_set()); self.assertTrue(source._closed)
            self.assertFalse(source._reads)

    async def test_joint_buffer_and_owned_symlink_checks_precede_start(self):
        with tempfile.TemporaryDirectory(dir='.') as directory:
            root = Path(directory).resolve(); manifest = fixture(root)
            with self.assertRaisesRegex(ValueError, 'reservation'):
                await self.prepare(root, manifest, capacity=639)
            (root/'alias.bin').symlink_to(root/'pcm.bin')
            manifest['records'][0]['payload_path'] = 'alias.bin'
            with self.assertRaises(OSError): await self.prepare(root, manifest)


if __name__ == '__main__': unittest.main()

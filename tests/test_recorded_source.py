import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from benchmarks.recorded_source import decode_manifest, read_capture, validate_manifest
from benchmarks.replay_clock import ReplayClock, validate_tasks


class RecordedCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir='.')
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name).resolve()/'capture.bin'
        self.payload = bytes(range(256))*3
        self.path.write_bytes(self.payload)
        self.manifest = dict(schema_version=1, recording_id='recording-1',
            origin_capture_ns=1000000000, duration_ns=100000000,
            sources=[dict(source_id='camera', sensor_id='camera-1', stream_epoch='original', format='bgr8_hwc',modality='video',calibration_id=None)],
            records=[dict(source_id='camera', sensor_id='camera-1', sequence=7,
                capture_ns=1000000000, offset_ns=0, format='bgr8_hwc',stream_epoch='original',
                payload_path=self.path.name, payload_bytes=len(self.payload),
                payload_sha256=hashlib.sha256(self.payload).hexdigest(), shape=[16,16,3])],
            availability=[dict(source_id='camera',intervals=[dict(start_ns=0,end_ns=100000000,state='available')])])

    def test_detached_manifest_and_actual_owned_bytes(self):
        frozen = validate_manifest(self.manifest)
        self.manifest['records'][0]['shape'][0] = 99
        self.assertEqual(frozen['records'][0]['shape'], [16,16,3])
        result = read_capture(frozen['records'][0],self.path.parent)
        self.path.write_bytes(b'x'*len(self.payload))
        self.assertEqual(result, self.payload)
        with self.assertRaisesRegex(ValueError, 'frozen hash'):
            read_capture(frozen['records'][0],self.path.parent)

    def test_changed_size_and_symlink_never_accepted(self):
        record = self.manifest['records'][0]
        self.path.write_bytes(self.payload+b'!')
        with self.assertRaises(ValueError):
            read_capture(record,self.path.parent)
        target = self.path.with_name('link.bin'); target.symlink_to(self.path)
        with self.assertRaises(OSError):
            read_capture(dict(record,payload_path=target.name),self.path.parent)

    def test_original_clock_sequence_and_layout_are_not_guessed(self):
        for changes in ({'capture_ns':1000000001}, {'sequence':True},
                        {'shape':[16,16,True]}, {'payload_bytes':767}, {'offset_ns':-1}):
            value = json.loads(json.dumps(self.manifest))
            value['records'][0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_manifest(value)
        self.manifest['records'] *= 2
        with self.assertRaisesRegex(ValueError, 'sequence'):
            validate_manifest(self.manifest)

    def test_strict_json_and_partition(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            decode_manifest(b'{"schema_version":1,"schema_version":1}')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            decode_manifest(b'{"value":NaN}')
        self.manifest['availability'][0]['intervals'][0]['start_ns'] = 1
        with self.assertRaisesRegex(ValueError, 'partition'):
            validate_manifest(self.manifest)

    def test_pcm_exact_format_and_chunk_bound(self):
        value = self.manifest
        value['sources'][0]['format'] = 'pcm_s16le_mono'
        value['sources'][0]['modality'] = 'audio'
        record = value['records'][0]
        record.pop('shape')
        record.update(format='pcm_s16le_mono',sample_rate=8000,sample_count=384)
        self.assertEqual(validate_manifest(value)['records'][0]['sample_count'],384)
        record.update(sample_count=801,payload_bytes=1602)
        with self.assertRaises(ValueError):
            validate_manifest(value)

    def test_reversible_clock_preserves_old_capture_and_rounding(self):
        wall = [1700000000123456789]; mono = [987654321]
        fake = SimpleNamespace(time_ns=lambda:wall[0], monotonic_ns=lambda:mono[0])
        with patch('benchmarks.replay_clock.time', fake):
            clock = ReplayClock(self.manifest)
            record = clock.capture(2345678)
            self.assertEqual(record['original_capture_ns'],1002345678)
            self.assertEqual(record['sdk_timestamp_ms'],1700000000125)
            self.assertEqual(record['submillisecond_remainder_ns'],345678)
            self.assertEqual(record['scheduled_monotonic_ns'],989999999)
            wall[0] += 30000000; mono[0] += 30000000
            self.assertEqual(clock.check(),mono[0])
            wall[0] += 50000001
            with self.assertRaisesRegex(RuntimeError,'discontinuity'):
                clock.check()

    def test_overlapping_task_inputs_are_frozen_not_rewritten(self):
        from benchmarks.recorded_source import digest
        original = {'question':'What was visible?'}
        a = dict(task_id='a',input=original,input_sha256=digest(original),
            opens_ns=0,snapshot_ns=10,deadline_ns=30,window='1h')
        b = dict(a,task_id='b',opens_ns=5,snapshot_ns=20,deadline_ns=40)
        result = validate_tasks([a,b],100)
        original['question']='changed'
        self.assertEqual(result[0]['input']['question'],'What was visible?')
        with self.assertRaisesRegex(ValueError,'digest'):
            validate_tasks([a],100)
        with self.assertRaisesRegex(ValueError,'window'):
            validate_tasks([dict(result[0],snapshot_ns=30)],100)

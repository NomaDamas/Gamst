import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from benchmarks.recorded_source import decode_manifest, read_capture, validate_manifest


class RecordedCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir='.')
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name).resolve()/'capture.bin'
        self.payload = bytes(range(256))*3
        self.path.write_bytes(self.payload)
        self.manifest = dict(schema_version=1, recording_id='recording-1',
            origin_capture_ns=1000000000, duration_ns=100000000,
            sources=[dict(source_id='camera', sensor_id='camera-1', stream_epoch='original', format='bgr8_hwc')],
            records=[dict(source_id='camera', sensor_id='camera-1', sequence=7,
                capture_ns=1000000000, offset_ns=0, format='bgr8_hwc',
                payload_path=str(self.path), payload_bytes=len(self.payload),
                payload_sha256=hashlib.sha256(self.payload).hexdigest(), shape=[16,16,3])],
            availability=[dict(start_ns=0,end_ns=100000000,state='available')])

    def test_detached_manifest_and_actual_owned_bytes(self):
        frozen = validate_manifest(self.manifest)
        self.manifest['records'][0]['shape'][0] = 99
        self.assertEqual(frozen['records'][0]['shape'], [16,16,3])
        result = read_capture(frozen['records'][0])
        self.path.write_bytes(b'x'*len(self.payload))
        self.assertEqual(result, self.payload)
        with self.assertRaisesRegex(ValueError, 'frozen hash'):
            read_capture(frozen['records'][0])

    def test_changed_size_and_symlink_never_accepted(self):
        record = self.manifest['records'][0]
        self.path.write_bytes(self.payload+b'!')
        with self.assertRaises(ValueError):
            read_capture(record)
        target = self.path.with_name('link.bin'); target.symlink_to(self.path)
        with self.assertRaises(OSError):
            read_capture(dict(record,payload_path=str(target)))

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
        self.manifest['availability'][0]['start_ns'] = 1
        with self.assertRaisesRegex(ValueError, 'partition'):
            validate_manifest(self.manifest)

    def test_pcm_exact_format_and_chunk_bound(self):
        value = self.manifest
        value['sources'][0]['format'] = 'pcm_s16le_mono'
        record = value['records'][0]
        record.pop('shape')
        record.update(format='pcm_s16le_mono',sample_rate=8000,sample_count=384)
        self.assertEqual(validate_manifest(value)['records'][0]['sample_count'],384)
        record.update(sample_count=801,payload_bytes=1602)
        with self.assertRaises(ValueError):
            validate_manifest(value)

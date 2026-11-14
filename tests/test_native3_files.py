import hashlib
import os
from pathlib import Path
import tempfile
import unittest

from benchmarks.native3_files import LIMITS, OriginalNativeFiles


class OriginalFiles(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir='.')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.inventory = {}
        for name in LIMITS:
            data = b'{"original":1}\n'
            (self.root/name).write_bytes(data)
            self.inventory[name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

    def test_stream_requires_complete_consumption_and_matching_hash(self):
        with OriginalNativeFiles(self.root, self.inventory) as files:
            lines = files.source_lines()
            self.assertEqual(next(lines), b'{"original":1}\n')
            self.assertNotIn('source-traces.jsonl', files.verified)
            self.assertEqual(list(lines), [])
            self.assertIn('source-traces.jsonl', files.verified)
            with self.assertRaises(ValueError): files.read('source-traces.jsonl')
            self.assertEqual(files.read('complete.json'), b'{"original":1}\n')
        with self.assertRaises(RuntimeError): files.read('complete.json')

    def test_changed_same_size_source_rejected_at_exhaustion(self):
        (self.root/'source-traces.jsonl').write_bytes(b'{"original":2}\n')
        with OriginalNativeFiles(self.root, self.inventory) as files:
            with self.assertRaisesRegex(ValueError, 'hash differs'): list(files.source_lines())
            self.assertNotIn('source-traces.jsonl', files.verified)

    def test_fifo_symlink_and_replaced_name_fail_closed(self):
        path = self.root/'complete.json';path.unlink();os.mkfifo(path, 0o600)
        with OriginalNativeFiles(self.root, self.inventory) as files:
            with self.assertRaises(ValueError):files.read('complete.json')
        path.unlink();path.symlink_to(self.root/'inputs.json')
        with OriginalNativeFiles(self.root, self.inventory) as files:
            with self.assertRaises(OSError):files.read('complete.json')
        with OriginalNativeFiles(self.root, self.inventory) as files:
            lines=files.source_lines();next(lines)
            original=self.root/'source-traces.jsonl'
            original.rename(self.root/'retained-original')
            original.write_bytes(b'{"original":1}\n')
            with self.assertRaisesRegex(ValueError,'changed'):list(lines)

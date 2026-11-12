import json
import tempfile
import unittest
from pathlib import Path
from humansense.storage import Store


def record(i, subject='user', value='측정'):
    return dict(id=str(i), subject_id=subject, timestamp_ms=i, type='observation.signal', observation={'value':value})


class StorageBoundsTests(unittest.TestCase):
    def test_sustained_input_evicts_oldest(self):
        store=Store(max_records=5)
        try:
            for i in range(1000): store.add(record(i))
            self.assertEqual([r['id'] for r in store.query()],list(map(str,range(995,1000))))
            self.assertEqual(store.status()['evicted_records'],995)
            self.assertEqual(len(store._sizes),5)
        finally: store.close()

    def test_utf8_bytes_and_oversized_record(self):
        budget=len(json.dumps(record(1),ensure_ascii=False).encode())*2
        store=Store(max_payload_bytes=budget)
        try:
            for i in range(100): store.add(record(i))
            actual=store.connection.execute('SELECT COALESCE(SUM(length(CAST(payload AS BLOB))),0) FROM records').fetchone()[0]
            self.assertEqual(store.payload_bytes,actual)
            self.assertLessEqual(actual,budget)
            self.assertFalse(store.add(record(101,value='가'*budget)))
            self.assertEqual(store.status()['oversized_records'],1)
        finally: store.close()

    def test_expiry_subject_delete_and_reopen_accounting(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            path=Path(folder)/'test.sqlite'
            store=Store(path,retention_hours=1/3600)
            store.add(record(0));store.add(record(500,'other'));store.add(record(1500))
            store.prune(1501)
            self.assertEqual(store.status()['records'],1)
            store.close()
            store=Store(path)
            self.assertEqual(store.status()['records'],1)
            store.delete('user')
            self.assertEqual(store.payload_bytes,0)
            self.assertEqual(store.status()['records'],0)
            store.close()

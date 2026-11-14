import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from benchmarks.native3_evidence import read_native3_evidence, ValidatedNative3Index
from benchmarks.native3_files import LIMITS
from benchmarks.perception_comparison import plan_digest
from test_instrumented_replay import setup, collect


class OriginalNativeIndex(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory(dir='.')
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()
        self.output=self.root/'collection'
        await collect(setup(self.root),self.output)

    def inventory(self):
        return {name:dict(bytes=(self.output/name).stat().st_size,
            sha256=hashlib.sha256((self.output/name).read_bytes()).hexdigest()) for name in LIMITS}

    def read(self):
        return read_native3_evidence(self.output,expected_files=self.inventory(),sidecar_ordinal=0)

    def rewrite_sidecars(self, mutate):
        path=self.output/'runtime-sidecars.jsonl'
        rows=[json.loads(line) for line in path.read_bytes().splitlines()]
        mutate(rows[0]);previous='0'*64
        for row in rows:
            row['previous_sidecar_sha256']=previous
            row['sidecar_sha256']=plan_digest({k:v for k,v in row.items() if k!='sidecar_sha256'})
            previous=row['sidecar_sha256']
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        path=self.output/'runtime-complete.json';value=json.loads(path.read_bytes())
        value['last_sidecar_sha256']=previous;path.write_text(json.dumps(value))

    async def test_actual_collector_entries_restore_instance_without_bill(self):
        index=self.read();data=index.snapshot();entries=index.entries()
        self.assertEqual(len(entries),1)
        entry=entries[0]
        self.assertEqual(entry['provider'],'test.client')
        self.assertEqual(len(entry['entry']['execution_instance_sha256']),64)
        rows=[json.loads(line) for line in (self.output/'runtime-sidecars.jsonl').read_bytes().splitlines()]
        attempt=rows[0]['runtime']['native_reports'][0]['attempts'][0]
        self.assertIsNone(attempt['execution_instance_sha256'])
        self.assertIsNone(attempt['cost_quantities'])
        self.assertEqual(index.lookup(entry['entry']['operation_id']),entry)
        self.assertFalse(data['original_channel_frames_verified'])
        self.assertFalse(data['transport_selection_preexecution_verified'])
        self.assertFalse(data['receipt_authenticity_verified'])
        entries[0]['provider']='changed'
        self.assertEqual(index.entries()[0]['provider'],'test.client')
        with self.assertRaises(TypeError):ValidatedNative3Index(data)

    async def test_relocated_collector_does_not_open_recorded_bootstrap_paths(self):
        inventory=self.inventory();old=self.output;self.output=self.root/'relocated';old.rename(self.output)
        self.assertFalse(old.exists())
        index=read_native3_evidence(self.output,expected_files=inventory,sidecar_ordinal=0)
        self.assertEqual(len(index.entries()),1)

    async def test_source_link_tampering_is_rejected_despite_rehashed_outer_chain(self):
        self.rewrite_sidecars(lambda row:row.update(original_response_sha256='a'*64))
        with self.assertRaisesRegex(ValueError,'sidecar/source binding'):self.read()

    async def test_original_native_event_tampering_cannot_hide_in_rehashed_sidecar(self):
        def change(row):
            row['runtime']['journal'][0]['fields']['provider']='other'
        self.rewrite_sidecars(change)
        with self.assertRaises(ValueError):self.read()

    async def test_original_file_hash_required_before_claiming_native_match(self):
        inventory=self.inventory()
        path=self.output/'runtime-sidecars.jsonl';path.write_bytes(path.read_bytes()+b'\n')
        with self.assertRaises(ValueError):read_native3_evidence(self.output,expected_files=inventory,sidecar_ordinal=0)

    async def test_actual_denied_dispatch_has_no_fabricated_native_entry(self):
        self.output=self.root/'denied-collection'
        await collect(setup(self.root,mode='denied'),self.output)
        index=self.read();data=index.snapshot()
        self.assertEqual(index.entries(),[])
        self.assertEqual(data['declared_provider_revisions'],[dict(provider='test.client',model_revision='v1')])
        self.assertEqual(data['owner']['task_id'],'0')
        self.assertFalse(data['original_channel_frames_verified'])

    async def test_producer_channel_flags_never_become_independent_wire_proof(self):
        self.rewrite_sidecars(lambda row:row['runtime'].update(hello_verified=False,eof=False,instrumentation_complete=False,
            incomplete_reason='incomplete_channel',validated_prefix_sha256='b'*64))
        data=self.read().snapshot()
        self.assertFalse(data['original_channel_frames_verified'])
        self.assertFalse(data['producer_reported_instrumentation_complete'])
        self.assertEqual(data['original_incomplete_reason'],'incomplete_channel')
        self.assertEqual(data['producer_reported_channel_prefix_sha256'],'b'*64)

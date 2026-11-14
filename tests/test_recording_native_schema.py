from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from benchmarks.recording_native_schema import (decode,snapshot,validate_recording_scope_protocol,digest)
from recording_native_fixtures import declarations,rehash


class Declarations(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir='.');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.value=declarations(self.root)

    def test_original_frozen_bundle_detaches_and_legacy_versions_reject(self):
        result=validate_recording_scope_protocol(self.value)
        self.value['collection']['recordings'][0]['task_ids'].clear()
        self.assertEqual(result['collection']['recordings'][0]['task_ids'],['0'])
        for doc,version in (('collection',1),('scope_protocol',2),('instrumentation_protocol',1)):
            bad=deepcopy(result);bad[doc]['schema_version']=version
            with self.assertRaises(ValueError):validate_recording_scope_protocol(bad)

    def test_hash_rebound_boolean_scope_unknown_profile_and_cross_arm_config(self):
        for kind in ('scope_bool','foreign_config','unknown_profile','changed_input','duplicate_recording','numeric_reconciliation'):
            bad=deepcopy(self.value)
            if kind=='scope_bool':bad['scope_protocol']['scopes'][1]['scope_index']=True
            elif kind=='foreign_config':bad['collection']['arms'][1]['config_sha256']='f'*64
            elif kind=='unknown_profile':
                bad['profiles']['mic']['profile']='arbitrary_import'
                bad['collection']['recordings'][0]['profiles_sha256']=digest(bad['profiles'])
            elif kind=='changed_input':
                bad['tasks'][0]['input']={'different':True};bad['tasks'][0]['input_sha256']=digest(bad['tasks'][0]['input'])
                bad['collection']['recordings'][0]['tasks_sha256']=digest(bad['tasks'])
            elif kind=='duplicate_recording':bad['collection']['recordings'].append(deepcopy(bad['collection']['recordings'][0]))
            else:bad['scope_protocol']['recording_reconciliation_sha256']='f'*64
            rehash(bad)
            with self.subTest(kind=kind),self.assertRaises(ValueError):validate_recording_scope_protocol(bad)

    def test_unknown_fields_duplicate_json_nonfinite_and_container_bounds(self):
        for body in (b'{"same":1,"same":2}',b'{"number":NaN}',b'{"number":Infinity}',b'\xff'):
            with self.assertRaises((ValueError,UnicodeError)):decode(body)
        for value in ({1:'not a string'}, {'number':float('nan')}, {'number':float('inf')}):
            with self.assertRaises(ValueError):snapshot(value)
        deep=[];cursor=deep
        for _ in range(35):child=[];cursor.append(child);cursor=child
        with self.assertRaises(ValueError):snapshot(deep)
        self.value['instrumentation_protocol']['trust_actual']=True
        with self.assertRaises(ValueError):validate_recording_scope_protocol(self.value)

    def test_global_capacity_and_fixed_legacy_binding_limits(self):
        for path,value in (('max_collection_entries',131073),('max_entries_per_run',True),('max_late_lookup_bytes',1)):
            bad=deepcopy(self.value);bad['collection']['capacity'][path]=value;rehash(bad)
            with self.assertRaises(ValueError):validate_recording_scope_protocol(bad)
        bad=deepcopy(self.value);bad['collection']['arms'][0]['provider_bindings'][0]['entry_boundary']='worker_backend';rehash(bad)
        with self.assertRaises(ValueError):validate_recording_scope_protocol(bad)

class JointAdmission(unittest.TestCase):
    def test_twenty_four_matched_short_recordings_balance_six_orders_without_reducing_gate(self):
        from collections import Counter
        from recording_native_fixtures import matched_recordings
        from benchmarks.recording_native_schema import preflight_recording_collection
        with tempfile.TemporaryDirectory(dir='.') as directory:
            bundles=matched_recordings(Path(directory));report=preflight_recording_collection(bundles)
            self.assertEqual(report['invocation_count'],72);self.assertEqual(report['original_task_count'],24)
            self.assertTrue(report['sample_gate_eligible']);self.assertEqual(bundles[0]['plan']['comparison']['min_pairs'],20)
            invocations=bundles[0]['collection']['invocations']
            orders=Counter(tuple(r['arm'] for r in invocations[n:n+3]) for n in range(0,72,3))
            self.assertEqual(set(orders.values()),{4});self.assertEqual(len(orders),6)
            for bundle in bundles:
                bundle['collection']['capacity']['max_collection_bytes']=report['predicted_artifact_bytes']-10000;rehash(bundle)
            with self.assertRaisesRegex(ValueError,'joint source/native'):preflight_recording_collection(bundles)

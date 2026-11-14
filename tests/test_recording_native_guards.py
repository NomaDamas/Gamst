"""Independent runtime admission and rehashed artifact boundary regressions."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from benchmarks.recording_evaluation import RecordingEvaluationSession
from benchmarks.recording_native_channel import NativeRunInventory
from benchmarks.recording_native_schema import digest,encoded,run_context
from benchmarks.recording_native_validation import validate_recording_native_inventory,canonical_lane_inventory
from benchmarks.recording_native_writer import RecordingNativeWriter
from benchmarks.replay_clock import ReplayClock
from recording_native_fixtures import declarations,rehash


def rechain(directory,mutate):
    path=directory/'native-000000.jsonl';rows=[json.loads(v) for v in path.read_bytes().splitlines()]
    header,footer=rows[0],rows[-1];events=rows[1:-1];mutate(events)
    previous=digest(['gamst.recording.events.v1',header['context']])
    for event in events:
        event['previous_sha256']=previous;event['sha256']=digest({k:v for k,v in event.items() if k!='sha256'});previous=event['sha256']
    prefix=b''.join(encoded(v)+b'\n' for v in [header]+events)
    footer.update(prefix_bytes=len(prefix),prefix_sha256=hashlib.sha256(prefix).hexdigest())
    data=prefix+encoded(footer)+b'\n';path.write_bytes(data)
    p=directory/'index-000000.json';page=json.loads(p.read_bytes())
    page['chunks'][0].update(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    page_bytes=encoded(page)+b'\n';p.write_bytes(page_bytes)
    t=directory/'terminal.json';terminal=json.loads(t.read_bytes());terminal['index_tail_sha256']=hashlib.sha256(page_bytes).hexdigest()
    t.write_bytes(encoded(terminal)+b'\n')


class Guards(unittest.IsolatedAsyncioTestCase):
    async def recording(self):
        temp=tempfile.TemporaryDirectory(dir='.');self.addCleanup(temp.cleanup);root=Path(temp.name)
        d=declarations(root,role='signal');source=d['source_manifest']
        other=dict(source['sources'][0],source_id='other',sensor_id='other-1');source['sources'].append(other)
        source['records'].append(dict(source['records'][0],source_id='other',sensor_id='other-1',offset_ns=250000000,capture_ns=1250000000))
        source['availability'].append(dict(source_id='other',intervals=deepcopy(source['availability'][0]['intervals'])))
        d['profiles']['other']=deepcopy(d['profiles']['mic'])
        d['collection']['recordings'][0].update(source_manifest_sha256=digest(source),profiles_sha256=digest(d['profiles']))
        for arm in d['collection']['arms']:arm['provider_bindings'][0].update(operation_kinds=['process'],semantic_lane=None)
        rehash(d);writer=await RecordingNativeWriter.open(root/'native',run_context(d,'a'*64,0))
        owner=RecordingEvaluationSession.from_frozen(d,writer,run_nonce='a'*64,execution_ordinal=0,retain_evaluation_hashes=True)
        self.addAsyncCleanup(owner.close);clock=ReplayClock(source);calls=[]
        async def callback():calls.append(True);return {'result':'fixture'}
        with self.assertRaisesRegex(ValueError,'frozen entry boundary'):owner.bind_backend(0,callback,'process')
        with self.assertRaisesRegex(ValueError,'frozen entry boundary'):owner._register_worker(0,'in_process')
        owner.bind_backend(0,callback,'callback');owner.bind_mapping(clock.mapping,clock.sha256)
        def reference(n,si):return dict(kind='capture',source_index=si,capture_ordinal=n,
            record_sha256=digest(source['records'][n]),mapping_sha256=clock.sha256)
        return owner,writer,callback,calls,reference(0,0),reference(5,1)

    def ticket(self,owner,ref):
        return owner.reserve_operation(0,owner.stream_scope(0),'process',[ref],
            digest(dict(binding=0,operation_kind='process',capture_refs=[ref])),ref['record_sha256'])

    def validate(self,owner,writer):
        return validate_recording_native_inventory(owner.declarations,writer.output,source_inventory=owner.declarations['source_manifest'])

    async def test_excluded_original_source_rejects_before_reservation_and_callback(self):
        owner,writer,callback,calls,good,bad=await self.recording()
        with self.assertRaisesRegex(ValueError,'capture locator'):self.ticket(owner,bad)
        self.assertEqual(calls,[]);self.assertEqual(owner._reservations,0)
        ticket=self.ticket(owner,good);await ticket.invoke(callback);ticket.published(True)
        owner.backend_closed(0,callback);await owner.close()
        self.assertEqual(self.validate(owner,writer).count,1)
        def forge(events):
            next(e for e in events if e['event']=='evidence_ref')['fields']['reference']=bad
            fields=next(e for e in events if e['event']=='reserved')['fields']
            fields.update(evidence_sha256=bad['record_sha256'],request_sha256=digest(dict(binding=0,operation_kind='process',capture_refs=[bad])))
        rechain(writer.output,forge)
        with self.assertRaisesRegex(ValueError,'original capture evidence'):self.validate(owner,writer)

    async def test_reader_rejects_rehashed_worker_boundary_change(self):
        owner,writer,callback,calls,good,bad=await self.recording()
        ticket=self.ticket(owner,good);await ticket.invoke(callback);ticket.published(True)
        owner.backend_closed(0,callback);await owner.close()
        def forge(events):next(e for e in events if e['event']=='worker_registered')['fields']['kind']='in_process'
        rechain(writer.output,forge)
        with self.assertRaisesRegex(ValueError,'kind differs from frozen entry boundary'):self.validate(owner,writer)

    async def test_dispatched_no_entry_liability_survives_release_and_frozen_snapshots(self):
        owner,writer,callback,calls,good,bad=await self.recording()
        with self.assertRaises(TypeError):owner.declarations['collection']['capacity']['max_entries_per_run']=999999
        with self.assertRaises(TypeError):owner.context['nonce']='b'*64
        with self.assertRaises(AttributeError):owner.context={}
        with self.assertRaises(AttributeError):owner.declarations={}
        with self.assertRaises(TypeError):NativeRunInventory({'parent_reaped':True})
        self.ticket(owner,good).release()
        ticket=self.ticket(owner,good);ticket.dispatched();ticket.release('cancelled')
        with self.assertRaises(RuntimeError):await ticket.invoke(callback)
        ticket=self.ticket(owner,good);await ticket.invoke(callback);ticket.published(True)
        owner.backend_closed(0,callback);await owner.close()
        lane=canonical_lane_inventory(self.validate(owner,writer),0,0)
        self.assertEqual(lane['counts']['reserved'],3);self.assertEqual(lane['counts']['dispatched'],2)
        self.assertEqual(lane['counts']['released_undispatched'],1);self.assertEqual(lane['counts']['released_dispatched'],1)
        self.assertEqual(lane['unknown_dispatch_liability_count'],1);self.assertEqual(lane['unknown_liability_count'],1)
        self.assertEqual(len(calls),1)

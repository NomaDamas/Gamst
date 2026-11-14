import hashlib
from pathlib import Path
import tempfile
import unittest

from benchmarks.recording_native_schema import digest,run_context
from benchmarks.recording_evaluation import RecordingEvaluationSession
from benchmarks.recording_native_writer import RecordingNativeWriter
from benchmarks.recording_native_validation import validate_recording_native_inventory,lookup_canonical_operation
from benchmarks.replay_clock import ReplayClock
from humansense.schema import timestamp
from recording_native_fixtures import declarations,rehash
import test_text_context_call as text_tests


class DeclaredReceipts(unittest.IsolatedAsyncioTestCase):
    asyncSetUp=text_tests.Calls.asyncSetUp
    asyncTearDown=text_tests.Calls.asyncTearDown
    queued=text_tests.Calls.queued

    async def recording(self,ordinal=0):
        directory=tempfile.TemporaryDirectory(dir='.');self.addCleanup(directory.cleanup);root=Path(directory.name)
        d=declarations(root)
        card={'request':dict(provider='test.client',revision='v1',unit='request',category='request',per_unit=1)}
        d['plan']['comparison']['cost']=dict(mode='itemized',currency='USD',scope='full_pipeline',
            accounting_protocol='Synthetic declared fixture only',rate_card=card)
        d['cost_card']=card;d['scope_protocol']['cost_card_sha256']=digest(card)
        d['scope_protocol']['retain_financial_documents']=True;rehash(d)
        nonce='abc'[ordinal]*64
        writer=await RecordingNativeWriter.open(root/'native',run_context(d,nonce,ordinal),retain_financial_documents=True)
        observer=RecordingEvaluationSession.from_frozen(d,writer,run_nonce=nonce,execution_ordinal=ordinal,
            retain_evaluation_hashes=True,retain_financial_documents=True)
        self.addAsyncCleanup(observer.close)
        return observer,writer,ReplayClock(d['source_manifest'])

    async def test_actual_entry_binding_late_declared_receipt_retains_unknown_money(self):
        observer,writer,clock=await self.recording();bound=[]
        async def actual(binding):bound.append(binding);return dict(answer='fixture')
        observer.bind_semantic_lane(self.c,0);observer.bind_backend(0,actual,'callback');observer.bind_mapping(clock.mapping,clock.sha256)
        rid,record=await self.queued();scope=observer.task_scope(observer.declarations['tasks'][0]['task_id'],0)
        ticket=observer.reserve_call(self.c,rid,provider='test.client',model_revision='v1',prompt_sha256='b'*64,evaluation_scope=scope)
        self.c.reserve(rid,provider='test.client',model_revision='v1',prompt_sha256='b'*64,maximum_charge=2,ttl_ms=1000,now_ms=timestamp())
        self.c.dispatch(rid,now_ms=timestamp());ticket.dispatched()
        await ticket.invoke_bound(actual);ticket.finish('completed')
        # Seal the original entry in an earlier chunk without retaining a call row.
        for _ in range(6):
            for _ in range(50):writer.emit('incomplete',dict(reason='unobserved_boundary',count=0))
            await writer.drain()
        source=b'{"synthetic_original":"no authenticated money"}\n'
        claim=digest(['gamst.recording.claim.v1','test.client','account','invoice','line'])
        receipt=dict(schema_version=1,kind='recording_declared_receipt',origin='synthetic_fixture',money_basis='unknown',
            provider='test.client',revision='v1',account_scope='account',invoice_id='invoice',
            source_document_sha256=hashlib.sha256(source).hexdigest(),source_document_bytes=len(source),
            lines=[dict(line_id='line',quantities={'request':1},claim_sha256=claim)])
        result=await observer.attach_declared_receipt(bound[0],receipt,source)
        self.assertTrue(result['unknown_liability'])
        with self.assertRaises(ValueError):await observer.attach_declared_receipt(bound[0],receipt,source)
        observer.backend_closed(0,actual);await observer.close()
        index=validate_recording_native_inventory(observer.declarations,writer.output,
            source_inventory=observer.declarations['source_manifest'])
        self.assertEqual(index.terminal['unsettled_entries'],1)
        self.assertEqual(self.c.snapshot()['budget']['reserved_liability'],2)
        self.assertTrue(lookup_canonical_operation(index,0)['unknown_liability'])
        with self.assertRaises(RuntimeError):await observer.attach_declared_receipt(bound[0],receipt,source)

    async def test_original_invoice_line_cannot_rebind_across_abc_owners(self):
        from benchmarks.recording_native_validation import validate_recording_native_collection
        runs=[]
        for ordinal in range(3):
            observer,writer,clock=await self.recording(ordinal);bound=[]
            async def actual(binding):bound.append(binding);return dict(answer='fixture')
            observer.bind_semantic_lane(self.c,0);observer.bind_backend(0,actual,'callback');observer.bind_mapping(clock.mapping,clock.sha256)
            rid,record=await self.queued();scope=observer.task_scope(observer.declarations['tasks'][0]['task_id'],0)
            ticket=observer.reserve_call(self.c,rid,provider='test.client',model_revision='v1',prompt_sha256='b'*64,evaluation_scope=scope)
            self.c.reserve(rid,provider='test.client',model_revision='v1',prompt_sha256='b'*64,maximum_charge=2,ttl_ms=1000,now_ms=timestamp())
            self.c.dispatch(rid,now_ms=timestamp());ticket.dispatched();await ticket.invoke_bound(actual);ticket.finish('completed')
            source=b'original declared invoice'
            receipt=dict(schema_version=1,kind='recording_declared_receipt',origin='synthetic_fixture',money_basis='unknown',
                provider='test.client',revision='v1',account_scope='account',invoice_id='invoice',
                source_document_sha256=hashlib.sha256(source).hexdigest(),source_document_bytes=len(source),
                lines=[dict(line_id='same-original-line',quantities={'request':1},claim_sha256=digest([
                    'gamst.recording.claim.v1','test.client','account','invoice','same-original-line']))])
            await observer.attach_declared_receipt(bound[0],receipt,source)
            observer.backend_closed(0,actual);await observer.close()
            runs.append(dict(declarations=observer.declarations,directory=writer.output,
                terminal_sha256=hashlib.sha256((writer.output/'terminal.json').read_bytes()).hexdigest(),
                source_inventory=observer.declarations['source_manifest']))
        with self.assertRaisesRegex(ValueError,'reused across arms'):
            validate_recording_native_collection(runs)
        self.assertEqual(self.c.snapshot()['budget']['reserved_liability'],6)

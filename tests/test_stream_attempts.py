import asyncio
import copy
from pathlib import Path
import tempfile
import unittest

from benchmarks.call_attempts import AttemptRecorder, cost_quantities as paired_quantities
from benchmarks.controller_attempts import RecordedContextClient
from benchmarks.perception_comparison import plan_digest
from benchmarks.stream_attempts import StreamAttemptRecorder, validate_report
from benchmarks.three_arm_attempts import cost_quantities as native2_quantities
from humansense import TextContextCalls
from test_itemized_costs import itemized
from test_stream_cost_ledger import fixture as stream_fixture
from test_three_arm_replay import setup
import test_text_context_call as text_tests


def fixture(root):
    plan,spec,inputs=setup(root,('ok',))
    plan['comparison']['cost']=itemized()[0]['cost'];spec['comparison_plan_sha256']=plan_digest(plan)
    stream,_=stream_fixture();stream.update(comparison_plan_sha256=plan_digest(plan),collection_sha256=plan_digest(spec))
    # This is a minimal frozen identity fixture. Full reconciliation semantics are
    # validated by the integration layer, not by the native recorder constructor.
    protocol=dict(schema_version=1,comparison_plan_sha256=plan_digest(plan),collection_sha256=plan_digest(spec),
        stream_protocol_sha256=plan_digest(stream),allocation_protocol_sha256='8'*64,
        currency_atomic_per_comparison_unit=dict(numerator=1,denominator=1),rate_map=[],lane_owners=[],
        provider_counter_groups=[],vlm_counter_bindings=[],recording_groups=[],whole_stream_cost_policy={})
    return plan,spec,inputs,stream,protocol


def rehash(report):
    report['sha256']=plan_digest({k:v for k,v in report.items() if k!='sha256'})
    return report


class NativeStreamAttempts(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory=tempfile.TemporaryDirectory(dir='.')
        self.addCleanup(self.directory.cleanup)
        self.root=Path(self.directory.name).resolve()
        self.plan,self.spec,self.inputs,self.stream,self.protocol=fixture(self.root)

    def args(self,**updates):
        args=dict(expected_plan_sha256=plan_digest(self.plan),expected_collection_sha256=plan_digest(self.spec),
            expected_stream_protocol_sha256=plan_digest(self.stream),expected_reconciliation_protocol_sha256=plan_digest(self.protocol),
            scope_kind='task',task_id='0',arm='A',stream_id='stream-A')
        args.update(updates)
        return args

    def recorder(self,**updates):
        return StreamAttemptRecorder(self.plan,self.spec,self.inputs,self.stream,self.protocol,**self.args(**updates))

    def validate(self,report,**updates):
        args=self.args(expected_report_sha256=report['sha256']);args.update(updates)
        return validate_report(self.plan,self.spec,self.inputs,self.stream,self.protocol,report,**args)

    async def completed(self,rec=None,receipt=True,**invoke_updates):
        rec=rec or self.recorder();tokens=[]
        async def callback(token):tokens.append(token);return 'private answer'
        args=dict(provider='remote',revision='frozen-v1',role='strong_vlm',phase='cold',evidence_sha256='a'*64)
        args.update(invoke_updates)
        await rec.invoke(callback,**args)
        token=tokens[0]
        if receipt:rec.report_lines(token,quantities={'request':1},line_claims=[plan_digest(['line',token])],
                                    execution_instance_sha256=plan_digest(['instance',token]))
        return rec,token

    async def test_native_task_header_late_lines_and_detached_snapshots(self):
        before=copy.deepcopy((self.plan,self.spec,self.inputs,self.stream,self.protocol))
        rec,token=await self.completed(receipt=False)
        pending=rec.snapshot()['attempts'][0]
        self.assertNotIn('receipt_sha256',pending)
        self.assertIsNone(pending['line_claims']);self.assertIsNone(pending['execution_instance_sha256'])
        rec.close()
        with self.assertRaises(ValueError):self.validate(rec.snapshot())
        quantities={'request':2};claims=['b'*64]
        rec.report_lines(token,quantities=quantities,line_claims=claims,execution_instance_sha256='c'*64)
        quantities['request']=999;claims.append('d'*64)
        report=rec.snapshot()
        self.assertEqual(report['schema_version'],3)
        self.assertEqual(report['scope_kind'],'task');self.assertEqual(report['task_id'],'0')
        self.assertEqual(report['input_sha256'],self.plan['comparison']['tasks'][0]['input_sha256'])
        self.assertEqual(report['recording_sha256'],self.stream['streams'][0]['recording_sha256'])
        self.assertEqual(report['configuration_sha256'],plan_digest(self.plan['systems']['A']))
        self.assertEqual(self.validate(report)['request'],2)
        self.assertEqual(report['attempts'][0]['line_claims'],['b'*64])
        self.assertNotIn('private answer',str(report))
        report['attempts'][0]['cost_quantities']['request']=7
        self.assertEqual(self.validate(rec.snapshot())['request'],2)
        self.assertEqual(before,(self.plan,self.spec,self.inputs,self.stream,self.protocol))
        with self.assertRaises(ValueError):rec.report(token,quantities={'request':1},receipt_sha256='b'*64)
        with self.assertRaises(ValueError):rec.report_lines(token,quantities={'request':1},line_claims=['d'*64],execution_instance_sha256='e'*64)

    async def test_stream_scope_and_original_arm_mapping(self):
        rec=self.recorder(scope_kind='stream',task_id=None,arm='C',stream_id='stream-C')
        rec.close();report=rec.snapshot()
        self.assertIsNone(report['task_id']);self.assertIsNone(report['input_sha256'])
        self.assertEqual(report['arm'],'C')
        self.assertTrue(all(value==0 for value in self.validate(report,scope_kind='stream',task_id=None,arm='C',stream_id='stream-C').values()))
        for updates in (dict(scope_kind='stream'),dict(task_id=None),dict(scope_kind='other'),
                        dict(stream_id='stream-B'),dict(arm='D'),dict(stream_id='missing'),dict(max_attempts=True)):
            with self.subTest(updates=updates),self.assertRaises(ValueError):self.recorder(**updates)

    async def test_frozen_protocol_and_report_scope_bindings(self):
        rec,_=await self.completed();rec.close();report=rec.snapshot()
        for field in ('expected_plan_sha256','expected_collection_sha256','expected_stream_protocol_sha256',
                      'expected_reconciliation_protocol_sha256','expected_report_sha256'):
            with self.subTest(field=field),self.assertRaises(ValueError):self.validate(report,**{field:'f'*64})
        for field,value in (('schema_version',2),('scope_kind','stream'),('task_id',None),('arm','B'),('stream_id','stream-B'),
                            ('input_sha256','f'*64),('recording_sha256','f'*64),('configuration_sha256','f'*64),
                            ('cost_sha256','f'*64),('stream_protocol_sha256','f'*64),('reconciliation_protocol_sha256','f'*64)):
            changed=copy.deepcopy(report);changed[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.validate(rehash(changed))
        self.protocol['collection_sha256']='f'*64
        with self.assertRaises(ValueError):self.recorder()
        self.protocol['collection_sha256']=plan_digest(self.spec);self.protocol['schema_version']=True
        with self.assertRaises(ValueError):self.recorder()

    async def test_closed_rejected_and_inflight_scopes_fail_closed(self):
        rec=self.recorder();entered=asyncio.Event();release=asyncio.Event();tokens=[]
        async def callback(token):tokens.append(token);entered.set();await release.wait()
        job=asyncio.create_task(rec.invoke(callback,provider='remote',revision='frozen-v1',role='strong_vlm',
                                          phase='cold',evidence_sha256='a'*64))
        await entered.wait()
        rec.report_lines(tokens[0],quantities={'request':1},line_claims=['b'*64],execution_instance_sha256='c'*64)
        rec.close()
        with self.assertRaises(ValueError):self.validate(rec.snapshot())
        release.set();await job
        self.assertEqual(self.validate(rec.snapshot())['request'],1)
        with self.assertRaises(RuntimeError):await self.completed(rec)
        self.assertEqual(rec.snapshot()['rejected'],1)
        with self.assertRaises(ValueError):self.validate(rec.snapshot())
        rec,_=await self.completed()
        with self.assertRaises(ValueError):self.validate(rec.snapshot())

    async def test_error_cancelled_retry_escalation_and_late_billing(self):
        rec=self.recorder();tokens=[];entered=asyncio.Event()
        async def blocked(token):tokens.append(token);entered.set();await asyncio.Event().wait()
        job=asyncio.create_task(rec.invoke(blocked,provider='remote',revision='frozen-v1',role='strong_vlm',
            phase='cold',evidence_sha256='a'*64))
        await entered.wait();job.cancel()
        with self.assertRaises(asyncio.CancelledError):await job
        async def error(token):tokens.append(token);raise RuntimeError('private failure')
        with self.assertRaises(RuntimeError):await rec.invoke(error,provider='remote',revision='frozen-v1',role='strong_vlm',
            phase='warm',evidence_sha256='a'*64,predecessor=tokens[0],relationship='retry')
        async def text(token):tokens.append(token);return 'private text'
        await rec.invoke(text,provider='text',revision='frozen-v1',role='text_llm',phase='warm',evidence_sha256='a'*64,
            predecessor=tokens[1],relationship='escalation')
        rec.close()
        with self.assertRaises(ValueError):self.validate(rec.snapshot())
        for i,token in enumerate(tokens):
            rec.report_lines(token,quantities={'request':1} if i<2 else {'text.output':3},
                line_claims=[plan_digest(['line',i])],execution_instance_sha256=plan_digest(['instance',i]))
        report=rec.snapshot();totals=self.validate(report)
        self.assertEqual([a['status'] for a in report['attempts']],['cancelled','error','success'])
        self.assertEqual((totals['request'],totals['text.output']),(2,3))
        self.assertNotIn('private',str(report))
        for field,value in (('predecessor',None),('evidence_sha256','f'*64),('started_ns',0)):
            changed=copy.deepcopy(report);changed['attempts'][1][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.validate(rehash(changed))

    async def test_line_claim_and_instance_ownership_and_explicit_zero_receipt(self):
        rec,first=await self.completed(receipt=False);_,second=await self.completed(rec,receipt=False)
        rec.report_lines(first,quantities={'request':0},line_claims=['b'*64],execution_instance_sha256='c'*64)
        for updates in (dict(line_claims=[]),dict(line_claims=['d'*64,'d'*64]),dict(line_claims=['b'*64]),
                        dict(execution_instance_sha256='c'*64),dict(quantities={}),dict(quantities={'request':True}),
                        dict(quantities={'request':-1}),dict(quantities={'request':10**12+1}),dict(line_claims=['not-a-digest'])):
            args=dict(quantities={'request':1},line_claims=['d'*64],execution_instance_sha256='e'*64);args.update(updates)
            with self.subTest(updates=updates),self.assertRaises(ValueError):rec.report_lines(second,**args)
        rec.report_lines(second,quantities={'request':1},line_claims=['d'*64],execution_instance_sha256='e'*64)
        rec.close();report=rec.snapshot()
        self.assertEqual(self.validate(report)['request'],1)
        for field,value in (('line_claims',['b'*64]),('execution_instance_sha256','c'*64)):
            changed=copy.deepcopy(report);changed['attempts'][1][field]=value
            with self.assertRaises(ValueError):self.validate(rehash(changed))

    async def test_strict_native_body_schema_and_bounded_quantities(self):
        rec,_=await self.completed();rec.close();report=rec.snapshot()
        for field,value in (('attempt_id','foreign:0'),('status','running'),('ended_ns',None),('started_ns',True),
                            ('phase','other'),('role','other'),('provider','missing'),('line_claims',None),
                            ('execution_instance_sha256',None),('cost_quantities',{'request':None}),
                            ('receipt_sha256','f'*64)):
            changed=copy.deepcopy(report);changed['attempts'][0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.validate(rehash(changed))
        for field,value in (('closed',False),('rejected',True),('scope','X'*32),('max_attempts',0),('clock','stream_elapsed')):
            changed=copy.deepcopy(report);changed[field]=value
            with self.assertRaises(ValueError):self.validate(rehash(changed))
        changed=copy.deepcopy(report);changed['attempts']*=1025
        with self.assertRaises(ValueError):self.validate(rehash(changed))
        changed=copy.deepcopy(report);changed['attempts'][0]['evidence_sha256']='x'*(16*1024*1024)
        with self.assertRaises(ValueError):self.validate(rehash(changed))

    async def test_total_line_claim_bound_and_paired_native2_compatibility(self):
        rec,first=await self.completed(receipt=False);_,second=await self.completed(rec,receipt=False)
        rec.report_lines(first,quantities={'request':1},line_claims=[f'{i:064x}' for i in range(10000)],execution_instance_sha256='c'*64)
        with self.assertRaises(ValueError):rec.report_lines(second,quantities={'request':1},line_claims=['d'*64],execution_instance_sha256='e'*64)
        other,_=await self.completed();other.close();report=other.snapshot()
        pair,_=itemized()
        with self.assertRaises(ValueError):paired_quantities(pair,report,expected_plan_sha256=plan_digest(pair),
            expected_report_sha256=report['sha256'],task_id='0',system='baseline')
        with self.assertRaises(ValueError):native2_quantities(self.plan,self.spec,self.inputs,report,
            expected_plan_sha256=plan_digest(self.plan),expected_collection_sha256=plan_digest(self.spec),
            expected_report_sha256=report['sha256'],task_id='0',arm='A')


class NativeStreamClient(unittest.IsolatedAsyncioTestCase):
    asyncSetUp=text_tests.Calls.asyncSetUp
    asyncTearDown=text_tests.Calls.asyncTearDown
    queued=text_tests.Calls.queued

    async def test_actual_recorded_sdk_client_native3_and_late_lines(self):
        with tempfile.TemporaryDirectory(dir='.') as directory:
            plan,spec,inputs,stream,protocol=fixture(Path(directory).resolve())
            args=dict(expected_plan_sha256=plan_digest(plan),expected_collection_sha256=plan_digest(spec),
                expected_stream_protocol_sha256=plan_digest(stream),expected_reconciliation_protocol_sha256=plan_digest(protocol),
                scope_kind='task',task_id='0',arm='C',stream_id='stream-C')
            rec=StreamAttemptRecorder(plan,spec,inputs,stream,protocol,**args)
            self.assertIsInstance(rec,AttemptRecorder)
            rid,record=await self.queued();requests=[]
            async def complete(request):requests.append(copy.deepcopy(request));return dict(text='private',usage=None)
            wrapped=RecordedContextClient(self.c,rid,rec,complete,phase='cold')
            adapter=TextContextCalls(self.h,'main',wrapped,provider='text',model_revision='frozen-v1');self.adapters.append(adapter)
            result=await adapter.execute(rid,record,'private question',maximum_charge=10)
            self.assertTrue(result['accepted']);rec.close();report=rec.snapshot()
            self.assertEqual(report['attempts'][0]['prompt_sha256'],plan_digest(requests[0]))
            self.assertIsNotNone(report['attempts'][0]['execution_sha256'])
            with self.assertRaises(ValueError):validate_report(plan,spec,inputs,stream,protocol,report,
                expected_report_sha256=report['sha256'],**args)
            rec.report_lines(wrapped.attempt_id,quantities={'text.output':3},line_claims=['d'*64],execution_instance_sha256='e'*64)
            report=rec.snapshot()
            self.assertEqual(validate_report(plan,spec,inputs,stream,protocol,report,
                expected_report_sha256=report['sha256'],**args)['text.output'],3)
            self.assertNotIn('receipt_sha256',report['attempts'][0])
            self.assertNotIn('private',str(report))

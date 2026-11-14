import asyncio
import copy
import unittest

from benchmarks.call_attempts import AttemptRecorder, cost_quantities
from benchmarks.perception_comparison import compare, plan_digest
from test_itemized_costs import itemized


def metadata(provider='small', **updates):
    values=dict(provider=provider,revision='frozen-v1',role='small_vlm',phase='cold',
                evidence_sha256='a'*64,prompt_sha256='b'*64,execution_sha256='c'*64)
    values.update(updates)
    return values


def reduce(plan, report, expected=None):
    return cost_quantities(plan,report,expected_plan_sha256=plan_digest(plan),
                           expected_report_sha256=expected or report['sha256'],task_id=report['task_id'],system=report['system'])


def rehash(report):
    report['sha256']=plan_digest({k:v for k,v in report.items() if k!='sha256'})
    return report


class Attempts(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.plan,self.rows=itemized()
        self.recorder=AttemptRecorder(self.plan,expected_plan_sha256=plan_digest(self.plan),task_id='0',system='candidate')

    async def test_failed_retry_escalation_receipts_and_comparison(self):
        r=self.recorder;tokens=[]
        async def failed(token):
            tokens.append(token)
            raise RuntimeError('private prompt must never appear')
        with self.assertRaises(RuntimeError):await r.invoke(failed,**metadata())
        r.report(tokens[0],quantities={'small.input':7},receipt_sha256='d'*64)
        async def success(token):
            tokens.append(token)
            return 'private answer'
        result=await r.invoke(success,**metadata(phase='warm',predecessor=tokens[0],relationship='retry'))
        self.assertEqual(result,'private answer')
        r.report(tokens[1],quantities={'small.input':3},receipt_sha256='e'*64)
        await r.invoke(success,**metadata('strong',role='strong_vlm',predecessor=tokens[1],relationship='escalation'))
        r.report(tokens[2],quantities={'strong.input':5,'cache.read':2,'cache.write':1},receipt_sha256='f'*64)
        r.close();report=r.snapshot();totals=reduce(self.plan,report)
        self.assertEqual([x['status'] for x in report['attempts']],['error','success','success'])
        self.assertEqual(totals['small.input'],10);self.assertEqual(totals['strong.input'],5)
        self.assertNotIn('private',str(report))
        for row in self.rows:row['cost_quantities']=dict(totals)
        result=compare(self.plan,self.rows,expected_plan_sha256=plan_digest(self.plan))
        self.assertFalse(result['meets_numeric_gates'])
        self.assertEqual(result['summaries']['candidate']['cost_quantities']['small.input'],200)

    async def test_concurrent_cancel_close_unknown_and_late_receipts(self):
        r=self.recorder;entered=asyncio.Event();tokens=[];gate=asyncio.Event()
        async def slow(token):
            tokens.append(token);entered.set();await gate.wait()
        task=asyncio.create_task(r.invoke(slow,**metadata()))
        await entered.wait();r.close()
        with self.assertRaises(ValueError):reduce(self.plan,r.snapshot())
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):await task
        report=r.snapshot();self.assertEqual(report['attempts'][0]['status'],'cancelled')
        self.assertIsNone(report['attempts'][0]['cost_quantities'])
        with self.assertRaises(ValueError):reduce(self.plan,report)
        r.report(tokens[0],quantities={'small.input':9},receipt_sha256='d'*64)
        self.assertEqual(reduce(self.plan,r.snapshot())['small.input'],9)
        with self.assertRaises(ValueError):r.report(tokens[0],quantities={'small.input':0},receipt_sha256='e'*64)

    async def test_capacity_never_executes_or_silently_certifies_overflow(self):
        r=AttemptRecorder(self.plan,expected_plan_sha256=plan_digest(self.plan),task_id='0',system='candidate',max_attempts=1)
        calls=[]
        async def work(token):
            calls.append(token);r.report(token,quantities={'small.input':1},receipt_sha256='d'*64)
        await r.invoke(work,**metadata())
        with self.assertRaises(RuntimeError):await r.invoke(work,**metadata())
        r.close();self.assertEqual(len(calls),1)
        self.assertEqual(r.snapshot()['rejected'],1)
        with self.assertRaises(ValueError):reduce(self.plan,r.snapshot())
        with self.assertRaises(RuntimeError):await r.invoke(work,**metadata())
        self.assertEqual(r.snapshot()['rejected'],2)

    async def test_concurrent_order_copy_and_immutable_plan_binding(self):
        r=self.recorder;gate=asyncio.Event();tokens=[]
        async def first(token):tokens.append(token);await gate.wait()
        task=asyncio.create_task(r.invoke(first,**metadata()))
        await asyncio.sleep(0)
        async def second(token):tokens.append(token)
        await r.invoke(second,**metadata(phase='warm'))
        gate.set();await task
        for token in tokens:r.report(token,quantities={'small.input':1},receipt_sha256='d'*64)
        r.close();report=r.snapshot()
        self.assertGreaterEqual(report['attempts'][0]['ended_ns'],report['attempts'][1]['ended_ns'])
        self.assertEqual(reduce(self.plan,report)['small.input'],2)
        report['attempts'][0]['cost_quantities']['small.input']=0
        self.assertEqual(reduce(self.plan,r.snapshot())['small.input'],2)
        altered=copy.deepcopy(self.plan);altered['cost']['rate_card']['small.input']['per_unit']=0
        with self.assertRaises(ValueError):reduce(altered,r.snapshot())

    async def test_invalid_receipts_lineage_and_tampering_fail_closed(self):
        r=self.recorder;tokens=[]
        async def work(token):tokens.append(token)
        await r.invoke(work,**metadata())
        for quantities in ({}, {'small.input':None}, {'small.input':True}, {'small.input':-1}, {'small.input':1,'media':1}):
            with self.assertRaises(ValueError):r.report(tokens[0],quantities=quantities,receipt_sha256='d'*64)
        for extra in (dict(predecessor=tokens[0]),dict(relationship='retry',predecessor='foreign'),
                      dict(relationship='retry',predecessor=tokens[0],evidence_sha256='d'*64),
                      dict(relationship='retry',predecessor=tokens[0],provider='strong')):
            with self.assertRaises(ValueError):await r.invoke(work,**metadata(**extra))
        r.report(tokens[0],quantities={'small.input':1},receipt_sha256='d'*64);r.close()
        report=r.snapshot()
        with self.assertRaises(ValueError):
            cost_quantities(self.plan,report,expected_plan_sha256=plan_digest(self.plan),expected_report_sha256=report['sha256'],task_id='0',system='baseline')
        bad=copy.deepcopy(report);bad['attempts'][0]['cost_quantities']['small.input']=2
        with self.assertRaises(ValueError):reduce(self.plan,bad)
        for update in ({'status':'running'},{'ended_ns':-1},{'attempt_id':'foreign:0'},
                       {'relationship':'retry','predecessor':tokens[0]}, {'cost_quantities':{}},
                       {'receipt_sha256':None},{'provider':'foreign'}):
            bad=copy.deepcopy(report);bad['attempts'][0].update(update)
            with self.assertRaises(ValueError):reduce(self.plan,rehash(bad))

    async def test_empty_closed_scope_is_explicit_zero_not_unknown(self):
        self.recorder.close();report=self.recorder.snapshot()
        self.assertEqual(report['attempts'],[])
        self.assertTrue(all(v==0 for v in reduce(self.plan,report).values()))
        # This is only the instrumented scope: external audit must detect bypasses.
        self.assertEqual(report['evidence'],'synthetic')


class CollectorBridge(unittest.IsolatedAsyncioTestCase):
    async def test_actual_runner_records_attempts_then_join_binds_quantities(self):
        import hashlib
        import json
        from pathlib import Path
        import sys
        import tempfile
        from benchmarks.paired_replay import run
        from benchmarks.replay_evidence import join, read_traces
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root) as directory:
            directory=Path(directory);script=directory/'driver.py';plan_path=directory/'plan.json'
            script.write_text('''import asyncio,json,sys
from pathlib import Path
from benchmarks.call_attempts import AttemptRecorder
request=json.loads(sys.stdin.readline())
plan=json.loads(Path(sys.argv[1]).read_text())
r=AttemptRecorder(plan,expected_plan_sha256=request['plan_sha256'],task_id=request['task_id'],system=request['system'])
async def main():
    ids=[]
    async def fail(token):
        ids.append(token)
        raise RuntimeError('synthetic provider failure')
    try:
        await r.invoke(fail,provider='small',revision='frozen-v1',role='small_vlm',phase='cold',evidence_sha256='a'*64)
    except RuntimeError:
        pass
    r.report(ids[0],quantities={'small.input':2},receipt_sha256='d'*64)
    r.close()
    return r.snapshot()
report=asyncio.run(main())
print(json.dumps(dict(schema_version=1,status='success',answer='Synthetic fixture, not a model answer',measurements={'attempt_report':report})))
''')
            plan,_=itemized();plan['tasks']=plan['tasks'][:1];data={'synthetic':True}
            plan['tasks'][0]['input_sha256']=plan_digest(data)
            executable=str(Path(sys.executable).resolve())
            command=dict(argv=[executable,str(script),str(plan_path)],cwd=str(root),environment={'PYTHONPATH':str(root)},
                artifacts={executable:hashlib.sha256(Path(executable).read_bytes()).hexdigest(),str(script):hashlib.sha256(script.read_bytes()).hexdigest()})
            plan['runner']=dict(schema_version=1,timeout_ms=10000,order='alternating',commands=dict(baseline=command,candidate=copy.deepcopy(command)))
            plan['evidence_join']=dict(schema_version=1,rubric_sha256='a'*64,measurement_protocol_sha256='b'*64,failure_policy='zero_quality_timeout_floor')
            plan_path.write_text(json.dumps(plan))
            digest=plan_digest(plan);inputs=[dict(task_id='0',input=data)]
            complete=await run(plan,inputs,directory/'run',expected_plan_sha256=digest)
            traces=list(read_traces(directory/'run/source-traces.jsonl'));entries=[]
            for trace in traces:
                self.assertEqual(trace['status'],'success',trace)
                report=trace['response']['measurements']['attempt_report']
                self.assertEqual(report['attempts'][0]['status'],'error')
                quantities=cost_quantities(plan,report,expected_plan_sha256=digest,expected_report_sha256=report['sha256'],task_id=trace['task_id'],system=trace['system'])
                entries.append(dict(trace_sha256=trace['trace_sha256'],quality_score=0,correct_critical_facts=[],
                    usage=dict(calls=1,input_tokens=2,output_tokens=0,cache_read_tokens=0,cache_write_tokens=0),
                    peak_memory_bytes=1,dropped_frames=0,resource_cost=None,cost_quantities=quantities))
            assessment=dict(schema_version=1,plan_sha256=digest,last_trace_sha256=complete['last_trace_sha256'],
                reviewer='synthetic fixture',rubric_sha256='a'*64,measurement_protocol_sha256='b'*64,
                source_evidence_sha256=plan_digest([t['response']['measurements'] for t in traces]),entries=entries)
            result=join(plan,inputs,traces,complete,assessment,expected_plan_sha256=digest,
                expected_trace_sha256=complete['last_trace_sha256'],expected_assessment_sha256=plan_digest(assessment))
            self.assertEqual([row['cost_quantities']['small.input'] for row in result['rows']],[2,2])
            self.assertFalse(result['comparison']['meets_numeric_gates'])

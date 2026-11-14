import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from benchmarks.call_attempts import AttemptRecorder, cost_quantities as paired_quantities
from benchmarks.controller_attempts import RecordedContextClient
from benchmarks.perception_comparison import plan_digest
from benchmarks.three_arm_attempts import ThreeArmAttemptRecorder, cost_quantities, join
from benchmarks.three_arm_evidence import read_traces
from humansense import TextContextCalls
from test_itemized_costs import itemized
from test_three_arm_replay import setup, collect
import test_text_context_call as text_tests


def rehash(report):
    report['sha256'] = plan_digest({k:v for k,v in report.items() if k != 'sha256'})
    return report


@unittest.skipUnless(os.name == 'posix', 'fixture collector requires POSIX')
class ThreeArmAttempts(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory(dir='.')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.plan, self.spec, self.inputs = setup(self.root, ('ok', 'error'))
        self.plan['comparison']['cost'] = itemized()[0]['cost']
        self.spec['comparison_plan_sha256'] = plan_digest(self.plan)
        self.run_path = self.root/'run'
        self.completion = await collect(self.plan, self.spec, self.inputs, self.run_path)
        self.traces = list(read_traces(self.run_path/'source-traces.jsonl'))
        self.reports = []
        self.assessment = dict(schema_version=1, comparison_plan_sha256=plan_digest(self.plan),
            collection_sha256=plan_digest(self.spec), last_trace_sha256=self.completion['last_trace_sha256'],
            reviewer='fixture scoring reviewer', rubric_sha256='1'*64,
            measurement_protocol_sha256='2'*64, source_evidence_sha256='3'*64, entries=[])
        for index, trace in enumerate(self.traces):
            rec = self.recorder(task_id=trace['task_id'], arm=trace['arm'])
            async def invoke(token):
                rec.report(token, quantities={'request':1}, receipt_sha256=plan_digest(['receipt',index]))
                if trace['status'] != 'success': raise RuntimeError('fixture failure')
            if trace['status'] == 'success':
                await rec.invoke(invoke,provider='remote',revision='frozen-v1',role='strong_vlm',
                                 phase='cold',evidence_sha256=trace['input_sha256'])
            else:
                with self.assertRaises(RuntimeError):
                    await rec.invoke(invoke,provider='remote',revision='frozen-v1',role='strong_vlm',
                                     phase='cold',evidence_sha256=trace['input_sha256'])
            rec.close(); report=rec.snapshot(); self.reports.append(report)
            self.assessment['entries'].append(dict(trace_sha256=trace['trace_sha256'],
                quality_score=int(trace['status']=='success'),
                correct_critical_facts=['fact'] if trace['status']=='success' else [],
                usage=dict(calls=1,input_tokens=1,output_tokens=1,cache_read_tokens=0,cache_write_tokens=0),
                peak_memory_bytes=100,dropped_frames=0,resource_cost=None,
                cost_quantities=self.reduce(report)))
        self.manifest = dict(schema_version=1,comparison_plan_sha256=plan_digest(self.plan),
            collection_sha256=plan_digest(self.spec),last_trace_sha256=self.completion['last_trace_sha256'],
            assessment_sha256=plan_digest(self.assessment),reviewer='fixture accounting reviewer',
            coverage_evidence_sha256='4'*64,entries=[dict(trace_sha256=t['trace_sha256'],
                report_sha256=r['sha256'],coverage_statement='Synthetic explicitly recorded callback only; coverage unverified')
                for t,r in zip(self.traces,self.reports)])

    def recorder(self,task_id='0',arm='A',**kwargs):
        return ThreeArmAttemptRecorder(self.plan,self.spec,self.inputs,
            expected_plan_sha256=plan_digest(self.plan),expected_collection_sha256=plan_digest(self.spec),
            task_id=task_id,arm=arm,**kwargs)

    def reduce(self,report,**kwargs):
        args=dict(expected_plan_sha256=plan_digest(self.plan),expected_collection_sha256=plan_digest(self.spec),
            expected_report_sha256=report['sha256'],task_id=report['task_id'],arm=report['arm'])
        args.update(kwargs)
        return cost_quantities(self.plan,self.spec,self.inputs,report,**args)

    def joined(self,**updates):
        args=dict(plan=self.plan,spec=self.spec,inputs=self.inputs,traces=self.traces,
            completion=self.completion,assessment=self.assessment,manifest=self.manifest,reports=self.reports,
            expected_plan_sha256=plan_digest(self.plan),expected_collection_sha256=plan_digest(self.spec),
            expected_trace_sha256=self.completion['last_trace_sha256'],
            expected_assessment_sha256=plan_digest(self.assessment),expected_manifest_sha256=plan_digest(self.manifest))
        args.update(updates)
        return join(**args)

    def replace_report(self,index,report):
        reports=copy.deepcopy(self.reports); reports[index]=rehash(report)
        manifest=copy.deepcopy(self.manifest); manifest['entries'][index]['report_sha256']=report['sha256']
        return dict(reports=reports,manifest=manifest,expected_manifest_sha256=plan_digest(manifest))

    async def test_original_collector_binding_lineage_canonical_a_and_immutability(self):
        before=copy.deepcopy((self.plan,self.spec,self.inputs,self.traces,self.completion,
                              self.assessment,self.manifest,self.reports))
        source={p.name:p.read_bytes() for p in self.run_path.iterdir()}
        result=self.joined()
        self.assertEqual(result['actual_acceptance'],'incomplete')
        self.assertFalse(result['receipts_verified']); self.assertFalse(result['instrumentation_coverage_verified'])
        self.assertEqual(len(result['attempt_lineage']),6)
        for i,(line,trace,report,row) in enumerate(zip(result['attempt_lineage'],self.traces,self.reports,result['rows'])):
            self.assertEqual(line['ordinal'],i); self.assertEqual(line['arm'],trace['arm'])
            self.assertEqual(line['task_id'],trace['task_id'])
            self.assertEqual(line['trace_sha256'],trace['trace_sha256'])
            self.assertEqual(line['report_sha256'],report['sha256'])
            self.assertEqual(line['row_sha256'],plan_digest(row))
            self.assertEqual(report['schema_version'],2); self.assertNotIn('system',report)
            self.assertNotIn('plan_sha256',report)
        self.assertEqual(result['comparison']['summaries']['A']['cost_quantities']['request'],2)
        for key in ('B_vs_A','C_vs_A'):
            self.assertEqual(result['comparison']['comparisons'][key]['summaries']['baseline']['cost_quantities']['request'],2)
        self.assertEqual(before,(self.plan,self.spec,self.inputs,self.traces,self.completion,
                                 self.assessment,self.manifest,self.reports))
        self.assertEqual(source,{p.name:p.read_bytes() for p in self.run_path.iterdir()})

    async def test_collector_subprocess_records_native_report_from_actual_request(self):
        script=self.root/'native-pipeline.py'
        script.write_text("""import asyncio,json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[4])
from benchmarks.perception_comparison import plan_digest
from benchmarks.three_arm_attempts import ThreeArmAttemptRecorder
request=json.loads(sys.stdin.readline())
plan=json.loads(Path(sys.argv[1]).read_text())
spec=json.loads(Path(sys.argv[2]).read_text())
inputs=json.loads(Path(sys.argv[3]).read_text())
rec=ThreeArmAttemptRecorder(plan,spec,inputs,
    expected_plan_sha256=request['comparison_plan_sha256'],
    expected_collection_sha256=request['collection_sha256'],
    task_id=request['task_id'],arm=request['arm'])
async def main():
    async def callback(token):
        rec.report(token,quantities={'request':1},receipt_sha256=plan_digest([request['task_id'],request['arm']]))
        return request['arm']+':'+request['task_id']
    answer=await rec.invoke(callback,provider='remote',revision='frozen-v1',role='strong_vlm',
        phase='cold',evidence_sha256=plan_digest(request['input']))
    rec.close()
    Path('report-'+request['task_id']+'-'+request['arm']+'.json').write_text(json.dumps(rec.snapshot()))
    print(json.dumps(dict(schema_version=1,status='success',answer=answer,measurements={})))
asyncio.run(main())
""")
        planpath=self.root/'native-plan.json';specpath=self.root/'native-spec.json';inputpath=self.root/'native-inputs.json'
        python=str(Path(sys.executable).resolve())
        command=dict(argv=[python,str(script),str(planpath),str(specpath),str(inputpath),str(Path.cwd())],
            cwd=str(self.root),environment={},artifacts={python:hashlib.sha256(Path(python).read_bytes()).hexdigest(),
            str(script):hashlib.sha256(script.read_bytes()).hexdigest()})
        self.spec['runner']['commands']={arm:copy.deepcopy(command) for arm in 'ABC'}
        planpath.write_text(json.dumps(self.plan));specpath.write_text(json.dumps(self.spec));inputpath.write_text(json.dumps(self.inputs))
        self.completion=await collect(self.plan,self.spec,self.inputs,self.root/'native-run')
        self.traces=list(read_traces(self.root/'native-run/source-traces.jsonl'))
        self.reports=[json.loads((self.root/('report-'+t['task_id']+'-'+t['arm']+'.json')).read_text()) for t in self.traces]
        self.assessment['collection_sha256']=plan_digest(self.spec)
        self.assessment['last_trace_sha256']=self.completion['last_trace_sha256']
        for entry,trace,report in zip(self.assessment['entries'],self.traces,self.reports):
            self.assertEqual(trace['status'],'success')
            entry['trace_sha256']=trace['trace_sha256'];entry['cost_quantities']=self.reduce(report)
        self.manifest.update(collection_sha256=plan_digest(self.spec),last_trace_sha256=self.completion['last_trace_sha256'],
                             assessment_sha256=plan_digest(self.assessment))
        for entry,trace,report in zip(self.manifest['entries'],self.traces,self.reports):
            entry.update(trace_sha256=trace['trace_sha256'],report_sha256=report['sha256'])
        result=self.joined()
        self.assertEqual(result['recorded_attempts'],6)
        self.assertEqual(result['unique_recorded_receipts'],6)
        self.assertEqual({arm:totals['request'] for arm,totals in result['canonical_cost_quantities'].items()},
                         {'A':2,'B':2,'C':2})

    async def test_frozen_manifest_and_report_bindings_fail_closed(self):
        for field in ('expected_plan_sha256','expected_collection_sha256','expected_trace_sha256',
                      'expected_assessment_sha256','expected_manifest_sha256'):
            with self.subTest(field=field),self.assertRaises(ValueError): self.joined(**{field:'f'*64})
        for field,value in (('arm','B'),('task_id','1'),('configuration_sha256','f'*64),
                ('input_sha256','f'*64),('comparison_plan_sha256','f'*64),('collection_sha256','f'*64),
                ('cost_sha256','f'*64),('schema_version',1),('evidence','measured')):
            report=copy.deepcopy(self.reports[0]); report[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError): self.joined(**self.replace_report(0,report))
        changed=copy.deepcopy(self.reports); changed[0]['attempts'][0]['cost_quantities']['request']=2
        with self.assertRaises(ValueError): self.joined(reports=changed)

    async def test_manifest_exact_mapping_and_coverage_bindings(self):
        for mode in ('missing','extra','duplicate','unknown','swapped','empty_coverage'):
            manifest=copy.deepcopy(self.manifest)
            if mode=='missing': manifest['entries'].pop()
            if mode=='extra': manifest['entries'].append(copy.deepcopy(manifest['entries'][0]))
            if mode=='duplicate': manifest['entries'][1]=copy.deepcopy(manifest['entries'][0])
            if mode=='unknown': manifest['entries'][0]['report_sha256']='f'*64
            if mode=='swapped': manifest['entries'][0]['report_sha256']=manifest['entries'][1]['report_sha256']
            if mode=='empty_coverage': manifest['entries'][0]['coverage_statement']=''
            with self.subTest(mode=mode),self.assertRaises(ValueError):
                self.joined(manifest=manifest,expected_manifest_sha256=plan_digest(manifest))
        for field in ('reviewer','coverage_evidence_sha256'):
            manifest=copy.deepcopy(self.manifest); manifest[field]='changed' if field=='reviewer' else 'f'*64
            with self.assertRaises(ValueError): self.joined(manifest=manifest)
        for reports in (self.reports[:-1],self.reports+[self.reports[0]],self.reports[1:]+[self.reports[1]]):
            with self.assertRaises(ValueError): self.joined(reports=reports)

    async def test_cross_arm_scope_attempt_and_receipt_collisions(self):
        for index in (1,3):  # Different arm, then different task.
            for mode in ('scope','receipt'):
                report=copy.deepcopy(self.reports[index])
                if mode=='scope':
                    report['scope']=self.reports[0]['scope']
                    report['attempts'][0]['attempt_id']=self.reports[0]['attempts'][0]['attempt_id']
                else: report['attempts'][0]['receipt_sha256']=self.reports[0]['attempts'][0]['receipt_sha256']
                with self.subTest(index=index,mode=mode),self.assertRaises(ValueError):
                    self.joined(**self.replace_report(index,report))
        # A duplicated receipt within one report is also an unsupported allocation.
        report=copy.deepcopy(self.reports[0]); row=copy.deepcopy(report['attempts'][0])
        row['attempt_id']=report['scope']+':1';report['attempts'].append(row)
        with self.assertRaises(ValueError): self.joined(**self.replace_report(0,report))

    async def test_unknown_inflight_rejected_or_unclosed_report_and_totals_mismatch(self):
        for field,value in (('closed',False),('rejected',1),('rejected',True)):
            report=copy.deepcopy(self.reports[0]);report[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError): self.joined(**self.replace_report(0,report))
        for field,value in (('status','running'),('ended_ns',None),('receipt_sha256',None),
                            ('cost_quantities',None),('cost_quantities',{'request':2})):
            report=copy.deepcopy(self.reports[0]);report['attempts'][0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError): self.joined(**self.replace_report(0,report))

    async def test_cancelled_and_error_attempts_late_receipts_retries_and_escalation(self):
        rec=self.recorder(); started=asyncio.Event(); tokens=[]
        async def blocked(token): tokens.append(token);started.set();await asyncio.Event().wait()
        task=asyncio.create_task(rec.invoke(blocked,provider='remote',revision='frozen-v1',role='strong_vlm',
            phase='cold',evidence_sha256='a'*64))
        await started.wait();task.cancel()
        with self.assertRaises(asyncio.CancelledError): await task
        async def fail(token): tokens.append(token);raise RuntimeError('failed retry')
        with self.assertRaises(RuntimeError):
            await rec.invoke(fail,provider='remote',revision='frozen-v1',role='strong_vlm',phase='warm',
                evidence_sha256='a'*64,predecessor=tokens[0],relationship='retry')
        async def finish(token):tokens.append(token);return 'ok'
        await rec.invoke(finish,provider='text',revision='frozen-v1',role='text_llm',phase='warm',
            evidence_sha256='a'*64,predecessor=tokens[1],relationship='escalation')
        rec.close()
        with self.assertRaises(ValueError):self.reduce(rec.snapshot())
        for i,token in enumerate(tokens):
            rec.report(token,quantities={'text.output':3} if i==2 else {'request':1},receipt_sha256=plan_digest(['late',i]))
        report=rec.snapshot(); totals=self.reduce(report)
        self.assertEqual([r['status'] for r in report['attempts']],['cancelled','error','success'])
        self.assertEqual(totals['request'],2);self.assertEqual(totals['text.output'],3)
        with self.assertRaises(ValueError):rec.report(tokens[0],quantities={'request':1},receipt_sha256='f'*64)

    async def test_empty_report_is_zero_recorded_attempts_not_coverage_proof(self):
        reports=[];assessment=copy.deepcopy(self.assessment);manifest=copy.deepcopy(self.manifest)
        for i,trace in enumerate(self.traces):
            rec=self.recorder(task_id=trace['task_id'],arm=trace['arm']);rec.close();report=rec.snapshot();reports.append(report)
            assessment['entries'][i]['cost_quantities']=self.reduce(report)
            manifest['entries'][i]['report_sha256']=report['sha256']
            manifest['entries'][i]['coverage_statement']='No callbacks recorded; uninstrumented work not excluded'
        manifest['assessment_sha256']=plan_digest(assessment)
        result=self.joined(reports=reports,assessment=assessment,manifest=manifest,
            expected_assessment_sha256=plan_digest(assessment),expected_manifest_sha256=plan_digest(manifest))
        self.assertFalse(result['instrumentation_coverage_verified']);self.assertFalse(result['receipts_verified'])
        self.assertEqual(result['actual_acceptance'],'incomplete')
        self.assertTrue(all(row['cost_quantities']['request']==0 for row in result['rows']))

    async def test_paired_v1_remains_native_and_rejects_three_arm_report(self):
        plan,_=itemized();rec=AttemptRecorder(plan,expected_plan_sha256=plan_digest(plan),task_id='0',system='baseline')
        async def invoke(token):rec.report(token,quantities={'request':2},receipt_sha256='d'*64)
        await rec.invoke(invoke,provider='remote',revision='frozen-v1',role='strong_vlm',phase='cold',evidence_sha256='e'*64)
        rec.close();report=rec.snapshot()
        totals=paired_quantities(plan,report,expected_plan_sha256=plan_digest(plan),expected_report_sha256=report['sha256'],task_id='0',system='baseline')
        self.assertEqual(report['schema_version'],1);self.assertEqual(totals['request'],2)
        with self.assertRaises(ValueError):
            cost_quantities(self.plan,self.spec,self.inputs,report,expected_plan_sha256=plan_digest(self.plan),
                expected_collection_sha256=plan_digest(self.spec),expected_report_sha256=report['sha256'],task_id='0',arm='A')
        report=self.reports[0]
        with self.assertRaises(ValueError):paired_quantities(plan,report,expected_plan_sha256=plan_digest(plan),expected_report_sha256=report['sha256'],task_id='0',system='baseline')

    async def test_cli_exclusive_outputs_and_immutable_sources(self):
        paths={}
        for name,obj in (('assessment',self.assessment),('manifest',self.manifest),('reports',self.reports)):
            paths[name]=self.root/(name+'.json');paths[name].write_text(json.dumps(obj))
        originals={str(p):p.read_bytes() for p in list(self.run_path.iterdir())+list(paths.values())}
        out=self.root/'joined'
        cmd=[sys.executable,'-m','benchmarks.three_arm_attempts','--run',str(self.run_path),
            '--assessment',str(paths['assessment']),'--manifest',str(paths['manifest']),'--reports',str(paths['reports']),
            '--plan-sha256',plan_digest(self.plan),'--collection-sha256',plan_digest(self.spec),
            '--trace-sha256',self.completion['last_trace_sha256'],'--assessment-sha256',plan_digest(self.assessment),
            '--manifest-sha256',plan_digest(self.manifest),'--output',str(out)]
        result=await asyncio.to_thread(subprocess.run,cmd,capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertFalse(json.loads(result.stdout)['instrumentation_coverage_verified'])
        self.assertEqual(json.loads((out/'attempt-lineage.json').read_text()),self.joined()['attempt_lineage'])
        before={p.name:p.read_bytes() for p in out.iterdir()}
        again=await asyncio.to_thread(subprocess.run,cmd,capture_output=True,text=True,timeout=20)
        self.assertNotEqual(again.returncode,0)
        self.assertEqual(before,{p.name:p.read_bytes() for p in out.iterdir()})
        self.assertEqual(originals,{p:Path(p).read_bytes() for p in originals})
        # Strict readers fail before creating any output, including nested duplicate keys
        # and JSON exponent overflow rather than just explicit NaN.
        for name in ('manifest','reports'):
            original=paths[name].read_bytes()
            for index,payload in enumerate((b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e999}',
                                             b' '*(16*1024*1024+1))):
                paths[name].write_bytes(payload)
                invalid_out=self.root/('invalid-'+name+'-'+str(index))
                invalid_cmd=cmd[:-1]+[str(invalid_out)]
                invalid=await asyncio.to_thread(subprocess.run,invalid_cmd,capture_output=True,text=True,timeout=20)
                self.assertNotEqual(invalid.returncode,0)
                self.assertFalse(invalid_out.exists())
            paths[name].write_bytes(original)

    async def test_bounded_manifest_and_reports(self):
        manifest=copy.deepcopy(self.manifest);manifest['reviewer']='x'*(16*1024*1024)
        with self.assertRaises(ValueError):self.joined(manifest=manifest,expected_manifest_sha256=plan_digest(manifest))
        report=copy.deepcopy(self.reports[0]);report['attempts']*=1025
        with self.assertRaises(ValueError):self.joined(**self.replace_report(0,report))
        for value in (float('nan'),float('inf'),True,-1):
            report=copy.deepcopy(self.reports[0]);report['attempts'][0]['cost_quantities']['request']=value
            with self.assertRaises(ValueError):
                self.joined(**self.replace_report(0,report))


class NativeRecordedClient(unittest.IsolatedAsyncioTestCase):
    asyncSetUp=text_tests.Calls.asyncSetUp
    asyncTearDown=text_tests.Calls.asyncTearDown
    queued=text_tests.Calls.queued

    async def test_actual_sdk_client_records_native_arm_and_late_receipt(self):
        with tempfile.TemporaryDirectory(dir='.') as directory:
            plan,spec,inputs=setup(Path(directory).resolve(),('ok',))
            plan['comparison']['cost']=itemized()[0]['cost'];spec['comparison_plan_sha256']=plan_digest(plan)
            rec=ThreeArmAttemptRecorder(plan,spec,inputs,expected_plan_sha256=plan_digest(plan),
                expected_collection_sha256=plan_digest(spec),task_id='0',arm='C')
            self.assertIsInstance(rec,AttemptRecorder)
            rid,record=await self.queued();requests=[]
            async def complete(request):requests.append(copy.deepcopy(request));return dict(text='fixture answer',usage=None)
            wrapped=RecordedContextClient(self.c,rid,rec,complete,phase='cold')
            adapter=TextContextCalls(self.h,'main',wrapped,provider='text',model_revision='frozen-v1')
            self.adapters.append(adapter)
            result=await adapter.execute(rid,record,'fixture question',maximum_charge=10)
            self.assertTrue(result['accepted']);rec.close();report=rec.snapshot()
            self.assertEqual(report['arm'],'C');self.assertEqual(report['attempts'][0]['prompt_sha256'],plan_digest(requests[0]))
            def reduce():
                report=rec.snapshot()
                return cost_quantities(plan,spec,inputs,report,expected_plan_sha256=plan_digest(plan),
                    expected_collection_sha256=plan_digest(spec),expected_report_sha256=report['sha256'],task_id='0',arm='C')
            with self.assertRaises(ValueError):reduce()
            rec.report(wrapped.attempt_id,quantities={'text.output':3},receipt_sha256='d'*64)
            self.assertEqual(reduce()['text.output'],3)

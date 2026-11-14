import asyncio
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from benchmarks.perception_comparison import plan_digest
from benchmarks.three_arm_comparison import compare_three_arm, _read
from benchmarks.three_arm_evidence import join, read_traces
from test_three_arm_replay import setup, collect


@unittest.skipUnless(os.name == 'posix', 'actual collector requires POSIX')
class ThreeArmEvidence(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory(dir='.')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.plan, self.spec, self.inputs = setup(self.root, ('ok', 'error'))
        await self.collect_fixture('run')

    async def collect_fixture(self, name):
        self.spec['comparison_plan_sha256'] = plan_digest(self.plan)
        self.run_path = self.root / name
        self.completion = await collect(self.plan, self.spec, self.inputs, self.run_path)
        self.traces = list(read_traces(self.run_path / 'source-traces.jsonl'))
        self.assessment = dict(schema_version=1, comparison_plan_sha256=plan_digest(self.plan),
            collection_sha256=plan_digest(self.spec), last_trace_sha256=self.completion['last_trace_sha256'],
            reviewer='synthetic fixture reviewer', rubric_sha256='1'*64,
            measurement_protocol_sha256='2'*64, source_evidence_sha256='3'*64,
            entries=[dict(trace_sha256=t['trace_sha256'], quality_score=int(t['status']=='success'),
                correct_critical_facts=['fact'] if t['status']=='success' else [],
                usage=dict(calls=1, input_tokens=1, output_tokens=1, cache_read_tokens=0, cache_write_tokens=0),
                peak_memory_bytes=100, dropped_frames=0, resource_cost=None) for t in self.traces])

    def joined(self, **updates):
        args = dict(plan=self.plan, spec=self.spec, inputs=self.inputs, traces=self.traces,
            completion=self.completion, assessment=self.assessment,
            expected_plan_sha256=plan_digest(self.plan), expected_collection_sha256=plan_digest(self.spec),
            expected_trace_sha256=self.completion['last_trace_sha256'],
            expected_assessment_sha256=plan_digest(self.assessment))
        args.update(updates)
        return join(**args)

    def rechain(self, traces):
        previous = '0'*64
        for t in traces:
            t['previous_trace_sha256'] = previous
            t['trace_sha256'] = plan_digest({k:v for k,v in t.items() if k != 'trace_sha256'})
            previous = t['trace_sha256']
        complete = copy.deepcopy(self.completion)
        complete['last_trace_sha256'] = previous
        a = copy.deepcopy(self.assessment)
        a['last_trace_sha256'] = previous
        for entry, trace in zip(a['entries'], traces): entry['trace_sha256'] = trace['trace_sha256']
        return dict(traces=traces, completion=complete, assessment=a,
            expected_trace_sha256=previous, expected_assessment_sha256=plan_digest(a))

    async def test_actual_collector_original_arms_immutable_inputs_numeric_parity(self):
        before = copy.deepcopy((self.plan,self.spec,self.inputs,self.traces,self.completion,self.assessment))
        source = {p.name:p.read_bytes() for p in self.run_path.iterdir()}
        result = self.joined()
        self.assertEqual(len(result['rows']), 6)
        self.assertEqual([r['arm'] for r in result['rows']], ['A','B','C','A','C','B'])
        self.assertEqual(result['rows_sha256'], plan_digest(result['rows']))
        self.assertEqual(len(result['lineage']), 6)
        for index, (trace, entry, row, lineage) in enumerate(zip(
                self.traces, self.assessment['entries'], result['rows'], result['lineage'])):
            self.assertEqual(lineage, dict(ordinal=index, task_id=trace['task_id'], arm=trace['arm'],
                trace_sha256=trace['trace_sha256'], assessment_entry_sha256=plan_digest(entry),
                row_sha256=plan_digest(row)))
        self.assertEqual(result['comparison'], compare_three_arm(self.plan, result['rows'],
            expected_plan_sha256=plan_digest(self.plan), expected_rows_sha256=plan_digest(result['rows'])))
        self.assertEqual(result['actual_acceptance'], 'incomplete')
        self.assertTrue(result['source_hash_integrity_verified'])
        self.assertFalse(result['source_verified'])
        self.assertFalse(result['comparison']['primary_meets_numeric_gates'])
        for trace, row in zip(self.traces, result['rows']):
            self.assertEqual(row['configuration_sha256'], trace['configuration_sha256'])
            self.assertEqual(row['latency_ms'], trace['answer_ready_ms'] if trace['status']=='success'
                else max(trace['process_lifetime_ms'], self.spec['runner']['timeout_ms']))
        self.assertEqual(before, (self.plan,self.spec,self.inputs,self.traces,self.completion,self.assessment))
        self.assertEqual(source, {p.name:p.read_bytes() for p in self.run_path.iterdir()})

    async def test_all_six_original_orders_and_position_counts(self):
        self.plan, self.spec, self.inputs = setup(self.root)
        await self.collect_fixture('six-orders')
        result = self.joined()
        arms = [row['arm'] for row in result['rows']]
        self.assertEqual([''.join(arms[i:i+3]) for i in range(0,18,3)],
                         ['ABC','ACB','BAC','BCA','CAB','CBA'])
        self.assertEqual(self.completion['arm_position_counts'], {arm:[2,2,2] for arm in 'ABC'})
        self.assertEqual(len(result['lineage']), 18)

    async def test_truncated_reordered_extra_and_unhashed_response_change(self):
        changed = copy.deepcopy(self.traces); changed[0]['response']['answer'] = 'forged'
        for traces in (self.traces[:-1], self.traces[::-1], self.traces+[self.traces[0]], changed):
            with self.subTest(traces=len(traces)), self.assertRaises(ValueError): self.joined(traces=traces)

    async def test_rehashed_forged_request_configuration_order_and_provenance(self):
        for field, value in (('request_sha256','f'*64), ('configuration_sha256','f'*64),
                ('input_sha256','f'*64), ('comparison_plan_sha256','f'*64), ('collection_sha256','f'*64),
                ('ordinal', True), ('arm','baseline'), ('task_id','other'),
                ('quality_status','scored'), ('measurement_status','verified')):
            traces = copy.deepcopy(self.traces); traces[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): self.joined(**self.rechain(traces))

    async def test_completion_counts_positions_end_digest_and_schema(self):
        for field,value in (('records',5), ('records',True), ('state','started'),
                ('last_trace_sha256','f'*64), ('complete_six_task_blocks',1), ('remaining_tasks',0),
                ('arm_position_counts',{'A':[2,0,0],'B':[0,2,0],'C':[0,0,2]}), ('schema_version',True)):
            completion=copy.deepcopy(self.completion); completion[field]=value
            with self.subTest(field=field), self.assertRaises(ValueError): self.joined(completion=completion)

    async def test_frozen_declarations_inputs_and_assessment_hash(self):
        for name in ('expected_plan_sha256','expected_collection_sha256','expected_trace_sha256','expected_assessment_sha256'):
            with self.subTest(name=name), self.assertRaises(ValueError): self.joined(**{name:'f'*64})
        inputs=copy.deepcopy(self.inputs); inputs[0]['input']['prompt']='changed'
        with self.assertRaises(ValueError): self.joined(inputs=inputs)
        for field in ('reviewer','rubric_sha256','measurement_protocol_sha256','source_evidence_sha256'):
            a=copy.deepcopy(self.assessment); a[field]='changed' if field=='reviewer' else 'f'*64
            with self.subTest(field=field), self.assertRaises(ValueError): self.joined(assessment=a)
        for field in ('rubric_sha256','measurement_protocol_sha256','last_trace_sha256','collection_sha256'):
            a=copy.deepcopy(self.assessment); a[field]='f'*64
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.joined(assessment=a,expected_assessment_sha256=plan_digest(a))

    async def test_one_assessment_per_trace_and_failed_credit(self):
        for mutation in ('missing','extra','duplicate','unknown','credit','facts'):
            a=copy.deepcopy(self.assessment)
            if mutation=='missing': a['entries'].pop()
            if mutation=='extra': a['entries'].append(copy.deepcopy(a['entries'][0]))
            if mutation=='duplicate': a['entries'][1]=copy.deepcopy(a['entries'][0])
            if mutation=='unknown': a['entries'][0]['trace_sha256']='f'*64
            if mutation=='credit': a['entries'][-1]['quality_score']=.5
            if mutation=='facts': a['entries'][-1]['correct_critical_facts']=['fact']
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.joined(assessment=a,expected_assessment_sha256=plan_digest(a))

    async def test_unknown_usage_memory_and_out_of_mode_cost_fail_closed(self):
        for field in ('calls','input_tokens','output_tokens','cache_read_tokens','cache_write_tokens'):
            a=copy.deepcopy(self.assessment); del a['entries'][0]['usage'][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.joined(assessment=a,expected_assessment_sha256=plan_digest(a))
        for field,value in (('peak_memory_bytes',None),('resource_cost',0),('dropped_frames',None)):
            a=copy.deepcopy(self.assessment); a['entries'][0][field]=value
            with self.assertRaises(ValueError): self.joined(assessment=a,expected_assessment_sha256=plan_digest(a))

    async def test_resource_and_itemized_costs_retain_failed_attempts(self):
        self.plan['comparison']['cost']['mode']='resource'
        await self.collect_fixture('resource')
        with self.assertRaises(ValueError): self.joined()
        for entry in self.assessment['entries']: entry['resource_cost']=1
        result=self.joined()
        self.assertTrue(all(row['resource_cost']==1 for row in result['rows']))
        from test_itemized_costs import itemized
        self.plan['comparison']['cost']=itemized()[0]['cost']
        await self.collect_fixture('itemized')
        for entry in self.assessment['entries']:
            entry['cost_quantities']={k:1 for k in self.plan['comparison']['cost']['rate_card']}
        result=self.joined()
        self.assertTrue(all(row['cost_quantities']['request']==1 for row in result['rows']))
        self.assertTrue(any(t['status']=='error' for t in self.traces))
        del self.assessment['entries'][-1]['cost_quantities']['local']
        with self.assertRaises(ValueError): self.joined()

    async def test_portable_join_does_not_reopen_execution_assets(self):
        # Frozen paths can be unavailable on the audit machine. Collector still
        # verified the actual artifact before execution; join certifies no live assets.
        self.plan, self.spec, self.inputs = setup(self.root, ('ok',))
        await self.collect_fixture('portable')
        (self.root/'pipeline.py').rename(self.root/'retired-pipeline.py')
        self.assertEqual(len(self.joined()['rows']),3)

    async def test_cli_exclusive_artifacts_and_original_source_bytes(self):
        a=self.root/'assessment.json'; a.write_text(json.dumps(self.assessment))
        source={p.name:p.read_bytes() for p in self.run_path.iterdir()}
        output=self.root/'joined'
        command=[sys.executable,'-m','benchmarks.three_arm_evidence','--run',str(self.run_path),
            '--assessment',str(a),'--plan-sha256',plan_digest(self.plan),
            '--collection-sha256',plan_digest(self.spec),'--trace-sha256',self.completion['last_trace_sha256'],
            '--assessment-sha256',plan_digest(self.assessment),'--output',str(output)]
        run=await asyncio.to_thread(subprocess.run,command,capture_output=True,text=True,timeout=20)
        self.assertEqual(run.returncode,0,run.stderr)
        self.assertEqual(json.loads(run.stdout)['actual_acceptance'],'incomplete')
        self.assertEqual(_read(output/'rows.json'),self.joined()['rows'])
        self.assertEqual(_read(output/'lineage.json'),self.joined()['lineage'])
        artifact_bytes={p.name:p.read_bytes() for p in output.iterdir()}
        again=await asyncio.to_thread(subprocess.run,command,capture_output=True,text=True,timeout=20)
        self.assertNotEqual(again.returncode,0)
        self.assertEqual(artifact_bytes,{p.name:p.read_bytes() for p in output.iterdir()})
        self.assertEqual(source,{p.name:p.read_bytes() for p in self.run_path.iterdir()})


class ThreeArmEvidenceBounds(unittest.TestCase):
    def test_strict_bounded_trace_reader(self):
        with tempfile.TemporaryDirectory(dir='.') as directory:
            path=Path(directory)/'traces.jsonl'
            for payload in (b'{}', b' '* (512*1024)+b'\n', b'{}\n'*3001,
                    b'{"a":1,"a":2}\n',b'{"a":NaN}\n',b'{"a":1e999}\n'):
                path.write_bytes(payload)
                with self.subTest(payload=payload[:30]),self.assertRaises(ValueError): list(read_traces(path))

    def test_strict_bounded_declaration_reader(self):
        with tempfile.TemporaryDirectory(dir='.') as directory:
            path=Path(directory)/'declaration.json'
            for payload in (b' '* (16*1024*1024+1),b'{"a":1,"a":2}',b'{"a":NaN}'):
                path.write_bytes(payload)
                with self.assertRaises(ValueError): _read(path)

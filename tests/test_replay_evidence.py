import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

from benchmarks.paired_replay import run
from benchmarks.perception_comparison import plan_digest
from benchmarks.replay_evidence import join,read_traces
from test_paired_replay import setup


@unittest.skipUnless(os.name=='posix','collector fixture requires POSIX')
class Evidence(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory=tempfile.TemporaryDirectory(dir='.')
        self.addCleanup(self.directory.cleanup)
        root=Path(self.directory.name).resolve()
        self.plan,self.inputs=setup(root,('ok','nonzero'))
        self.plan['evidence_join']=dict(schema_version=1,rubric_sha256='a'*64,
            measurement_protocol_sha256='b'*64,failure_policy='zero_quality_timeout_floor')
        self.digest=plan_digest(self.plan)
        self.completion=await run(self.plan,self.inputs,root/'run',expected_plan_sha256=self.digest)
        self.traces=list(read_traces(root/'run/source-traces.jsonl'))
        self.assessment=dict(schema_version=1,plan_sha256=self.digest,
            last_trace_sha256=self.completion['last_trace_sha256'],reviewer='synthetic fixture reviewer',
            rubric_sha256='a'*64,measurement_protocol_sha256='b'*64,source_evidence_sha256='c'*64,
            entries=[dict(trace_sha256=t['trace_sha256'],quality_score=int(t['status']=='success'),
                correct_critical_facts=['fact'] if t['status']=='success' else [],
                usage=dict(calls=1,input_tokens=1,output_tokens=1,cache_read_tokens=0,cache_write_tokens=0),
                peak_memory_bytes=100,dropped_frames=0,resource_cost=None) for t in self.traces])

    def joined(self,**updates):
        args=dict(plan=self.plan,inputs=self.inputs,traces=self.traces,completion=self.completion,
                  assessment=self.assessment,expected_plan_sha256=self.digest,
                  expected_trace_sha256=self.completion['last_trace_sha256'],
                  expected_assessment_sha256=plan_digest(self.assessment))
        args.update(updates); return join(**args)

    async def test_real_collector_join_preserves_every_pair_and_cannot_certify_fixture(self):
        result=self.joined()
        self.assertEqual(len(result['rows']),4)
        self.assertEqual(result['rows_sha256'],plan_digest(result['rows']))
        self.assertFalse(result['comparison']['meets_numeric_gates'])
        for trace,row in zip(self.traces,result['rows']):
            self.assertEqual(row['system'],trace['system'])
            if trace['status']=='success': self.assertEqual(row['latency_ms'],trace['answer_ready_ms'])
            else:
                self.assertGreaterEqual(row['latency_ms'],self.plan['runner']['timeout_ms'])
                self.assertEqual(row['quality_score'],0); self.assertEqual(row['correct_critical_facts'],[])

    async def test_truncation_reorder_insertion_and_tampered_response_rejected(self):
        variants=[self.traces[:-1],list(reversed(self.traces)),self.traces+[self.traces[0]]]
        altered=copy.deepcopy(self.traces); altered[0]['response']['answer']='changed'; variants.append(altered)
        for traces in variants:
            with self.assertRaises(ValueError): self.joined(traces=traces)
        completion=copy.deepcopy(self.completion); completion['state']='started'
        with self.assertRaises(ValueError): self.joined(completion=completion)
        with self.assertRaises(ValueError): self.joined(expected_trace_sha256='f'*64)
        inputs=copy.deepcopy(self.inputs); inputs[0]['input']['prompt']='different'
        with self.assertRaises(ValueError): self.joined(inputs=inputs)

    async def test_assessment_requires_exact_trace_rubric_and_reviewed_digest(self):
        for change in ('digest','rubric','source','duplicate','missing'):
            a=copy.deepcopy(self.assessment)
            if change=='digest': a['reviewer']='changed after review'
            if change=='rubric': a['rubric_sha256']='d'*64
            if change=='source': a['last_trace_sha256']='d'*64
            if change=='duplicate': a['entries'][1]=a['entries'][0]
            if change=='missing': a['entries'].pop()
            expected=plan_digest(self.assessment) if change=='digest' else plan_digest(a)
            with self.assertRaises(ValueError): self.joined(assessment=a,expected_assessment_sha256=expected)

    async def test_unknown_usage_or_memory_cannot_become_zero(self):
        for missing in ('input_tokens','cache_read_tokens','cache_write_tokens','calls'):
            a=copy.deepcopy(self.assessment); a['entries'][0]['usage'].pop(missing)
            with self.assertRaises(ValueError): self.joined(assessment=a,expected_assessment_sha256=plan_digest(a))
        a=copy.deepcopy(self.assessment); a['entries'][0]['peak_memory_bytes']=None
        with self.assertRaises(ValueError): self.joined(assessment=a,expected_assessment_sha256=plan_digest(a))

    async def test_resource_mode_requires_measured_cost_for_every_trace(self):
        plan=copy.deepcopy(self.plan); plan['cost']['mode']='resource'; digest=plan_digest(plan)
        root=Path(self.directory.name).resolve()/'resource-run'
        complete=await run(plan,self.inputs,root,expected_plan_sha256=digest)
        traces=list(read_traces(root/'source-traces.jsonl'))
        assessment=copy.deepcopy(self.assessment)
        assessment.update(plan_sha256=digest,last_trace_sha256=complete['last_trace_sha256'])
        for entry,trace in zip(assessment['entries'],traces): entry['trace_sha256']=trace['trace_sha256']
        overrides=dict(plan=plan,traces=traces,completion=complete,assessment=assessment,
                       expected_plan_sha256=digest,expected_trace_sha256=complete['last_trace_sha256'])
        with self.assertRaises(ValueError):
            self.joined(**overrides,expected_assessment_sha256=plan_digest(assessment))
        for entry in assessment['entries']: entry['resource_cost']=1
        result=self.joined(**overrides,expected_assessment_sha256=plan_digest(assessment))
        self.assertTrue(all(row['resource_cost']==1 for row in result['rows']))
        self.assertFalse(result['comparison']['meets_numeric_gates'])

    async def test_failed_answers_cannot_receive_quality_or_critical_fact_credit(self):
        failed=next(i for i,t in enumerate(self.traces) if t['status']!='success')
        for update in ({'quality_score':.5},{'correct_critical_facts':['fact']}):
            a=copy.deepcopy(self.assessment); a['entries'][failed].update(update)
            with self.assertRaises(ValueError): self.joined(assessment=a,expected_assessment_sha256=plan_digest(a))


class Bounds(unittest.TestCase):
    def test_streamed_source_lines_reject_incomplete_or_oversized_records(self):
        with tempfile.TemporaryDirectory(dir='.') as directory:
            path=Path(directory)/'traces.jsonl'
            for payload in (b'{}',b' '* (512*1024)+b'\n',b'{}\n'*2001):
                path.write_bytes(payload)
                with self.assertRaises(ValueError): list(read_traces(path))

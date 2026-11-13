import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.paired_replay import run, validate_collection
from benchmarks.perception_comparison import plan_digest
from test_perception_comparison import fixture


SCRIPT = '''import json,sys,time
request=json.loads(sys.stdin.readline())
mode=request['input'].get('mode','ok')
if mode=='timeout': time.sleep(60)
if mode=='overflow': print('x'*70000); sys.exit(0)
if mode=='nan': print('{"schema_version":1,"status":"success","answer":"x","measurements":{"energy":NaN}}'); sys.exit(0)
response=dict(schema_version=1,status='success',answer=request['input']['prompt'],measurements={})
print(json.dumps(response),flush=True)
if mode=='trailing': print('extra')
if mode=='nonzero': sys.exit(2)
'''


def setup(root, modes=('ok','ok')):
    path=root/'pipeline.py'; path.write_text(SCRIPT)
    executable=str(Path(sys.executable).resolve())
    command=dict(argv=[executable,str(path)],cwd=str(root),environment={},
                 artifacts={executable:hashlib.sha256(Path(executable).read_bytes()).hexdigest(),
                            str(path):hashlib.sha256(path.read_bytes()).hexdigest()})
    plan,_=fixture(); plan['tasks']=plan['tasks'][:len(modes)]
    inputs=[]
    for task,mode in zip(plan['tasks'],modes):
        payload={'prompt':'task '+task['id'],'recorded_media':{'bytes_base64':'AA=='},'mode':mode}
        task['input_sha256']=plan_digest(payload)
        inputs.append({'task_id':task['id'],'input':payload})
    plan['runner']=dict(schema_version=1,timeout_ms=1000,order='alternating',
                        commands={'baseline':command,'candidate':copy.deepcopy(command)})
    return plan,inputs


@unittest.skipUnless(os.name=='posix','POSIX owned process groups required')
class Collection(unittest.IsolatedAsyncioTestCase):
    async def test_pair_order_frozen_inputs_hash_chain_and_unscored_measurements(self):
        with tempfile.TemporaryDirectory(dir='.') as directory:
            root=Path(directory).resolve(); plan,inputs=setup(root)
            out=root/'run'; digest=plan_digest(plan)
            result=await run(plan,inputs,out,expected_plan_sha256=digest)
            self.assertEqual(result['state'],'collected_unscored')
            traces=[json.loads(line) for line in (out/'source-traces.jsonl').read_text().splitlines()]
            self.assertEqual([(r['task_id'],r['system']) for r in traces],
                             [('0','baseline'),('0','candidate'),('1','candidate'),('1','baseline')])
            previous='0'*64
            for row in traces:
                recorded=row.pop('trace_sha256'); self.assertEqual(recorded,plan_digest(row))
                self.assertEqual(row['previous_trace_sha256'],previous); previous=recorded
                self.assertEqual(row['status'],'success'); self.assertGreater(row['answer_ready_ms'],0)
                self.assertGreaterEqual(row['process_lifetime_ms'],row['answer_ready_ms'])
                self.assertEqual(row['response']['measurements'],{})
                self.assertNotIn('quality_score',row); self.assertEqual(row['quality_status'],'unscored')
            self.assertEqual(previous,result['last_trace_sha256'])
            self.assertEqual(json.loads((out/'plan.json').read_text()),plan)
            self.assertEqual((out/'source-traces.jsonl').stat().st_mode & 0o777,0o600)
            with self.assertRaises(FileExistsError): await run(plan,inputs,out,expected_plan_sha256=digest)

    async def test_all_failed_pairs_retained_without_invented_accounting(self):
        for mode,status in (('timeout','timeout'),('overflow','error'),('nan','error'),
                            ('trailing','error'),('nonzero','error')):
            with tempfile.TemporaryDirectory(dir='.') as directory:
                root=Path(directory).resolve(); plan,inputs=setup(root,(mode,))
                if mode=='timeout': plan['runner']['timeout_ms']=100
                await run(plan,inputs,root/'run',expected_plan_sha256=plan_digest(plan))
                rows=[json.loads(line) for line in (root/'run/source-traces.jsonl').read_text().splitlines()]
                self.assertEqual(len(rows),2)
                self.assertTrue(all(row['status']==status for row in rows),mode)
                self.assertTrue(all('usage' not in row for row in rows))
                if mode=='nan': self.assertTrue(all(row['response'] is None for row in rows))

    async def test_changed_input_plan_artifact_and_missing_pairs_never_launch(self):
        with tempfile.TemporaryDirectory(dir='.') as directory:
            root=Path(directory).resolve(); plan,inputs=setup(root)
            for mutation in ('input','plan','missing','artifact'):
                p,i=copy.deepcopy(plan),copy.deepcopy(inputs); digest=plan_digest(p)
                if mutation=='input': i[0]['input']['prompt']='changed'
                if mutation=='plan': p['systems']['candidate']['policy']='changed'
                if mutation=='missing': i.pop()
                if mutation=='artifact': (root/'pipeline.py').write_text('changed')
                with patch('asyncio.create_subprocess_exec') as launch:
                    with self.assertRaises(ValueError): await run(p,i,root/'run',expected_plan_sha256=digest)
                    launch.assert_not_called()
                self.assertFalse((root/'run').exists())

    async def test_cancellation_reaps_owned_child_and_leaves_incomplete_run(self):
        with tempfile.TemporaryDirectory(dir='.') as directory:
            root=Path(directory).resolve(); plan,inputs=setup(root,('timeout',))
            processes=[]; started=asyncio.Event(); original=asyncio.create_subprocess_exec
            async def launch(*args,**kwargs):
                p=await original(*args,**kwargs); processes.append(p); started.set(); return p
            with patch('asyncio.create_subprocess_exec',side_effect=launch):
                task=asyncio.create_task(run(plan,inputs,root/'run',expected_plan_sha256=plan_digest(plan)))
                await asyncio.wait_for(started.wait(),2)
                # Let the collector acquire the created subprocess handle.
                await asyncio.sleep(.02)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError): await task
            self.assertEqual(len(processes),1)
            self.assertIsNotNone(processes[0].returncode)
            self.assertFalse((root/'run/complete.json').exists())


class Validation(unittest.TestCase):
    def test_environment_and_executable_must_be_explicit(self):
        with tempfile.TemporaryDirectory(dir='.') as directory:
            root=Path(directory).resolve(); plan,inputs=setup(root)
            for changes in ({'argv':['python3']},{'environment':{'BAD=KEY':'x'}},
                            {'artifacts':{}},{'cwd':'.'}):
                p=copy.deepcopy(plan); p['runner']['commands']['baseline'].update(changes)
                with self.assertRaises(ValueError): validate_collection(p,inputs,plan_digest(p))

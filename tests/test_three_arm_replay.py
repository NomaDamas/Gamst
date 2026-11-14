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
from unittest.mock import patch

from benchmarks.three_arm_replay import run, validate_collection_spec, PERMUTATIONS, ORDER
from benchmarks.perception_comparison import plan_digest
from test_three_arm_comparison import fixture

SCRIPT = '''import json,sys,time
r=json.loads(sys.stdin.readline());mode=r['input'].get('mode','ok')
if mode=='timeout':time.sleep(60)
if mode=='overflow':print('x'*70000);sys.exit(0)
if mode=='malformed':print('broken');sys.exit(0)
print(json.dumps(dict(schema_version=1,status='error' if mode=='error' else 'success',
 answer=r['arm']+':'+r['task_id'],measurements={'request':r})),flush=True)
'''


def setup(root, modes=('ok',)*6):
    plan,_ = fixture('synthetic')
    plan['comparison']['tasks'] = plan['comparison']['tasks'][:len(modes)]
    inputs = []
    for task,mode in zip(plan['comparison']['tasks'],modes):
        payload = {'prompt': 'same visible input', 'mode': mode}
        task['input_sha256'] = plan_digest(payload)
        inputs.append({'task_id': task['id'], 'input': payload})
    script = root/'pipeline.py'; script.write_text(SCRIPT)
    python = str(Path(sys.executable).resolve())
    command = dict(argv=[python,str(script)],cwd=str(root),environment={},artifacts={
        python:hashlib.sha256(Path(python).read_bytes()).hexdigest(),
        str(script):hashlib.sha256(script.read_bytes()).hexdigest()})
    spec = dict(schema_version=1,comparison_plan_sha256=plan_digest(plan),
        runner=dict(timeout_ms=1000,order=ORDER,commands={arm:copy.deepcopy(command) for arm in 'ABC'}),
        evidence_join=dict(schema_version=1,rubric_sha256='1'*64,measurement_protocol_sha256='2'*64,
                           failure_policy='zero_quality_timeout_floor'))
    return plan,spec,inputs


async def collect(plan,spec,inputs,out):
    return await run(plan,spec,inputs,out,expected_plan_sha256=plan_digest(plan),expected_collection_sha256=plan_digest(spec))


@unittest.skipUnless(os.name=='posix','owned POSIX process groups required')
class ThreeArmReplay(unittest.IsolatedAsyncioTestCase):
    async def test_actual_commands_all_six_orders_original_requests_and_chain(self):
        with tempfile.TemporaryDirectory(dir='.') as directory:
            root=Path(directory).resolve();plan,spec,inputs=setup(root)
            result=await collect(plan,spec,inputs,root/'run')
            rows=[json.loads(line) for line in (root/'run/source-traces.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),18)
            self.assertEqual([''.join(r['arm'] for r in rows[i:i+3]) for i in range(0,18,3)],list(PERMUTATIONS))
            self.assertEqual(result['arm_position_counts'],{arm:[2,2,2] for arm in 'ABC'})
            self.assertEqual(result['actual_acceptance'],'incomplete')
            previous='0'*64
            for i,row in enumerate(rows):
                digest=row.pop('trace_sha256');self.assertEqual(digest,plan_digest(row))
                self.assertEqual(row['previous_trace_sha256'],previous);previous=digest
                self.assertEqual(row['ordinal'],i);self.assertEqual(row['status'],'success')
                request=row['response']['measurements']['request']
                self.assertEqual(row['request_sha256'],plan_digest(request))
                self.assertEqual(request['arm'],row['arm'])
                self.assertEqual(request['comparison_plan_sha256'],plan_digest(plan))
                self.assertEqual(request['collection_sha256'],plan_digest(spec))
                self.assertEqual(request['configuration'],plan['systems'][row['arm']])
                self.assertEqual(request['configuration_sha256'],plan_digest(request['configuration']))
                self.assertEqual(request['input'],inputs[int(row['task_id'])]['input'])
                self.assertNotIn('critical_facts',request);self.assertNotIn('quality_score',row)
            self.assertEqual(result['last_trace_sha256'],previous)
            self.assertEqual(json.loads((root/'run/comparison-plan.json').read_text()),plan)
            for path in (root/'run').iterdir():self.assertEqual(path.stat().st_mode&0o777,0o600)
            with self.assertRaises(FileExistsError):await collect(plan,spec,inputs,root/'run')

    async def test_preflight_rejects_mutation_before_output_or_launch(self):
        with tempfile.TemporaryDirectory(dir='.') as directory:
            root=Path(directory).resolve();plan,spec,inputs=setup(root,('ok',))
            for change in ('plan_hash','spec_hash','input','extra','arm','command','artifact'):
                p,s,items=copy.deepcopy((plan,spec,inputs));ph,sh=plan_digest(p),plan_digest(s)
                if change=='plan_hash':ph='f'*64
                if change=='spec_hash':sh='f'*64
                if change=='input':items[0]['input']['prompt']='changed'
                if change=='extra':s['extra']=True;sh=plan_digest(s)
                if change=='arm':s['runner']['commands']['D']=s['runner']['commands'].pop('C');sh=plan_digest(s)
                if change=='command':s['runner']['commands']['A']['argv']=['python'];sh=plan_digest(s)
                if change=='artifact':s['runner']['commands']['A']['artifacts'][str(root/'pipeline.py')]='f'*64;sh=plan_digest(s)
                with patch('asyncio.create_subprocess_exec') as spawn:
                    with self.subTest(change=change),self.assertRaises(ValueError):
                        await run(p,s,items,root/'run',expected_plan_sha256=ph,expected_collection_sha256=sh)
                    spawn.assert_not_called()
                self.assertFalse((root/'run').exists())

    async def test_failure_timeout_and_malformed_reports_retained(self):
        with tempfile.TemporaryDirectory(dir='.') as directory:
            root=Path(directory).resolve();plan,spec,inputs=setup(root,('error','malformed','overflow','timeout'))
            spec['runner']['timeout_ms']=200
            result=await collect(plan,spec,inputs,root/'run')
            rows=[json.loads(line) for line in (root/'run/source-traces.jsonl').read_text().splitlines()]
            self.assertEqual([r['status'] for r in rows],['error']*9+['timeout']*3)
            self.assertEqual(result['records'],12);self.assertEqual(result['remaining_tasks'],4)
            self.assertEqual(result['state'],'collected_unscored')
            self.assertTrue(all(r['quality_status']=='unscored' for r in rows))
            self.assertTrue(all('cost' not in r for r in rows))

    async def test_snapshot_and_cancel_leave_prefix_no_next_arm(self):
        with tempfile.TemporaryDirectory(dir='.') as directory:
            root=Path(directory).resolve();plan,spec,inputs=setup(root,('ok','ok'))
            original=copy.deepcopy((plan,spec,inputs));real=asyncio.create_subprocess_exec;children=[]
            async def spawn(*args,**kwargs):
                child=await real(*args,**kwargs);children.append(child)
                if len(children)==1:
                    plan['systems']['A']['model']='mutated';inputs[0]['input']['prompt']='mutated'
                    spec['runner']['commands']['B']['argv']=['invalid']
                if len(children)==3:asyncio.get_running_loop().call_soon(task.cancel)
                return child
            with patch('asyncio.create_subprocess_exec',spawn):
                task=asyncio.create_task(collect(plan,spec,inputs,root/'run'))
                with self.assertRaises(asyncio.CancelledError):await task
            self.assertEqual(len(children),3)
            self.assertTrue(all(child.returncode is not None for child in children))
            self.assertFalse((root/'run/complete.json').exists())
            rows=[json.loads(line) for line in (root/'run/source-traces.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),2)
            self.assertEqual(json.loads((root/'run/comparison-plan.json').read_text()),original[0])
            self.assertEqual(rows[0]['response']['measurements']['request']['input'],original[2][0]['input'])
            self.assertEqual(rows[1]['arm'],'B')

    async def test_cli_and_partial_cycle_position_counts(self):
        with tempfile.TemporaryDirectory(dir='.') as directory:
            root=Path(directory).resolve();plan,spec,inputs=setup(root,('ok',))
            for name,value in (('plan',plan),('spec',spec),('inputs',inputs)):(root/(name+'.json')).write_text(json.dumps(value))
            command=[sys.executable,'-m','benchmarks.three_arm_replay','--plan',str(root/'plan.json'),
                '--collection-spec',str(root/'spec.json'),'--inputs',str(root/'inputs.json'),
                '--plan-sha256',plan_digest(plan),'--collection-sha256',plan_digest(spec),'--output',str(root/'run')]
            result=await asyncio.to_thread(subprocess.run,command,capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stderr)
            summary=json.loads(result.stdout)
            self.assertEqual(summary['arm_position_counts'],{'A':[1,0,0],'B':[0,1,0],'C':[0,0,1]})
            self.assertEqual(summary['remaining_tasks'],1)
            self.assertEqual(summary['complete_six_task_blocks'],0)

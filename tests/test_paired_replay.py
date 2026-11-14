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

from benchmarks.paired_replay import run, validate_collection, _execute, _stop_owned
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

    async def test_spawn_completion_cancel_keeps_prefix_and_never_starts_next_arm(self):
        for delay_cancel in (False,True):
            with tempfile.TemporaryDirectory(dir='.') as directory:
                root=Path(directory).resolve(); plan,inputs=setup(root)
                plan['runner']['timeout_ms']=300
                processes=[]; original=asyncio.create_subprocess_exec
                loop=asyncio.get_running_loop()
                async def launch(*args,**kwargs):
                    if len(processes)==1:
                        args=(sys.executable,'-c','import time; time.sleep(60)')
                    child=await original(*args,**kwargs);processes.append(child)
                    if len(processes)==2:
                        if delay_cancel:loop.call_soon(loop.call_soon,task.cancel)
                        else:loop.call_soon(task.cancel)
                    return child
                with patch('asyncio.create_subprocess_exec',launch):
                    task=asyncio.create_task(run(plan,inputs,root/'run',expected_plan_sha256=plan_digest(plan)))
                    with self.assertRaises(asyncio.CancelledError):await task
                self.assertEqual(len(processes),2)
                self.assertTrue(all(p.returncode is not None and p._transport.is_closing() for p in processes))
                self.assertFalse((root/'run/complete.json').exists())
                prefix=(root/'run/source-traces.jsonl').read_text().splitlines()
                self.assertEqual(len(prefix),1)
                row=json.loads(prefix[0]);digest=row.pop('trace_sha256')
                self.assertEqual(digest,plan_digest(row));self.assertEqual(row['status'],'success')

    async def test_collect_completion_cancel_preserves_cancellation_and_owned_jobs(self):
        # Complete actual pipes/process tasks, then cancel the caller before the
        # wait boundary resumes. Also cover malformed and nonzero child results.
        for mode in ('ok','nan','nonzero'):
            with tempfile.TemporaryDirectory(dir='.') as directory:
                root=Path(directory).resolve();plan,inputs=setup(root,('ok',mode))
                original_gather=asyncio.gather;original_spawn=asyncio.create_subprocess_exec
                loop=asyncio.get_running_loop();groups=[];processes=[]
                def gather(*args,**kwargs):
                    future=original_gather(*args,**kwargs)
                    if len(args)==4 and not kwargs:
                        groups.append((args,future))
                        # Cancel the third invocation, after a full first pair.
                        if len(groups)==3:
                            future.add_done_callback(lambda f:loop.call_soon(task.cancel))
                    return future
                async def spawn(*args,**kwargs):
                    child=await original_spawn(*args,**kwargs);processes.append(child);return child
                with patch('asyncio.gather',gather),patch('asyncio.create_subprocess_exec',spawn):
                    task=asyncio.create_task(run(plan,inputs,root/'run',expected_plan_sha256=plan_digest(plan)))
                    with self.assertRaises(asyncio.CancelledError):await task
                self.assertEqual(len(processes),3)
                self.assertEqual(len(groups),3)
                self.assertTrue(all(p.returncode is not None and p._transport.is_closing() for p in processes))
                self.assertTrue(all(f.done() and all(t.done() for t in tasks) for tasks,f in groups))
                self.assertFalse((root/'run/complete.json').exists())
                prefix=(root/'run/source-traces.jsonl').read_text().splitlines()
                self.assertEqual(len(prefix),2)
                previous='0'*64
                for line in prefix:
                    row=json.loads(line);digest=row.pop('trace_sha256')
                    self.assertEqual(digest,plan_digest(row));self.assertEqual(row['previous_trace_sha256'],previous)
                    previous=digest

    async def test_exhausted_shared_deadline_does_not_start_collection(self):
        with tempfile.TemporaryDirectory(dir='.') as directory:
            root=Path(directory).resolve();plan,inputs=setup(root,('timeout',))
            original_gather=asyncio.gather;original_spawn=asyncio.create_subprocess_exec
            collections=[];processes=[];ticks=iter((0,2_000_000_000))
            def gather(*args,**kwargs):
                if len(args)==4 and not kwargs:collections.append(args)
                return original_gather(*args,**kwargs)
            async def spawn(*args,**kwargs):
                child=await original_spawn(*args,**kwargs);processes.append(child);return child
            with patch('asyncio.gather',gather),patch('asyncio.create_subprocess_exec',spawn), \
                    patch('benchmarks.paired_replay.time.perf_counter_ns',side_effect=lambda:next(ticks,3_000_000_000)):
                result=await _execute(plan['runner']['commands']['baseline'],{'input':inputs[0]['input']},1000)
            self.assertEqual(result['status'],'timeout')
            self.assertEqual(result['diagnostic'],'deadline_exceeded')
            self.assertEqual(collections,[])
            self.assertEqual(len(processes),1)
            self.assertIsNotNone(processes[0].returncode)
            self.assertTrue(processes[0]._transport.is_closing())

    async def test_cleanup_failure_still_joins_all_collection_tasks(self):
        with tempfile.TemporaryDirectory(dir='.') as directory:
            root=Path(directory).resolve();plan,inputs=setup(root,('timeout',))
            groups=[];processes=[];waits=[];collection_entered=asyncio.Event()
            original_gather=asyncio.gather;original_spawn=asyncio.create_subprocess_exec
            original_wait=asyncio.wait
            def gather(*args,**kwargs):
                future=original_gather(*args,**kwargs)
                if len(args)==4 and not kwargs:
                    groups.append((args,future))
                    collection_entered.set()
                return future
            async def spawn(*args,**kwargs):
                child=await original_spawn(*args,**kwargs);processes.append(child);return child
            async def wait(tasks,*,timeout):
                waits.append(timeout)
                if len(waits)==1:
                    # Await real spawn independent of host load. The enclosing
                    # watchdog bounds the fixture, not the production deadline.
                    return await original_wait(tasks)
                self.assertEqual(len(waits),2)
                await collection_entered.wait()
                # Scheduler probe: inject timeout only after all four real
                # collection jobs and their gather exist. This is not a timing
                # benchmark; the exhausted shared deadline has separate coverage.
                return set(),set(tasks)
            try:
                with patch('asyncio.gather',gather),patch('asyncio.create_subprocess_exec',spawn), \
                        patch('asyncio.wait',wait), \
                        patch('benchmarks.paired_replay.time.perf_counter_ns',return_value=0), \
                        patch('benchmarks.paired_replay._stop_owned',side_effect=PermissionError('fixture')):
                    with self.assertRaisesRegex(PermissionError,'fixture'):
                        await asyncio.wait_for(_execute(plan['runner']['commands']['baseline'],
                            {'input':inputs[0]['input']},100),5)
                self.assertEqual(waits,[.1,.1])
                self.assertTrue(collection_entered.is_set())
                self.assertEqual(len(groups),1)
                self.assertEqual(len(groups[0][0]),4)
                self.assertTrue(all(f.done() and all(t.done() for t in tasks) for tasks,f in groups))
                self.assertEqual(len(processes),1)
            finally:
                for child in processes:
                    await _stop_owned(child)
                    self.assertIsNotNone(child.returncode)
                    self.assertTrue(child._transport.is_closing())

    async def test_group_permission_error_is_retried_but_not_silently_ignored(self):
        proc=await asyncio.create_subprocess_exec(sys.executable,'-c','import time; time.sleep(60)',
            stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,start_new_session=True)
        original=os.killpg; count=0
        def transient(pid,sig):
            nonlocal count
            count+=1
            if count==1: raise PermissionError('fixture transient')
            return original(pid,sig)
        try:
            with patch('benchmarks.paired_replay.os.killpg',side_effect=transient): await _stop_owned(proc)
            self.assertEqual(count,2); self.assertIsNotNone(proc.returncode)
            with patch('benchmarks.paired_replay.os.killpg',side_effect=PermissionError('fixture persistent')):
                with self.assertRaises(PermissionError): await _stop_owned(proc)
        finally:
            if proc.returncode is None: await _stop_owned(proc)

    async def _assert_child_stopped(self,pid):
        import subprocess
        for _ in range(100):
            state=subprocess.run(['ps','-p',str(pid),'-o','stat='],capture_output=True,text=True).stdout.strip()
            if not state or state.startswith('Z'): return
            await asyncio.sleep(.01)
        self.fail('owned descendant is still executing')

    def _descendant_command(self,root,status):
        script=root/'descendant.py'; marker=root/'pid.json'
        script.write_text("import subprocess,sys,os,json\nfrom pathlib import Path\n"
            "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],"
            "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
            "Path(sys.argv[1]).write_text(json.dumps([os.getpid(),child.pid]))\n"
            "print(json.dumps(dict(schema_version=1,status=sys.argv[2],answer='done',measurements={})),flush=True)\n")
        return dict(argv=[sys.executable,str(script),str(marker),status],cwd=str(root),environment={}),marker

    async def _wait_marker(self,marker):
        for _ in range(200):
            if marker.exists():
                try: return json.loads(marker.read_text())
                except json.JSONDecodeError: pass
            await asyncio.sleep(.01)
        self.fail('fixture did not start its owned descendant')

    async def test_success_and_error_response_exit_clean_detached_stdio_descendants(self):
        import signal
        for status in ('success','error'):
            with tempfile.TemporaryDirectory(dir='.') as directory:
                root=Path(directory).resolve(); command,marker=self._descendant_command(root,status)
                try:
                    result=await _execute(command,{},1000)
                    group,child=await self._wait_marker(marker)
                    self.assertEqual(result['status'],status)
                    await self._assert_child_stopped(child)
                finally:
                    if marker.exists():
                        try: os.killpg(json.loads(marker.read_text())[0],signal.SIGKILL)
                        except ProcessLookupError: pass

    async def test_cancel_and_timeout_during_native_pipe_connection_retain_group_handle(self):
        import signal
        for mode in ('cancel','timeout'):
            with tempfile.TemporaryDirectory(dir='.') as directory:
                root=Path(directory).resolve(); command,marker=self._descendant_command(root,'success')
                loop=asyncio.get_running_loop(); original=loop.connect_write_pipe
                entered,release=asyncio.Event(),asyncio.Event()
                async def delayed(*args,**kwargs):
                    result=await original(*args,**kwargs)
                    entered.set(); await release.wait(); return result
                task=None
                try:
                    with patch.object(loop,'connect_write_pipe',side_effect=delayed):
                        task=asyncio.create_task(_execute(command,{},100 if mode=='timeout' else 1000))
                        await asyncio.wait_for(entered.wait(),2)
                        group,child=await self._wait_marker(marker)
                        if mode=='cancel':
                            task.cancel(); await asyncio.sleep(.01); task.cancel()
                        else: await asyncio.sleep(.15)
                        await asyncio.sleep(.01)
                        self.assertFalse(task.done())  # must own unfinished creation until release
                        release.set()
                        if mode=='cancel':
                            with self.assertRaises(asyncio.CancelledError): await task
                        else: self.assertEqual((await task)['status'],'timeout')
                    await self._assert_child_stopped(child)
                finally:
                    release.set()
                    if task is not None:
                        if not task.done(): task.cancel()
                        await asyncio.gather(task,return_exceptions=True)
                    if marker.exists():
                        try: os.killpg(json.loads(marker.read_text())[0],signal.SIGKILL)
                        except ProcessLookupError: pass


class Validation(unittest.TestCase):
    def test_environment_and_executable_must_be_explicit(self):
        with tempfile.TemporaryDirectory(dir='.') as directory:
            root=Path(directory).resolve(); plan,inputs=setup(root)
            for changes in ({'argv':['python3']},{'environment':{'BAD=KEY':'x'}},
                            {'artifacts':{}},{'cwd':'.'}):
                p=copy.deepcopy(plan); p['runner']['commands']['baseline'].update(changes)
                with self.assertRaises(ValueError): validate_collection(p,inputs,plan_digest(p))

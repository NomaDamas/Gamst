"""Replay CLI contracts with the real worker and an explicitly fake SLAM System."""
import asyncio
import hashlib
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch
import test_orbslam_context as fixture
from benchmarks.orbslam_replay import replay,load_manifest


@unittest.skipIf(fixture.np is None or shutil.which('c++') is None,'requires numpy/OpenCV/compiler')
class Replay(unittest.IsolatedAsyncioTestCase):
    def assets(self,slow=False,**changes):
        config=dict(fixture.assets(self,slow=slow),**changes)
        root=Path(config['settings_path']).parent
        image=bytes([2])*(96*96*3);(root/'frame.bgr').write_bytes(image)
        m=dict(schema_version=1,config=config,sensor=dict(source_id='cam',sensor_id='video',coordinate_frame='camera',stream_epoch='recording1'),
            shape=[96,96,3],captures=[dict(path='frame.bgr',sha256=hashlib.sha256(image).hexdigest(),capture_ms=1000+i*100,
                capture_sequence=100+i,reset_before=False) for i in range(3)])
        path=root/'manifest.json';path.write_text(json.dumps(m));return root,path,m

    async def test_cli_rotation_and_truthful_artifact(self):
        root,path,m=self.assets(max_frames=1);out=root/'report.jsonl'
        child=await asyncio.create_subprocess_exec(sys.executable,'-m','benchmarks.orbslam_replay','--manifest',str(path),'--output',str(out),
            '--rotate',stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
        stdout,stderr=await asyncio.wait_for(child.communicate(),30)
        self.assertEqual(child.returncode,0,stderr.decode());self.assertEqual(json.loads(stdout)['completed_observations'],3)
        rows=[json.loads(line) for line in out.read_text().splitlines()]
        self.assertEqual(rows[0]['status'],'incomplete');self.assertEqual(rows[-1]['status'],'complete')
        self.assertEqual([r['value']['provenance']['capture_sequence'] for r in rows[1:-1]],[100,101,102])
        self.assertEqual([r['value']['provenance']['rotation']['session_number'] for r in rows[1:-1]],[1,2,3])
        self.assertIsNone(rows[-1]['capture_to_context_latency_ms']);self.assertFalse(rows[-1]['quality_validated'])
        before=out.read_bytes()
        with self.assertRaises(FileExistsError):await replay(path,out,rotate=True)
        self.assertEqual(out.read_bytes(),before)

    async def test_partial_hash_failure_reaps_and_records_failed_run(self):
        root,path,m=self.assets();m['captures'][1]['sha256']='0'*64;path.write_text(json.dumps(m))
        children=[];spawn=asyncio.create_subprocess_exec
        async def owned(*args,**kwargs):
            p=await spawn(*args,**kwargs);children.append(p);return p
        with patch('asyncio.create_subprocess_exec',owned):
            with self.assertRaisesRegex(ValueError,'size/hash'):await replay(path,root/'failed.jsonl')
        self.assertTrue(all(p.returncode is not None for p in children))
        rows=[json.loads(line) for line in (root/'failed.jsonl').read_text().splitlines()]
        self.assertEqual(rows[-1]['status'],'failed');self.assertEqual(rows[-1]['completed_observations'],1)

    async def test_manifest_order_escape_bounds_budget_and_reset(self):
        root,path,m=self.assets(max_frames=2)
        with self.assertRaisesRegex(ValueError,'budget'):await replay(path,root/'budget.jsonl')
        self.assertFalse((root/'budget.jsonl').exists())
        for field,value in (('path','../escape'),('capture_ms',True),('reset_before',1)):
            changed=json.loads(json.dumps(m));changed['captures'][1][field]=value;path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):load_manifest(path)
        m['config']['max_frames']=3;m['captures'][1]['capture_ms']=100;m['captures'][1]['capture_sequence']=0
        path.write_text(json.dumps(m))
        with self.assertRaises(ValueError):load_manifest(path)
        m['captures'][1]['reset_before']=True;path.write_text(json.dumps(m))
        await replay(path,root/'reset.jsonl')
        rows=[json.loads(line) for line in (root/'reset.jsonl').read_text().splitlines()]
        self.assertEqual(rows[2]['value']['provenance']['reset_count'],1)
        self.assertEqual(rows[2]['value']['provenance']['capture_ms'],100)

    async def test_cancelled_replay_reaps_and_cannot_mark_complete(self):
        root,path,m=self.assets(slow=True)
        payload=bytes([253])*(96*96*3);(root/'frame.bgr').write_bytes(payload)
        for row in m['captures']:row['sha256']=hashlib.sha256(payload).hexdigest()
        path.write_text(json.dumps(m));children=[];spawn=asyncio.create_subprocess_exec;ready=asyncio.Event()
        async def owned(*args,**kwargs):
            child=await spawn(*args,**kwargs);children.append(child);ready.set();return child
        with patch('asyncio.create_subprocess_exec',owned):
            task=asyncio.create_task(replay(path,root/'cancelled.jsonl'))
            await asyncio.wait_for(ready.wait(),5);await asyncio.sleep(.2);task.cancel()
            with self.assertRaises(asyncio.CancelledError):await task
        self.assertTrue(all(c.returncode is not None for c in children))
        rows=[json.loads(line) for line in (root/'cancelled.jsonl').read_text().splitlines()]
        self.assertEqual(rows[-1]['status'],'failed');self.assertEqual(rows[-1]['error_type'],'CancelledError')
        self.assertEqual(rows[-1]['completed_observations'],0)

    async def test_cancellation_first_arriving_during_cleanup_remains_failed(self):
        from humansense import ProcessProvider
        root,path,m=self.assets()
        original=ProcessProvider.close
        for count in (1,4):
            entered=asyncio.Event();release=asyncio.Event();children=[]
            async def delayed_close(provider):
                children.append(provider._process);entered.set()
                await release.wait();await original(provider)
            out=root/('cleanup-cancel-'+str(count)+'.jsonl')
            with patch.object(ProcessProvider,'close',delayed_close):
                task=asyncio.create_task(replay(path,out))
                await asyncio.wait_for(entered.wait(),5)
                for _ in range(count):task.cancel();await asyncio.sleep(0)
                release.set()
                with self.assertRaises(asyncio.CancelledError):await task
            self.assertTrue(all(child.returncode is not None for child in children))
            rows=[json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual(rows[-1]['completed_observations'],3)
            self.assertEqual(rows[-1]['status'],'failed');self.assertEqual(rows[-1]['error_type'],'CancelledError')

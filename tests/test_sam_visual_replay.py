import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch,AsyncMock

import test_sam_visual_window as fixtures
import test_sam_audio_context as audio
from humansense.sam_visual_packet import unpack_visual_packet,visual_policy
from humansense.sam_visual_replay import recording,replay


@unittest.skipIf(fixtures.fixtures.np is None,'optional numpy missing')
class Replay(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir='.')
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).absolute()
        f=fixtures.Window();f.setUp();self.packet=f.packet
        (self.root/'one.gsamv').write_bytes(self.packet)
        self.doc=dict(schema_version=1,config=dict(audio.CONFIG,prompt='',
            visual_policy=visual_policy(unpack_visual_packet(self.packet)[0])),
            packets=[dict(file='one.gsamv',sha256=hashlib.sha256(self.packet).hexdigest())])
        self.manifest=self.root/'manifest.json'
        self.write()

    def write(self):
        raw=json.dumps(self.doc).encode();self.manifest.write_bytes(raw)
        self.digest=hashlib.sha256(raw).hexdigest()

    async def test_actual_owned_replay_preserves_historical_capture_and_discards_media(self):
        code='''
from humansense import provider_worker as w
from humansense.providers.sam_audio_native import SAMAudioNative
from test_sam_visual_context import Backend
SAMAudioNative.from_local=classmethod(lambda cls,config:Backend())
w.main()
'''
        real=asyncio.create_subprocess_exec;children=[]
        async def spawn(*args,**kwargs):
            env=dict(kwargs.get('env',{}));env['PYTHONPATH']=os.path.abspath('tests')+os.pathsep+os.path.abspath('.')
            kwargs['env']=env
            child=await real(sys.executable,'-u','-c',code,**kwargs);children.append(child);return child
        with patch('asyncio.create_subprocess_exec',spawn):
            rows=[row async for row in replay(str(self.manifest),self.digest,consent=True)]
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['capture_timestamp_ms'],2000)
        self.assertEqual(rows[0]['value']['prompt']['visual']['feature_frames'],25)
        self.assertEqual(rows[0]['value']['media'],{})
        self.assertFalse(rows[0]['media_retained'])
        self.assertFalse(rows[0]['trained_quality_verified'])
        self.assertTrue(all(c.returncode is not None for c in children))

    async def test_invalid_manifest_or_payload_never_launches_worker(self):
        with patch('asyncio.create_subprocess_exec',new_callable=AsyncMock) as spawn:
            with self.assertRaises(PermissionError):
                _=[v async for v in replay(str(self.manifest),self.digest)]
            (self.root/'one.gsamv').write_bytes(self.packet[:-1])
            with self.assertRaises(ValueError):
                _=[v async for v in replay(str(self.manifest),self.digest,consent=True)]
            spawn.assert_not_called()

    async def test_names_duplicate_entries_and_declared_policy_are_strict(self):
        self.doc['packets']*=2;self.write()
        with self.assertRaises(ValueError):recording(str(self.manifest),self.digest)
        self.doc['packets']=self.doc['packets'][:1]
        self.doc['packets'][0]['file']='../one.gsamv';self.write()
        with self.assertRaises(ValueError):recording(str(self.manifest),self.digest)

    async def test_cli_emits_only_metadata_jsonl_through_owned_worker(self):
        # Run the real CLI in a parent subprocess; only the trained backend
        # constructor inside its owned worker is replaced by the explicit fixture.
        code = r"""
import asyncio, os, sys
from humansense.sam_visual_replay import main
real = asyncio.create_subprocess_exec
worker_code = "from humansense import provider_worker as w; from humansense.providers.sam_audio_native import SAMAudioNative; from test_sam_visual_context import Backend; SAMAudioNative.from_local=classmethod(lambda cls,config:Backend()); w.main()"
async def spawn(*args, **kwargs):
    env = dict(kwargs.get('env', {}))
    env['PYTHONPATH'] = os.path.abspath('tests') + os.pathsep + os.path.abspath('.')
    kwargs['env'] = env
    return await real(sys.executable, '-u', '-c', worker_code, **kwargs)
asyncio.create_subprocess_exec = spawn
main()
"""
        child=await asyncio.create_subprocess_exec(sys.executable,'-c',code,
            '--manifest',str(self.manifest),'--sha256',self.digest,'--consent',
            stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
        try:
            out,err=await asyncio.wait_for(child.communicate(),15)
        finally:
            if child.returncode is None:
                child.kill();await child.wait()
        self.assertEqual(child.returncode,0,err.decode())
        lines=out.decode().splitlines();self.assertEqual(len(lines),1)
        row=json.loads(lines[0])
        self.assertEqual(row['capture_timestamp_ms'],2000)
        self.assertEqual(row['value']['media'],{})
        self.assertFalse(row['trained_quality_verified'])
        self.assertNotIn('masked_videos',lines[0])

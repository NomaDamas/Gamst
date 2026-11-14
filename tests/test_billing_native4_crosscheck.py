"""Real native4 SDK/HTTP originals, never fabricated native index rows."""
import asyncio
import hashlib
import json
from pathlib import Path
import time
import subprocess
import sys
import socket
from copy import deepcopy
import unittest
from unittest.mock import patch

import test_openai_responses_native4 as native4_tests
from benchmarks.billing_native4_crosscheck import crosscheck_recording_transport
from benchmarks.billing_native4_bundle import persist_recording_crosscheck
from benchmarks.billing_aggregate import encoded
from humansense.responses_evidence import ResponsesEvidenceSession
from humansense.openai_responses import TRANSPORT
from humansense.openai_responses_codec import PROJECTION


class RecordingCrosscheck(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = native4_tests.Native4Transport.asyncSetUp
    asyncTearDown = native4_tests.Native4Transport.asyncTearDown
    sdk = native4_tests.Native4Transport.sdk
    originals = native4_tests.Native4Transport.originals

    async def collect(self, visual=False, denied=False, cancelled=False, connect_failure=False, incomplete=False, cancel_before_entry=False):
        queued = await self.sdk(visual=visual)
        if incomplete:
            original_close=self.observer.close
            async def incomplete_close(*args,**kwargs):
                return await original_close(cleanup_finished=False)
            self.observer.close=incomplete_close
        unavailable=None
        if connect_failure:
            unavailable=socket.socket(); unavailable.bind(('127.0.0.1',0))
            self.client._port=unavailable.getsockname()[1]; self.client._timeout=.05
        if cancelled: self.hold = asyncio.Event()
        task = asyncio.create_task(self.adapter.execute(*queued, 'original question',
            maximum_charge=None if denied else 5, evaluation_scope=self.scope))
        if cancelled:
            await asyncio.wait_for(self.entered.wait(), 3); task.cancel()
            with self.assertRaises(asyncio.CancelledError): await task
        elif cancel_before_entry:
            with self.assertRaises(asyncio.CancelledError): await task
        elif connect_failure:
            try:
                with self.assertRaises((ValueError,TimeoutError)): await task
            finally: unavailable.close()
        else: await task
        self.index, _ = await self.originals()
        self.root = Path(self.tmp.name).resolve(); self.transport = self.sink.directory
        self.declarations = self.root/'original-declarations.json'
        self.declarations.write_bytes(encoded(self.observer.declarations)+b'\n')
        self.declarations_sha = hashlib.sha256(self.declarations.read_bytes()).hexdigest()
        self.terminal_sha = hashlib.sha256((self.writer.output/'terminal.json').read_bytes()).hexdigest()
        self.selection = dict(owner_kind='recording_native4', owner=self.scope.binding_owner(),
            provider='openai', model_revision='fixture-model', role='small_vlm' if visual else 'text_llm',
            projection_version=PROJECTION, transport_version=TRANSPORT)
        now = int(time.time())
        page = dict(object='page', data=[dict(object='bucket', start_time=now-60, end_time=now,
            results=[dict(object='organization.costs.result', project_id='fixture-project',
                amount=dict(value=.1,currency='usd'))])], has_more=False,next_page=None)
        raw = json.dumps(page).encode(); (self.root/'costs.json').write_bytes(raw)
        self.manifest = dict(schema_version=1, kind='openai_aggregate_acquisition', account_scope='fixture-account',
            origin='synthetic_fixture', acquired_at=now, queries=[dict(id='costs', endpoint='costs',
                start_time=now-60,end_time=now,bucket_width='1d',group_by=['project_id'],filters={},
                pages=[dict(request_cursor=None,path='costs.json',bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())])])

    def check(self):
        return crosscheck_recording_transport(self.manifest, self.root, self.transport,
            expected_inventory_sha256=hashlib.sha256((self.transport/'inventory.json').read_bytes()).hexdigest(),
            native_index=self.index,expected_binding=self.selection,declarations_path=self.declarations,
            declarations_sha256=self.declarations_sha,terminal_sha256=self.terminal_sha)

    async def test_original_text_entry_and_http_match_without_financial_claim(self):
        await self.collect(); report=self.check()
        self.assertTrue(report['local_binding_complete'])
        self.assertEqual(report['local_counts']['native_entries_matched'],1)
        self.assertEqual(report['native_sources']['terminal_sha256'],self.terminal_sha)
        self.assertIsNone(report['entries'][0]['per_attempt_cost'])
        self.assertFalse(report['request_costs_verified'])
        self.assertEqual(report['native_lane']['unknown_liability_count'],1)

    async def test_original_visual_png_entry_matches(self):
        try: import cv2; import numpy
        except ImportError: self.skipTest('optional vision unavailable')
        await self.collect(visual=True); report=self.check()
        self.assertTrue(report['local_binding_complete'])
        self.assertTrue(json.loads(self.calls[0][1])['input'][0]['content'][1]['image_url'].startswith('data:image/png;base64,'))

    async def test_denied_and_cancelled_are_separate_from_actual_send(self):
        await self.collect(denied=True); report=self.check()
        self.assertTrue(report['local_binding_complete'])
        self.assertEqual(report['local_counts']['native_entries_matched'],0)
        self.assertEqual(report['local_counts']['not_entered'],1)

    async def test_cancel_after_http_preserves_original_entry_and_unknown_cost(self):
        await self.collect(cancelled=True); report=self.check()
        self.assertTrue(report['local_binding_complete'])
        self.assertEqual(report['local_counts']['cancelled'],1)
        self.assertEqual(report['local_counts']['possible_sends'],1)
        self.assertEqual(report['local_counts']['responses_observed'],0)

    async def test_original_file_identity_rechecked_before_join(self):
        await self.collect()
        self.terminal_sha='f'*64
        with self.assertRaisesRegex(ValueError,'hash mismatch'): self.check()
        self.terminal_sha=hashlib.sha256((self.writer.output/'terminal.json').read_bytes()).hexdigest()
        self.declarations.write_bytes(self.declarations.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'hash mismatch'): self.check()

    async def test_whole_lane_omission_and_selection_mutation_cannot_hide_call(self):
        await self.collect()
        self.transport=self.root/'empty-transport'
        empty=ResponsesEvidenceSession(self.transport,owner_kind='recording_native4',
            expected_owner=self.selection['owner'],retain_response_bytes=True,retain_financial_hashes=True)
        await empty.close()
        original_sha=hashlib.sha256(encoded(self.selection)).hexdigest()
        from benchmarks.billing_aggregate import build_report
        def mutate(*args,**kwargs):
            self.selection['owner']['scope_index']=0
            self.selection['role']='strong_vlm'
            return build_report(*args,**kwargs)
        with patch('benchmarks.billing_native4_crosscheck.build_report',side_effect=mutate): report=self.check()
        self.assertFalse(report['local_binding_complete'])
        self.assertEqual(report['local_counts']['native_entries_without_transport'],1)
        self.assertEqual(report['transport_selection_sha256'],original_sha)

    def persist(self, output):
        return persist_recording_crosscheck(self.manifest,self.root,self.transport,
            expected_inventory_sha256=hashlib.sha256((self.transport/'inventory.json').read_bytes()).hexdigest(),
            native_index=self.index,expected_binding=self.selection,declarations_path=self.declarations,
            declarations_sha256=self.declarations_sha,terminal_sha256=self.terminal_sha,output_directory=output)

    async def test_persistence_recomputes_originals_and_rejects_joint_tampering(self):
        await self.collect(); output=self.root/'crosscheck'
        before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.transport.iterdir()}
        report=self.persist(output); self.assertEqual(self.persist(output),report)
        self.assertEqual(output.stat().st_mode&0o777,0o700)
        self.assertEqual((output/'crosscheck.json').stat().st_mode&0o777,0o600)
        report['request_costs_verified']=True; raw=encoded(report)
        (output/'crosscheck.json').write_bytes(raw)
        completion=json.loads((output/'completion.json').read_bytes())
        completion['files']['crosscheck.json']=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        (output/'completion.json').write_bytes(encoded(completion))
        with self.assertRaises(ValueError): self.persist(output)
        self.assertEqual(before,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.transport.iterdir()})

    async def test_offline_cli_reloads_original_declarations_terminal_and_transport(self):
        await self.collect(); output=self.root/'cli-crosscheck'
        acquisition=self.root/'acquisition.json'; acquisition.write_bytes(encoded(self.manifest))
        selection=self.root/'selection.json'; selection.write_bytes(encoded(self.selection))
        options={'manifest':acquisition,'assets':self.root,'transport':self.transport,
            'transport-inventory-sha256':hashlib.sha256((self.transport/'inventory.json').read_bytes()).hexdigest(),
            'native-root':self.writer.output,'declarations':self.declarations,'declarations-sha256':self.declarations_sha,
            'terminal-sha256':self.terminal_sha,'selection':selection,'output':output}
        command=[sys.executable,'-m','benchmarks.billing_native4_bundle']
        for key,value in options.items(): command.extend(['--'+key,str(value)])
        result=await asyncio.to_thread(subprocess.run,command,capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads((output/'crosscheck.json').read_bytes()),self.check())

    async def test_nested_original_output_rejected_without_changing_sources(self):
        await self.collect()
        for original in (self.index.directory,self.transport):
            before={p.relative_to(original).as_posix() for p in original.rglob('*')}
            for output in (original,original/'nested-crosscheck'):
                with self.assertRaisesRegex(ValueError,'outside original'): self.persist(output)
                self.assertEqual(before,{p.relative_to(original).as_posix() for p in original.rglob('*')})

    async def test_original_owner_scope_binding_and_context_must_match(self):
        await self.collect(); original=deepcopy(self.selection)
        for change in ('scope','binding','context','model','role'):
            self.selection=deepcopy(original)
            if change=='scope':self.selection['owner']['scope_index']=0
            elif change=='binding':self.selection['owner']['binding_index']=1
            elif change=='context':self.selection['owner']['context']['nonce']='f'*64
            elif change=='model':self.selection['model_revision']='other-model'
            else:self.selection['role']='strong_vlm'
            with self.assertRaises(ValueError):self.check()

    async def test_actual_incomplete_close_cannot_certify_complete_lane(self):
        await self.collect(incomplete=True); report=self.check()
        self.assertEqual(self.index.terminal['state'],'incomplete')
        self.assertFalse(report['local_binding_complete'])
        self.assertTrue(report['native_lane']['unobserved_entry_inventory_possible'])
        self.assertEqual(report['local_counts']['native_entries_matched'],1)

    async def test_actual_connection_failure_has_entry_without_send(self):
        await self.collect(connect_failure=True); report=self.check()
        self.assertEqual(report['local_counts']['native_entries_matched'],1)
        self.assertEqual(report['local_counts']['possible_sends'],0)
        self.assertEqual(report['local_counts']['responses_observed'],0)
        self.assertEqual(report['local_counts']['error']+report['local_counts']['timeout'],1)
        self.assertEqual(report['native_lane']['unknown_liability_count'],1)

    async def test_cancel_after_dispatch_before_entry_retains_dispatch_liability(self):
        async def cancel_before_entry(ticket,callback):
            asyncio.current_task().cancel()
            await asyncio.sleep(0)
        with patch('benchmarks.recording_evaluation._Call.invoke_bound',new=cancel_before_entry):
            await self.collect(cancel_before_entry=True)
        report=self.check()
        self.assertEqual(report['local_counts']['native_entries_matched'],0)
        self.assertEqual(report['local_counts']['possible_sends'],0)
        self.assertEqual(report['native_lane']['unknown_dispatch_liability_count'],1)
        self.assertEqual(report['native_lane']['counts']['released_dispatched'],1)
        self.assertEqual(report['native_lane']['unknown_liability_count'],0)
        self.assertFalse(report['request_costs_verified'])

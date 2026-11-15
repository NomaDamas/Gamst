"""Fixed built-in original-recording child. No caller command/module is accepted."""
import argparse
import asyncio
from pathlib import Path
import time

from gamst.streams import _finish
from .recorded_source import digest
from .recording_runtime_inputs import OwnedRuntimeInputs
from .recording_runtime_native import RuntimeChildNativeChannel
from .recording_control_channel import ChildCoordinatorChannel
from .recording_native_writer import RecordingNativeWriter,ArtifactBudget
from .recording_evaluation import RecordingEvaluationSession
from .recording_runtime_artifacts import RecordingRuntimeArtifacts
from .recording_arm_runtime import RecordingArmRuntime,numeric_inputs


async def run_arm(*,runtime_inputs,output,execution_sha256,invocation_ordinal,startup_deadline_ns):
    owned=native=coordinator=writer=owner=artifacts=runtime=None
    run_task=stop_task=None;terminal_sent=False
    try:
        coordinator=await ChildCoordinatorChannel.from_environment(startup_deadline_ns=startup_deadline_ns)
        bootstrap=coordinator.bootstrap
        if (bootstrap['execution_sha256'],bootstrap['invocation_ordinal'])!=(execution_sha256,invocation_ordinal):
            raise ValueError('fixed child arguments differ from original coordinator invocation')
        owned=await OwnedRuntimeInputs.open(runtime_inputs,expected_execution_sha256=execution_sha256,invocation_ordinal=invocation_ordinal)
        documents=owned.receipt.documents();declarations=owned.receipt.declarations();bundles=owned.receipt.source_bundles()
        execution=documents[f'invocations/{invocation_ordinal:02}/execution.json']
        native=await RuntimeChildNativeChannel.from_runtime_environment(latency_policy=documents['runtime/latency.json'],
            source_bundles=bundles,numeric_inputs=numeric_inputs(documents))
        owned.validate_native_paths(native.bootstrap['paths'])
        for key in ('nonce','invocation_ordinal'):
            if native.bootstrap[key]!=bootstrap[key]:raise ValueError('native/coordinator original invocation mismatch')
        if (digest(documents['runtime/policy.json']),digest(documents['runtime/latency.json']))!=(bootstrap['runtime_policy_sha256'],bootstrap['latency_policy_sha256']):
            raise ValueError('coordinator original policy mismatch')
        artifacts=await RecordingRuntimeArtifacts.open(output,execution,budget=ArtifactBudget(),retain_evaluation_artifacts=True)
        writer=await RecordingNativeWriter.open(Path(output)/'native',native.context)
        native.follow_writer(writer)
        owner=RecordingEvaluationSession.from_runtime_frozen(declarations,writer,latency_policy=documents['runtime/latency.json'],
            source_bundles=bundles,numeric_inputs=numeric_inputs(documents),run_nonce=bootstrap['nonce'],execution_ordinal=invocation_ordinal,
            retain_evaluation_hashes=True,retain_financial_documents=declarations['scope_protocol']['retain_financial_documents'])
        runtime=await RecordingArmRuntime.open(owned,owner=owner,artifacts=artifacts)
        await coordinator.send_wait('ready',dict(execution_sha256=execution_sha256,asset_inventory_sha256=execution['asset_inventory_sha256'],
            clock_domain=bootstrap['clock_domain'],clock_implementation=time.get_clock_info('monotonic').implementation))
        start=await coordinator.receive()
        if start is None or start.snapshot()['type']!='start':raise RuntimeError('parent did not authorize playback START')
        async def playback_started(mapping,source_mappings,clock):
            await coordinator.send_wait('playback_started',dict(execution_sha256=execution_sha256,mapping=mapping,mapping_sha256=clock.sha256,
                source_mappings=source_mappings,source_mappings_sha256=digest(runtime.source_mappings),anchor_monotonic_ns=clock.monotonic_ns,clock_domain=bootstrap['clock_domain']))
        async def answer_ready(task_id,result,result_sha256):
            await coordinator.send_wait('answer_ready',dict(task_id=task_id,result=result,result_sha256=result_sha256))
        run_task=asyncio.create_task(runtime.start(playback_started=playback_started,answer_ready=answer_ready))
        async def watch_stop():
            await coordinator.wait_stop()
            if not run_task.done():run_task.cancel()
        stop_task=asyncio.create_task(watch_stop())
        try:await run_task
        except (Exception,asyncio.CancelledError):
            # Preserve the original failed prefix; actual owned cleanup below
            # remains mandatory before a failed terminal can be emitted.
            if not runtime.failures:runtime._fail('cancelled' if run_task.cancelled() else 'backend_failed')
        finally:
            stop_task.cancel();await _finish(asyncio.gather(stop_task,return_exceptions=True))
            await runtime.close()
        await owner.close();await native.close()
        ref,summary,index=await runtime.finalize(writer.output)
        await owned.close()
        await coordinator.send_wait('runtime_terminal',dict(execution_sha256=execution_sha256,child_summary=ref,child_summary_sha256=ref['file_sha256']))
        terminal_sent=True
        await coordinator.close()
        return 0 if summary['state']=='closed' else 1
    finally:
        if run_task is not None and not run_task.done():run_task.cancel();await _finish(asyncio.gather(run_task,return_exceptions=True))
        if stop_task is not None and not stop_task.done():stop_task.cancel();await _finish(asyncio.gather(stop_task,return_exceptions=True))
        for item in (runtime,owner):
            if item is not None:
                try:await item.close()
                except Exception:pass
        if native is not None:
            try:await native.close() if owner is not None else await native.abort()
            except Exception:await native.abort()
        if not terminal_sent:
            if artifacts is not None:await artifacts.abort()
            if coordinator is not None:await coordinator.abort()
        if owned is not None:await owned.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-inputs',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--execution-sha',required=True);parser.add_argument('--invocation',type=int,required=True)
    parser.add_argument('--startup-deadline-ns',type=int,required=True)
    args=parser.parse_args()
    return asyncio.run(run_arm(runtime_inputs=args.runtime_inputs,output=args.output,execution_sha256=args.execution_sha,
        invocation_ordinal=args.invocation,startup_deadline_ns=args.startup_deadline_ns))


if __name__=='__main__':raise SystemExit(main())

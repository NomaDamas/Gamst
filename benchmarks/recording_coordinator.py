"""Fixed persistent recording parent: original inputs, two channels, owned reap.

This executes only the installed recording_arm module. Local synthetic tests
are not actual evaluation. Actual evaluation/data acquisition follows the
repository's GCP-only policy; no provider or model is installed here.
"""
import asyncio
import hashlib
import os
from pathlib import Path
import secrets
import re
import stat
import sys
import sysconfig
import time

from .recording_artifacts import _finish,_write_all,_encode
from .recording_native_schema import digest,encoded,decode,keys,integer,sha,run_context,snapshot
from .recording_native_writer import ArtifactBudget
from .recording_runtime_inputs import OwnedRuntimeInputs,_absolute_directory,_open_file,_identity
from .recording_runtime_native import preflight_runtime_collection,RuntimeParentNativeChannel
from .recording_control_channel import ParentCoordinatorChannel,CoordinatorError,ENV_FD
from .recording_native_channel import FD_ENVIRONMENT_KEY
from .recording_runtime_artifacts import (RecordingRuntimeArtifacts,preflight_runtime_artifacts,KINDS,
    read_ref,read_publication,iter_outcome_rows,validate_task_artifact,validate_metadata,_line)
from .paired_replay import _stop_owned,_drain


async def _owned_thread(function,*args,**kwargs):
    task=asyncio.create_task(asyncio.to_thread(function,*args,**kwargs))
    try:return await asyncio.shield(task)
    except asyncio.CancelledError:
        await _finish(task);raise


def _numeric(documents):
    if documents['runtime/policy.json']['schema_version']==1:return None
    return dict(opportunities=documents['runtime/opportunities.json'],exposure_policy=documents['runtime/exposure.json'],
        financial_adapter_inventory=documents['runtime/financial-adapters.json'])


def _declarations(documents,recording_index):
    result=dict(plan=documents['runtime/plan.json'],collection=documents['runtime/collection.json'],
        scope_protocol=documents['runtime/scopes.json'],instrumentation_protocol=documents['runtime/instrumentation.json'],
        runtime_policy=documents['runtime/policy.json'],source_manifest=documents[f'recordings/{recording_index:02}/manifest.json'],
        tasks=documents[f'recordings/{recording_index:02}/tasks.json'],profiles=documents[f'recordings/{recording_index:02}/profiles.json'])
    if 'runtime/accounting.json' in documents:result['stream_protocol']=documents['runtime/accounting.json']
    return result


def _native_paths(root,recording_index,documents):
    names=dict(plan='runtime/plan.json',collection='runtime/collection.json',scope_protocol='runtime/scopes.json',
        instrumentation_protocol='runtime/instrumentation.json',runtime_policy='runtime/policy.json',
        source_manifest=f'recordings/{recording_index:02}/manifest.json',tasks=f'recordings/{recording_index:02}/tasks.json',profiles=f'recordings/{recording_index:02}/profiles.json')
    if 'runtime/accounting.json' in documents:names['stream_protocol']='runtime/accounting.json'
    return {key:str(Path(root)/value) for key,value in names.items()}


def _ticks(periodic,manifest,fill=False):
    if periodic is None:return 0
    period=periodic['selection_period_ns' if fill else 'period_ns'];phase=periodic['phase_ns']
    if not fill and periodic['time_origin_policy']=='capture_epoch':phase=(phase-manifest['origin_capture_ns'])%period
    duration=manifest['duration_ns']
    return 0 if phase>=duration else (duration-1-phase)//period+1


def _runtime_bound(documents,invocation):
    """Derived joint bounds, not96 independent maximum-sized runtime copies."""
    c=documents['runtime/collection.json'];u=documents['runtime/policy.json'];n=invocation['recording_index']
    source=documents[f'recordings/{n:02}/manifest.json'];tasks=documents[f'recordings/{n:02}/tasks.json']
    arm=next(a for a in u['arms'] if a['arm']==invocation['arm'])
    policy=next(p for p in arm['recording_policies'] if p['recording_id']==source['recording_id'])
    records=len(source['records']);task_count=len(tasks)
    ticks=_ticks(policy['periodic'],source)+_ticks(policy['fill_chain'],source,True)
    # Per delivered sample: process, endpoint and bounded finish; fill has
    # one direct measurement plus initial/reinspection/optional strong.
    periodic=_ticks(policy['periodic'],source);fill=_ticks(policy['fill_chain'],source,True)
    entries=min(c['capacity']['max_entries_per_run'],3*records+periodic+4*fill+task_count)
    requests=min(entries,periodic+3*fill+task_count)
    publications=3*entries+records+task_count
    counts=dict.fromkeys(KINDS,0)
    counts.update(ticks=ticks,adapters=len(arm['lanes']),deliveries=12*records,transport=entries,
        publications=publications,channels=4)
    counts['delivery-summaries']=len(policy['deliveries']);counts['mapping-updates']=min(1024,entries)
    snapshot_bytes=min(4*1024**2,task_count*(32768+128*1024+128*1024))
    # Requests are optional, but a successful run cannot exceed this reserved
    # maximum by choosing to retain them later. Fill case metadata is included.
    retained=publications*128*1024+requests*256*1024+min(ticks,entries)*16*1024
    task_bytes=snapshot_bytes+task_count*3*65536
    result=preflight_runtime_artifacts(counts,retained_record_bytes=retained,
        task_artifact_bytes=min(task_bytes,4*1024**2))
    result['maximum_bytes']+=max(0,task_bytes-4*1024**2)+4*65536
    if result['maximum_bytes']>16*1024**2:raise ValueError('derived runtime metadata cannot fit16MiB')
    result['publication_rows']=publications
    return result


def _startup_budget_ns(documents,invocation):
    """One immutable deadline includes only selected frozen worker startup."""
    u=documents['runtime/policy.json']
    arm=next(a for a in u['arms'] if a['arm']==invocation['arm'])
    rid=documents['runtime/collection.json']['recordings'][invocation['recording_index']]['recording_id']
    policy=next(p for p in arm['recording_policies'] if p['recording_id']==rid)
    selected=[] if policy['periodic'] is None else [policy['periodic']['lane_id']]
    total=30_000_000_000
    for lane_id in selected:
        lane=next(l for l in arm['lanes'] if l['lane_id']==lane_id)
        adapter=next(a for a in u['adapters'] if a['adapter_id']==lane['adapter_id'])
        if adapter['implementation']=='smolvlm_process_v1':
            timeout=adapter['config']['startup_timeout_ms'];integer(timeout,1,120000)
            total+=timeout*1_000_000
    return total


def _admit_join_workspace(input_bytes,bounds):
    """Retained issued runs accumulate; sequential execution does not free them."""
    integer(input_bytes,0,256*1024**2)
    retained=0;peak=input_bytes
    for bound in bounds:
        current=integer(bound['peak_workspace_bytes'],0,256*1024**2)
        held=integer(bound['retained_workspace_bytes'],0,128*1024**2)
        peak=max(peak,input_bytes+retained+current)
        retained+=held
        if peak>256*1024**2 or retained>128*1024**2:
            raise ValueError('joint parent validation/retained run workspace exceeds bound')
    return dict(peak_workspace_bytes=peak,retained_workspace_bytes=retained)


async def _preflight(inputs):
    docs=inputs.receipt.documents();bundles=inputs.receipt.source_bundles();c=docs['runtime/collection.json']
    _environment(docs['runtime/policy.json']) # Fail before output or subprocess activation.
    declarations=[_declarations(docs,n) for n in range(len(c['recordings']))]
    report=await _owned_thread(preflight_runtime_collection,declarations,
        latency_policy=docs['runtime/latency.json'],source_bundles=bundles,numeric_inputs=_numeric(docs))
    bounds=[_runtime_bound(docs,inv) for inv in c['invocations']]
    from .recording_runtime_validation import preflight_recording_run_validation
    validation=[]
    for inv,bound in zip(c['invocations'],bounds):
        value=preflight_recording_run_validation(declarations[inv['recording_index']],
            docs[f"invocations/{inv['ordinal']:02}/execution.json"],publication_rows=bound['publication_rows'])
        bound['validation']=value;validation.append(value)
    _admit_join_workspace(inputs.workspace_bytes,validation)
    runtime=sum(b['maximum_bytes'] for b in bounds)
    # Existing native preflight retains one source set, two native sets. Parent
    # source import is a second source set; it is checked here, charged on copy.
    predicted=report['predicted_artifact_bytes']+report['predicted_source_bytes']+2*runtime+1024**2
    if predicted>c['capacity']['max_collection_bytes']:raise ValueError('joint child/parent source/native/runtime copies cannot fit original budget')
    from .recording_dependencies import verify_dependencies,VERIFICATION_WORKSPACE_BYTES
    if inputs.workspace_bytes+VERIFICATION_WORKSPACE_BYTES>256*1024**2:raise ValueError('parent metadata/dependency workspace exceeds256MiB')
    assets=docs['runtime/assets.json']
    await _owned_thread(verify_dependencies,assets['dependencies'],allowed_roots=(Path(sysconfig.get_paths()['data']),),metadata_roots=(Path(sysconfig.get_paths()['purelib']),))
    await _owned_thread(_verify_assets,inputs._root_path,assets)
    from .owned_capture import OwnedCaptureSource,CaptureCapacity
    for original in c['recordings']:
        source=bundles[original['recording_id']]['manifest']
        maxima={s['source_id']:0 for s in source['sources']}
        for record in source['records']:maxima[record['source_id']]=max(maxima[record['source_id']],record['payload_bytes'])
        size=max(1,4*sum(maxima.values()))
        if size+inputs.workspace_bytes>docs['runtime/policy.json']['limits']['max_owned_media_bytes']:raise ValueError('parent source plus original metadata cannot fit')
        fd=inputs.duplicate_payload_root(original['recording_id']);owner=None
        try:owner=await OwnedCaptureSource.preflight(source,fd,CaptureCapacity(size),journal=None)
        finally:
            os.close(fd)
            if owner is not None:await owner.close()
    budget=ArtifactBudget(c['capacity']['max_collection_bytes'])
    budget.bind_collection(report['collection_sha256'],report['parent_budget_precharge'])
    budget.reserve(runtime) # child runtime copies, whose writer is in another process
    return docs,bundles,declarations,bounds,budget


def _verify_assets(root,assets):
    for asset in assets['assets']:
        path=Path(asset['path']);path=path if path.is_absolute() else Path(root)/path
        fd=_absolute_directory(str(path.parent))
        try:child=os.open(path.name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=fd)
        finally:os.close(fd)
        try:
            before=os.fstat(child)
            if not stat.S_ISREG(before.st_mode) or before.st_size!=asset['bytes']:raise ValueError('original installed asset regular-file/size mismatch')
            hashed=hashlib.sha256();count=0
            while count<=before.st_size:
                data=os.read(child,min(65536,before.st_size+1-count))
                if not data:break
                count+=len(data);hashed.update(data)
            if count!=before.st_size or _identity(before)!=_identity(os.fstat(child)) or hashed.hexdigest()!=asset['sha256']:raise ValueError('original installed asset bytes changed')
        finally:os.close(child)


def _environment(policy):
    if FD_ENVIRONMENT_KEY in os.environ or ENV_FD in os.environ:raise ValueError('parent cannot inherit an existing recording transport')
    result={k:os.environ[k] for k in ('HOME','USER','LOGNAME','TMPDIR','LANG','LC_ALL','LC_CTYPE','PATH','SYSTEMROOT') if k in os.environ}
    for adapter in policy['adapters']:
        name=adapter['config'].get('api_key_env')
        if name is not None:
            # Frozen credential identifiers do not grant loader/search-path or
            # process-activation authority to the fixed child.
            blocked={'PATH','HOME','USER','LOGNAME','TMPDIR','TMP','TEMP','SHELL','ENV','BASH_ENV','ZDOTDIR',
                     'LANG','LC_ALL','LC_CTYPE','SYSTEMROOT','COMSPEC','PATHEXT','IFS','CDPATH',
                     'LIBPATH','SHLIB_PATH','NODE_OPTIONS','RUBYOPT','PERL5LIB','PERL5OPT'}
            if (type(name) is not str or not re.fullmatch(r'(?:API_KEY|TOKEN|[A-Z][A-Z0-9_]{0,112}(?:_API_KEY|_TOKEN))',name)
                    or name in blocked or name.startswith(('PYTHON','LD_','DYLD_','GAMST_'))):
                raise ValueError('credential environment cannot control fixed child activation')
            if name not in os.environ:raise ValueError('explicitly frozen credential environment is unavailable')
            result[name]=os.environ[name]
    return result


def _validate_result(directory,ref,task,task_ordinal,execution_sha256,source_manifest_sha256,mapping_sha256):
    """Original task/evidence bytes, before the parent's acceptance timestamp."""
    if ref['path']!='tasks/task-%04d-result.json'%task_ordinal:raise ValueError('result aliases another original task')
    result=read_ref(directory,ref);validate_task_artifact('result',result,execution_sha256)
    if (result['task_id'],result['input_sha256'],result['scope_index'])!=(task['task_id'],task['input_sha256'],task_ordinal+1):raise ValueError('original task input/scope changed')
    retained=[dict(ref)]
    evidence=None
    if result['evidence'] is not None:
        evref=result['evidence']
        if evref['path']!='tasks/task-%04d-evidence.json'%task_ordinal:raise ValueError('evidence aliases another original task')
        evidence=read_ref(directory,evref);validate_task_artifact('evidence',evidence,execution_sha256);retained.append(evref)
        if (evidence['task_id'],evidence['input_sha256'],evidence['scope_index'],evidence['mapping_sha256'])!=(task['task_id'],task['input_sha256'],task_ordinal+1,mapping_sha256):raise ValueError('original mapped evidence changed')
        docs={}
        for key,kind,limit in (('context','context',32768),('snapshot_envelope','envelope',128*1024),('snapshot_record','record',128*1024)):
            original=evidence[key]
            if original['path']!='tasks/task-%04d-%s.json'%(task_ordinal,kind):raise ValueError('snapshot artifact aliases another task')
            docs[kind]=read_ref(directory,original,maximum=limit);retained.append(original)
        from gamst.call_controller import _digest
        context,envelope,record=docs['context'],docs['envelope'],docs['record']
        keys(envelope,('kind','task_id','input_sha256','task_sha256','source_manifest_sha256','mapping_sha256','context_sha256',
            'snapshot_timestamp_ms','scheduled_offset_ns','observed_begin_ns','observed_end_ns','watermarks','context'))
        if (envelope['kind']!='frozen_context_bundle_not_new_sensor_measurement' or envelope['task_id']!=task['task_id'] or
            envelope['input_sha256']!=task['input_sha256'] or envelope['task_sha256']!=digest(task) or
            envelope['source_manifest_sha256']!=source_manifest_sha256 or envelope['mapping_sha256']!=mapping_sha256 or
            envelope['context']!=context or envelope['context_sha256']!=_digest(context) or
            evidence['context_object_sha256']!=_digest(context) or evidence['snapshot_record_sha256']!=_digest(record) or
            record.get('observation',{}).get('value')!=envelope):raise ValueError('original Context/registered snapshot evidence differs')
        if (envelope['scheduled_offset_ns']!=task['snapshot_ns'] or envelope['observed_begin_ns']!=evidence['snapshot_observed_begin_ns'] or
            envelope['observed_end_ns']!=evidence['snapshot_observed_end_ns'] or envelope['snapshot_timestamp_ms']!=context.get('timestamp_ms')):
            raise ValueError('original snapshot observation timestamps differ')
        if result['status']=='answer' and (result['operation_instance']!=evidence['operation_instance'] or any(evidence[n] is None for n in ('controller_request_id','request_evidence_sha256','prompt_sha256','operation_instance'))):raise ValueError('answer missing actual operation/request evidence')
    elif result['status']=='answer':raise ValueError('accepted answer requires original snapshot evidence')
    return result,evidence,retained


def _receipt(execution_sha256,task,*,result=None,result_sha256=None,received_ns=None,anchor=None,failed_ns=None,reason='no_receipt'):
    floor=task['deadline_ns']-task['opens_ns']
    if anchor is None:
        status='not_executed';latency=floor;answer=None
    elif result is not None and result['status']=='answer':
        if received_ns<anchor+task['opens_ns']:raise ValueError('task result predates original opening')
        status='answer' if received_ns<=anchor+task['deadline_ns'] else 'late_answer'
        latency=received_ns-(anchor+task['opens_ns']);answer=result['answer_sha256']
        if status=='late_answer':latency=max(floor,latency);reason='timeout'
        else:reason='completed'
    else:
        status='failed';answer=None;later=max(v for v in (failed_ns,received_ns,anchor+task['deadline_ns']) if v is not None)
        latency=max(floor,later-(anchor+task['opens_ns']))
    value=dict(schema_version=1,kind='recording_parent_task_receipt',execution_sha256=execution_sha256,
        task_id=task['task_id'],input_sha256=task['input_sha256'],result_sha256=result_sha256,received_ns=received_ns,
        failed_observed_ns=None if status=='answer' else failed_ns or received_ns,status=status,
        failure_reason=reason,answer_sha256=answer,latency_ns=latency)
    validate_task_artifact('receipt',value,execution_sha256);return value


async def _collect_coordinator(channel,directory,execution,tasks,manifest,receipts,expected):
    identities={task['task_id']:(n,task) for n,task in enumerate(tasks)}
    mapping=None;summary_ref=None;summary=None
    while True:
        received=await channel.receive()
        if received is None:break
        frame=received.snapshot();kind=frame['type'];fields=frame['fields']
        if kind=='ready':
            if fields['asset_inventory_sha256']!=execution['asset_inventory_sha256']:raise ValueError('child actual asset inventory differs from original execution')
            await channel.send_wait('start',dict(execution_sha256=digest(execution),clock_domain=channel.bootstrap['clock_domain']))
        elif kind=='playback_started':
            mapping=await _owned_thread(read_ref,directory,fields['mapping'])
            validate_metadata('mapping',mapping,digest(execution))
            mappings=await _owned_thread(read_ref,directory,fields['source_mappings'])
            validate_metadata('source-mappings',mappings,digest(execution))
            if (digest(mapping)!=fields['mapping_sha256'] or digest(mappings)!=fields['source_mappings_sha256'] or
                mapping['monotonic_anchor_ns']!=fields['anchor_monotonic_ns'] or mapping['source_manifest_sha256']!=digest(manifest) or
                mapping['recording_id']!=manifest['recording_id'] or mapping['duration_ns']!=manifest['duration_ns'] or mapping['origin_capture_ns']!=manifest['origin_capture_ns']):raise ValueError('actual original mapping binding mismatch')
            expected.extend((fields['mapping'],fields['source_mappings']))
        elif kind=='answer_ready':
            if fields['task_id'] not in identities or fields['task_id'] in receipts or mapping is None:raise ValueError('one original mapped task result required')
            ordinal,task=identities[fields['task_id']]
            if fields['result_sha256']!=fields['result']['file_sha256']:raise ValueError('answer frame result hash differs from original file')
            result,evidence,refs=await _owned_thread(_validate_result,directory,fields['result'],task,ordinal,digest(execution),digest(manifest),digest(mapping))
            accepted=time.monotonic_ns() # Immediately after complete referenced artifact validation.
            receipts[task['task_id']]=_receipt(digest(execution),task,result=result,result_sha256=fields['result_sha256'],received_ns=accepted,
                anchor=mapping['monotonic_anchor_ns'],failed_ns=accepted if result['status']=='failed' else None,reason=result['reason'])
            expected.extend(refs)
        elif kind=='runtime_terminal':
            summary_ref=fields['child_summary'];summary=await _owned_thread(read_ref,directory,summary_ref)
            validate_metadata('child-summary',summary,digest(execution));expected.append(summary_ref)
    if summary is None:raise ValueError('missing original child runtime terminal')
    return summary_ref,summary,mapping


def _channel_rows(native,coordinator,begin,end):
    cs=coordinator.status();ns=native.snapshot();rows=[]
    for channel,direction,frames,size,last,terminal,eof,failed in (
        ('native2','parent_to_child',int(getattr(native,'bootstrap_sent_bytes',0)>0),getattr(native,'bootstrap_sent_bytes',0),None,False,False,native.reason not in (None,'native_incomplete')),
        ('native2','child_to_parent',native.sequence+int(native.hello),native.received,native.sequence-1 if native.sequence else None,native.terminal is not None,native.eof,native.reason not in (None,'native_incomplete')),
        ('coordinator1','parent_to_child',cs['sent_frames']+int(cs['sent_bytes']>0),cs['sent_bytes'],coordinator._send_seq-1 if coordinator._send_seq else None,False,False,bool(cs['failure'])),
        ('coordinator1','child_to_parent',cs['received_frames'],cs['received_bytes'],coordinator._receive_seq-1 if coordinator._receive_seq else None,cs['terminal_seen'],cs['eof_seen'],bool(cs['failure']))):
        if direction=='child_to_parent' and not (terminal and eof):failed=True
        reason='completed' if not failed else 'timeout' if (native.reason=='deadline_exceeded' if channel=='native2' else cs['failure']=='timeout') else 'malformed'
        rows.append(dict(channel=channel,direction=direction,frames=frames,bytes=size,last_sequence=last,terminal_seen=terminal,eof_seen=eof,
            state='failed' if failed else 'closed',reason=reason,observed_begin_ns=begin,observed_end_ns=end))
    return rows


async def _run_one(inputroot,stage,final,docs,bundles,declarations,budget,invocation,*,retain_financial_documents):
    ordinal=invocation['ordinal'];execution=docs[f'invocations/{ordinal:02}/execution.json'];d=declarations[invocation['recording_index']]
    nonce=secrets.token_hex(32);now=time.monotonic_ns();startup=now+_startup_budget_ns(docs,invocation)
    run_deadline=startup+d['source_manifest']['duration_ns']+d['instrumentation_protocol']['close_timeout_ms']*1_000_000
    ctx=run_context(d,nonce,ordinal);environment=_environment(docs['runtime/policy.json'])
    writer=await RecordingRuntimeArtifacts.open(final,execution,budget=budget,retain_evaluation_artifacts=True)
    native=None;coordinator=None;process=None;tasks=[];receipts={};expected=[];summary=None;summary_ref=None;mapping=None
    error=None;reaped_ns=None;resources_ns=None;copied={};cancelled=None;joined=None
    try:
        native=RuntimeParentNativeChannel(ctx,d,paths=_native_paths(inputroot,invocation['recording_index'],docs),output=Path(final)/'native',
            deadline_ns=run_deadline,budget=budget,retain_financial_documents=retain_financial_documents,collection_declarations=declarations,
            latency_policy=docs['runtime/latency.json'],source_bundles=bundles,numeric_inputs=_numeric(docs))
        coordinator=ParentCoordinatorChannel(dict(schema_version=1,type='runtime_bootstrap',nonce=nonce,invocation_ordinal=ordinal,
            execution_sha256=digest(execution),runtime_policy_sha256=execution['runtime_policy_sha256'],latency_policy_sha256=digest(docs['runtime/latency.json']),
            clock_domain='host-monotonic',startup_deadline_ns=startup,run_deadline_ns=run_deadline))
        options=native.prepare(environment);control=coordinator.spawn_options()
        options['env'].update(control['env_delta']);options['pass_fds']+=control['pass_fds']
        argv=(sys.executable,'-m','benchmarks.recording_arm','--runtime-inputs',str(inputroot),'--output',str(stage),
            '--execution-sha',digest(execution),'--invocation',str(ordinal),'--startup-deadline-ns',str(startup))
        now=time.monotonic_ns() # Actual parent spawn-call boundary, after declaration preflight.
        spawning=asyncio.create_task(asyncio.create_subprocess_exec(*argv,**options,cwd=str(Path(__file__).resolve().parents[1]),
            stdin=asyncio.subprocess.DEVNULL,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE,start_new_session=True))
        try:process=await asyncio.shield(spawning)
        except asyncio.CancelledError:
            process=await _finish(spawning);native.spawned(process);raise
        native.spawned(process)
        native_job=asyncio.create_task(native.collect());tasks.append(native_job)
        await coordinator.spawned()
        control_job=asyncio.create_task(_collect_coordinator(coordinator,stage,execution,d['tasks'],d['source_manifest'],receipts,expected));tasks.append(control_job)
        tasks.extend(asyncio.create_task(_drain(stream,65536)) for stream in (process.stdout,process.stderr))
        wait_job=asyncio.create_task(process.wait());tasks.append(wait_job)
        remaining=max(0,(run_deadline-time.monotonic_ns())/1e9)
        joined=asyncio.gather(*tasks)
        try:await asyncio.wait_for(asyncio.shield(joined),remaining)
        finally:
            if joined.done():joined.exception()
        summary_ref,summary,mapping=control_job.result()
        if process.returncode!=0 or native.reason is not None or summary['state']!='closed':error='child_failed'
    except asyncio.CancelledError as exc:cancelled=exc;error='cancelled'
    except (TimeoutError,asyncio.TimeoutError):error='timeout'
    except Exception:error='protocol_failed'
    finally:
        async def cleanup():
            nonlocal reaped_ns,resources_ns,error
            if process is not None:
                if error and process.returncode is None and coordinator is not None:
                    try:
                        if coordinator.status()['sent_bytes']:
                            await asyncio.wait_for(coordinator.send_wait('cancel',{'reason':'deadline' if error=='timeout' else 'parent_shutdown' if error=='cancelled' else 'protocol_failed'}),d['instrumentation_protocol']['close_timeout_ms']/1000)
                    except Exception:pass
                    try:await asyncio.wait_for(asyncio.shield(process.wait()),d['instrumentation_protocol']['close_timeout_ms']/1000)
                    except (Exception,asyncio.CancelledError):pass
                try:
                    await _stop_owned(process);reaped_ns=time.monotonic_ns()
                    if native is not None:await native.record_reaped(process)
                except Exception:error='cleanup_incomplete'
            if coordinator is not None:
                try:
                    if not error:await coordinator.close()
                    else:await coordinator.abort()
                except Exception:error=error or 'protocol_failed'
            if native is not None:await native.abort()
            for task in tasks:
                if not task.done():task.cancel()
            await asyncio.gather(*tasks,return_exceptions=True)
            if joined is not None:await asyncio.gather(joined,return_exceptions=True)
            resources_ns=time.monotonic_ns()
        await _finish(asyncio.create_task(cleanup()))
    try:
        if native is None or coordinator is None:raise RuntimeError('coordinator construction failed before original inventory')
        if process is not None and native.parent_reaped and Path(stage).is_dir():
            try:copied=await writer.ingest_child(stage,reaped_inventory=native.snapshot(),expected_refs=expected,allow_financial=retain_financial_documents and docs['runtime/policy.json']['schema_version']==2)
            except Exception:error=error or 'protocol_failed'
        if summary_ref is None and 'child-summary.json' in copied:
            summary_ref=copied['child-summary.json'];summary=read_ref(final,summary_ref)
        anchor=coordinator.anchor_monotonic_ns
        finished=time.monotonic_ns();receipt_refs=[]
        for n,task in enumerate(d['tasks']):
            value=receipts.get(task['task_id']) or _receipt(digest(execution),task,anchor=anchor,failed_ns=finished,reason=('child_failed' if error=='cleanup_incomplete' else error) or 'no_receipt')
            receipt_refs.append(await writer.write_task(n,'receipt',value))
        native_ref=await writer.write_native_inventory(native.snapshot())
        handshake=await writer.write_metadata('handshake',coordinator.handshake())
        for row in _channel_rows(native,coordinator,now,resources_ns):writer.record_channel(row)
        if 'execution.json' not in copied:await writer.write_metadata('execution',execution)
        indices=await writer.seal_inventory()
        run=dict(schema_version=1,kind='recording_run_inventory',execution_sha256=digest(execution),invocation_ordinal=ordinal,
            nonce=nonce,spawn_started_ns=now,process_reaped_ns=reaped_ns,parent_resources_closed_ns=resources_ns,
            child_summary=summary_ref,native_inventory=native_ref,source_inventory=copied.get('source/terminal.json'),
            receipts=receipt_refs,channel_outcomes=indices['channels'],handshake=handshake,transport_outcomes=indices['transport'],
            exit_code=None if process is None else process.returncode,state='failed' if error else 'closed',failure_counts={} if error is None else {error:1})
        run_ref=await writer.write_metadata('run-inventory',run)
        await writer.close(cleanup_finished=True)
        if cancelled is not None:raise cancelled
        return run_ref,run,native.snapshot(),now
    except BaseException:
        await writer.abort();raise


def _create_output(output):
    output=Path(output);parent=_absolute_directory(str(output.parent))
    try:os.mkdir(output.name,mode=0o700,dir_fd=parent)
    finally:os.close(parent)
    fd=_absolute_directory(str(output))
    try:
        os.fchmod(fd,0o700);os.mkdir('runs',mode=0o700,dir_fd=fd);os.mkdir('.staging',mode=0o700,dir_fd=fd)
    finally:os.close(fd)


def _retained_bytes(output,limit):
    root=_absolute_directory(str(output));count=0;files=0
    def visit(fd,depth):
        nonlocal count,files
        if depth>4:raise ValueError('bounded fixed recording directory depth required')
        with os.scandir(fd) as rows:
            for row in rows:
                info=row.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    child=os.open(row.name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=fd)
                    try:visit(child,depth+1)
                    finally:os.close(child)
                elif stat.S_ISREG(info.st_mode):
                    count+=info.st_size;files+=1
                    if count>limit or files>96*65536:raise ValueError('retained original artifact bound exhausted')
                else:raise ValueError('only original regular artifact files may be inventoried')
    try:visit(root,0)
    finally:os.close(root)
    return count


def _write_collection(output,value,budget):
    data=_line(value,1024**2);budget.reserve(len(data));fd=_absolute_directory(str(output))
    try:
        child=os.open('execution-inventory.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=fd)
        try:os.fchmod(child,0o600);_write_all(child,data);os.fsync(child)
        finally:os.close(child)
    finally:os.close(fd)
    return dict(path='execution-inventory.json',file_sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))


async def run_recording_collection(inputroot,output,*,expected_execution_sha256,
                                   retain_evaluation_artifacts=False,retain_financial_documents=False):
    """Run each frozen invocation exactly once, sequentially after owned reap.

    expected_execution_sha256 pins original invocation0 before opening inputs;
    all remaining execution identities come from that same validated receipt.
    """
    if retain_evaluation_artifacts is not True:raise PermissionError('explicit external evaluation retention required')
    if type(retain_financial_documents) is not bool:raise ValueError('explicit financial retention boolean required')
    from .recording_runtime_validation import validate_recording_run
    root=Path(inputroot).absolute();out=Path(output).absolute();sha(expected_execution_sha256)
    inputs=await OwnedRuntimeInputs.open(str(root),expected_execution_sha256=expected_execution_sha256,invocation_ordinal=0)
    try:
        docs,bundles,declarations,bounds,budget=await _preflight(inputs)
        if retain_financial_documents is not docs['runtime/scopes.json']['retain_financial_documents']:raise PermissionError('financial retention must match original frozen policy')
        await _owned_thread(_create_output,out)
        executions=[];runs=[];validated_runs=[];retained_validation=0;complete=True
        for inv in docs['runtime/collection.json']['invocations']:
            n=inv['ordinal'];run_ref,run,native_inventory,activation_ns=await _run_one(str(root),out/'.staging'/f'{n:02}',out/'runs'/f'{n:02}',
                docs,bundles,declarations,budget,inv,retain_financial_documents=retain_financial_documents)
            validated=await _owned_thread(validate_recording_run,out/'runs'/f'{n:02}',documents=docs,source_bundles=bundles,
                invocation_ordinal=n,parent_native_inventory=native_inventory,activation_started_ns=activation_ns)
            actual_retained=validated.retained_workspace_bytes
            integer(actual_retained,0,128*1024**2)
            if actual_retained>bounds[n]['validation']['retained_workspace_bytes']:
                raise ValueError('validated run exceeds original retained-workspace admission')
            retained_validation+=actual_retained
            if retained_validation>128*1024**2:raise ValueError('retained validated collection exceeds128MiB')
            validated_runs.append(validated)
            prefix=f'runs/{n:02}/'
            runs.append(dict(run_ref,path=prefix+run_ref['path']))
            execution=docs[f'invocations/{n:02}/execution.json'];data=_line(execution,65536)
            executions.append(dict(path=prefix+'execution.json',file_sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
            complete=complete and run['state']=='closed' and validated.runtime_completeness=='complete'
            if run['process_reaped_ns'] is None:raise RuntimeError('owned process reap unverified; further activation refused')
        manifest=dict(schema_version=1,kind='recording_execution_inventory',comparison_plan_sha256=digest(docs['runtime/plan.json']),
            runtime_policy_sha256=digest(docs['runtime/policy.json']),collection_sha256=digest(docs['runtime/collection.json']),
            scope_protocol_sha256=digest(docs['runtime/scopes.json']),instrumentation_protocol_sha256=digest(docs['runtime/instrumentation.json']),
            executions=executions,runs=runs,assessments=[],total_bytes=await _owned_thread(_retained_bytes,out,budget.limit),structural_state='complete' if complete else 'incomplete',numeric_eligible=False)
        # Include this exclusive final manifest itself without treating reserved
        # budget as observed retained bytes. Decimal length converges monotonically.
        retained=manifest['total_bytes']
        for _ in range(8):
            total=retained+len(_line(manifest,1024**2))
            if total==manifest['total_bytes']:break
            manifest['total_bytes']=total
        else:raise ValueError('final inventory size did not stabilize')
        ref=await _owned_thread(_write_collection,out,manifest,budget)
        return dict(manifest=manifest,reference=ref,validated_runs=tuple(validated_runs),artifact_budget=budget,actual_acceptance='incomplete')
    finally:await inputs.close()


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-inputs',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--expected-execution-sha256',required=True)
    parser.add_argument('--retain-evaluation-artifacts',action='store_true')
    parser.add_argument('--retain-financial-documents',action='store_true')
    args=parser.parse_args()
    asyncio.run(run_recording_collection(args.runtime_inputs,args.output,
        expected_execution_sha256=args.expected_execution_sha256,
        retain_evaluation_artifacts=args.retain_evaluation_artifacts,
        retain_financial_documents=args.retain_financial_documents))


if __name__=='__main__':main()

"""Original A/B/C collection plus separately bound owned SDK runtime artifacts."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import time

from .paired_replay import _execute, _encoded, _keys, _write_new
from .perception_comparison import plan_digest, _digest
from .three_arm_replay import validate_collection_spec, ORDER, PERMUTATIONS, MAX_TRACE_BYTES
from .runtime_channel import (ParentChannel, CollectionBudget, build_bootstrap,
    validate_instrumentation_protocol, read_declaration_paths, MAX_BYTES, HASH_FIELDS, FD_ENVIRONMENT_KEY)
from .three_arm_evidence import _strict_decode


def validate(plan,spec,inputs,stream,allocation,reconciliation,instrumentation,*,hashes):
    _keys(hashes,HASH_FIELDS,'instrumented collection identities')
    values=[];total=0
    for value in (plan,spec,inputs,stream,allocation,reconciliation,instrumentation):
        data=_encoded(value);total+=len(data)+1
        if total>MAX_BYTES:raise ValueError('runtime declaration inventory exceeds16MiB')
        values.append(_strict_decode(data))
    for value,field in zip(values,HASH_FIELDS):
        if plan_digest(value)!=_digest(hashes[field]): raise ValueError('changed frozen runtime declaration')
    plan,spec,by_id=validate_collection_spec(values[0],values[1],values[2],
        expected_plan_sha256=hashes['comparison_plan_sha256'],expected_collection_sha256=hashes['collection_sha256'])
    if any(FD_ENVIRONMENT_KEY in command['environment'] for command in spec['runner']['commands'].values()):
        raise ValueError('evaluation descriptor environment key is reserved for explicit transport')
    from .stream_cost_ledger import validate_protocol
    from .stream_cost_allocation import _protocol as allocation_protocol
    from .stream_attempts import PROTOCOL_KEYS
    stream=validate_protocol(values[3],expected_protocol_sha256=hashes['stream_protocol_sha256'])
    allocation,reconciliation=values[4:6]
    if stream['comparison_plan_sha256']!=hashes['comparison_plan_sha256'] or stream['collection_sha256']!=hashes['collection_sha256']:
        raise ValueError('stream belongs to another source collection')
    streams,_,targets,_=allocation_protocol(plan,stream,allocation,hashes['allocation_protocol_sha256'])
    # This is declaration identity, not monetary reconciliation. Unknown bills
    # and required resource proxies intentionally remain unresolved in scope A.
    _keys(reconciliation,PROTOCOL_KEYS,'reconciliation declaration')
    if type(reconciliation['schema_version']) is not int or reconciliation['schema_version']!=1:
        raise ValueError('reconciliation schema1 required')
    for key in HASH_FIELDS[:2]+HASH_FIELDS[3:5]:
        if reconciliation[key]!=hashes[key]: raise ValueError('reconciliation frozen binding mismatch')
    instrumentation=validate_instrumentation_protocol(values[6],expected_sha256=hashes['instrumentation_protocol_sha256'])
    for key in HASH_FIELDS:
        if key in instrumentation and instrumentation[key]!=hashes[key]: raise ValueError('instrumentation frozen binding mismatch')
    if len(plan['comparison']['tasks'])*3>3000:
        raise ValueError('runtime source/declaration inventory exceeds bound')
    mapping={(t['task_id'],streams[t['stream_id']]['arm']):t['stream_id'] for t in targets.values() if t['kind']=='task'}
    return plan,spec,values[2],stream,allocation,reconciliation,instrumentation,by_id,mapping


async def run(comparison_plan,collection_spec,inputs,stream_protocol,allocation_protocol,
              reconciliation_protocol,instrumentation_protocol,output_directory,*,
              expected_plan_sha256,expected_collection_sha256,expected_stream_protocol_sha256,
              expected_allocation_protocol_sha256,expected_reconciliation_protocol_sha256,
              expected_instrumentation_protocol_sha256,expected_inputs_sha256):
    if os.name!='posix': raise RuntimeError('instrumented collection requires owned POSIX processes')
    hashes=dict(zip(HASH_FIELDS,(expected_plan_sha256,expected_collection_sha256,expected_inputs_sha256,
        expected_stream_protocol_sha256,expected_allocation_protocol_sha256,expected_reconciliation_protocol_sha256,
        expected_instrumentation_protocol_sha256)))
    plan,spec,inputs,stream,allocation,reconciliation,instrumentation,by_id,mapping=validate(
        comparison_plan,collection_spec,inputs,stream_protocol,allocation_protocol,reconciliation_protocol,
        instrumentation_protocol,hashes=hashes)
    root=Path(output_directory).absolute();root.mkdir(mode=0o700,parents=False,exist_ok=False)
    artifacts=dict(plan=plan,collection=spec,inputs=inputs,stream_protocol=stream,
        allocation_protocol=allocation,reconciliation_protocol=reconciliation,instrumentation_protocol=instrumentation)
    filenames=dict(plan='comparison-plan.json',collection='collection-spec.json',inputs='inputs.json',
        stream_protocol='stream-protocol.json',allocation_protocol='allocation-protocol.json',
        reconciliation_protocol='reconciliation-protocol.json',instrumentation_protocol='instrumentation-protocol.json')
    paths={key:str(root/name) for key,name in filenames.items()}
    for key,value in artifacts.items():_write_new(Path(paths[key]),value)
    identity=dict(schema_version=1,comparison_plan_sha256=expected_plan_sha256,collection_sha256=expected_collection_sha256)
    _write_new(root/'collection.json',dict(identity,state='started',order=ORDER,scoring='unscored',started_unix_ns=time.time_ns()))
    count=0;previous='0'*64;side_previous='0'*64;side_bytes=0;side_count=0;all_complete=True
    positions={arm:[0,0,0] for arm in 'ABC'};budget=CollectionBudget();state='incomplete'
    with (root/'source-traces.jsonl').open('xb') as source,(root/'runtime-sidecars.jsonl').open('xb') as side:
        os.chmod(source.name,0o600);os.chmod(side.name,0o600)
        def append_side(channel,record,status):
            nonlocal side_previous,side_bytes,side_count,all_complete
            snapshot=channel.snapshot()
            value=dict(schema_version=1,ordinal=side_count,**hashes,task_id=channel.bootstrap['task_id'],
                arm=channel.bootstrap['arm'],stream_id=channel.bootstrap['stream_id'],
                request_sha256=channel.bootstrap['request_sha256'],nonce=channel.bootstrap['nonce'],
                original_trace_sha256=None if record is None else record['trace_sha256'],
                original_response_sha256=None if record is None else record['response_sha256'],
                source_status=status,previous_sidecar_sha256=side_previous,runtime=snapshot)
            digest=plan_digest(value);line=_encoded(dict(value,sidecar_sha256=digest))+b'\n'
            if side_bytes+len(line)>MAX_BYTES: raise ValueError('runtime sidecar artifact exceeds16MiB')
            side.write(line);side.flush();os.fsync(side.fileno())
            side_bytes+=len(line);side_count+=1;side_previous=digest
            all_complete=all_complete and snapshot['instrumentation_complete']
        try:
            for index,task in enumerate(plan['comparison']['tasks']):
                for position,arm in enumerate(PERMUTATIONS[index%6]):
                    if budget.exhausted: raise ValueError('runtime collection capacity exhausted; future work refused')
                    # Sidecars replace wire envelopes with original source/hash
                    # bindings. Reserve their bounded extra metadata before work.
                    budget.reserve_artifact_overhead(4096)
                    configuration=plan['systems'][arm];configuration_sha256=plan_digest(configuration)
                    request=dict(identity,task_id=task['id'],arm=arm,configuration_sha256=configuration_sha256,
                        configuration=configuration,input=by_id[task['id']])
                    bootstrap=build_bootstrap(paths=paths,hashes=hashes,task_id=task['id'],arm=arm,
                        stream_id=mapping[task['id'],arm],request_sha256=plan_digest(request),
                        hello_timeout_ms=instrumentation['hello_timeout_ms'])
                    channel=ParentChannel(bootstrap,budget=budget)
                    try:
                        result=await _execute(spec['runner']['commands'][arm],request,spec['runner']['timeout_ms'],runtime_channel=channel)
                    except BaseException:
                        channel.close();append_side(channel,None,'cancelled_or_failed_before_trace');raise
                    record=dict(ordinal=count,task_id=task['id'],arm=arm,
                        comparison_plan_sha256=expected_plan_sha256,collection_sha256=expected_collection_sha256,
                        configuration_sha256=configuration_sha256,input_sha256=task['input_sha256'],
                        request_sha256=plan_digest(request),previous_trace_sha256=previous,**result)
                    digest=plan_digest(record);record=dict(record,trace_sha256=digest);line=_encoded(record)+b'\n'
                    if len(line)>MAX_TRACE_BYTES:raise ValueError('source trace exceeds512KiB')
                    source.write(line);source.flush();os.fsync(source.fileno())
                    count+=1;previous=digest;positions[arm][position]+=1
                    append_side(channel,record,result['status'])
            summary=dict(identity,state='collected_unscored',records=count,last_trace_sha256=previous,
                order=ORDER,arm_position_counts=positions,complete_six_task_blocks=len(plan['comparison']['tasks'])//6,
                remaining_tasks=len(plan['comparison']['tasks'])%6,actual_acceptance='incomplete',
                certification='Original unscored command reports only. Source inference, sampling execution, receipts, '
                    'resource scope, independent scoring and recording-cluster uncertainty remain unverified.')
            _write_new(root/'complete.json',summary);state='collected_unscored'
            return summary
        finally:
            _write_new(root/'runtime-complete.json',dict(schema_version=1,**hashes,state=state,records=side_count,
                last_sidecar_sha256=side_previous,last_trace_sha256=previous,received_runtime_bytes=budget.received,
                admitted_runtime_frame_bytes=budget.used,
                instrumentation_complete=state=='collected_unscored' and all_complete and not budget.exhausted,
                actual_acceptance='incomplete',receipt_authenticity_verified=False,measurement_truth_verified=False,
                instrumentation_coverage_verified=False,isolation_verified=False,whole_pipeline_cost_verified=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','collection-spec','inputs','stream-protocol','allocation-protocol','reconciliation-protocol',
                 'instrumentation-protocol','plan-sha256','collection-sha256','inputs-sha256',
                 'stream-protocol-sha256','allocation-protocol-sha256','reconciliation-protocol-sha256',
                 'instrumentation-protocol-sha256','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    paths=dict(plan=args.plan,collection=args.collection_spec,inputs=args.inputs,stream_protocol=args.stream_protocol,
        allocation_protocol=args.allocation_protocol,reconciliation_protocol=args.reconciliation_protocol,
        instrumentation_protocol=args.instrumentation_protocol)
    declarations=read_declaration_paths(paths)
    result=asyncio.run(run(*declarations.values(),args.output,
        expected_plan_sha256=args.plan_sha256,expected_collection_sha256=args.collection_sha256,
        expected_inputs_sha256=args.inputs_sha256,expected_stream_protocol_sha256=args.stream_protocol_sha256,
        expected_allocation_protocol_sha256=args.allocation_protocol_sha256,
        expected_reconciliation_protocol_sha256=args.reconciliation_protocol_sha256,
        expected_instrumentation_protocol_sha256=args.instrumentation_protocol_sha256))
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()

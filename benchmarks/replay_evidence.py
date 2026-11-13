"""Bind complete collected pairs to externally reviewed scoring/measurement records."""
import argparse
import json
from pathlib import Path

from .perception_comparison import _digest, _number, _read, _text, compare, plan_digest, validate_plan
from .paired_replay import _decode, _keys, _write_new


TRACE_KEYS = ('ordinal','task_id','system','plan_sha256','input_sha256','request_sha256',
    'previous_trace_sha256','status','diagnostic','response','answer_ready_ms','process_lifetime_ms',
    'returncode','response_sha256','stderr_sha256','measurement_status','quality_status','trace_sha256')
ASSESSMENT_KEYS = ('trace_sha256','quality_score','correct_critical_facts','usage',
                   'peak_memory_bytes','dropped_frames','resource_cost')


def join(plan, inputs, traces, completion, assessment, *, expected_plan_sha256,
         expected_trace_sha256, expected_assessment_sha256):
    """Verify source binding, not the truth or independence of an external assessment."""
    digest,tasks,_,mode=validate_plan(plan,expected_plan_sha256=expected_plan_sha256)
    _digest(expected_trace_sha256); _digest(expected_assessment_sha256)
    policy=plan.get('evidence_join')
    _keys(policy,('schema_version','rubric_sha256','measurement_protocol_sha256','failure_policy'),'evidence_join')
    if type(policy['schema_version']) is not int or policy['schema_version']!=1:
        raise ValueError('unsupported evidence join schema')
    _digest(policy['rubric_sha256']); _digest(policy['measurement_protocol_sha256'])
    if policy['failure_policy']!='zero_quality_timeout_floor': raise ValueError('unsupported frozen failure policy')
    runner=plan.get('runner',{})
    timeout=runner.get('timeout_ms')
    if (runner.get('order')!='alternating' or type(timeout) is not int or not 1<=timeout<=3600000 or
            type(runner.get('schema_version')) is not int or runner['schema_version']!=1):
        raise ValueError('matching paired runner declaration required')
    _keys(completion,('schema_version','state','records','plan_sha256','last_trace_sha256','certification'),'completion')
    if (type(completion['schema_version']) is not int or completion['schema_version']!=1 or
            completion['state']!='collected_unscored' or type(completion['records']) is not int or
            completion['records']!=2*len(tasks) or completion['plan_sha256']!=digest or
            completion['last_trace_sha256']!=expected_trace_sha256):
        raise ValueError('incomplete or mismatched collection')
    if plan_digest(assessment)!=expected_assessment_sha256:
        raise ValueError('assessment differs from reviewed digest')
    _keys(assessment,('schema_version','plan_sha256','last_trace_sha256','reviewer','rubric_sha256',
                     'measurement_protocol_sha256','source_evidence_sha256','entries'),'assessment')
    if (type(assessment['schema_version']) is not int or assessment['schema_version']!=1 or
            assessment['plan_sha256']!=digest or assessment['last_trace_sha256']!=expected_trace_sha256 or
            assessment['rubric_sha256']!=policy['rubric_sha256'] or
            assessment['measurement_protocol_sha256']!=policy['measurement_protocol_sha256']):
        raise ValueError('assessment does not match frozen source/rubric/measurement protocol')
    _text(assessment['reviewer'],'reviewer'); _digest(assessment['source_evidence_sha256'])
    entries=assessment['entries']
    if type(entries) is not list or len(entries)!=2*len(tasks): raise ValueError('assess every source trace exactly once')
    assessments={}
    for entry in entries:
        _keys(entry,ASSESSMENT_KEYS,'assessment entry'); key=_digest(entry['trace_sha256'])
        if key in assessments: raise ValueError('duplicate assessed trace')
        assessments[key]=entry
    if type(inputs) is not list or len(inputs)!=len(tasks): raise ValueError('complete recorded inputs required')
    by_id={}
    for item in inputs:
        _keys(item,('task_id','input'),'recorded input')
        key=item['task_id']
        if type(key) is not str or key not in tasks or key in by_id or type(item['input']) is not dict:
            raise ValueError('unknown/duplicate input')
        if plan_digest(item['input'])!=tasks[key]['input_sha256']: raise ValueError('changed recorded input')
        by_id[key]=item['input']
    expected=[(task,system) for i,task in enumerate(plan['tasks']) for system in
              (('baseline','candidate') if i%2==0 else ('candidate','baseline'))]
    rows=[]; previous='0'*64
    for ordinal,trace in enumerate(traces):
        if ordinal>=len(expected): raise ValueError('extra source trace')
        _keys(trace,TRACE_KEYS,'source trace')
        task,system=expected[ordinal]
        key=_digest(trace['trace_sha256'])
        content={k:v for k,v in trace.items() if k!='trace_sha256'}
        if (plan_digest(content)!=key or trace['previous_trace_sha256']!=previous or
                type(trace['ordinal']) is not int or trace['ordinal']!=ordinal or
                trace['task_id']!=task['id'] or trace['system']!=system or
                trace['plan_sha256']!=digest or trace['input_sha256']!=task['input_sha256']):
            raise ValueError('source chain/order/input/plan mismatch')
        request=dict(schema_version=1,task_id=task['id'],input=by_id[task['id']],system=system,
                     plan_sha256=digest,configuration=plan['systems'][system])
        if trace['request_sha256']!=plan_digest(request): raise ValueError('source request mismatch')
        if trace['measurement_status']!='pipeline_reported_unverified' or trace['quality_status']!='unscored':
            raise ValueError('expected original unscored source trace')
        if trace['status'] not in ('success','error','timeout'): raise ValueError('unknown source status')
        lifetime=_number(trace['process_lifetime_ms'],'process lifetime',minimum=1e-9)
        ready=trace['answer_ready_ms']
        if ready is not None: _number(ready,'answer ready',minimum=1e-9,maximum=lifetime)
        response=trace['response']
        if trace['status']=='success':
            _keys(response,('schema_version','status','answer','measurements'),'source response')
            if (type(response['schema_version']) is not int or response['schema_version']!=1 or
                    response['status']!='success' or type(response['answer']) is not str or
                    type(response['measurements']) is not dict or ready is None or
                    type(trace['returncode']) is not int or trace['returncode']!=0 or trace['diagnostic'] is not None):
                raise ValueError('inconsistent source success')
            _digest(trace['response_sha256'])
        if key not in assessments: raise ValueError('missing trace-bound assessment')
        scored=assessments.pop(key)
        failed=trace['status']!='success'
        if failed and (scored['quality_score']!=0 or scored['correct_critical_facts']!=[]):
            raise ValueError('failed tasks must retain zero quality and no correct critical facts')
        if mode!='resource' and scored['resource_cost'] is not None:
            raise ValueError('resource_cost must be null outside declared resource mode')
        row=dict(task_id=task['id'],system=system,plan_sha256=digest,input_sha256=task['input_sha256'],
            latency_ms=max(lifetime,timeout) if failed else ready,
            **{k:scored[k] for k in ('quality_score','correct_critical_facts','usage','peak_memory_bytes','dropped_frames')})
        if mode=='resource': row['resource_cost']=scored['resource_cost']
        rows.append(row); previous=key
    if len(rows)!=len(expected) or previous!=expected_trace_sha256 or assessments:
        raise ValueError('missing/truncated source traces or unmatched assessment')
    # Existing strict accounting and numerical gates remain the single authority.
    comparison=compare(plan,rows,expected_plan_sha256=digest)
    return dict(rows=rows,rows_sha256=plan_digest(rows),plan_sha256=digest,
        last_trace_sha256=previous,assessment_sha256=expected_assessment_sha256,
        comparison=comparison,
        certification='Hash binding verified only; independent scoring, measurement truth, artifact provenance and real inference still require audit.')


def read_traces(path):
    with Path(path).open('rb') as stream:
        count=0
        while True:
            line=stream.readline(512*1024+1)
            if not line: return
            count+=1
            if len(line)>512*1024 or count>2000 or not line.endswith(b'\n'):
                raise ValueError('source trace line/count bound or incomplete line')
            yield _decode(line)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','assessment','plan-sha256','trace-sha256','assessment-sha256','output'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args(); root=Path(args.run)
    result=join(_read(root/'plan.json'),_read(root/'inputs.json'),read_traces(root/'source-traces.jsonl'),
        _read(root/'complete.json'),_read(args.assessment),expected_plan_sha256=args.plan_sha256,
        expected_trace_sha256=args.trace_sha256,expected_assessment_sha256=args.assessment_sha256)
    output=Path(args.output); output.mkdir(mode=0o700,parents=False,exist_ok=False)
    rows=result.pop('rows'); _write_new(output/'rows.json',rows); _write_new(output/'join.json',result)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__': main()

"""Isolated synthetic lifecycle replay. No model, pricing or quality measurement.

Run only as a dedicated command: virtual clock/UUID patches are process-global.
"""
import asyncio
import json
import sys
import uuid
from unittest.mock import patch

from humansense import HumanSense, SensorMetadata
from humansense.capabilities import CapabilitySpec, OutputField, bounded_json, identifier
from humansense.call_budget import CallBudget
from humansense.call_controller import CallController
from humansense.call_policy import CallPolicy
from humansense.mcp import MCPServer
from .perception_comparison import plan_digest

MAX_INPUT = 262144
MAX_OUTPUT = 65536


def validate(scenario):
    if type(scenario) is not dict or set(scenario) != {'schema_version','policy','budget','limits','actions'}:
        raise ValueError('exact replay scenario fields required')
    if type(scenario['schema_version']) is not int or scenario['schema_version'] != 1:
        raise ValueError('replay schema1 required')
    if len(json.dumps(scenario,allow_nan=False).encode()) > MAX_INPUT:
        raise ValueError('scenario too large')
    for name in ('policy','budget','limits'):
        if type(scenario[name]) is not dict:
            raise ValueError('explicit configuration objects required')
    CallPolicy(**scenario['policy']); CallBudget(**scenario['budget'])
    if not set(scenario['limits']) <= {'max_sources','max_requests','receipt_limit','retention_ms'}:
        raise ValueError('unsupported controller limits')
    actions = scenario['actions']
    if type(actions) is not list or not 1 <= len(actions) <= 48:
        raise ValueError('1..48 replay actions required')
    clock = 0
    fields = {
        'observe': ({'op','at','name','subject','source','epoch','capture','value','rules'}, {'modality','include_visual'}),
        'reserve': ({'op','at','request','maximum_charge'}, {'ttl_ms'}),
        'dispatch': ({'op','at','request'}, set()),
        'finish': ({'op','at','request','success'}, set()),
        'settle': ({'op','at','request','components'}, set()),
        'cancel': ({'op','at','request'}, set()),
        'expire': ({'op','at'}, set()),
        'pause': ({'op','at'}, set()), 'resume': ({'op','at'}, set()),
        'reset': ({'op','at'}, set()), 'close': ({'op','at'}, set()),
        'delete': ({'op','at','subject'}, set()),
    }
    for raw in actions:
        a=bounded_json(raw,max_bytes=4096)
        if type(a) is not dict or a.get('op') not in fields:
            raise ValueError('unknown replay action')
        required, optional=fields[a['op']]
        if not required <= set(a) or set(a)-required-optional:
            raise ValueError('invalid action fields')
        if type(a['at']) is not int or not clock <= a['at'] <= 86400000:
            raise ValueError('nondecreasing virtual milliseconds in one day required')
        clock=a['at']
        if a['op']=='observe':
            for key in ('name','subject','source','epoch'):identifier(a[key])
            if type(a['capture']) is not int or not 0 <= a['capture'] <= 86400000:
                raise ValueError('bounded capture time required')
            if a.get('modality','video') not in ('video','audio','text') or type(a.get('include_visual',True)) is not bool:
                raise ValueError('supported fixture modality and explicit visual flag required')
            if type(a['rules']) is not dict or not set(a['rules']) <= {'change_score','query','needs_visual','unknown_candidate','conflict_rule'}:
                raise ValueError('explicit supported rule inputs required')
        elif 'request' in a:identifier(a['request'])
        if a['op']=='finish' and type(a['success']) is not bool:
            raise ValueError('explicit synthetic completion flag required')
        if a['op']=='delete' and a['subject'] is not None:identifier(a['subject'])
    return scenario


def _brief(state):
    budget=state['budget']
    return dict(closed=state['closed'], sources=len(state['sources']),
        requests=[dict(id=r['id'], state=r['state'], desired_route=r['decision']['desired_route'],
            allowed_route=r['allowed_route'], dispatched_route=r['dispatched_route'],
            blockers=r['admission_blockers'], unknown_unresolved=r['unknown_candidate_unresolved'],
            semantic_audit_due=r['semantic_audit_due'], capture_ms=r['decision']['evidence']['capture_ms'],
            record_sha256=r['decision']['evidence']['record_sha256']) for r in state['requests']],
        detached=len(state['detached_financial_requests']),
        accounting={k:budget[k] for k in ('accepted','dispatched','settled','released','spent_known',
            'reserved_liability','unknown_reports','unresolved_dispatched','overrun')},
        denials=state['admission_denials'], overflow=state['overflow_rejections'])


async def _replay(scenario):
    validate(scenario)
    # Reproducibility is deliberately isolated to this synthetic command process.
    now=[0]; serial=[0]
    def clock(value=None):return now[0] if value is None else value
    def identity():
        serial[0]+=1
        return uuid.UUID(int=serial[0])
    with patch('uuid.uuid4',side_effect=identity), patch('humansense.engine.timestamp',side_effect=clock):
        async with HumanSense(consent=True) as human:
            for modality in ('video','audio','text'):
                spec=CapabilitySpec('replay.'+modality,'replay.'+modality,modality,'replay.fixture','1',
                    (OutputField('content','object'),),'frame',freshness_ms=1000)
                human.register_capability(spec)
            c=CallController(policy=CallPolicy(**scenario['policy']),budget=CallBudget(**scenario['budget']),**scenario['limits'])
            human.register_call_controller('replay',c)
            server=MCPServer(human)
            handles={}; requests={}; observations={}; trace=[]; previous='0'*64
            for index,a in enumerate(scenario['actions']):
                now[0]=a['at']; op=a['op']; result=None
                try:
                    if op=='observe':
                        if a['name'] in observations:raise ValueError('observation name reused')
                        modality=a.get('modality','video')
                        r=await human.push_observation('replay.'+modality,{'content':a['value']},subject_id=a['subject'],
                            sensor=SensorMetadata(a['source'],modality,index,coordinate_frame='pixels',stream_epoch=a['epoch']),
                            timestamp_ms=a['capture'])
                        observations[a['name']]=r
                        key=(a['subject'],a['source'],modality,a['epoch'],r['observation']['processing_epoch'])
                        if key not in handles:
                            # Source epoch replacement is explicit, never financial reset.
                            for old,h in list(handles.items()):
                                if old[:3]==key[:3]:
                                    if h in c.snapshot()['sources']:c.close_source(h,now_ms=now[0])
                                    del handles[old]
                            handles[key]=c.open_source(r,now_ms=now[0])
                        ref=dict(source_id=a['source'],sensor_id=modality,stream_epoch=a['epoch'],
                            capture_ms=a['capture'],sha256=plan_digest(a['value']))
                        if modality!='video' or not a.get('include_visual',True):ref=None
                        result=c.submit(handles[key],r,sequence=index,now_ms=now[0],visual_reference=ref,**a['rules'])
                        if 'request_id' in result:requests[a['name']]=result['request_id']
                    elif op=='reserve':
                        result=c.reserve(requests[a['request']],provider='replay.synthetic',model_revision='1',
                            prompt_sha256=plan_digest({'request':a['request']}),maximum_charge=a['maximum_charge'],
                            ttl_ms=a.get('ttl_ms',1000),now_ms=now[0])
                    elif op=='dispatch':result=c.dispatch(requests[a['request']],now_ms=now[0])
                    elif op=='finish':
                        result=c.finish(requests[a['request']],success=a['success'],
                            evidence_sha256=plan_digest(observations[a['request']]),
                            result_sha256=plan_digest({'synthetic_result':a['success']}),now_ms=now[0])
                    elif op=='settle':result=c.settle(requests[a['request']],components=a['components'],now_ms=now[0])
                    elif op=='cancel':result=c.cancel(requests[a['request']],now_ms=now[0])
                    elif op=='expire':result=c.expire(now_ms=now[0])
                    elif op=='pause':result=await human.pause();handles.clear()
                    elif op=='resume':result=await human.resume()
                    elif op=='reset':result=await human.reset_processing();handles.clear()
                    elif op=='delete':result=await human.delete_data(a['subject']);handles.clear()
                    elif op=='close':await human.close();handles.clear();result={'closed':True}
                    outcome={'status':'returned','result_sha256':plan_digest(result)}
                    if type(result) is dict:
                        outcome.update({k:result[k] for k in ('status','blockers','accepted','semantic_capture_ms') if k in result})
                except (ValueError,RuntimeError,PermissionError,KeyError,TypeError) as exc:
                    # Retain every failed action; no favorable success-only filtering.
                    outcome={'status':'error','error_type':type(exc).__name__,'error_sha256':plan_digest(str(exc))}
                state=c.snapshot()
                status=await server.call_tool('get_call_status',{'controller_id':'replay'})
                row=dict(index=index,virtual_ms=now[0],op=op,action_sha256=plan_digest(a),
                    outcome=outcome,state=_brief(state),state_sha256=plan_digest(state),
                    mcp_status_sha256=plan_digest(status),previous_sha256=previous)
                previous=plan_digest(row);row['trace_sha256']=previous;trace.append(row)
            final=_brief(c.snapshot())
            await server.close()
            return dict(schema_version=1,evidence='synthetic',scenario_sha256=plan_digest(scenario),
                clock='virtual_not_latency',identity='deterministic_fixture_UUIDs',trace=trace,
                last_trace_sha256=previous,final=final,actual_model_executed=False,
                quality_score=None,measured_cost=None,quality_preservation_proven=False)


def main():
    raw=sys.stdin.buffer.readline(MAX_INPUT+1)
    if len(raw)>MAX_INPUT or not raw.endswith(b'\n'):
        raise ValueError('bounded newline-delimited input required')
    def reject(value):raise ValueError('nonfinite JSON')
    package=json.loads(raw,parse_constant=reject)
    # Collector adapter binds all frozen request identity, but never certifies truth.
    paired=type(package) is dict and set(package)=={'schema_version','task_id','system','plan_sha256','configuration','input'}
    scenario=package['input'] if paired else package
    if paired:
        from humansense.call_policy import _sha
        if type(package['schema_version']) is not int or package['schema_version']!=1 or package['system'] not in ('baseline','candidate'):
            raise ValueError('invalid paired request')
        _sha(package['plan_sha256'])
        if type(package['task_id']) is not str or not 1 <= len(package['task_id']) <= 4096:
            raise ValueError('bounded collector task ID required')
        config=package['configuration']
        if type(config) is not dict or config.get('model')!='synthetic-controller-fixture' or type(config.get('controller_replay_policy')) is not dict:
            raise ValueError('explicit synthetic model identity and replay policy required')
        validate(scenario)
        scenario=dict(scenario,policy=config['controller_replay_policy'])
    report=asyncio.run(_replay(scenario))
    if paired:
        report['collector_request_sha256']=plan_digest(package)
        report['recorded_input_sha256']=plan_digest(package['input'])
        report={'schema_version':1,'status':'success','answer':'Synthetic controller replay; no model answer.', 'measurements':report}
    encoded=json.dumps(report,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if len(encoded)>MAX_OUTPUT:raise ValueError('replay report exceeds output budget; split scenario')
    sys.stdout.buffer.write(encoded+b'\n')


if __name__=='__main__':main()

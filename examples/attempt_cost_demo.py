"""Run python -m examples.attempt_cost_demo. Synthetic callbacks, no model calls."""
import asyncio
import json

from benchmarks.call_attempts import AttemptRecorder, cost_quantities
from benchmarks.perception_comparison import plan_digest


def plan():
    config={key:'synthetic-fixture' for key in ('model','revision','precision','hardware',
            'policy','warm_state','sampling','window_lookahead')}
    return dict(schema_version=1,evidence='synthetic',held_out=False,evaluation_id='attempt-demo',
        scoring_protocol='No quality scoring',latency_protocol='Callback duration only',
        trace_provenance='Synthetic application callbacks',systems={'baseline':config,'candidate':dict(config)},
        min_pairs=20,bootstrap_repetitions=500,bootstrap_seed=0,
        tasks=[dict(id='demo',input_sha256='a'*64,critical_facts=[])],
        cost=dict(mode='itemized',currency='USD',scope='full_pipeline',
            accounting_protocol='Fixture receipts only; not measured currency',rate_card={
                'small.attempt':dict(provider='fixture.small',revision='1',unit='attempt',category='request',per_unit=.01),
                'strong.attempt':dict(provider='fixture.strong',revision='1',unit='attempt',category='request',per_unit=.1)}))


async def main():
    frozen=plan();digest=plan_digest(frozen)
    recorder=AttemptRecorder(frozen,expected_plan_sha256=digest,task_id='demo',system='candidate')
    failed=[]
    async def small(token):
        failed.append(token)
        raise RuntimeError('synthetic timeout')
    kwargs=dict(revision='1',phase='cold',evidence_sha256='a'*64,prompt_sha256='b'*64)
    try:
        await recorder.invoke(small,provider='fixture.small',role='small_vlm',**kwargs)
    except RuntimeError:
        pass
    # A failed request may still be billed. This is a declared fixture receipt.
    recorder.report(failed[0],quantities={'small.attempt':1},receipt_sha256='c'*64)
    async def strong(token):
        recorder.report(token,quantities={'strong.attempt':1},receipt_sha256='d'*64)
        return 'synthetic response'
    await recorder.invoke(strong,provider='fixture.strong',role='strong_vlm',
        predecessor=failed[0],relationship='escalation',**kwargs)
    recorder.close();report=recorder.snapshot()
    totals=cost_quantities(frozen,report,expected_plan_sha256=digest,expected_report_sha256=report['sha256'],task_id='demo',system='candidate')
    # Self-supplied digest is adequate for this demonstration, not independent audit.
    return dict(evidence='synthetic',actual_model_executed=False,report=report,
                cost_quantities=totals,quality_preservation_proven=False)


if __name__=='__main__':print(json.dumps(asyncio.run(main()),sort_keys=True))

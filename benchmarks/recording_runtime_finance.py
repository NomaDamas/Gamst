"""Explicit synthetic finance for the owned recording runtime.

Frozen scenarios describe possible charges. Only actual SDK entry and settlement
create an invoice, and only the complete runtime join may issue a financial view.
"""
from types import MappingProxyType

from gamst.call_controller import CallController
from gamst.native_entry import validate_owner
from gamst.recording_settlement import RecordingSettlementRecorder,validate_synthetic_charge_scenario
from .recording_financial import (FinancialArtifacts,_issue_financial_run_view,decode,digest,
    encode,integer,keys,snapshot,validate_adapters)
from .recording_fixture_receipts import FinancialFixtureReceiptRouter
from .recording_native_schema import digest as original_digest
from .recording_native_writer import ArtifactBudget
from .recording_runtime_artifacts import validate_execution
from .recording_runtime_contract import validate_fixture_responses


SCENARIOS_PATH='runtime/charge-scenarios.json'
MAX_SCENARIOS=256
MAX_SCENARIO_BYTES=65536


def _bindings(runtime,collection,invocation):
    arm=next((a for a in runtime['arms'] if a['arm']==invocation['arm']),None)
    native=next((a for a in collection['arms'] if a['arm']==invocation['arm']),None)
    if arm is None or native is None:raise ValueError('original runtime/native arm required')
    table={b['binding_id']:i for i,b in enumerate(native['provider_bindings'])}
    if len(table)!=len(native['provider_bindings']):raise ValueError('unique original native binding required')
    lanes={lane['lane_id']:(lane,table.get(lane['binding_id'])) for lane in arm['lanes']}
    if any(i is None for _,i in lanes.values()):raise ValueError('semantic lane lacks its original native binding')
    adapters={a['adapter_id']:a for a in runtime['adapters']}
    roles={'text_answer':'text_llm','periodic_small':'small_vlm','fill_small':'small_vlm','fill_strong':'strong_vlm'}
    for lane,index in lanes.values():
        binding=native['provider_bindings'][index];adapter=adapters[lane['adapter_id']]
        if ((binding['provider'],binding['model_revision'],binding['role'],binding['config_sha256'],binding['operation_kinds'])!=
                (adapter['provider'],adapter['model_revision'],roles[lane['role']],original_digest(adapter['config']),['semantic'])):
            raise ValueError('semantic scenario crosses frozen actual provider/model/role/config binding')
    return arm,lanes


def validate_runtime_charge_scenarios(value,*,runtime_policy,collection,responses,
        accounting_policy,financial_adapter_inventory):
    """Validate the U2-only, <=64KiB content-addressed scenario inventory.

    Return detached immutable canonical scenario bytes. Every frozen non-null
    response hash must resolve exactly once, including other original invocations.
    No model, controller, artifact writer or invoice is activated here.
    """
    value=snapshot(value,MAX_SCENARIO_BYTES)
    encode(value,MAX_SCENARIO_BYTES,newline=True)
    keys(value,('schema_version','kind','scenarios'),'runtime charge scenarios')
    if (type(value['schema_version']) is not int or value['schema_version']!=1 or
            value['kind']!='recording_synthetic_charge_scenarios' or
            runtime_policy.get('schema_version')!=2 or
            runtime_policy.get('evaluation_mode')!='matched_source_exposure_full_lifetime_v2'):
        raise ValueError('prospectively frozen U2 synthetic charge scenarios required')
    if type(value['scenarios']) is not list or len(value['scenarios'])>MAX_SCENARIOS:
        raise ValueError('joint scenario inventory exceeds256 bodies')
    adapters=validate_adapters(financial_adapter_inventory,expected_sha256=accounting_policy['financial_adapter_inventory_sha256'])
    if (runtime_policy['accounting_policy_sha256']!=digest(accounting_policy) or
            collection['runtime_policy_sha256']!=original_digest(runtime_policy)):
        raise ValueError('scenario inventory crosses frozen runtime/accounting collection')
    if type(responses) not in (list,tuple) or len(responses)>64:raise ValueError('bounded original response inventories required')
    response_map={}
    for response in responses:
        checked=validate_fixture_responses(response);h=original_digest(checked)
        if h in response_map:raise ValueError('duplicate original fixture response inventory')
        response_map[h]=checked
    runtime_adapters={a['adapter_id']:a for a in runtime_policy['adapters']}
    required={a['config']['responses_sha256'] for a in runtime_adapters.values() if a['implementation']=='fixture_completion_v1'}
    if set(response_map)!=required:raise ValueError('all and only original frozen response inventories required')
    invocations={i['ordinal']:i for i in collection['invocations']}
    streams={s['stream_id']:s for s in accounting_policy['streams']}
    components={c['id']:c for c in accounting_policy['components']}
    found={};claims=set()
    for scenario in value['scenarios']:
        scenario=validate_synthetic_charge_scenario(scenario);h=digest(scenario)
        if h in found:raise ValueError('duplicate frozen synthetic scenario')
        claim=tuple(scenario[k] for k in ('provider','account_scope','invoice_id','line_id'))
        if claim in claims:raise ValueError('one frozen original invoice claim per possible charge')
        claims.add(claim);found[h]=encode(scenario)
    referenced=set()
    for response_sha,response in response_map.items():
        for row in response['rows']:
            h=row['synthetic_receipt_sha256']
            if h is None:continue
            if h not in found:raise ValueError('frozen response has no original charge scenario body')
            inv=invocations.get(row['invocation_ordinal'])
            if inv is None:raise ValueError('scenario has no original invocation')
            arm,lanes=_bindings(runtime_policy,collection,inv)
            selected=lanes.get(row['lane_id'])
            if selected is None:raise ValueError('scenario has no original semantic lane')
            lane,binding=selected;adapter=runtime_adapters[lane['adapter_id']]
            if (adapter['implementation']!='fixture_completion_v1' or adapter['config']['responses_sha256']!=response_sha or
                    adapter['config']['settlement']!='synthetic_explicit'):
                raise ValueError('scenario response crosses original fixture adapter')
            scenario=decode(found[h]);operation=scenario['operation_key']
            if operation!=dict(invocation_ordinal=inv['ordinal'],binding_index=binding,call_ordinal=row['call_ordinal']):
                raise ValueError('scenario crosses original invocation/binding/call ordinal')
            stream=streams.get(inv['stream_id']);component=components.get(scenario['component_id'])
            if (stream is None or scenario['stream_id']!=stream['stream_id'] or component is None or component['kind']!='monetary' or component['coverage']!='event'):
                raise ValueError('scenario crosses original monetary stream/component')
            requirement=stream['requirements'][component['id']]
            matches=[l for l in requirement['lanes'] if (l['resource_id'],l['lane_id'])==(scenario['resource_id'],scenario['lane_id'])]
            if requirement['state']!='required' or len(matches)!=1:raise ValueError('scenario lacks its frozen applicable resource lane')
            financial_adapter=adapters.get(scenario['adapter_id'])
            if (scenario['provider'],scenario['revision'])!=(financial_adapter['provider'],financial_adapter['revision']):
                raise ValueError('scenario differs from original F1 provider/revision')
            if (scenario['currency'],scenario['currency_unit'],scenario['pricing_revision'])!=(accounting_policy['currency'],accounting_policy['currency_unit'],accounting_policy['rate_revision']):
                raise ValueError('scenario differs from original monetary unit/revision')
            referenced.add(h)
    if referenced!=set(found):raise ValueError('unreferenced or omitted frozen synthetic scenario')
    return tuple(found[h] for h in sorted(found))


class RuntimeFixtureFinance:
    """Own the actual recorder/router and its bounded original financial files."""
    def __init__(self):raise TypeError('use open_fixture_receipts')

    @property
    def router(self):return self._router

    @property
    def artifacts(self):return self._artifacts

    @property
    def recorders(self):return self._recorders

    def register_request(self,binding_index,request_id,decision_sha256):
        if self._closed:raise RuntimeError('recording financial owner closed')
        self._router.register_request(self._invocation,binding_index,request_id,decision_sha256)

    def actual_settlements(self):return self._router.actual_settlements(self._invocation)

    @property
    def controller_settlement_inventory_sha256(self):return digest(list(self.actual_settlements()))

    def retained_references(self):return self._router.retained_references()

    def close(self):
        # Caller first joins all adapters/controllers. Closing an artifact owner
        # never declares remote work, a native prefix or a financial lane closed.
        if not self._closed:self._closed=True;self._artifacts.close()

    def __enter__(self):return self
    def __exit__(self,*args):self.close()


def open_fixture_receipts(documents,execution,controllers,semantic_bindings,native_context,
        root,artifact_budget,*,retain_financial_documents):
    """Build actual recorder ownership once per original controller before use.

    documents are the detached original OwnedRuntimeInputs receipt documents;
    semantic_bindings maps original lane_id to its actual native binding index.
    root is an already-created owned directory and shares the collection budget.
    """
    if retain_financial_documents is not True:raise ValueError('explicit financial document retention required')
    if type(artifact_budget) is not ArtifactBudget:raise TypeError('shared actual artifact budget required')
    execution=validate_execution(snapshot(execution));runtime=documents['runtime/policy.json'];collection=documents['runtime/collection.json']
    for field,path in (('runtime_policy_sha256','runtime/policy.json'),('collection_sha256','runtime/collection.json'),
            ('scope_protocol_sha256','runtime/scopes.json'),('instrumentation_protocol_sha256','runtime/instrumentation.json')):
        if execution[field]!=original_digest(documents[path]):raise ValueError('financial execution crosses original owned inputs')
    invocation=collection['invocations'][integer(execution['invocation_ordinal'],maximum=95)]
    if any(execution[k]!=invocation[k] for k in ('recording_index','arm','stream_id')):raise ValueError('original financial invocation required')
    native_context=dict(native_context)
    validate_owner('recording_native4',dict(context=native_context,scope_index=0,binding_index=0))
    if native_context['invocation_ordinal']!=invocation['ordinal'] or any(native_context[k]!=execution[k] for k in
            ('collection_sha256','scope_protocol_sha256','instrumentation_protocol_sha256')):
        raise ValueError('financial native context crosses original execution')
    responses=[value for path,value in documents.items() if path.startswith('responses/')]
    encoded_scenarios=validate_runtime_charge_scenarios(documents[SCENARIOS_PATH],runtime_policy=runtime,collection=collection,
        responses=responses,accounting_policy=documents['runtime/accounting.json'],financial_adapter_inventory=documents['runtime/financial-adapters.json'])
    arm,lanes=_bindings(runtime,collection,invocation)
    if type(semantic_bindings) is not dict or semantic_bindings!={lid:index for lid,(_,index) in lanes.items()}:
        raise ValueError('all original actual semantic binding indices required')
    if type(controllers) is not dict or set(controllers)!={c['controller_id'] for c in arm['controllers']}:
        raise ValueError('all original actual controllers required')
    if any(type(c) is not CallController for c in controllers.values()) or len({id(c) for c in controllers.values()})!=len(controllers):
        raise TypeError('distinct actual declared SDK controllers required')
    budgets={original_digest(value):value for path,value in documents.items() if path.startswith('budgets/')}
    policy=documents['runtime/accounting.json'];recorders={}
    # Validate every constructor precondition before installing any irreversible
    # one-controller owner binding. No partial setup on a later budget mismatch.
    for declared in arm['controllers']:
        budget=budgets.get(declared['budget_sha256']);controller=controllers[declared['controller_id']]
        if budget is None or (budget['unit'],budget['pricing_revision'])!=(policy['currency_unit'],policy['rate_revision']):
            raise ValueError('frozen controller budget differs from financial monetary unit/revision')
        with controller._lock:
            state=controller.snapshot()
            if state['closed'] or state['sources'] or state['requests'] or state['budget']['accepted']:
                raise ValueError('financial recorder ownership must precede actual controller activation')
            if controller._recording_settlement_recorder is not None:raise ValueError('actual controller already has recording settlement owner')
            if (controller._budget.unit,controller._budget.pricing_revision)!=(budget['unit'],budget['pricing_revision']):
                raise ValueError('actual controller budget differs from original monetary unit/revision')
    store=FinancialArtifacts(root,budget=artifact_budget)
    try:
        for declared in arm['controllers']:
            cid=declared['controller_id']
            recorders[cid]=RecordingSettlementRecorder(controllers[cid],execution_sha256=original_digest(execution),controller_id=cid,
                unit=policy['currency_unit'],pricing_revision=policy['rate_revision'],max_receipts=MAX_SCENARIOS)
        bindings={(invocation['ordinal'],index):recorders[lane['controller_id']] for lane,index in lanes.values()}
        scenarios=[decode(value) for value in encoded_scenarios if decode(value)['operation_key']['invocation_ordinal']==invocation['ordinal']]
        router=FinancialFixtureReceiptRouter(scenarios=scenarios,adapter_inventory=documents['runtime/financial-adapters.json'],
            recorders=bindings,contexts={invocation['ordinal']:dict(native_context)},artifacts=store,
            retain_financial_documents=True,max_receipts=MAX_SCENARIOS)
    except BaseException:store.close();raise
    owner=object.__new__(RuntimeFixtureFinance);owner._artifacts=store;owner._router=router
    owner._recorders=MappingProxyType(recorders);owner._invocation=invocation['ordinal'];owner._closed=False
    return owner


def _execution_inventory(value,documents,runs):
    """Check the sealed preassessment manifest against every typed original run."""
    value=snapshot(value,65536)
    keys(value,('schema_version','kind','comparison_plan_sha256','runtime_policy_sha256','collection_sha256',
        'scope_protocol_sha256','instrumentation_protocol_sha256','executions','runs','assessments','total_bytes',
        'structural_state','numeric_eligible'),'original execution inventory')
    if (type(value['schema_version']) is not int or value['schema_version']!=1 or
            value['kind']!='recording_execution_inventory' or value['assessments']!=[] or value['numeric_eligible'] is not False):
        raise ValueError('immutable preassessment execution inventory required')
    for field,path in (('comparison_plan_sha256','runtime/plan.json'),('runtime_policy_sha256','runtime/policy.json'),
            ('collection_sha256','runtime/collection.json'),('scope_protocol_sha256','runtime/scopes.json'),
            ('instrumentation_protocol_sha256','runtime/instrumentation.json')):
        if value[field]!=original_digest(documents[path]):raise ValueError('execution inventory crosses original frozen declarations')
    if value['structural_state'] not in ('complete','incomplete'):raise ValueError('explicit actual structural state required')
    integer(value['total_bytes'],maximum=4*1024**3)
    invocations=documents['runtime/collection.json']['invocations']
    if (type(value['executions']) is not list or type(value['runs']) is not list or
            len(value['executions'])!=len(invocations) or len(value['runs'])!=len(invocations) or len(runs)!=len(invocations)):
        raise ValueError('all original execution/run inventory references required')
    seen_paths=set()
    from .recording_financial import reference
    import hashlib
    for invocation,validated,execution_ref,run_ref in zip(invocations,runs,value['executions'],value['runs']):
        execution=validated.execution;run=validated.run_inventory
        if validated.financial_values['invocation_ordinal']!=invocation['ordinal'] or execution!=documents[f"invocations/{invocation['ordinal']:02}/execution.json"]:
            raise ValueError('typed runtime joins cross original execution set/order')
        prefix=f"runs/{invocation['ordinal']:02}/"
        if execution_ref.get('path')!=prefix+'execution.json' or run_ref.get('path')!=prefix+'run-inventory.json':
            raise ValueError('fixed original execution/run paths required')
        for ref,original in ((execution_ref,execution),(run_ref,run)):
            reference(ref);raw=encode(original,newline=True)
            if ref['path'] in seen_paths or (ref['file_sha256'],ref['bytes'])!=(hashlib.sha256(raw).hexdigest(),len(raw)):
                raise ValueError('execution inventory substituted or aliased original execution/run reference')
            seen_paths.add(ref['path'])
    if value['structural_state']=='complete' and any(r.run_inventory['state']!='closed' for r in runs):
        raise ValueError('failed original run cannot become complete collection inventory')
    return value


def finalize_financial_runs(validated_runs,execution_inventory,*,documents):
    """Issue all financial views only from full typed runtime joins and manifest.

    The runtime join derives SDK settlement authority from the original source
    inventory anchored in the sealed original child summary, never attachments.
    """
    from .recording_runtime_validation import ValidatedRecordingRun
    if documents['runtime/policy.json'].get('schema_version')!=2:
        raise ValueError('financial runtime finalization requires original prospective U2')
    runs=tuple(validated_runs)
    if not runs or any(type(run) is not ValidatedRecordingRun for run in runs):
        raise TypeError('all original typed complete runtime joins required')
    inventory=_execution_inventory(execution_inventory,documents,runs);inventory_sha=original_digest(inventory)
    return tuple(_issue_financial_run_view(**dict(run.financial_values),execution_inventory_sha256=inventory_sha,
        controller_settlement_inventory_sha256=run.controller_settlement_inventory_sha256) for run in runs)


def reconcile_runtime_finance(validated_runs,execution_inventory,*,documents,store,ledger,financial_inventory,
        validity_checkpoint,expected_ledger_sha256,expected_checkpoint_sha256,validity_entries,late_notices,
        assessment,expected_assessment_sha256):
    """Run the strict lifetime and unchanged paired-quality evaluators together.

    Original invoice/finality artifacts and externally reviewed ledger/checkpoint
    identities remain explicit inputs. This helper cannot manufacture closure or
    independent financial/scoring review from a completed child process.
    """
    from .recording_lifetime_ledger import validate_lifetime_ledger
    from .recording_exposure_report import compare_quality_latency,reconcile_exposure
    runs=tuple(validated_runs);views=finalize_financial_runs(runs,execution_inventory,documents=documents)
    plan=documents['runtime/plan.json'];policy=documents['runtime/accounting.json'];exposure=documents['runtime/exposure.json']
    totals=validate_lifetime_ledger(policy,exposure,documents['runtime/policy.json'],documents['runtime/collection.json'],
        documents['runtime/scopes.json'],documents['runtime/instrumentation.json'],views,[r.native_index for r in runs],
        ledger,financial_inventory,validity_checkpoint,plan=plan,recordings=documents['runtime/opportunities.json'],
        adapters_document=documents['runtime/financial-adapters.json'],store=store,expected_policy_sha256=digest(policy),
        expected_exposure_sha256=digest(exposure),expected_ledger_sha256=expected_ledger_sha256,
        expected_checkpoint_sha256=expected_checkpoint_sha256,validity_entries=validity_entries,late_notices=late_notices)
    by_ordinal={view.invocation_ordinal:(view,{r['task_id']:r for r in run.task_receipts}) for view,run in zip(views,runs)}
    receipts=[]
    for row in assessment['rows']:
        selected=by_ordinal.get(row['invocation_ordinal'])
        if selected is None or row['task_id'] not in selected[1]:raise ValueError('assessment has no original validated parent receipt')
        view,original=selected;receipts.append((view.arm,original[row['task_id']]))
    quality=compare_quality_latency(plan,assessment,receipts,expected_assessment_sha256=expected_assessment_sha256)
    report=reconcile_exposure(plan,exposure,documents['runtime/opportunities.json'],totals,quality,
        runtime_policy_sha256=original_digest(documents['runtime/policy.json']),collection_sha256=original_digest(documents['runtime/collection.json']),
        scope_protocol_sha256=original_digest(documents['runtime/scopes.json']),instrumentation_protocol_sha256=original_digest(documents['runtime/instrumentation.json']),
        execution_inventory_sha256=original_digest(execution_inventory))
    return totals,quality,report

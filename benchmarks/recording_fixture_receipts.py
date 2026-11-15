"""Explicit frozen F1 scenarios joined to actual SDK settlement transitions."""
from types import MappingProxyType

from gamst.recording_settlement import RecordingSettlementRecorder,validate_synthetic_charge_scenario
from gamst.native_entry import NativeEntryBinding
from gamst.schema import timestamp
from .recording_transports import FixtureReceiptRouter
from .recording_financial import (FinancialArtifacts,decode,digest,encode,integer,keys,sha,snapshot,validate_adapters)


class FinancialFixtureReceiptRouter(FixtureReceiptRouter):
    """Owned runtime seam, never activated from arbitrary JSON or environment.

    Recorder bindings and expected native contexts are frozen before activation.
    Original request/decision registration remains explicit before execute().
    """
    def __init__(self,*,scenarios,adapter_inventory,recorders,contexts,artifacts,
                 retain_financial_documents,max_receipts=16384):
        if retain_financial_documents is not True:
            raise ValueError('explicit frozen financial document retention required')
        if type(artifacts) is not FinancialArtifacts: raise TypeError('owned bounded financial artifacts required')
        self._store=artifacts;self._maximum=integer(max_receipts,1,16384)
        self._inventory=snapshot(adapter_inventory);self._inventory_sha=digest(self._inventory)
        validate_adapters(self._inventory,expected_sha256=self._inventory_sha)
        self._inventory_bytes=encode(self._inventory,newline=True)
        if type(recorders) is not dict or not 1<=len(recorders)<=6144:raise ValueError('bounded frozen recorder bindings required')
        if type(contexts) is not dict or not 1<=len(contexts)<=96:raise ValueError('bounded original native contexts required')
        fixed={};recorder_invocations={}
        for key,recorder in recorders.items():
            if type(key) is not tuple or len(key)!=2:raise ValueError('exact invocation/binding recorder key required')
            invocation,binding=key;integer(invocation,maximum=95);integer(binding,maximum=63)
            if type(recorder) is not RecordingSettlementRecorder:raise TypeError('actual SDK settlement recorder required')
            if invocation not in contexts:raise ValueError('recorder has no original native context')
            if id(recorder) in recorder_invocations and recorder_invocations[id(recorder)]!=invocation:
                raise ValueError('a controller settlement recorder cannot span arm invocations')
            recorder_invocations[id(recorder)]=invocation
            fixed[key]=recorder
        self._recorders=MappingProxyType(fixed)
        self._contexts=MappingProxyType({integer(n,maximum=95):encode(dict(c)) for n,c in contexts.items()})
        if type(scenarios) not in (list,tuple) or len(scenarios)>self._maximum:raise ValueError('bounded frozen synthetic scenarios required')
        frozen={}
        for scenario in scenarios:
            checked=validate_synthetic_charge_scenario(scenario);h=digest(checked)
            key=checked['operation_key'];owner=(key['invocation_ordinal'],key['binding_index'])
            if owner not in fixed or h in frozen:raise ValueError('unique original frozen scenario/recorder required')
            frozen[h]=encode(checked)
        self._scenarios=MappingProxyType(frozen);self._rows={};self._ordinal=0

    def register_request(self,invocation_ordinal,binding_index,request_id,request_decision_sha256):
        recorder=self._recorders.get((invocation_ordinal,binding_index))
        if recorder is None:raise ValueError('unregistered immutable recorder binding')
        recorder.register_request(request_id,request_decision_sha256)

    async def settle(self,binding,scenario_sha256,*,call_ordinal):
        if type(binding) is not NativeEntryBinding:raise TypeError('actual SDK-issued native entry required')
        value=binding.snapshot()
        if value['owner_kind']!='recording_native4':raise ValueError('no legacy financial promotion')
        owner=value['owner'];context=owner['context'];invocation=context['invocation_ordinal']
        if self._contexts.get(invocation)!=encode(context):raise ValueError('synthetic receipt crosses original native context')
        key=(invocation,owner['binding_index']);recorder=self._recorders.get(key)
        raw=self._scenarios.get(sha(scenario_sha256))
        if recorder is None or raw is None:raise ValueError('scenario/actual recorder was not frozen before activation')
        scenario=decode(raw);operation=scenario['operation_key']
        if (operation['invocation_ordinal'],operation['binding_index'],operation['call_ordinal'])!=(*key,integer(call_ordinal,maximum=16383)):
            raise ValueError('scenario crosses original actual binding/call ordinal')
        instance=value['entry']['instance_sha256']
        if instance in self._rows or self._ordinal>=self._maximum:raise ValueError('single bounded original receipt per actual entry required')
        request=recorder.bind_entry(binding,call_ordinal=call_ordinal)
        receipt=recorder.generate_synthetic_receipt(request,scenario,expected_scenario_sha256=scenario_sha256,
            adapter_inventory=self._inventory_bytes,expected_adapter_inventory_sha256=self._inventory_sha)
        source_bytes,evidence_bytes=recorder.generated_artifacts(request)
        source=self._store.write_invoice(decode(source_bytes));evidence=decode(evidence_bytes)
        if evidence['source_document']!=source:raise ValueError('retained original source differs from SDK-issued bytes')
        ordinal=self._store.allocate_receipt_ordinal();self._ordinal+=1
        evidence_ref=self._store.write('financial/evidence/evidence-%06d.json'%ordinal,evidence)
        # Actual budget transition precedes receipt publication and response waits.
        # A later write failure retains the known SDK charge and fails the run.
        recorder.settle(request,receipt,now_ms=timestamp())
        settled=next(row for row in recorder.receipts() if row['instance']==instance)
        settlement_ref=self._store.write('financial/settlements/receipt-%06d.json'%ordinal,settled)
        self._rows[instance]=encode(dict(instance=instance,source_document=source,evidence=evidence_ref,
            controller_settlement=settlement_ref))

    def retained_references(self):
        return tuple(decode(self._rows[key]) for key in sorted(self._rows))

    def actual_settlements(self,invocation_ordinal):
        integer(invocation_ordinal,maximum=95);rows=[];seen=set();recorders=set()
        for (invocation,_),recorder in self._recorders.items():
            if invocation!=invocation_ordinal or id(recorder) in recorders:continue
            recorders.add(id(recorder))
            for row in recorder.receipts():
                if row['instance'] in seen:raise ValueError('actual controller settlement instance duplicated')
                seen.add(row['instance']);rows.append(row)
        return tuple(sorted(rows,key=lambda row:row['instance']))

    def actual_settlement_inventory_sha256(self,invocation_ordinal):
        return digest(list(self.actual_settlements(invocation_ordinal)))

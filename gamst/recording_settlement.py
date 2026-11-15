"""Opt-in authoritative recording settlement, initially synthetic F1 only.

No benchmark dependency and no change to ordinary controller settlement. A
receipt is issued only after its original synthetic line has been checked; the
controller records its actual budget transition while the request still exists.
"""
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import re
import time

from .capabilities import bounded_json,identifier

AUTHORITY = 'gamst.synthetic_invoice_fixture_v1'
ADAPTER = 'synthetic_invoice_fixture_v1'
INVOICE_FIELDS = ('schema_version','kind','evidence_class','authority','provider','revision',
                  'account_scope','invoice_id','currency','currency_unit','lines')
LINE_FIELDS = ('line_id','stream_id','component_id','resource_id','lane_id','operation_key',
               'service_start_ns','service_end_ns','service_event_ns','source_quantity','quantity','billed_atomic')
FINALITY_FIELDS = ('schema_version','kind','evidence_class','authority','execution_sha256','stream_id',
    'accounting_policy_sha256','source_document_sha256','operation_inventory_sha256',
    'lane_closure_inventory_sha256','service_end_ns','issued_ns','state')
EVIDENCE_FIELDS = ('schema_version','kind','adapter_id','adapter_inventory_sha256','origin','source_document',
    'provider','revision','account_scope','invoice_id','line_id','claim_sha256','owner_instance','stream_id',
    'quantities','billed_atomic','currency','currency_unit','service_start_ns','service_end_ns',
    'service_event_ns','observed_received_ns','finality_evidence_sha256')
INPUT_GRAMMAR = dict(schema_version=1,kind='synthetic_invoice_fixture_input_grammar',authority=AUTHORITY,
    invoice_fields=list(INVOICE_FIELDS),line_fields=list(LINE_FIELDS),
    operation_key=['invocation_ordinal','binding_index','call_ordinal'],finality_fields=list(FINALITY_FIELDS),
    rational='reduced_unsigned_256',invoice_limit_bytes=65536,invoice_max_lines=64)
OUTPUT_GRAMMAR = dict(schema_version=1,kind='synthetic_invoice_fixture_output_grammar',
    evidence_fields=list(EVIDENCE_FIELDS),origin='synthetic_fixture',claim_domain='gamst.recording.claim.v1',
    receipt_authenticity_verified=False)
_ISSUER = object()
_COMPONENTS = {'input','output','cache_read','cache_write','media','text_consumer','local_compute','other'}
SCENARIO_FIELDS = ('schema_version','kind','evidence_class','authority','adapter_id','provider','revision',
    'account_scope','invoice_id','line_id','stream_id','component_id','resource_id','lane_id','operation_key',
    'source_quantity','quantity','billed_atomic','currency','currency_unit','pricing_revision','components')
INPUT_GRAMMAR['scenario_fields'] = list(SCENARIO_FIELDS)


def _keys(value,fields):
    if type(value) is not dict or set(value) != set(fields): raise ValueError('exact recording receipt fields required')


def _int(value,maximum=2**63-1):
    if type(value) is not int or not 0 <= value <= maximum: raise ValueError('bounded exact recording receipt integer required')
    return value


def _sha(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}',value) is None: raise ValueError('lowercase SHA256 required')
    return value


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()


def _digest(value): return hashlib.sha256(_canonical(value)).hexdigest()


def _parse(data):
    if type(data) is not bytes or len(data) > 65536: raise ValueError('bounded original synthetic receipt bytes required')
    def pairs(items):
        result = {}
        for key,value in items:
            if key in result: raise ValueError('duplicate original receipt key')
            result[key] = value
        return result
    def integer(text):
        if len(text.lstrip('-')) > 78: raise ValueError('bounded receipt integer lexeme required')
        value = int(text)
        if not -(2**256-1) <= value <= 2**256-1: raise ValueError('bounded receipt integer required')
        return value
    def forbidden(text): raise ValueError('floating/nonfinite receipt value forbidden')
    try:
        value = json.loads(data.decode('utf-8'),object_pairs_hook=pairs,parse_int=integer,
                           parse_float=forbidden,parse_constant=forbidden)
    except (UnicodeDecodeError,json.JSONDecodeError,RecursionError) as error:
        raise ValueError('strict original receipt JSON required') from error
    return bounded_json(value,max_bytes=65536)


def validate_synthetic_invoice(value):
    """Shared fixed F1 primitive, reused by the benchmark evidence validator."""
    value = bounded_json(value,max_bytes=65536)
    _keys(value,INVOICE_FIELDS)
    if (type(value['schema_version']) is not int or value['schema_version'] != 1 or
            value['kind'] != 'synthetic_recording_invoice' or value['evidence_class'] != 'fixture' or
            value['authority'] != AUTHORITY): raise ValueError('synthetic F1 original invoice required')
    for key in ('provider','revision','account_scope','invoice_id','currency','currency_unit'): identifier(value[key])
    if type(value['lines']) is not list or len(value['lines']) > 64: raise ValueError('bounded original invoice lines required')
    seen = set()
    for line in value['lines']:
        _keys(line,LINE_FIELDS)
        for key in ('line_id','stream_id','component_id','resource_id','lane_id'): identifier(line[key])
        if line['line_id'] in seen: raise ValueError('unique original invoice line required')
        seen.add(line['line_id'])
        operation = line['operation_key']
        if operation is not None:
            _keys(operation,('invocation_ordinal','binding_index','call_ordinal'))
            for key,maximum in (('invocation_ordinal',95),('binding_index',63),('call_ordinal',16383)):
                _int(operation[key],maximum)
        start,end,event = (line[k] for k in ('service_start_ns','service_end_ns','service_event_ns'))
        if event is None:
            if _int(start) >= _int(end): raise ValueError('positive original service interval required')
        elif start is not None or end is not None: raise ValueError('exact original service event required')
        else: _int(event)
        quantity = line['source_quantity']; _keys(quantity,('numerator','denominator'))
        n = _int(quantity['numerator'],2**256-1); d = _int(quantity['denominator'],2**256-1)
        if not d or (Fraction(n,d).numerator,Fraction(n,d).denominator) != (n,d): raise ValueError('reduced original quantity required')
        _int(line['quantity']); _int(line['billed_atomic'])
    return value


@dataclass(frozen=True,slots=True,init=False,eq=False)
class FrozenRecordingReceipt:
    _recorder: object
    _request: str
    _instance: str
    _evidence_sha256: str
    _components: tuple
    _claim: str
    _observed_ns: int

    def __init__(self,recorder,request,instance,evidence_sha256,components,claim,observed_ns,*,_issuer=None):
        if _issuer is not _ISSUER: raise TypeError('recording receipts are SDK issued')
        for key,value in (('_recorder',recorder),('_request',request),('_instance',instance),
                ('_evidence_sha256',evidence_sha256),('_components',tuple(sorted(components.items()))),
                ('_claim',claim),('_observed_ns',observed_ns)):
            object.__setattr__(self,key,value)


def validate_synthetic_charge_scenario(value):
    value = bounded_json(value,max_bytes=65536)
    _keys(value,SCENARIO_FIELDS)
    if (type(value['schema_version']) is not int or value['schema_version'] != 1 or
            value['kind'] != 'synthetic_recording_charge_scenario' or value['evidence_class'] != 'fixture' or
            value['authority'] != AUTHORITY): raise ValueError('explicit synthetic charge scenario required')
    for key in ('adapter_id','provider','revision','account_scope','invoice_id','line_id','stream_id','component_id',
                'resource_id','lane_id','currency','currency_unit','pricing_revision'): identifier(value[key])
    # Reuse the original-line grammar for all quantity/owner-key bounds.
    line = {key:deepcopy(value[key]) for key in LINE_FIELDS if key not in ('service_start_ns','service_end_ns','service_event_ns')}
    line.update(service_start_ns=None,service_end_ns=None,service_event_ns=0)
    document = {key:value[key] for key in ('provider','revision','account_scope','invoice_id','currency','currency_unit')}
    document.update(schema_version=1,kind='synthetic_recording_invoice',evidence_class='fixture',authority=AUTHORITY,lines=[line])
    validate_synthetic_invoice(document)
    if value['operation_key'] is None: raise ValueError('semantic scenario requires actual callback operation key')
    components = value['components']
    if type(components) is not dict or not components or not set(components) <= _COMPONENTS:
        raise ValueError('explicit scenario charge components required')
    if sum(_int(amount,10**18) for amount in components.values()) != value['billed_atomic']:
        raise ValueError('scenario components must equal original billed amount')
    return value


class RecordingSettlementRecorder:
    """Bound to one controller; retains only opaque financial ownership."""
    def __init__(self,controller,*,execution_sha256,controller_id,unit,pricing_revision,max_receipts=16384):
        from .call_controller import CallController
        if type(controller) is not CallController: raise TypeError('actual CallController required')
        self._controller = controller
        self.execution_sha256 = _sha(execution_sha256)
        self.controller_id = identifier(controller_id)
        self.unit = identifier(unit); self.pricing_revision = identifier(pricing_revision)
        self._maximum = _int(max_receipts,16384)
        if not self._maximum: raise ValueError('positive recording receipt bound required')
        self._bindings = {}; self._issued = {}; self._records = {}; self._claims = set(); self._binding_entries = {}
        self._registered = {}; self._generated = {}; self._generated_bytes = 0
        with controller._lock:
            if controller._recording_settlement_recorder is not None:
                raise ValueError('controller recording settlement owner cannot be rebound')
            if (controller._budget.unit,controller._budget.pricing_revision) != (self.unit,self.pricing_revision):
                raise ValueError('frozen receipt monetary unit/revision differs from actual budget')
            controller._recording_settlement_recorder = self

    def register_request(self,request_id,request_decision_sha256):
        """Freeze exact request identity before adapter.execute, never currenttask."""
        identifier(request_id); _sha(request_decision_sha256)
        with self._controller._lock:
            request = self._controller._request(request_id)
            if request['state'] != 'pending' or request['decision']['decision_sha256'] != request_decision_sha256:
                raise ValueError('original pending request/decision required')
            if request_decision_sha256 in self._registered or request_id in self._registered.values():
                raise ValueError('single immutable request decision registration required')
            if len(self._registered) >= self._maximum: raise ValueError('recording request lookup capacity exhausted')
            self._registered[request_decision_sha256] = request_id

    def bind_entry(self,entry_binding,*,call_ordinal=None):
        from .native_entry import NativeEntryBinding
        if type(entry_binding) is not NativeEntryBinding: raise TypeError('actual immutable native entry required')
        decision = entry_binding.snapshot()['entry'].get('request_decision_sha256')
        with self._controller._lock:
            request_id = self._registered.get(decision)
            if request_id is None: raise ValueError('actual entry has no registered original request')
            self.bind_request(request_id,entry_binding,call_ordinal=call_ordinal)
            return request_id

    def generate_synthetic_receipt(self,request_id,scenario,*,expected_scenario_sha256,
                                   adapter_inventory,expected_adapter_inventory_sha256):
        """Generate downstream original fixture bytes at actual entered time.

        The upstream response table hashes this scenario, never a future invoice
        whose actual service timestamp did not exist at declaration time.
        """
        scenario = validate_synthetic_charge_scenario(scenario)
        if _digest(scenario) != _sha(expected_scenario_sha256): raise ValueError('frozen synthetic charge scenario changed')
        if (scenario['currency_unit'],scenario['pricing_revision']) != (self.unit,self.pricing_revision):
            raise ValueError('scenario monetary identity differs from actual controller')
        with self._controller._lock:
            binding = self._bindings.get(request_id)
            if binding is None or request_id in self._generated: raise ValueError('single actual entered scenario generation required')
            key = tuple(scenario['operation_key'][name] for name in ('invocation_ordinal','binding_index','call_ordinal'))
            if key != binding[1:]: raise ValueError('scenario does not name original actual binding/call ordinal')
            now = time.monotonic_ns()
            line = {name:deepcopy(scenario[name]) for name in LINE_FIELDS if name not in ('service_start_ns','service_end_ns','service_event_ns')}
            line.update(service_start_ns=None,service_end_ns=None,service_event_ns=now)
            invoice = {name:scenario[name] for name in ('provider','revision','account_scope','invoice_id','currency','currency_unit')}
            invoice.update(schema_version=1,kind='synthetic_recording_invoice',evidence_class='fixture',authority=AUTHORITY,lines=[line])
            source = _canonical(invoice)+b'\n'; source_hash = hashlib.sha256(source).hexdigest()
            claim = _digest(['gamst.recording.claim.v1',scenario['provider'],scenario['account_scope'],scenario['invoice_id'],scenario['line_id']])
            evidence = dict(schema_version=1,kind='recording_financial_evidence',adapter_id=scenario['adapter_id'],
                adapter_inventory_sha256=expected_adapter_inventory_sha256,origin='synthetic_fixture',
                source_document=dict(path='financial/source/source-'+source_hash+'.json',file_sha256=source_hash,bytes=len(source)),
                provider=scenario['provider'],revision=scenario['revision'],account_scope=scenario['account_scope'],
                invoice_id=scenario['invoice_id'],line_id=scenario['line_id'],claim_sha256=claim,owner_instance=binding[0],
                stream_id=scenario['stream_id'],quantities={scenario['component_id']:scenario['source_quantity']},
                billed_atomic=scenario['billed_atomic'],currency=scenario['currency'],currency_unit=scenario['currency_unit'],
                service_start_ns=None,service_end_ns=None,service_event_ns=now,observed_received_ns=now,finality_evidence_sha256=None)
            evidence_bytes = _canonical(evidence)+b'\n'
            if self._generated_bytes+len(source)+len(evidence_bytes) > 64*1024**2:
                raise ValueError('synthetic receipt retained byte capacity exhausted')
            receipt = self.synthetic_receipt(request_id,source_document=source,financial_evidence=evidence_bytes,
                adapter_inventory=adapter_inventory,expected_adapter_inventory_sha256=expected_adapter_inventory_sha256,
                components=scenario['components'])
            self._generated_bytes += len(source)+len(evidence_bytes)
            self._generated[request_id] = (source,evidence_bytes)
            return receipt

    def generated_artifacts(self,request_id):
        with self._controller._lock:
            if request_id not in self._generated: raise ValueError('no generated original fixture artifacts')
            return self._generated[request_id]

    def bind_request(self,request_id,entry_binding,*,call_ordinal=None):
        """Bind an actual dispatched request while it is still owned/opaque."""
        from .native_entry import NativeEntryBinding
        identifier(request_id)
        if type(entry_binding) is not NativeEntryBinding:
            raise TypeError('original immutable actual-entry binding required')
        entry = entry_binding.snapshot()
        if entry['owner_kind'] != 'recording_native4':
            raise ValueError('recording native4 entry required; no legacy promotion')
        instance = entry['entry']['instance_sha256']
        c = self._controller
        with c._lock:
            # Association must be captured while original sensory/request data
            # exists. Later privacy deletion retains only this opaque binding.
            request = c._request(request_id)
            if request['state'] != 'dispatched' or request['charged']:
                raise ValueError('actual unsettled dispatched recording request required')
            execution = request['execution']
            if (entry['entry']['request_decision_sha256'] != execution['decision_sha256'] or
                    any(entry[key] != execution[key] for key in ('provider','model_revision','prompt_sha256')) or
                    entry['role'] != execution['route']):
                raise ValueError('actual entry differs from original controller request')
            if request_id in self._bindings or instance in (value[0] for value in self._bindings.values()):
                raise ValueError('single immutable request/instance binding required')
            if len(self._bindings) >= self._maximum: raise ValueError('recording settlement capacity exhausted')
            lane = (entry['owner']['context']['invocation_ordinal'],entry['owner']['binding_index'])
            previous,count = self._binding_entries.get(lane,(-1,0))
            if entry['entry']['entry_ordinal'] <= previous:
                raise ValueError('recording settlement binding must follow actual entry order')
            rank = count if call_ordinal is None else _int(call_ordinal,16383)
            if not count <= rank <= entry['entry']['entry_ordinal']:
                raise ValueError('explicit actual transport ordinal must preserve original entry order')
            self._binding_entries[lane] = (entry['entry']['entry_ordinal'],rank+1)
            self._bindings[request_id] = (instance,*lane,rank)

    def synthetic_receipt(self,request_id,*,source_document,financial_evidence,adapter_inventory,
                          expected_adapter_inventory_sha256,components):
        """Validate original F1 evidence before issuing a one-use receipt.

        All three inputs are exact retained bytes. No native attachment, HTTP
        usage, declared price, or unknown real provider can issue this token.
        """
        c = self._controller
        invoice = validate_synthetic_invoice(_parse(source_document))
        evidence = _parse(financial_evidence); inventory = _parse(adapter_inventory)
        if _digest(inventory) != _sha(expected_adapter_inventory_sha256): raise ValueError('frozen F inventory differs')
        _keys(inventory,('schema_version','kind','adapters'))
        if type(inventory['schema_version']) is not int or inventory['schema_version'] != 1 or inventory['kind'] != 'recording_financial_adapters':
            raise ValueError('frozen F1 inventory required')
        if type(inventory['adapters']) is not list or not 1 <= len(inventory['adapters']) <= 64: raise ValueError('bounded F1 adapters required')
        adapter_ids = set()
        for candidate in inventory['adapters']:
            _keys(candidate,('adapter_id','implementation','implementation_sha256','provider','revision','evidence_class',
                             'input_schema_sha256','output_schema_sha256','closure_capability'))
            for field in ('adapter_id','provider','revision'): identifier(candidate[field])
            if candidate['adapter_id'] in adapter_ids: raise ValueError('unique frozen adapter identity required')
            adapter_ids.add(candidate['adapter_id'])
            if (candidate['implementation'] != ADAPTER or candidate['evidence_class'] != 'fixture' or
                    candidate['input_schema_sha256'] != _digest(INPUT_GRAMMAR) or
                    candidate['output_schema_sha256'] != _digest(OUTPUT_GRAMMAR) or
                    candidate['closure_capability'] not in ('explicit_original_lines_and_finality','none')):
                raise ValueError('complete frozen synthetic F1 adapter inventory required')
            _sha(candidate['implementation_sha256'])
        _keys(evidence,EVIDENCE_FIELDS)
        matches = [a for a in inventory['adapters'] if a.get('adapter_id') == evidence['adapter_id']]
        if len(matches) != 1: raise ValueError('unique frozen original-line adapter required')
        adapter = matches[0]
        _keys(adapter,('adapter_id','implementation','implementation_sha256','provider','revision','evidence_class',
                      'input_schema_sha256','output_schema_sha256','closure_capability'))
        if (adapter['implementation'] != ADAPTER or adapter['evidence_class'] != 'fixture' or
                adapter['input_schema_sha256'] != _digest(INPUT_GRAMMAR) or adapter['output_schema_sha256'] != _digest(OUTPUT_GRAMMAR)):
            raise ValueError('only synthetic F1 receipt parser is recognized')
        _sha(adapter['implementation_sha256'])
        for key in ('provider','revision','account_scope','invoice_id','currency','currency_unit'):
            if evidence[key] != invoice[key]: raise ValueError('original receipt identity changed')
        if (adapter['provider'],adapter['revision']) != (invoice['provider'],invoice['revision']): raise ValueError('adapter provider differs')
        if (type(evidence['schema_version']) is not int or evidence['schema_version'] != 1 or
                evidence['kind'] != 'recording_financial_evidence' or evidence['origin'] != 'synthetic_fixture' or
                evidence['adapter_inventory_sha256'] != expected_adapter_inventory_sha256 or
                evidence['currency_unit'] != self.unit): raise ValueError('exact original fixture evidence/unit required')
        if evidence['finality_evidence_sha256'] is not None:
            raise ValueError('initial F1 finality is a downstream closure join, never a cyclic receipt pointer')
        ref = evidence['source_document']; _keys(ref,('path','file_sha256','bytes'))
        original_hash = hashlib.sha256(source_document).hexdigest()
        if (ref['path'] != 'financial/source/source-'+original_hash+'.json' or
                ref['file_sha256'] != original_hash or type(ref['bytes']) is not int or ref['bytes'] != len(source_document)):
            raise ValueError('retained original invoice bytes differ')
        lines = [line for line in invoice['lines'] if line['line_id'] == evidence['line_id']]
        if len(lines) != 1: raise ValueError('single original receipt line required')
        line = lines[0]
        for key in ('stream_id','billed_atomic','service_start_ns','service_end_ns','service_event_ns'):
            if evidence[key] != line[key]: raise ValueError('original receipt line changed')
        if evidence['quantities'] != {line['component_id']:line['source_quantity']}: raise ValueError('original receipt quantities differ')
        claim = _digest(['gamst.recording.claim.v1',invoice['provider'],invoice['account_scope'],invoice['invoice_id'],line['line_id']])
        if evidence['claim_sha256'] != claim: raise ValueError('original logical claim differs')
        if type(components) is not dict or not components or not set(components) <= _COMPONENTS:
            raise ValueError('exact receipt billing components required')
        components = {key:_int(value,10**18) for key,value in components.items()}
        if sum(components.values()) != line['billed_atomic']: raise ValueError('receipt components do not equal original billed amount')
        observed = _int(evidence['observed_received_ns'])
        with c._lock:
            binding = self._bindings.get(request_id)
            instance = None if binding is None else binding[0]
            operation = line['operation_key']
            if (binding is None or evidence['owner_instance'] != instance or operation is None or
                    (operation['invocation_ordinal'],operation['binding_index'],operation['call_ordinal']) != binding[1:]):
                raise ValueError('receipt not bound to original actual request/instance')
            if request_id in self._issued or claim in self._claims: raise ValueError('single original receipt per semantic operation required')
            receipt = FrozenRecordingReceipt(self,request_id,instance,_digest(evidence),components,claim,observed,_issuer=_ISSUER)
            self._issued[request_id] = receipt; self._claims.add(claim)
            return receipt

    def _prepare(self,controller,request_id,receipt,components):
        if (type(components) is not dict or controller is not self._controller or type(receipt) is not FrozenRecordingReceipt or
                receipt._recorder is not self or receipt._request != request_id or
                self._issued.get(request_id) is not receipt or request_id in self._records or
                tuple(sorted(components.items())) != receipt._components):
            raise ValueError('SDK-issued original receipt/actual request ownership required')

    def _record(self,request_id,receipt,transition):
        # Called under the controller lock BEFORE deleting detached lookup or
        # retiring the request. All capacity/identity checks preceded settlement.
        if transition['state'] != 'settled': raise RuntimeError('known original receipt failed actual settlement')
        record = dict(schema_version=1,kind='recording_controller_settlement',execution_sha256=self.execution_sha256,
            instance=receipt._instance,controller_id=self.controller_id,request_id=request_id,
            financial_evidence_sha256=receipt._evidence_sha256,unit=self.unit,pricing_revision=self.pricing_revision,
            components=dict(receipt._components),charged=True,observed_ns=time.monotonic_ns(),
            controller_transition_sha256=_digest(transition))
        self._records[request_id] = (record,deepcopy(transition))

    def settle(self,request_id,receipt,*,now_ms):
        return self._controller.settle(request_id,components=dict(receipt._components),now_ms=now_ms,
                                       recording_receipt=receipt)

    def receipts(self):
        with self._controller._lock:
            return tuple(deepcopy(record) for record,transition in self._records.values())

    def transition(self,request_id):
        with self._controller._lock:
            if request_id not in self._records: raise ValueError('no actual recorded settlement')
            return deepcopy(self._records[request_id][1])

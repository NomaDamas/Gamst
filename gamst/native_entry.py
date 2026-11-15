"""Immutable actual-entry correlation; not billing truth or a security sandbox."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
import json
import re


def encoded(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()


def digest(value):return hashlib.sha256(encoded(value)).hexdigest()


def _keys(value,keys):
    if type(value) is not dict or set(value)!=set(keys.split()):raise ValueError('exact entry binding fields required')


def _hash(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}',value) is None:raise ValueError('entry digest required')


def _id(value):
    if type(value) is not str or not 1<=len(value)<=256 or not value.isascii() or any(ord(c)<33 or ord(c)>126 for c in value):raise ValueError('bounded opaque identifier required')


def validate_owner(kind,owner):
    if kind=='task_native3':
        _keys(owner,'comparison_plan_sha256 collection_sha256 stream_protocol_sha256 reconciliation_protocol_sha256 instrumentation_protocol_sha256 scope_kind task_id arm stream_id input_sha256 recording_sha256 configuration_sha256 cost_sha256')
        for key,value in owner.items():
            if key.endswith('sha256'):
                if value is None and key in ('instrumentation_protocol_sha256','input_sha256'):continue
                _hash(value)
        if owner['scope_kind'] not in ('task','stream') or owner['arm'] not in ('A','B','C'):raise ValueError('native3 scope required')
        _id(owner['stream_id'])
        if owner['scope_kind']=='task':_id(owner['task_id']);_hash(owner['input_sha256'])
        elif owner['task_id'] is not None or owner['input_sha256'] is not None:raise ValueError('stream owner cannot acquire task')
    elif kind=='recording_native4':
        _keys(owner,'context scope_index binding_index')
        c=owner['context'];_keys(c,'schema_version kind nonce invocation_ordinal collection_sha256 scope_protocol_sha256 instrumentation_protocol_sha256')
        if type(c['schema_version']) is not int or c['schema_version']!=4 or c['kind']!='recording_native':raise ValueError('native4 context required')
        for key in ('nonce','collection_sha256','scope_protocol_sha256','instrumentation_protocol_sha256'):_hash(c[key])
        for v in (c['invocation_ordinal'],owner['scope_index'],owner['binding_index']):
            if type(v) is not int or not 0<=v<=2**63-1:raise ValueError('native4 bounded ordinal required')
    else:raise ValueError('unsupported entry owner')
    if len(encoded(owner))>4096:raise ValueError('owner bound exceeded')
    return json.loads(encoded(owner))


_ISSUER=object()
@dataclass(frozen=True,init=False)
class NativeEntryBinding:
    """Produced by an actual observer entry; detached snapshots are read-only copies."""
    _bytes: bytes
    def __init__(self,value,*,_issuer=None):
        if _issuer is not _ISSUER:raise TypeError('binding must be issued by an actual native entry')
        _keys(value,'schema_version owner_kind owner entry provider model_revision role prompt_sha256')
        if type(value['schema_version']) is not int or value['schema_version']!=1:raise ValueError('binding schema1 required')
        validate_owner(value['owner_kind'],value['owner'])
        for name in ('provider','model_revision','role'):_id(value[name])
        _hash(value['prompt_sha256']);entry=value['entry']
        if value['owner_kind']=='task_native3':
            _keys(entry,'operation_id attempt_id execution_instance_sha256 execution_sha256')
            _id(entry['operation_id']);_id(entry['attempt_id'])
            _hash(entry['execution_instance_sha256']);_hash(entry['execution_sha256'])
        else:
            _keys(entry,'ticket_ordinal entry_ordinal instance_sha256 request_decision_sha256 entered_event_sha256')
            for name in ('ticket_ordinal','entry_ordinal'):
                if type(entry[name]) is not int or not 0<=entry[name]<=2**63-1:raise ValueError('bounded entry ordinal required')
            for name in ('instance_sha256','request_decision_sha256','entered_event_sha256'):_hash(entry[name])
        raw=encoded(value)
        if len(raw)>8192:raise ValueError('entry binding exceeds8192 bytes')
        object.__setattr__(self,'_bytes',raw)
    def snapshot(self):return json.loads(self._bytes)


def _issue_entry_binding(value):
    """Internal producer seam. Does not authorize user-authored JSON as real entry."""
    return NativeEntryBinding(value,_issuer=_ISSUER)


class PreparedCall(ABC):
    @abstractmethod
    async def invoke(self,binding):pass
    @abstractmethod
    async def release(self):pass


class BoundCompletion(ABC):
    """Opt-in completion that requires the observer's actual-entry binding."""
    @abstractmethod
    def prepare(self,request):pass
    async def __call__(self,request):
        raise RuntimeError('bound completion requires an original native entry')

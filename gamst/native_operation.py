"""Immutable actual-operation locator for accepted-publication provenance.

It is separate from semantic NativeEntryBinding: direct source/OpenCV callbacks
have no semantic prompt. No observation body or financial authenticity is added.
"""
from dataclasses import dataclass
import json
from .native_entry import encoded, _hash, validate_owner


_ISSUER = object()
_FIELDS = frozenset(('context', 'scope_index', 'binding_index', 'ticket_ordinal',
    'entry_ordinal', 'instance_sha256', 'worker_ordinal', 'entered_event_sha256',
    'request_sha256', 'evidence_sha256', 'prompt_sha256'))


@dataclass(frozen=True, slots=True, init=False)
class NativeOperationRef:
    _bytes: bytes

    def __init__(self, value, *, _issuer=None):
        if _issuer is not _ISSUER: raise TypeError('actual observer entry must issue operation reference')
        if type(value) is not dict or set(value) != _FIELDS:
            raise ValueError('exact native operation reference required')
        validate_owner('recording_native4', dict(context=value['context'],
            scope_index=value['scope_index'], binding_index=value['binding_index']))
        for name in ('ticket_ordinal', 'entry_ordinal', 'worker_ordinal'):
            if type(value[name]) is not int or not 0 <= value[name] <= 2**63-1:
                raise ValueError('bounded native operation ordinal required')
        for name in ('instance_sha256', 'entered_event_sha256', 'request_sha256', 'evidence_sha256'):
            _hash(value[name])
        if value['prompt_sha256'] is not None: _hash(value['prompt_sha256'])
        raw = encoded(value)
        if len(raw) > 8192: raise ValueError('native operation reference exceeds8192 bytes')
        object.__setattr__(self, '_bytes', raw)

    def snapshot(self): return json.loads(self._bytes)

    @property
    def instance_sha256(self): return self.snapshot()['instance_sha256']

    @property
    def binding_index(self): return self.snapshot()['binding_index']


def _issue_operation_reference(value):
    return NativeOperationRef(value, _issuer=_ISSUER)

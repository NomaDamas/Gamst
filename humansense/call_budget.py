"""Opt-in, in-memory liability accounting. Never executes a model or prices usage."""
from collections import deque
from copy import deepcopy
import threading
import uuid

from .capabilities import identifier
from .call_policy import _sha

_COMPONENTS = frozenset(('input', 'output', 'cache_read', 'cache_write', 'media',
                         'text_consumer', 'local_compute', 'other'))


def _integer(value, maximum=10**18, minimum=0):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError('bounded exact integer required')
    return value


class CallBudget:
    """One accounting scope; caller supplies verified prices/settlements.

    Request sequence is strictly increasing across all sources in this scope.
    A source reset must NOT replace this ledger. No crash persistence is claimed.
    Unknown dispatched charges retain their ceiling and an active slot.
    """
    def __init__(self, *, unit, pricing_revision, ceiling, max_calls=1000,
                 max_inflight=4, receipt_limit=128):
        self._unit = identifier(unit)
        self._pricing_revision = identifier(pricing_revision)
        self._ceiling = _integer(ceiling)
        self._max_calls = _integer(max_calls, 1000000, 1)
        self._max_inflight = _integer(max_inflight, 1024, 1)
        self._recent = deque(maxlen=_integer(receipt_limit, 1024, 1))
        self._scope = uuid.uuid4().hex
        self._lock = threading.RLock()
        self._active = {}
        self._last_sequence = -1
        self._clock = 0
        self._accepted = self._dispatched = self._settled = self._released = 0
        self._spent = self._reserved = self._unknown_reports = 0
        self._components = {key: 0 for key in sorted(_COMPONENTS)}
        self._overrun = self._closed = False

    @property
    def unit(self):
        return self._unit

    @property
    def pricing_revision(self):
        return self._pricing_revision

    @property
    def ceiling(self):
        return self._ceiling

    @property
    def max_calls(self):
        return self._max_calls

    @property
    def max_inflight(self):
        return self._max_inflight

    def _time(self, now_ms):
        now = _integer(now_ms, 2**53-1)
        if now < self._clock:
            raise ValueError('accounting clock must not move backwards')
        self._clock = now
        return now

    def _get(self, token):
        if type(token) is not str or token not in self._active:
            raise ValueError('unknown, retired or foreign reservation')
        return self._active[token]

    def reserve(self, *, sequence, decision_sha256, evidence_sha256, provider,
                model_revision, maximum_charge, now_ms, ttl_ms=1000):
        sequence = _integer(sequence, 2**53-1)
        _sha(decision_sha256); _sha(evidence_sha256)
        identifier(provider); identifier(model_revision)
        maximum_charge = _integer(maximum_charge)
        ttl_ms = _integer(ttl_ms, 3600000, 1)
        with self._lock:
            now = self._time(now_ms)
            if now + ttl_ms > 2**53-1:
                raise ValueError('expiry exceeds clock range')
            if sequence <= self._last_sequence:
                raise ValueError('request sequence replay or reordering')
            if self._closed or self._overrun:
                raise RuntimeError('ledger closed or reservation overrun')
            if self._accepted >= self.max_calls or len(self._active) >= self.max_inflight:
                raise RuntimeError('call or inflight limit')
            if self._spent + self._reserved + maximum_charge > self.ceiling:
                raise RuntimeError('budget exhausted')
            self._last_sequence = sequence
            self._accepted += 1
            self._reserved += maximum_charge
            token = f'{self._scope}:{sequence}'
            r = dict(token=token, sequence=sequence, state='reserved',
                     decision_sha256=decision_sha256, evidence_sha256=evidence_sha256,
                     provider=provider, model_revision=model_revision,
                     maximum_charge=maximum_charge, expires_ms=now+ttl_ms,
                     reserved_ms=now, unit=self.unit, pricing_revision=self.pricing_revision)
            self._active[token] = r
            return deepcopy(r)

    def dispatch(self, token, *, now_ms):
        with self._lock:
            now = self._time(now_ms); r = self._get(token)
            if self._closed or self._overrun:
                raise RuntimeError('ledger closed or reservation overrun')
            if r['state'] != 'reserved':
                raise ValueError('reservation already dispatched')
            if now >= r['expires_ms']:
                self._release(r, 'expired', now)
                raise ValueError('reservation expired before dispatch')
            r.update(state='dispatched', dispatched_ms=now)
            self._dispatched += 1
            return deepcopy(r)

    def _release(self, r, reason, now):
        self._reserved -= r['maximum_charge']
        self._released += 1
        r.update(state='released', release_reason=reason, completed_ms=now)
        self._recent.append(deepcopy(r)); del self._active[r['token']]

    def release(self, token, *, now_ms):
        with self._lock:
            now = self._time(now_ms); r = self._get(token)
            if r['state'] != 'reserved':
                raise ValueError('dispatched liability cannot be released')
            self._release(r, 'undispatched', now)

    def expire(self, *, now_ms):
        with self._lock:
            now = self._time(now_ms)
            tokens = [t for t, r in self._active.items()
                      if r['state'] == 'reserved' and now >= r['expires_ms']]
            for t in tokens:
                self._release(self._active[t], 'expired', now)
            return len(tokens)

    def settle(self, token, *, components, now_ms):
        """Exact total components, or None for unknown liability (including errors).

        Partial/unknown components cannot produce a final priced settlement.
        Retries/chained calls require separately reserved increasing sequences.
        """
        if components is not None:
            if type(components) is not dict or not components or not set(components) <= _COMPONENTS:
                raise ValueError('explicit nonempty known charge components required')
            components = {key: _integer(value) for key, value in components.items()}
            charge = sum(components.values())
        with self._lock:
            now = self._time(now_ms); r = self._get(token)
            if r['state'] not in ('dispatched', 'unknown'):
                raise ValueError('only dispatched calls can settle')
            if components is None:
                if r['state'] != 'unknown':
                    self._unknown_reports += 1
                r.update(state='unknown', last_unknown_ms=now)
                return deepcopy(r)
            self._reserved -= r['maximum_charge']
            self._spent += charge
            self._settled += 1
            for key, value in components.items():
                self._components[key] += value
            self._overrun |= charge > r['maximum_charge'] or self._spent+self._reserved > self.ceiling
            r.update(state='settled', components=components, actual_charge=charge,
                     completed_ms=now, reservation_exceeded=charge > r['maximum_charge'])
            self._recent.append(deepcopy(r)); del self._active[token]
            return deepcopy(r)

    def close(self, *, now_ms):
        """Stop admission; release only undispatched calls. Late settlement remains valid."""
        with self._lock:
            now = self._time(now_ms)
            self._closed = True
            for r in list(self._active.values()):
                if r['state'] == 'reserved':
                    self._release(r, 'closed_undispatched', now)
            return self.snapshot()

    def redact_evidence(self):
        """Drop correlation provenance without deleting charges or liabilities."""
        with self._lock:
            for record in list(self._active.values()) + list(self._recent):
                for key in ('decision_sha256', 'evidence_sha256', 'provider', 'model_revision'):
                    record.pop(key, None)
                record['evidence_redacted'] = True

    def snapshot(self):
        with self._lock:
            return deepcopy(dict(scope=self._scope, unit=self.unit,
                pricing_revision=self.pricing_revision, ceiling=self.ceiling,
                spent_known=self._spent, reserved_liability=self._reserved,
                available=max(0, self.ceiling-self._spent-self._reserved),
                accepted=self._accepted, dispatched=self._dispatched,
                settled=self._settled, released=self._released,
                unknown_reports=self._unknown_reports,
                unresolved_dispatched=sum(r['state'] in ('dispatched', 'unknown') for r in self._active.values()),
                known_components=self._components, overrun=self._overrun,
                closed=self._closed, active=list(self._active.values()),
                recent=list(self._recent), omitted_receipts=self._settled+self._released-len(self._recent),
                max_calls=self.max_calls, max_inflight=self.max_inflight,
                executes_model=False, durable=False, quality_preservation_proven=False))

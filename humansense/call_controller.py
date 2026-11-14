"""Explicit, bounded recommendation lifecycle; never calls a model or controls sensors."""
from collections import deque
from copy import deepcopy
import hashlib
import threading
import uuid

from .capabilities import bounded_json, identifier
from .call_policy import CallPolicy, _canonical, _ms, _sha
from .call_budget import CallBudget, _integer


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _owner(record, decision):
    e = decision['evidence']
    models = record.get('models')
    if type(models) is not list or not 1 <= len(models) <= 16:
        raise ValueError('bounded registered model provenance required')
    for model in models:
        if type(model) is not dict:
            raise ValueError('model provenance object required')
        identifier(model.get('id')); identifier(model.get('version'))
    base = tuple(e[k] for k in ('subject_id', 'source_id', 'sensor_id', 'capability_id'))
    signature = _digest(dict(base=base, models=sorted(models, key=_canonical), stream_epoch=e['stream_epoch'],
        processing_epoch=e['processing_epoch'], coordinate_frame=e['coordinate_frame'],
        calibration_id=e['calibration_id'], capability=record['observation']['capability'],
        modality=record['observation']['modality'],
        channels=record['observation']['sensor'].get('channels', [])))
    return base, signature


class CallController:
    """Owns source history, bounded requests and an exclusive fresh CallBudget.

    Application must retain/verify media and explicitly perform authorized calls.
    Lifecycle acknowledgements are trusted application inputs, not model claims.
    """
    def __init__(self, *, policy, budget, max_sources=32, max_requests=128,
                 receipt_limit=128, retention_ms=60000):
        if type(policy) is not CallPolicy or type(budget) is not CallBudget:
            raise ValueError('explicit policy and fresh exclusive budget required')
        if budget.snapshot()['accepted'] or budget.snapshot()['closed']:
            raise ValueError('budget must be fresh and exclusively owned')
        self._policy, self._budget = policy, budget
        self._max_sources = _integer(max_sources, 128, 1)
        self._max_requests = _integer(max_requests, 256, 1)
        self._retention = _integer(retention_ms, 3600000, 1)
        self._recent = deque(maxlen=_integer(receipt_limit, 256, 1))
        self._lock = threading.RLock()
        self._scope = uuid.uuid4().hex
        self._sources, self._requests = {}, {}
        self._source_seq = self._request_seq = self._budget_seq = 0
        self._submission_seq = -1
        self._clock = self._retired = self._overflow = 0
        self._closed = False
        self._runtime_guard = self._runtime_clock = self._record_guard = self._runtime_epoch = None
        self._detached = {}

    def bind_runtime(self, *, guard, clock, validate_record, epoch):
        """One-way binding of a fresh instance to an application-owned runtime."""
        with self._lock:
            if self._runtime_guard is not None or self._source_seq or self._closed or self._budget.snapshot()['accepted']:
                raise ValueError('runtime binding requires a fresh unbound controller')
            if not all(callable(f) for f in (guard, clock, validate_record, epoch)):
                raise ValueError('runtime callbacks required')
            self._runtime_guard, self._runtime_clock, self._record_guard = guard, clock, validate_record
            self._runtime_epoch = epoch

    def _admit(self, record=None):
        if self._runtime_guard is not None:
            self._runtime_guard()
        if record is not None and self._record_guard is not None:
            self._record_guard(record)

    def _runtime_active(self):
        try:
            self._admit()
            return True
        except (RuntimeError, PermissionError):
            return False

    def _time(self, now_ms):
        now = _ms(now_ms)
        if self._runtime_clock is not None:
            now = max(self._clock, _ms(self._runtime_clock()))
        if now < self._clock:
            raise ValueError('controller clock regression')
        self._clock = now
        return now

    def _source(self, handle):
        if type(handle) is not str or handle not in self._sources:
            raise ValueError('unknown or closed source handle')
        return self._sources[handle]

    def _request(self, token):
        if type(token) is not str or token not in self._requests:
            raise ValueError('unknown or retired request')
        return self._requests[token]

    def open_source(self, record, *, now_ms):
        record = bounded_json(record, max_bytes=131072)
        with self._lock:
            now = self._time(now_ms)
            if self._closed:
                raise RuntimeError('controller closed')
            self._admit(record)
            decision = self._policy.recommend(record, now_ms=now)
            base, owner = _owner(record, decision)
            if any(s['base'] == base for s in self._sources.values()):
                raise ValueError('close previous source before epoch/config replacement')
            if len(self._sources) >= self._max_sources:
                raise RuntimeError('source capacity')
            self._source_seq += 1
            handle = f'{self._scope}:source:{self._source_seq}'
            self._sources[handle] = dict(base=base, owner=owner, active=False,
                last_dispatch=None, last_semantic=None, last_capture=None)
            return handle

    def submit(self, handle, record, *, sequence, now_ms, **rules):
        allowed = {'change_score', 'query', 'needs_visual', 'unknown_candidate',
                   'conflict_rule', 'visual_reference'}
        if not set(rules) <= allowed:
            raise ValueError('caller cannot override owned lifecycle history')
        rules = bounded_json(rules)
        record = bounded_json(record, max_bytes=131072)
        with self._lock:
            now = self._time(now_ms); s = self._source(handle)
            if self._closed:
                raise RuntimeError('controller closed')
            self._admit(record)
            sequence = _integer(sequence, 2**53-1)
            if sequence <= self._submission_seq:
                raise ValueError('submission replay or reordering')
            d = self._policy.recommend(record, now_ms=now, change_active=s['active'],
                last_dispatch_ms=s['last_dispatch'], last_semantic_capture_ms=s['last_semantic'], **rules)
            if _owner(record, d)[1] != s['owner']:
                raise ValueError('source epoch/configuration mismatch')
            capture = d['evidence']['capture_ms']
            self._submission_seq = sequence
            fingerprint = _digest(dict(record=d['evidence']['record_sha256'], rules=rules))
            for r in self._requests.values():
                if r['source'] == handle and r['fingerprint'] == fingerprint and not r['finished'] and not r['discard']:
                    return dict(status='coalesced_exact', request_id=r['id'], decision=deepcopy(r['decision']))
            attempts = [r for r in self._requests.values()
                        if r['source'] == handle and r['fingerprint'] == fingerprint]
            attempts += [r for r in self._recent
                         if r['source'] == handle and r.get('fingerprint') == fingerprint]
            prior = max(attempts, key=lambda r:r['order']) if attempts else None
            retry = prior is not None and self._retryable(prior)
            if s['last_capture'] is not None and capture < s['last_capture'] and not retry:
                raise ValueError('capture order regression')
            threshold = prior is not None and (prior.get('trigger_reason') or prior.get('decision', {}).get('reason')) == 'change_threshold_entered'
            if retry and threshold and d['desired_route'] == 'none':
                # Re-admit a retained failed threshold demand, without resetting
                # source hysteresis or replacing its original evidence.
                d = self._policy.recommend(record, now_ms=now, change_active=False,
                    last_dispatch_ms=s['last_dispatch'], last_semantic_capture_ms=s['last_semantic'], **rules)
            if any(b != 'cooldown' for b in d['blockers']):
                return dict(status='rejected', reason='invalid_dispatch_evidence', decision=d)
            if d['desired_route'] == 'none':
                # Never let future/expired records advance hysteresis/history.
                age = now-capture
                if record['quality']['status'] == 'sufficient' and 0 <= age <= min(self._policy.max_age_ms, record['observation']['capability']['freshness_ms']):
                    s.update(active=d['proposed_change_active'], last_capture=capture)
                return dict(status='context_only', decision=d)
            priority = 0 if rules.get('conflict_rule') or rules.get('unknown_candidate') else (1 if rules.get('query') else 2)
            if len(self._requests) + len(self._detached) >= self._max_requests:
                victims = [r for r in self._requests.values() if r['state'] == 'pending' and r['priority'] > priority]
                if victims:
                    victim = max(victims, key=lambda r:(r['priority'], r['submitted_ms']))
                    self._retire(victim, 'preempted_uninspected')
            if len(self._requests) + len(self._detached) >= self._max_requests:
                self._overflow += 1
                return dict(status='rejected', reason='request_capacity', decision=d)
            # A newer attempt permanently supersedes older retry eligibility.
            # Evicting its receipt must not resurrect an older unknown bill.
            for old in list(self._requests.values()) + list(self._recent):
                if old['source'] == handle and old.get('fingerprint') == fingerprint:
                    old['superseded'] = True
            self._request_seq += 1
            token = f'{self._scope}:request:{self._request_seq}'
            priority = 0 if rules.get('conflict_rule') or rules.get('unknown_candidate') else (1 if rules.get('query') else 2)
            self._requests[token] = dict(id=token, source=handle, fingerprint=fingerprint,
                state='pending', order=self._request_seq, decision=d, record=record, rules=rules,
                freshness_ms=record['observation']['capability']['freshness_ms'],
                submitted_ms=now, expires_ms=min(2**53-1, now+self._retention), priority=priority,
                reservation=None, execution=None, finished=False, charged=False,
                discard=False, outcome=None)
            # The event is now retained even if cooldown or price blocks execution.
            if not retry:
                s.update(active=d['proposed_change_active'], last_capture=capture)
            return dict(status='queued', request_id=token, decision=deepcopy(d))

    def _blockers(self, r, now):
        s = self._sources.get(r['source'])
        if self._closed or s is None:
            return ['source_closed']
        capture = r['decision']['evidence']['capture_ms']
        blockers = []
        if now >= r['expires_ms']:
            blockers.append('retention_expired')
        if now-capture > min(self._policy.max_age_ms, r['freshness_ms']):
            blockers.append('stale_capture')
        if s['last_dispatch'] is not None and now-s['last_dispatch'] < self._policy.cooldown_ms:
            blockers.append('cooldown')
        return blockers

    def reserve(self, token, *, provider, model_revision, prompt_sha256,
                maximum_charge, now_ms, ttl_ms=1000):
        _sha(prompt_sha256); identifier(provider); identifier(model_revision)
        with self._lock:
            now = self._time(now_ms); r = self._request(token)
            self._admit(r.get('record'))
            if r['state'] != 'pending':
                raise ValueError('request already reserved or dispatched')
            blockers = self._blockers(r, now)
            if blockers:
                return dict(status='blocked', blockers=blockers)
            execution = dict(request_id=token, route=r['decision']['desired_route'],
                decision_sha256=r['decision']['decision_sha256'], provider=provider,
                model_revision=model_revision, prompt_sha256=prompt_sha256)
            self._budget_seq += 1
            reservation = self._budget.reserve(sequence=self._budget_seq,
                decision_sha256=_digest(execution), evidence_sha256=r['decision']['evidence']['record_sha256'],
                provider=provider, model_revision=model_revision, maximum_charge=maximum_charge,
                now_ms=now, ttl_ms=ttl_ms)
            r.update(state='reserved', reservation=reservation, execution=execution)
            return dict(status='reserved', execution=deepcopy(execution), reservation=deepcopy(reservation))

    def dispatch(self, token, *, now_ms):
        with self._lock:
            now = self._time(now_ms); r = self._request(token)
            self._admit(r.get('record'))
            if r['state'] != 'reserved':
                raise ValueError('reserved request required')
            blockers = self._blockers(r, now)
            if now >= r['reservation']['expires_ms']:
                blockers.append('reservation_expired')
            if blockers:
                self._budget.release(r['reservation']['token'], now_ms=now)
                r.update(state='pending', reservation=None, execution=None)
                return dict(status='blocked', blockers=blockers)
            self._budget.dispatch(r['reservation']['token'], now_ms=now)
            r['state'] = 'dispatched'
            observation = r.pop('record')
            self._sources[r['source']]['last_dispatch'] = now
            return dict(status='dispatched', execution=deepcopy(r['execution']), observation=deepcopy(observation),
                evidence=deepcopy(r['decision']['evidence']), visual_reference=deepcopy(r['decision']['visual_reference']))

    def finish(self, token, *, success, evidence_sha256, result_sha256, now_ms):
        if type(success) is not bool:
            raise ValueError('explicit success required')
        _sha(evidence_sha256); _sha(result_sha256)
        with self._lock:
            now = self._time(now_ms); r = self._request(token)
            if r['state'] != 'dispatched' or r['finished']:
                raise ValueError('single dispatched completion required')
            if evidence_sha256 != r['decision']['evidence']['record_sha256']:
                raise ValueError('completion evidence mismatch')
            s = self._sources.get(r['source'])
            valid = success and not r['discard'] and not self._closed and s is not None and self._runtime_active()
            if self._runtime_epoch is not None:
                valid = valid and r['decision']['evidence']['processing_epoch'] == self._runtime_epoch()
            semantic = valid and r['decision']['desired_route'] in ('small_vlm', 'strong_vlm')
            if semantic:
                capture = r['decision']['evidence']['capture_ms']
                s['last_semantic'] = max(capture, s['last_semantic'] or 0)
            r.update(finished=True, outcome=dict(success=success, accepted=valid,
                result_sha256=result_sha256, completed_ms=now,
                semantic_capture_ms=r['decision']['evidence']['capture_ms'] if semantic else None))
            result = deepcopy(r['outcome'])
            if r['charged']:
                self._retire(r, 'completed')
            return result

    def settle(self, token, *, components, now_ms):
        with self._lock:
            now = self._time(now_ms)
            if type(token) is str and token in self._detached:
                result = self._budget.settle(self._detached[token], components=components, now_ms=now)
                if result['state'] == 'settled':
                    del self._detached[token]
                return result
            r = self._request(token)
            if r['state'] != 'dispatched' or r['charged']:
                raise ValueError('single dispatched financial settlement required')
            result = self._budget.settle(r['reservation']['token'], components=components, now_ms=now)
            if result['state'] == 'settled':
                r['charged'] = True
                if r['finished']:
                    self._retire(r, 'completed')
            return result

    @staticmethod
    def _retryable(r):
        if r.get('superseded', False):
            return False
        if 'retryable' in r:
            return r['retryable']
        return (r['discard'] and not r['finished']) or (r['finished'] and not r['outcome']['accepted'])

    def _retire(self, r, reason):
        self._retired += 1
        self._recent.append(dict(id=r['id'], source=r['source'], reason=reason,
            order=r['order'], fingerprint=r['fingerprint'], retryable=self._retryable(r),
            trigger_reason=r['decision']['reason'],
            evidence=deepcopy(r['decision']['evidence']), outcome=deepcopy(r['outcome'])))
        del self._requests[r['id']]

    def cancel(self, token, *, now_ms):
        with self._lock:
            now = self._time(now_ms); r = self._request(token)
            if r['state'] == 'dispatched':
                r['discard'] = True
                return dict(status='discard_output', liability_preserved=True)
            if r['state'] == 'reserved':
                self._budget.release(r['reservation']['token'], now_ms=now)
            self._retire(r, 'cancelled_undispatched')
            return dict(status='cancelled')

    def expire(self, *, now_ms):
        with self._lock:
            now = self._time(now_ms)
            changed = 0
            for r in list(self._requests.values()):
                if r['state'] == 'dispatched':
                    continue
                if now >= r['expires_ms']:
                    if r['state'] == 'reserved':
                        self._budget.release(r['reservation']['token'], now_ms=now)
                    self._retire(r, 'expired_uninspected')
                    changed += 1
                elif r['state'] == 'reserved' and now >= r['reservation']['expires_ms']:
                    self._budget.release(r['reservation']['token'], now_ms=now)
                    r.update(state='pending', reservation=None, execution=None)
                    changed += 1
            return changed

    def close_source(self, handle, *, now_ms):
        with self._lock:
            now = self._time(now_ms); self._source(handle)
            for r in list(self._requests.values()):
                if r['source'] == handle:
                    self.cancel(r['id'], now_ms=now)
            del self._sources[handle]

    def reset_sources(self, *, now_ms):
        with self._lock:
            now = self._time(now_ms)
            for handle in list(self._sources):
                self.close_source(handle, now_ms=now)

    def forget_subject(self, subject_id=None, *, now_ms):
        """Erase sensory associations; retain only opaque unsettled financial handles.

        Ledger evidence is conservatively redacted across the whole controller,
        since bounded historical receipts cannot prove subject ownership forever.
        """
        with self._lock:
            now = self._time(now_ms)
            handles = {h for h,s in self._sources.items()
                       if subject_id is None or s['base'][0] == subject_id}
            for r in list(self._requests.values()):
                subject = r['decision']['evidence']['subject_id']
                if subject_id is not None and subject != subject_id:
                    continue
                if r['state'] == 'reserved':
                    self._budget.release(r['reservation']['token'], now_ms=now)
                if r['state'] == 'dispatched' and not r['charged']:
                    self._detached[r['id']] = r['reservation']['token']
                del self._requests[r['id']]
            for h in handles:
                del self._sources[h]
            self._recent = deque((r for r in self._recent
                if subject_id is not None and r['evidence']['subject_id'] != subject_id), maxlen=self._recent.maxlen)
            self._budget.redact_evidence()

    def close(self, *, now_ms):
        with self._lock:
            now = self._time(now_ms)
            for handle in list(self._sources):
                self.close_source(handle, now_ms=now)
            self._closed = True
            self._budget.close(now_ms=now)

    def snapshot(self, *, now_ms=None):
        with self._lock:
            if now_ms is not None or self._runtime_clock is not None:
                self._time(self._clock if now_ms is None else now_ms)
            requests = sorted(self._requests.values(), key=lambda r:(r['priority'], r['submitted_ms'], r['id']))
            return deepcopy(dict(closed=self._closed, sources=self._sources,
                requests=[dict(id=r['id'], source=r['source'], state=r['state'], priority=r['priority'],
                    decision=r['decision'], blockers=self._blockers(r, self._clock),
                    expires_ms=r['expires_ms'], finished=r['finished'], charged=r['charged'],
                    discard=r['discard'], execution=r['execution']) for r in requests],
                detached_financial_requests=list(self._detached),
                recent=list(self._recent), omitted_receipts=self._retired-len(self._recent),
                overflow_rejections=self._overflow, budget=self._budget.snapshot(),
                executes_model=False, changes_sensor_cadence=False, durable=False))

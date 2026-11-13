"""Bounded, thread-safe VLM diagnostics; no prompts, images or response text."""
from collections import deque
from copy import deepcopy
import threading
import time


_FIELDS = ('prompt_tokens', 'completion_tokens', 'cached_prompt_tokens', 'cache_write_tokens')


def reported_usage(payload):
    """Extract provider-reported counts without inventing missing cache charges."""
    if payload is None:
        return None, 'missing'
    if not isinstance(payload, dict):
        return None, 'invalid'
    details = payload.get('prompt_tokens_details')
    if details is not None and not isinstance(details, dict):
        return None, 'invalid'
    details = details or {}
    values = dict(prompt_tokens=payload.get('prompt_tokens'),
        completion_tokens=payload.get('completion_tokens'),
        cached_prompt_tokens=details.get('cached_tokens'),
        cache_write_tokens=details.get('cache_write_tokens'))
    for value in values.values():
        if value is not None and (type(value) is not int or not 0 <= value <= 10**12):
            return None, 'invalid'
    if values['prompt_tokens'] is not None and values['cached_prompt_tokens'] is not None:
        if values['cached_prompt_tokens'] > values['prompt_tokens']:
            return None, 'invalid'
    # Cache-write accounting differs across compatible providers. Preserve its
    # reported count, but never subtract/readjust it into a billing bucket here.
    known = {key: value for key, value in values.items() if value is not None}
    return (values, 'reported' if len(known) == len(_FIELDS) else 'partial') if known else (None, 'missing')


class VLMCallMetrics:
    """Operational lifetime counters and a bounded completion-order sample.

    start/finish are called exactly once by the owning synchronous call wrapper.
    An async cancellation may leave that synchronous request running; it remains
    in flight until the wrapper exits, even if the output is later discarded.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._records = deque(maxlen=128)
        self._started = self._completed = self._errors = self._http_attempts = 0
        self._totals = {key: 0 for key in _FIELDS}
        self._observations = {key: 0 for key in _FIELDS}
        self._usage_states = {key: 0 for key in ('reported', 'partial', 'missing', 'invalid')}

    def start(self):
        started = time.monotonic()
        with self._lock:
            self._started += 1
            return self._started, started

    def http_attempt(self):
        with self._lock:
            self._http_attempts += 1

    def finish(self, token, *, success, usage, request_started=False):
        call_id, started = token
        elapsed_ms = max(0., (time.monotonic()-started)*1000)
        counts, state = reported_usage(usage)
        record = dict(call_id=call_id, outcome='success' if success else 'error',
            http_request_attempted=request_started,
            call_processing_ms=elapsed_ms, usage=counts, usage_status=state)
        with self._lock:
            self._completed += 1
            self._errors += not success
            self._usage_states[state] += 1
            for key, value in (counts or {}).items():
                if value is not None:
                    self._totals[key] += value
                    self._observations[key] += 1
            self._records.append(record)

    def snapshot(self):
        with self._lock:
            return dict(started=self._started, completed=self._completed, http_attempts=self._http_attempts,
                in_flight=self._started-self._completed, errors=self._errors,
                usage_states=dict(self._usage_states),
                reported_token_totals=dict(self._totals),
                token_observation_counts=dict(self._observations),
                recent_calls=deepcopy(list(self._records)[-16:]),
                omitted_calls=max(0, self._completed-16),
                timing_scope='call preparation + HTTP + response parsing; excludes capture/queue/consumer',
                usage_scope='provider-reported counts; missing counts are unknown, totals cover known fields only',
                cost=None, task_answer_ready_latency_ms=None)

"""Explicit evaluation bridge for actual evidence-bound SDK client invocations."""
import inspect
import threading

from humansense.call_controller import CallController, _digest
from .call_attempts import AttemptRecorder


class RecordedContextClient:
    """One controller request/client attempt; caller owns client and both ledgers."""
    def __init__(self,controller,request_id,recorder,completion,*,phase,
                 predecessor=None,relationship='root'):
        if not isinstance(controller,CallController) or not isinstance(recorder,AttemptRecorder):
            raise ValueError('CallController and AttemptRecorder required')
        if type(request_id) is not str or not request_id or len(request_id)>256:
            raise ValueError('bounded request ID required')
        if not (inspect.iscoroutinefunction(completion) or inspect.iscoroutinefunction(getattr(completion,'__call__',None))):
            raise ValueError('asynchronous completion required')
        if phase not in ('cold','warm','unknown') or relationship not in ('root','retry','escalation'):
            raise ValueError('explicit supported phase and relationship required')
        if predecessor is not None and (type(predecessor) is not str or len(predecessor)>256):
            raise ValueError('bounded predecessor required')
        if (relationship=='root') != (predecessor is None):
            raise ValueError('root has no predecessor; linked attempt requires one')
        self._controller=controller;self._request_id=request_id;self._recorder=recorder;self._completion=completion
        self._phase=phase;self._predecessor=predecessor;self._relationship=relationship
        self._used=False;self._attempt_id=None;self._lock=threading.RLock()

    @property
    def attempt_id(self):
        with self._lock:return self._attempt_id

    async def __call__(self,request):
        with self._lock:
            if self._used:raise RuntimeError('one client invocation per binding; create an explicit new attempt')
            self._used=True
        state=self._controller.snapshot()
        pending=next((r for r in state['requests'] if r['id']==self._request_id),None)
        if (state['closed'] or pending is None or pending['state']!='dispatched' or pending['finished']
                or pending['discard'] or pending['blockers']):
            raise ValueError('live dispatched controller request required')
        execution=pending['execution'];evidence=pending['decision']['evidence']
        if (type(request) is not dict or execution is None or
                execution['route'] not in ('text_llm','small_vlm','strong_vlm') or
                request.get('provider')!=execution['provider'] or
                request.get('model_revision')!=execution['model_revision'] or
                _digest(request)!=execution['prompt_sha256']):
            raise ValueError('client request differs from dispatched execution')
        async def invoke(token):
            with self._lock:self._attempt_id=token
            return await self._completion(request)
        return await self._recorder.invoke(invoke,provider=execution['provider'],revision=execution['model_revision'],
            role=execution['route'],phase=self._phase,evidence_sha256=evidence['record_sha256'],
            prompt_sha256=execution['prompt_sha256'],execution_sha256=_digest(execution),
            predecessor=self._predecessor,relationship=self._relationship)

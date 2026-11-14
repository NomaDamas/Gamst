"""Explicitly owned local subprocesses isolate incompatible ML dependencies."""
import asyncio
import os
import uuid
from .call_controller import _digest
from pathlib import Path
import tempfile

from .capabilities import bounded_json
from .provider_wire import envelope, sizes, parse_header, encode_frame, spec_value
from .provider_profiles import capability
from .streams import _finish


class ProcessProvider:
    """Async SensorStreams provider. No shell, network service or implicit restart.

    A selected Python environment must already contain humansense-sdk and the
    requested optional dependencies. Raw input travels only over owned pipes;
    the validated response is ordinary bounded sensory context.
    """
    capability = staticmethod(capability)
    # Legacy transports can initialize their process fields directly instead
    # of calling open(). Only the explicit SAM open path installs an owner.
    _separation_owner = None

    @classmethod
    async def open(cls, python, profile, config, *, startup_timeout=120, inference_timeout=30,
                   separation_owner=None, evaluation_observer=None):
        if profile == 'pg.physical.local':
            if separation_owner is not None:raise ValueError('PG has no private separation media')
            if evaluation_observer is not None:
                raise ValueError('PG canonical worker evaluation instrumentation is unavailable')
            from .pg_physical_process import PGPhysicalProcess
            return await PGPhysicalProcess.open(python,config,startup_timeout=startup_timeout,inference_timeout=inference_timeout)
        from .separation_media import PROFILE, PROTOCOL, SeparationMediaOwner
        if profile == PROFILE:
            if type(separation_owner) is not SeparationMediaOwner:
                raise PermissionError('SAM separation requires explicit local media ownership')
        elif separation_owner is not None:
            raise ValueError('private separation media is unavailable for this JSON profile')
        for timeout in (startup_timeout, inference_timeout):
            if type(timeout) not in (int, float) or not .01 <= timeout <= 600:
                raise ValueError('worker timeouts must be 0.01..600 seconds')
        # Preserve a virtualenv's symlink spelling; resolving its target selects
        # the base interpreter and silently loses the isolated environment.
        python = str(Path(python).expanduser().absolute())
        if not os.path.isfile(python) or not os.access(python, os.X_OK):
            raise ValueError('select an existing executable Python environment')
        self = cls()
        self.spec = capability(profile, config)
        self.config = bounded_json(config)
        self.profile, self.timeout = profile, inference_timeout
        self._lock = asyncio.Lock()
        self._process = None
        self._close_task = None
        self._sequence, self.closed = 0, False
        self._evaluation_observer=evaluation_observer
        self._worker_id=uuid.uuid4().hex
        self._lifecycle_ticket=None
        self._separation_owner = separation_owner
        if separation_owner is not None:
            separation_owner._authorize_configuration(self.config)
            separation_owner._claim()
        worker_environment = os.environ.copy()
        if separation_owner is not None:
            worker_environment.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                                      HF_HUB_DISABLE_TELEMETRY='1')
        if profile in ('qwen3.asr', 'qwen3.asr.endpoint'):
            worker_environment.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                                      HF_HUB_DISABLE_TELEMETRY='1', VLLM_NO_USAGE_STATS='1',
                                      VLLM_ENABLE_V1_MULTIPROCESSING='0')
        if evaluation_observer is not None:
            try:
                self._lifecycle_ticket=evaluation_observer.reserve_work("worker_lifecycle",dict(worker_id=self._worker_id,
                    profile=profile,configuration_sha256=_digest(self.config)))
            except BaseException:
                if separation_owner is not None:separation_owner.close()
                raise
        spawn = staging = None
        try:
            if separation_owner is not None:
                # The parent owns cleanup even if startup cancellation kills the
                # worker before its tokenizer constructor can return.
                staging = tempfile.TemporaryDirectory(prefix='.gamst-sam-tokenizer-', dir='.')
                worker_environment['GAMST_SAM_TOKENIZER_DIR'] = str(Path(staging.name).absolute())
            if self._lifecycle_ticket is not None:self._lifecycle_ticket.event('spawn_begin')
            spawn = asyncio.create_task(asyncio.create_subprocess_exec(python, '-u', '-m',
                'humansense.provider_worker', stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL, env=worker_environment, close_fds=True))
            self._process = await asyncio.shield(spawn)
            if self._lifecycle_ticket is not None:self._lifecycle_ticket.event("handle_acquired")
            extra = {'separation_protocol': PROTOCOL} if separation_owner is not None else {}
            response = await self._request('start', profile=profile, config=self.config,
                                           timeout=startup_timeout, **extra)
            if response != {'spec': spec_value(self.spec), **extra}:
                raise RuntimeError('worker capability/provenance differs from selected contract')
            if self._lifecycle_ticket is not None:self._lifecycle_ticket.event("handshake_ready")
            return self
        except BaseException:
            self._observe_lifecycle("handshake_failed")
            if self._process is None and spawn is not None:
                try:
                    self._process = await _finish(spawn)
                    self._observe_lifecycle("handle_acquired")
                except Exception:
                    pass
            await _finish(asyncio.create_task(self._stop()))
            if self._process is None and (spawn is None or (spawn.done() and (spawn.cancelled() or spawn.exception() is not None))):
                if self._lifecycle_ticket is not None:
                    try:self._lifecycle_ticket.finish("error")
                    except BaseException:pass
            raise
        finally:
            if staging is not None:
                staging.cleanup()

    def _observe_lifecycle(self, event):
        ticket=getattr(self,"_lifecycle_ticket",None)
        if ticket is not None:
            try:ticket.event(event)
            except BaseException:pass

    async def _stop(self):
        self.closed = True
        owner = getattr(self, '_separation_owner', None)
        if owner is not None:
            owner.close()
        process, self._process = self._process, None
        if process is None:
            return
        if process.stdin:
            process.stdin.close()
        # Process has no public close-reader method. Its owned asyncio transport
        # exposes pipe closure; close stdout even when StreamReader has paused
        # after malformed excessive output, rather than relying on GC to do it.
        transport = process._transport
        for descriptor in (1, 2):
            pipe = transport.get_pipe_transport(descriptor)
            if pipe is not None:
                pipe.close()
        try:
            # A close acknowledgement, EOF or protocol failure often means the
            # worker is already exiting. Let asyncio's child watcher reap it
            # before asking Popen to signal (and poll) the same child.
            try:
                await asyncio.wait_for(process.wait(), .1)
                self._observe_lifecycle("reap_confirmed")
                return
            except asyncio.TimeoutError:
                pass
            if process.returncode is None:
                try:
                    process.terminate()  # Only this explicitly spawned child handle.
                except ProcessLookupError:
                    pass
            try:
                await asyncio.wait_for(process.wait(), 1)
            except asyncio.TimeoutError:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await process.wait()
            self._observe_lifecycle("reap_confirmed")
        finally:
            transport.close()
            ticket=getattr(self,"_lifecycle_ticket",None)
            if ticket is not None:
                try:ticket.finish("closed")
                except BaseException:pass

    async def _exchange(self, operation, payload, fields):
        process = self._process
        self._sequence += 1
        sequence = self._sequence
        process.stdin.write(envelope(dict(id=sequence, operation=operation, **fields), payload))
        await process.stdin.drain()
        head_size, payload_size = sizes(await process.stdout.readexactly(8))
        owner = self._separation_owner
        if owner is not None:
            from .separation_media import MAX_PAIR_BYTES, MEDIA_HEADER_LIMIT
            if head_size > MEDIA_HEADER_LIMIT or payload_size > MAX_PAIR_BYTES:
                raise ValueError('separation response exceeds private wire bounds')
        if payload_size and (owner is None or operation != 'process'):
            raise ValueError('worker response cannot contain raw sensor payload')
        response = parse_header(await process.stdout.readexactly(head_size))
        if type(response) is not dict or type(response.get('id')) is not int or response.get('id') != sequence or type(response.get('ok')) is not bool:
            raise ValueError('invalid or mismatched worker response')
        if not response['ok']:
            kind = response.get('error_type', 'unknown')
            if not isinstance(kind, str) or len(kind) > 128:
                kind = 'unknown'
            raise RuntimeError(f'{self.profile} worker failed ({kind}); verify profile configuration, '
                               'local checkpoints and optional dependencies in the selected Python environment')
        result = response.get('result')
        if owner is not None and operation == 'process':
            from .separation_media import unpack_pair
            if set(response) != {'id', 'ok', 'result', 'separation'} or not payload_size:
                raise ValueError('separation process requires a typed pair response')
            # Exact expected size is checked before reading any waveform bytes.
            binding = fields['separation_binding']
            if payload_size != binding['sample_count'] * 8:
                raise ValueError('separation response length differs from request')
            binary = await process.stdout.readexactly(payload_size)
            return unpack_pair(result, response['separation'], binary, binding, self.spec), binding
        if 'separation' in response:
            raise ValueError('unnegotiated separation metadata')
        if owner is not None and operation != 'start' and result is not None:
            raise ValueError('whole-window separation controls cannot publish media')
        return self.spec.validate(result) if operation in ('process', 'process_regions', 'finish', 'endpoint') and result is not None else result

    async def _request(self, operation, *, payload=b'', timeout=None, input_frame=None, evaluation_operation_id=None, **fields):
        owner = self._separation_owner if input_frame is not None else None
        request_revision = None
        if owner is not None:
            # Admission precedes the lock: a request that starts before/during
            # reset must not become a fresh request merely by waiting behind it.
            owner._check()
            if owner.reset_pending:
                raise PermissionError('separation request invalidated by provider reset')
            request_revision = owner.revision
        async with self._lock:
            if self.closed or self._process is None:
                raise RuntimeError('provider worker is closed; explicitly reopen its stream')
            if owner is not None and (owner.reset_pending or request_revision != owner.revision):
                raise PermissionError('separation request invalidated by provider reset')
            if input_frame is not None:
                fields['frame'], payload = encode_frame(input_frame)
                if self._separation_owner is not None:
                    from .separation_media import request_binding
                    owner = self._separation_owner
                    fields['separation_binding'] = request_binding(fields['frame'], payload, config=self.config,
                        generation=owner.generation, config_id=self.spec.version,
                        owner_revision=owner.revision)
                    owner._bind(fields['separation_binding'])
                    owner._reserve(fields['separation_binding']['sample_count'],
                        request_bytes=len(payload) if 'visual_policy' in self.config else 0)
            ticket=None
            observer=getattr(self,"_evaluation_observer",None)
            if observer is not None:
                try:
                    ticket=observer.reserve_work("worker_operation",dict(worker_id=self._worker_id,
                        operation=operation,parent_operation_id=evaluation_operation_id))
                except BaseException:
                    if self._separation_owner is not None:self._separation_owner._release()
                    if operation != "start":await _finish(asyncio.create_task(self._stop()))
                    raise
            async def invoke():
                if ticket is not None:ticket.event("operation_started")
                return await self._exchange(operation,payload,fields)
            exchange = asyncio.create_task(invoke())
            outcome="error"

            try:
                # Explicit ownership preserves caller cancellation even when the
                # response completes in the same loop turn (including Python 3.10).
                done, _ = await asyncio.wait({exchange}, timeout=self.timeout if timeout is None else timeout)
                if not done:
                    raise asyncio.TimeoutError()
                result = exchange.result()
                outcome="completed"
                if self._separation_owner is not None and operation == 'process':
                    pair, binding = result
                    if binding['owner_revision'] != self._separation_owner.revision:
                        return None  # Reset intent superseded this complete window.
                    observation = self.spec.validate(pair.observation)
                    observation['window'].update(capture_start_ms=binding['capture_start_ms'],
                        capture_end_ms=binding['capture_end_ms'], sample_rate=48000,
                        sample_count=binding['sample_count'], padded_sample_count=pair.padded_samples,
                        trimmed_padding_samples=pair.padded_samples-binding['sample_count'],
                        source_evidence=binding)
                    observation['media'] = self._separation_owner._admit(pair, binding)
                    return self.spec.validate(observation)
                return result
            except BaseException as exc:
                outcome="cancelled" if isinstance(exc,asyncio.CancelledError) else "timeout" if isinstance(exc,asyncio.TimeoutError) else "error"
                exchange.cancel()
                await _finish(asyncio.gather(exchange, return_exceptions=True))
                if operation != "start":await _finish(asyncio.create_task(self._stop()))
                raise
            finally:
                if self._separation_owner is not None:
                    self._separation_owner._release()
                if ticket is not None:
                    try:ticket.finish(outcome)
                    except BaseException:
                        if operation != "start":await _finish(asyncio.create_task(self._stop()))
                        raise

    async def process(self, frame, *, evaluation_operation_id=None):
        return await self._request('process', input_frame=frame,evaluation_operation_id=evaluation_operation_id)

    async def process_regions(self, frame, evidence, *, evaluation_operation_id=None):
        evidence = bounded_json(evidence)
        return await self._request('process_regions', input_frame=frame, evidence=evidence,evaluation_operation_id=evaluation_operation_id)

    async def endpoint(self, *, evaluation_operation_id=None):
        """Serialize a nonterminal utterance endpoint; unsupported workers fail closed."""
        return await self._request('endpoint',evaluation_operation_id=evaluation_operation_id)

    async def finish(self, *, evaluation_operation_id=None):
        return await self._request('finish',evaluation_operation_id=evaluation_operation_id)

    async def reset(self, *, evaluation_operation_id=None):
        if self._separation_owner is not None:
            self._separation_owner._reset()
            try:
                await self._request('reset',evaluation_operation_id=evaluation_operation_id)
                self._separation_owner._reset_complete()
            except BaseException:
                await _finish(asyncio.create_task(self._stop()))
                raise
        else:
            await self._request('reset',evaluation_operation_id=evaluation_operation_id)

    def revoke_separation_media(self):
        """Synchronous lifecycle intent, before waiting for a stalled worker."""
        if self._separation_owner is not None:
            self._separation_owner.close()

    def discard_separation_result(self, result):
        if self._separation_owner is not None and isinstance(result, dict):
            self._separation_owner.discard(result.get('media', {}).get('reference'))

    async def _close(self, evaluation_operation_id=None):
        try:
            if self._separation_owner is not None and self._lock.locked():
                # Revocation already happened synchronously in close(). Do not
                # wait behind a potentially stalled whole-window inference to
                # send its close request; terminate only this owned child.
                await _finish(asyncio.create_task(self._stop()))
                return
            if not self.closed:
                await self._request('close', timeout=2,evaluation_operation_id=evaluation_operation_id)
        finally:
            await _finish(asyncio.create_task(self._stop()))

    async def close(self, *, evaluation_operation_id=None):
        if self._separation_owner is not None:
            self._separation_owner.close()
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close(evaluation_operation_id))
        try:
            await asyncio.shield(self._close_task)
        except asyncio.CancelledError:
            await _finish(self._close_task)
            raise

"""Explicitly owned local subprocesses isolate incompatible ML dependencies."""
import asyncio
import os
from pathlib import Path

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

    @classmethod
    async def open(cls, python, profile, config, *, startup_timeout=120, inference_timeout=30):
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
        worker_environment = os.environ.copy()
        if profile in ('qwen3.asr', 'qwen3.asr.endpoint'):
            worker_environment.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                                      HF_HUB_DISABLE_TELEMETRY='1', VLLM_NO_USAGE_STATS='1',
                                      VLLM_ENABLE_V1_MULTIPROCESSING='0')
        spawn = asyncio.create_task(asyncio.create_subprocess_exec(python, '-u', '-m',
            'humansense.provider_worker', stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL, env=worker_environment))
        try:
            self._process = await asyncio.shield(spawn)
            response = await self._request('start', profile=profile, config=self.config,
                                           timeout=startup_timeout)
            if response != {'spec': spec_value(self.spec)}:
                raise RuntimeError('worker capability/provenance differs from selected contract')
            return self
        except BaseException:
            if self._process is None:
                try:
                    self._process = await _finish(spawn)
                except Exception:
                    pass
            await _finish(asyncio.create_task(self._stop()))
            raise

    async def _stop(self):
        self.closed = True
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
        finally:
            transport.close()

    async def _exchange(self, operation, payload, fields):
        process = self._process
        self._sequence += 1
        sequence = self._sequence
        process.stdin.write(envelope(dict(id=sequence, operation=operation, **fields), payload))
        await process.stdin.drain()
        head_size, payload_size = sizes(await process.stdout.readexactly(8))
        if payload_size:
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
        return self.spec.validate(result) if operation in ('process', 'finish', 'endpoint') and result is not None else result

    async def _request(self, operation, *, payload=b'', timeout=None, input_frame=None, **fields):
        async with self._lock:
            if self.closed or self._process is None:
                raise RuntimeError('provider worker is closed; explicitly reopen its stream')
            if input_frame is not None:
                fields['frame'], payload = encode_frame(input_frame)
            try:
                return await asyncio.wait_for(self._exchange(operation, payload, fields),
                                              self.timeout if timeout is None else timeout)
            except BaseException:
                await _finish(asyncio.create_task(self._stop()))
                raise

    async def process(self, frame):
        return await self._request('process', input_frame=frame)

    async def endpoint(self):
        """Serialize a nonterminal utterance endpoint; unsupported workers fail closed."""
        return await self._request('endpoint')

    async def finish(self):
        return await self._request('finish')

    async def reset(self):
        await self._request('reset')

    async def _close(self):
        try:
            if not self.closed:
                await self._request('close', timeout=2)
        finally:
            await _finish(asyncio.create_task(self._stop()))

    async def close(self):
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        try:
            await asyncio.shield(self._close_task)
        except asyncio.CancelledError:
            await _finish(self._close_task)
            raise

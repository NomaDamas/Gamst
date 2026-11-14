"""Supervisor-owned PG canonical worker and immutable staging lifetime."""
import asyncio
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from time import monotonic as _now

from .pg_artifact import validate_manifest,_encoded,_json
from .pg_canonical import validate_canonical
from .sam_audio_artifact import _absolute,_regular_path
from .streams import _finish

REQUEST_LIMIT=8*1024**2


async def _deadline(task,seconds):
    if seconds<=0:raise TimeoutError('PG owned worker deadline exceeded')
    done,_=await asyncio.wait({task},timeout=seconds)
    if not done:raise TimeoutError('PG owned worker deadline exceeded')
    return task.result()


class PGCanonicalProcess:
    """Fixed local model worker. No generation interface until a later delivery."""
    @classmethod
    async def open(cls,python,graph,graph_sha,lock,lock_sha,canonical,canonical_sha,*,stage_root,startup_timeout=120):
        if type(startup_timeout) not in (int,float) or not .01<=startup_timeout<=600:
            raise ValueError('PG startup timeout must be0.01..600 seconds')
        graph=validate_manifest(graph,expected_sha256=graph_sha)
        canonical=validate_canonical(canonical,expected_sha256=canonical_sha)
        if canonical['graph_manifest_sha256']!=graph_sha or canonical['runtime_lock_sha256']!=lock_sha:
            raise ValueError('canonical graph/runtime binding mismatch')
        lock_raw=_encoded(lock)
        if len(lock_raw)>256*1024 or hashlib.sha256(lock_raw).hexdigest()!=lock_sha:
            raise ValueError('frozen worker lock mismatch')
        lock=_json(lock_raw,256*1024)
        root=_absolute(stage_root);_regular_path(root,directory=True)
        source=_absolute(graph['bundle_root'])
        if root==source or root.is_relative_to(source) or source.is_relative_to(root):
            raise ValueError('source and staging directory overlap')
        request=_encoded(dict(graph=graph,graph_sha=graph_sha,lock=lock,lock_sha=lock_sha,
            canonical=canonical,canonical_sha=canonical_sha))
        if len(request)>REQUEST_LIMIT:raise ValueError('PG request exceeds bound')
        total=sum(row['size'] for row in canonical['shards'])
        if canonical['budgets']['disk_bytes']<total+len(request):
            raise ValueError('planned disk budget excludes supervisor request bytes')
        if shutil.disk_usage(root).free<total+len(request):raise ValueError('insufficient free staging disk')
        python=str(Path(python).expanduser().absolute())
        if not os.path.isfile(python) or not os.access(python,os.X_OK):raise ValueError('existing Python worker required')
        self=cls();self._process=None;self._close_task=None;self._read_task=None;self.closed=False
        self._stage=Path(tempfile.mkdtemp(prefix='.gamst-pg-',dir=root))
        self._owned={'request.json',*(f'shard-{i:02d}.safetensors' for i in range(len(canonical['shards'])))}
        spawn=None
        try:
            path=self._stage/'request.json'
            with path.open('xb') as stream:stream.write(request)
            os.chmod(path,0o400)
            env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
            deadline=_now()+startup_timeout
            spawn=asyncio.create_task(asyncio.create_subprocess_exec(python,'-u','-m','humansense.pg_canonical_worker',
                '--request',str(path),'--sha256',hashlib.sha256(request).hexdigest(),
                stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.DEVNULL,
                env=env,limit=65536))
            self._process=await _deadline(spawn,deadline-_now())
            if _now()>=deadline:raise TimeoutError('PG owned worker deadline exceeded')
            self._read_task=asyncio.create_task(self._process.stdout.readline())
            raw=await _deadline(self._read_task,deadline-_now())
            if _now()>=deadline:raise TimeoutError('PG owned worker deadline exceeded')
            if not raw or len(raw)>65536:raise ValueError('missing/bounded PG worker response required')
            response=_json(raw,65536)
            if type(response) is not dict or response.get('status')!='loaded':
                raise RuntimeError('PG worker unavailable: '+str(response.get('reason','startup_failed') if type(response) is dict else 'invalid_response'))
            expected={'status','canonical_sha256','graph_schema_sha256','planned_storage_bytes',
                'trained_identity_verified','reference_parity_verified','runtime_compatibility_verified',
                'resource_admission','actual_acceptance'}
            if (set(response)!=expected or response['canonical_sha256']!=canonical_sha or
                response['graph_schema_sha256']!=canonical['graph_schema_sha256'] or
                type(response['planned_storage_bytes']) is not int or not 0<response['planned_storage_bytes']<=512*1024**3 or
                any(response[name] is not False for name in ('trained_identity_verified','reference_parity_verified','runtime_compatibility_verified')) or
                response['resource_admission']!='unavailable' or response['actual_acceptance']!='incomplete'):
                raise ValueError('PG loaded response differs from frozen contract')
            self.receipt=response
            return self
        except BaseException as initial:
            acquisition_error=None
            if self._process is None and spawn is not None:
                try:self._process=await _finish(spawn)
                except BaseException as error:acquisition_error=error
            # Exec failure must not bypass cleanup merely because acquisition
            # re-raises. A late acquired handle is used ONLY to stop/reap it.
            try:await _finish(asyncio.create_task(self.close()))
            except BaseException as cleanup:raise initial from cleanup
            if acquisition_error is not None and acquisition_error is not initial:
                raise initial from acquisition_error
            raise

    async def _stop(self):
        process=self._process
        if process is not None:
            if process.stdin:process.stdin.close()
            transport=process._transport
            for descriptor in (1,2):
                pipe=transport.get_pipe_transport(descriptor)
                if pipe is not None:pipe.close()
            if process.returncode is None:
                try:process.terminate()
                except ProcessLookupError:pass
            waiter=asyncio.create_task(process.wait())
            try:
                try:await _deadline(waiter,1)
                except TimeoutError:
                    if process.returncode is None:
                        try:process.kill()
                        except ProcessLookupError:pass
                    await waiter
            finally:
                if process.returncode is not None:transport.close()
            if process.returncode is None:raise RuntimeError('PG worker not reaped; preserve staging')
            self._process=None
        if self._read_task is not None:
            await asyncio.gather(self._read_task,return_exceptions=True)
        entries=list(self._stage.iterdir()) if self._stage.exists() else []
        if any(path.name not in self._owned or path.is_symlink() or not path.is_file() for path in entries):
            raise RuntimeError('unexpected staging entry preserved after worker reap')
        for path in entries:path.unlink()
        if self._stage.exists():self._stage.rmdir()
        self.closed=True

    async def close(self):
        if self._close_task is None:self._close_task=asyncio.create_task(self._stop())
        try:await asyncio.shield(self._close_task)
        except asyncio.CancelledError as initial:
            try:await _finish(self._close_task)
            except BaseException as cleanup:raise initial from cleanup
            raise

"""Turn a role installed by setup into a Pipeline module.

A profile role (``{"model", "provider_profile", "python", "config"}``, written by
:func:`gamst.crew.setup.apply_plan`) becomes one of two module kinds the
:class:`gamst.pipeline.Pipeline` already runs:

* an isolated worker: :class:`WorkerModule` around a
  :class:`gamst.ProcessProvider` in the role's own interpreter, a *provider* module
  (``spec`` + ``process(frame) -> value``) committed with ``push_observation``;
* an in-process candidate (registry ``in_process``, e.g. ``vision.rotation``, which reads
  the records other modules commit): the named class built on the caller's runtime, a
  *plugin* module committed like a registered plugin. One that runs its own native process
  (``owns_process``, e.g. the MediaPipe graph runner) gets a bounded lane like a worker.

``ProfileRuntime`` (``gamst mcp --profile``, ``gamst dashboard --profile``) and recipes
build their modules here, so a Crew-launched worker is one Pipeline module however it is
started.
"""
from __future__ import annotations

import asyncio
import importlib

from ..plugins import InputFrame
from ..private_payload import PRIVATE_ONLY
from .registry import ROLES


class RoleUnavailable(RuntimeError):
    """A profile role that cannot become a module; the reason is reported, never dropped."""


def bounded_image(image, max_pixels=1_048_576, max_side=2048):
    """Workers accept at most ~1 MP; boxes are normalized so the scale does not matter."""
    height, width = image.shape[:2]
    scale = min(1., (max_pixels / (height * width)) ** .5, max_side / max(height, width))
    if scale >= 1:
        return image
    import cv2
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


class WorkerModule:
    """One owned ProcessProvider worker as a Pipeline provider module."""
    owns_process = True

    def __init__(self, worker):
        if getattr(worker, 'private', False):
            raise PermissionError('a private worker answers its owning module only')
        self.worker, self.spec = worker, worker.spec
        self.name, self.version = self.spec.provider, self.spec.version
        self.modalities = (self.spec.modality,)

    async def process(self, frame):
        return await self._ask(self.worker, frame)

    async def _ask(self, worker, frame):
        data = frame.data
        if self.spec.modality == 'video' and hasattr(data, 'shape'):
            data = bounded_image(data)
        return await worker.process(InputFrame(data, frame.modality, frame.timestamp_ms,
                                                    subject_id=frame.subject_id, sensor=frame.sensor,
                                                    sample_rate=frame.sample_rate))

    async def reset(self):
        # The SensorStreams rule: only a stateful provider holds state across frames.
        if self.spec.stateful:
            await self.worker.reset()

    async def close(self):
        await self.worker.close()


class ReplicatedWorker(WorkerModule):
    """N workers of one stateless spec sharing a lane's frames; the Pipeline commits them in capture order.

    ``process`` hands the frame to a free worker, so each frame runs once. A worker that failed (its process
    ended) is not used again; the frame it had is reported by the Pipeline as failed, like any other.
    """

    def __init__(self, workers):
        specs = [worker.spec for worker in workers]
        if not workers or any(spec != specs[0] for spec in specs):
            raise ValueError('replicas are workers of one spec')
        if specs[0].stateful:
            raise ValueError(f'{specs[0].id} is stateful: its frames cannot be shared between replicas')
        super().__init__(workers[0])
        self.workers, self.replicas = list(workers), len(workers)
        self._free = asyncio.Queue()
        for worker in workers:
            self._free.put_nowait(worker)

    async def process(self, frame):
        if all(worker.closed for worker in self.workers):
            raise RuntimeError('every replica has ended')
        worker = await self._free.get()
        while worker.closed:
            worker = await self._free.get()
        try:
            return await self._ask(worker, frame)
        finally:
            self._free.put_nowait(worker)   # a closed one is skipped above

    async def reset(self):
        pass   # stateless

    async def close(self):
        await asyncio.gather(*(worker.close() for worker in self.workers))


def candidate_for(role, model):
    """The registry candidate behind a profile role, or None (custom or unknown model)."""
    entry = ROLES.get(role)
    return None if entry is None else entry.candidates.get(model)


def _import(path):
    module, _, name = path.partition(':')
    return getattr(importlib.import_module(module), name)


async def open_private_worker(python, provider_profile, config, *, startup_timeout=120, inference_timeout=30):
    """Start one worker for the module that owns it: ``query`` only (:mod:`gamst.private_payload`)."""
    from ..process_provider import ProcessProvider
    return await ProcessProvider.open(python, provider_profile, config, startup_timeout=startup_timeout,
                                      inference_timeout=inference_timeout, private=True)


async def open_worker(python, provider_profile, config, *, startup_timeout=120, inference_timeout=30, replicas=1):
    """Start one isolated worker (``replicas``: N of a stateless spec) and wrap it as a :class:`WorkerModule`."""
    from ..process_provider import ProcessProvider, capability
    if type(replicas) is not int or not 1 <= replicas <= 16:
        raise ValueError('replicas must be an integer from 1 to 16')
    if replicas > 1 and capability(provider_profile, config).stateful:
        raise ValueError(f'{provider_profile} is stateful: its frames cannot be shared between replicas')
    workers = []
    try:
        for _ in range(replicas):
            workers.append(await ProcessProvider.open(python, provider_profile, config,
                                                      startup_timeout=startup_timeout,
                                                      inference_timeout=inference_timeout))
    except BaseException:
        for worker in workers:
            await worker.close()
        raise
    return WorkerModule(workers[0]) if replicas == 1 else ReplicatedWorker(workers)


async def role_module(human, role, entry, *, config=None, python=None, startup_timeout=120,
                      inference_timeout=30):
    """Build the Pipeline module for one profile role.

    Returns ``dict(module, spec, modality, capability, in_process)``; ``spec`` is None for a
    plugin module. ``config`` / ``python`` override the profile's values (a recipe narrows
    labels or the device). Raises :class:`RoleUnavailable` with the reason.
    """
    if ROLES.get(role) is not None and ROLES[role].kind != 'module':
        raise RoleUnavailable(f'{ROLES[role].kind} role: its packages serve the Pipeline process; it runs no model')
    config = dict(entry.get('config') or {}) if config is None else dict(config)
    chosen = candidate_for(role, entry.get('model'))
    if chosen is not None and chosen.in_process:
        plugin = _import(chosen.in_process)(human, **config)
        spec = getattr(plugin, 'spec', None)
        return dict(module=plugin, spec=None, modality=tuple(plugin.modalities)[0],
                    capability=None if spec is None else spec.id, in_process=True,
                    owns_process=bool(getattr(plugin, 'owns_process', False)))
    from ..providers.model_workers import PROFILES as MODEL_PROFILES
    if entry.get('provider_profile') in MODEL_PROFILES | PRIVATE_ONLY:
        raise RoleUnavailable('answers one recipe module (enrolled reference and consent needed); '
                              'its embeddings or audio are never context')
    python = python or entry.get('python')
    if not python or not entry.get('provider_profile'):
        raise RoleUnavailable('no isolated worker in profile; register its in-process adapter explicitly')
    module = await open_worker(python, entry['provider_profile'], config, startup_timeout=startup_timeout,
                               inference_timeout=inference_timeout)
    return dict(module=module, spec=module.spec, modality=module.spec.modality, capability=module.spec.id,
                in_process=False)

"""Turn a role installed by setup into a Pipeline module.

A profile role (``{"model", "provider_profile", "python", "config"}``, written by
:func:`gamst.crew.setup.apply_plan`) becomes one of two module kinds the
:class:`gamst.pipeline.Pipeline` already runs:

* an isolated worker: :class:`WorkerModule` around a
  :class:`gamst.ProcessProvider` in the role's own interpreter, a *provider* module
  (``spec`` + ``process(frame) -> value``) committed with ``push_observation``;
* an in-process candidate (registry ``in_process``, e.g. ``vision.rotation``, which reads
  the records other modules commit): the named class built on the caller's runtime, a
  *plugin* module committed like a registered plugin.

``ProfileRuntime`` (``gamst mcp --profile``, ``gamst dashboard --profile``) and recipes
build their modules here, so a Crew-launched worker is one Pipeline module however it is
started.
"""
from __future__ import annotations

import importlib

from ..plugins import InputFrame
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

    def __init__(self, worker):
        self.worker, self.spec = worker, worker.spec
        self.name, self.version = self.spec.provider, self.spec.version
        self.modalities = (self.spec.modality,)

    async def process(self, frame):
        data = frame.data
        if self.spec.modality == 'video' and hasattr(data, 'shape'):
            data = bounded_image(data)
        return await self.worker.process(InputFrame(data, frame.modality, frame.timestamp_ms,
                                                    subject_id=frame.subject_id, sensor=frame.sensor,
                                                    sample_rate=frame.sample_rate))

    async def reset(self):
        # The SensorStreams rule: only a stateful provider holds state across frames.
        if self.spec.stateful:
            await self.worker.reset()

    async def close(self):
        await self.worker.close()


def candidate_for(role, model):
    """The registry candidate behind a profile role, or None (custom or unknown model)."""
    entry = ROLES.get(role)
    return None if entry is None else entry.candidates.get(model)


def _import(path):
    module, _, name = path.partition(':')
    return getattr(importlib.import_module(module), name)


async def open_worker(python, provider_profile, config, *, startup_timeout=120, inference_timeout=30):
    """Start one isolated worker and wrap it as a :class:`WorkerModule`."""
    from ..process_provider import ProcessProvider
    worker = await ProcessProvider.open(python, provider_profile, config, startup_timeout=startup_timeout,
                                        inference_timeout=inference_timeout)
    return WorkerModule(worker)


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
                    capability=None if spec is None else spec.id, in_process=True)
    if config.get('landmarks'):
        raise RoleUnavailable('keypoint output is for the recipe module that asked for it; it is never context')
    python = python or entry.get('python')
    if not python or not entry.get('provider_profile'):
        raise RoleUnavailable('no isolated worker in profile; register its in-process adapter explicitly')
    module = await open_worker(python, entry['provider_profile'], config, startup_timeout=startup_timeout,
                               inference_timeout=inference_timeout)
    return dict(module=module, spec=module.spec, modality=module.spec.modality, capability=module.spec.id,
                in_process=False)

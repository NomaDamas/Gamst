"""Run the models chosen by agentic setup inside one Gamst runtime.

GamstCrew (``gamst setup`` / MCP ``plan_setup`` + ``apply_setup``) writes
``~/.config/gamst/profile.json``::

    {"version": 1, "hardware": {...},
     "roles": {"vision.detect": {"model": "rfdetr.nano", "provider_profile": "rfdetr.nano",
                                 "python": "/.../envs/rfdetr/bin/python", "config": {...}},
               "vision.rotation": {"model": "gamst.rotation", "provider_profile": "vision.rotation",
                                   "python": null, "config": {}}}}

``ProfileRuntime.open`` turns every role into an ordinary engine plugin, so any
intake (``push_video``/``push_audio``, MCP ``start_capture``, the dashboard)
feeds the selected models with no separate submission path:

* a role with a worker Python runs in its own ProcessProvider subprocess behind
  a QueuedPlugin, so a slow model never blocks capture; its validated output is
  committed with ``push_observation`` under the worker's capability;
* ``vision.rotation`` is model-free and runs in-process as RotationPlugin,
  reading the latest detector observations that the line above committed.

Roles without a worker (direct in-process adapters) are reported as skipped,
never silently dropped.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import uuid

from .capabilities import SensorMetadata
from .plugins import InputFrame

PROFILE_VERSION = 1


def default_profile_path() -> Path:
    return Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')) / 'gamst' / 'profile.json'


def read_profile(path=None) -> dict:
    path = Path(path or default_profile_path()).expanduser()
    value = json.loads(path.read_text(encoding='utf-8'))
    if value.get('version') != PROFILE_VERSION or type(value.get('roles')) is not dict:
        raise ValueError(f'{path}: not a version {PROFILE_VERSION} GAMST profile; run gamst setup')
    for role, entry in value['roles'].items():
        if type(entry) is not dict or not isinstance(entry.get('model'), str) or type(entry.get('config', {})) is not dict:
            raise ValueError(f'{path}: role {role} must name a model and a config object')
    return value


class WorkerPlugin:
    """Engine plugin around one owned ProcessProvider worker (use behind QueuedPlugin)."""

    def __init__(self, human, role, worker):
        self.human, self.role, self.worker = human, role, worker
        self.spec = worker.spec
        self.name, self.version = f'{role}:{self.spec.provider}', self.spec.version
        self.modalities = (self.spec.modality,)
        self._epoch = uuid.uuid4().hex
        self._sequences = {}
        human.register_capability(self.spec)

    async def process(self, frame):
        sequence = self._sequences.get(frame.subject_id, -1) + 1
        self._sequences[frame.subject_id] = sequence
        sensor = SensorMetadata(source_id='gamst-profile', sensor_id=self.role, sequence=sequence,
                                capture_sequence=sequence, coordinate_frame='image', stream_epoch=self._epoch)
        data = frame.data
        if self.spec.modality == 'video':
            data = _bounded_image(data)
        value = await self.worker.process(InputFrame(data, frame.modality, frame.timestamp_ms,
                                                     subject_id=frame.subject_id, sensor=sensor,
                                                     sample_rate=frame.sample_rate))
        if value is not None:
            await self.human.push_observation(self.spec.id, value, sensor=sensor,
                                              subject_id=frame.subject_id, timestamp_ms=frame.timestamp_ms)
        return []

    def reset(self):
        self._sequences.clear()
        self._epoch = uuid.uuid4().hex

    async def close(self):
        await self.worker.close()


def _bounded_image(image, max_pixels=1_048_576, max_side=2048):
    """Workers accept at most ~1 MP; boxes are normalized so the scale does not matter."""
    height, width = image.shape[:2]
    scale = min(1., (max_pixels / (height * width)) ** .5, max_side / max(height, width))
    if scale >= 1:
        return image
    import cv2
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


class ProfileRuntime:
    """Registered plugins for one profile; ``close`` stops every worker it started."""

    def __init__(self, human, profile):
        self.human, self.profile = human, profile
        self.plugins, self.status = [], {}

    @classmethod
    async def open(cls, human, path=None, *, roles=None, startup_timeout=120, inference_timeout=30,
                   queue_capacity=1):
        from .process_provider import ProcessProvider
        from .providers.rotation import RotationPlugin
        from .realtime import QueuedPlugin
        self = cls(human, read_profile(path))
        selected = self.profile['roles'] if roles is None else {r: self.profile['roles'][r] for r in roles}
        try:
            # Workers first: rotation reads their committed observations.
            for role, entry in selected.items():
                if role == 'vision.rotation':
                    continue
                if not entry.get('python') or not entry.get('provider_profile'):
                    self.status[role] = dict(model=entry['model'], state='skipped',
                                             reason='no isolated worker in profile; register its in-process adapter explicitly')
                    continue
                worker = await ProcessProvider.open(entry['python'], entry['provider_profile'], entry.get('config', {}),
                                                    startup_timeout=startup_timeout, inference_timeout=inference_timeout)
                plugin = QueuedPlugin(human, WorkerPlugin(human, role, worker), capacity=queue_capacity)
                human.register(plugin)
                self.plugins.append(plugin)
                self.status[role] = dict(model=entry['model'], state='running', capability=worker.spec.id)
            if 'vision.rotation' in selected:
                entry = selected['vision.rotation']
                plugin = RotationPlugin(human, **entry.get('config', {}))
                human.register(plugin)
                self.plugins.append(plugin)
                self.status['vision.rotation'] = dict(model=entry['model'], state='running', capability=plugin.spec.id)
        except BaseException:
            await self.close()
            raise
        return self

    async def close(self):
        plugins, self.plugins = self.plugins, []
        for plugin in plugins:
            if plugin in self.human.plugins:
                self.human.plugins.remove(plugin)
            close = getattr(plugin, 'close', None)
            if close:
                result = close()
                if hasattr(result, '__await__'):
                    await result

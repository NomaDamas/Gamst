"""Run the models chosen by setup inside one Gamst runtime, as one Pipeline.

Setup (``gamst setup``, MCP ``plan_setup`` + ``apply_setup``, ``gamst recipe setup``; all
:mod:`gamst.crew.setup`) writes ``~/.config/gamst/profile.json``::

    {"version": 1, "hardware": {...},
     "roles": {"vision.detect": {"model": "rfdetr.nano", "provider_profile": "rfdetr.nano",
                                 "python": "/.../envs/rfdetr/bin/python", "config": {...}},
               "vision.rotation": {"model": "gamst.rotation", "provider_profile": "vision.rotation",
                                   "python": null, "config": {}}}}

``ProfileRuntime.open`` builds a :class:`gamst.pipeline.Pipeline` named ``profile`` whose
host-fed inputs (``camera``, ``microphone``) are fed by the engine's own intake
(``push_video``/``push_audio``, MCP ``start_capture``, the dashboard) through
``PipelineIntake``; each role becomes a module via :func:`gamst.crew.modules.role_module`, the
same helper recipes use:

* a role with a worker Python runs in its own ProcessProvider subprocess, in a bounded Pipeline
  lane (lossless, bounded) so a slow model never blocks capture; its validated output
  is committed with ``push_observation`` under the worker's capability;
* an in-process candidate (``vision.rotation``) runs on the caller's runtime, reading the latest
  detector observations the workers committed.

Roles that cannot run (input roles, direct in-process adapters with no worker) are reported as
skipped, never silently dropped. The Pipeline is the runtime's ``human.pipeline``, so MCP
``get_pipeline`` / ``configure_pipeline_module`` see and switch the roles.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

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


def role_python(role, model, path=None):
    """The interpreter setup recorded for ``model`` in ``role``, or None when it is not set up."""
    try:
        entry = read_profile(path)['roles'].get(role)
    except FileNotFoundError:
        return None
    return entry.get('python') if entry and entry['model'] == model else None


class ProfileRuntime:
    """The profile's roles as one Pipeline fed by the engine's intake; ``close`` stops every worker."""

    INPUTS = {'video': 'camera', 'audio': 'microphone'}

    def __init__(self, human, profile):
        self.human, self.profile = human, profile
        self.pipeline, self.plugins, self.status = None, [], {}

    @classmethod
    async def open(cls, human, path=None, *, roles=None, startup_timeout=120, inference_timeout=30,
                   queue_capacity='auto'):
        from .crew.modules import RoleUnavailable, candidate_for, role_module
        from .pipeline import Pipeline
        self = cls(human, read_profile(path))
        selected = self.profile['roles'] if roles is None else {r: self.profile['roles'][r] for r in roles}
        self.pipeline = Pipeline(human, 'profile')

        def in_process(role):
            chosen = candidate_for(role, selected[role]['model'])
            return bool(chosen is not None and chosen.in_process)

        try:
            # Workers first: an in-process module (rotation) reads the records they commit.
            for role in sorted(selected, key=in_process):
                entry = selected[role]
                try:
                    built = await role_module(human, role, entry, startup_timeout=startup_timeout,
                                              inference_timeout=inference_timeout)
                except RoleUnavailable as exc:
                    self.status[role] = dict(model=entry['model'], state='skipped', reason=str(exc))
                    continue
                input_id = self.INPUTS.get(built['modality'])
                if input_id is None:
                    close = getattr(built['module'], 'close', None)
                    if close:
                        result = close()
                        if hasattr(result, '__await__'):
                            await result
                    self.status[role] = dict(model=entry['model'], state='skipped',
                                             reason=f"no engine intake for {built['modality']} input")
                    continue
                self._input(input_id, built['modality'])
                # A worker runs in its own bounded lane so a slow model never blocks capture.
                self.pipeline.add_module(role, built['module'], consumes=[input_id], spec=built['spec'],
                                         queue=None if built['in_process'] and not built.get('owns_process')
                                         else queue_capacity)
                self.status[role] = dict(model=entry['model'], state='running', capability=built['capability'])
        except BaseException:
            await self.close()
            raise
        return self

    def _input(self, input_id, modality):
        from .pipeline import PipelineIntake
        if input_id in self.pipeline.inputs:
            return
        self.pipeline.add_input(input_id, modality=modality)
        plugin = PipelineIntake(self.pipeline, input_id, modality)
        self.human.register(plugin)
        self.plugins.append(plugin)

    async def close(self):
        plugins, self.plugins = self.plugins, []
        for plugin in plugins:
            if plugin in self.human.plugins:
                self.human.plugins.remove(plugin)
        pipeline, self.pipeline = self.pipeline, None
        if pipeline is not None:
            await pipeline.close()

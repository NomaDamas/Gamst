# Handoff: feat/absorb-gamstcrew (GamstCrew absorbed into GAMST)

> Superseded in part by [handoff-feat-one-env-per-model.md](handoff-feat-one-env-per-model.md): the owner decisions
> below are settled there. Correction: the demo venv holds both opencv-python 4.11.0.86 and opencv-contrib-python
> 5.1.0.95, so a `current` apply would not have added a second OpenCV; it already has two.

Worktree `.local-run/wt-crew`, branch `feat/absorb-gamstcrew` (from `feat/target-speaker-keyword-and-pose-count`
@ 30b705f). Owner decision: "흡수해. 분리하는 건 의미없다. 세팅까지 gamst에서." The owner archives the Crew repo
later; agents do not. Not pushed; no PR.

Full design and the dedupe table: [docs/crew.md](crew.md). This file is the state for whoever continues.

## Done
- `gamst/crew/`: registry, setup, os_profiles, diagnostics, resources, host_limits, manager, process_runtime,
  launcher, transport, shared_memory_transport, media_bus, media_source, parallel, parallel_worker, packet,
  `profiles/` (example manifest + locks, package data in pyproject), `modules.py` (new).
- One setup path: `gamst setup` (in-process `gamst.crew.setup.main`), MCP `plan_setup`/`apply_setup` in
  `gamst/mcp.py`, `gamst recipe setup` (crew plan/apply), `gamst demo rotation --fetch-models`
  (`download_weight`). The registry is the only place with weight URLs/SHA256.
- One runtime: `ProfileRuntime` builds a Pipeline (`PipelineIntake` + `WorkerModule` lanes); recipes build their
  worker modules with the same helpers. Pipeline gained `add_module(queue=N)`, `reset()`, `drain()`.
- Dropped: `gamst/crew_bridge.py` + test; Crew's `TemporalAligner` (helpers moved onto `gamst.temporal`),
  `CoreProviderRuntime`, duplicate checksum code, standalone Crew MCP server; recipes' own pip/fetch/checkout,
  `POSE_ASSETS`; the demo's own downloader.
- Crew tests ported: 12 `tests/test_crew_*.py` files, plus new `test_crew_modules.py` (worker as a Pipeline
  module, lanes, intake); 78 tests in the 13 files. `test_recipes.py` / `test_profile_runtime.py` adjusted to the new plan
  shape (same semantics: coin-flip = rotation + yoloe + efficienttam, rotation = no weights, person-action-count
  = MediaPipe pose).
- Docs: `docs/crew.md`, `docs/crew/*.md` (Crew's docs, paths updated), README setup section, notes on history
  docs that still name GamstCrew commits/issues.

## Test results (2026-11-25, `~/.cache/gamst/demo/venv/bin/python`, `PYTHONPATH=.`)
- `python -m unittest discover -s tests -q`: Ran 1240, errors 15, skipped 38. The 15 are the pre-existing ones:
  test_stereo_context 7, test_stereo_metric 6, test_sam_audio_native 2. Baseline before this branch: Ran 1163,
  same 15 errors, skipped 40 (the 2 removed skips were the optional-Crew bridge tests).
- `test_crew_*.py`: 78 tests OK.
- Manual end-to-end (not in the suite): `gamst recipe setup person-action-count` with installs patched downloaded
  `pose_landmarker_full.task` for real through crew (9398198 bytes, sha 5134a3…); `ProfileRuntime` on that
  profile ran the `mediapipe.body.video` worker in a Pipeline lane (2 processed, 1 dropped of 3 pushed frames);
  `gamst mcp --profile` answered `get_pipeline` with the profile Pipeline.

## Not done / not verifiable here
- `ParallelCrew`, cgroup host limits, container launches, venv-mode apply over the network: code faithful to Crew,
  not run (see docs/crew.md, "Not verified locally").
- A real `isolation="current"` apply was not run into the shared demo venv on purpose: the registry pins
  `opencv-python>=4.8,<5` while that venv has OpenCV 5.1 from another package, so pip would add a second OpenCV.

## Owner decisions needed
1. `BASE` worker requirement is now `gamst-sdk==<running version>` (0.6.0) instead of Crew's fixed 0.5.0. A
   `venv` apply needs that version on PyPI (or an index/checkout); confirm 0.6.0 will be published, or choose
   another pin.
2. OpenCV pin `opencv-python>=4.8,<5` (Crew's) vs the OpenCV 5.1 actually in the demo venv: keep `<5` or relax.
3. Recipe weights default to one flat directory `~/.cache/gamst/models/yoloe` (the directory `gamst demo rotation`
   already reads), now also holding pose files. Rename the directory (one re-download) or keep it.
4. `gamst setup` installs with `isolation="venv"` (one venv per environment, Crew behaviour) while
   `gamst recipe setup` uses `isolation="current"` (this interpreter). Both write the same profile.

## Adding a registry role (for the agent adding speaker verification, target speech, ASR words, person match,
Sapiens2, semantic candidates incl. Qwen3.8-Flash-Next)
Add a `Role(...)` to `ROLES` in `gamst/crew/registry.py` with `_candidate(...)` entries (packages, devices, RAM/VRAM/disk
MiB + source, license, verification text, `_weight(name, url, sha256, size, size_source)`; `automatic=False` for
manual bundles; `in_process="module:Class"` only for modules that must read committed records). If apply must turn
files into a worker config, add a branch to `_role_config` in `gamst/crew/setup.py`. A hosted endpoint candidate
(like `qwen3.5.endpoint`) has no weights and `config.endpoint`. Recipe parts then name `role=` / `candidate=` in
`gamst/recipes/parts.py`. Nothing else downloads weights.

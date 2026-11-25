# Handoff: feat/absorb-gamstcrew (GamstCrew absorbed into GAMST)

Worktree: `.local-run/wt-crew`, branch `feat/absorb-gamstcrew` (from `feat/target-speaker-keyword-and-pose-count` @ 30b705f).
Owner decision: "흡수해. 분리하는 건 의미없다. 세팅까지 gamst에서." Crew repo is archived by the owner later (not by agents). No push, no PR.

## Status

WIP checkpoint. See the "Done" / "Left" lists; the final state is updated at the end of the work.

### Done
- `gamst/crew/` created from GamstCrew (registry, setup, os_profiles, diagnostics, resources, host_limits,
  manager, process_runtime, launcher, transport, shared_memory_transport, media_bus, media_source,
  parallel, parallel_worker, packet, profiles/runtime-profiles.example.json + locks/).
- `gamst/crew/modules.py`: `WorkerModule` (ProcessProvider worker as a Pipeline provider module),
  `role_module` (profile role -> Pipeline module), `RoleUnavailable`.
- `gamst/pipeline.py`: `add_module(queue=N)` bounded lanes, `reset()`, `drain()`, `PipelineIntake`
  (engine intake -> host-fed Pipeline input).
- `gamst/profile.py`: `ProfileRuntime` now builds one Pipeline named `profile` (no WorkerPlugin/QueuedPlugin).
- `gamst setup` runs `gamst.crew.setup.main` in-process; MCP `plan_setup` / `apply_setup` in `gamst/mcp.py`.
- `gamst recipe setup` = `make_plan(roles=..., dependencies=False, isolation="current")` + `apply_plan(models_dir=...)`.
  `recipes.parts` keeps only build + sampling + `role`/`candidate`. `POSE_ASSETS`, `_fetch`, extras/pip/sources gone.
- `gamst/demos/rotation.py`: `MODEL_ASSETS` / `REFINE_ASSETS` are views of the registry; `fetch_models` uses `download_weight`.
- Registry additions: `mediapipe.pose_full` (float16/1, sha 5134a3…, 9398198 bytes measured), heavy URL moved
  from `float16/latest` to `float16/1` (sha 64437af…, 30664242 bytes measured), role `vision.video_segmentation`
  (`efficienttam.ti_512`), input roles `input.video` (opencv), `input.microphone` (sounddevice), candidate
  field `in_process`, role field `kind`, helpers `candidate()` / `weight_assets()`.

### Left (at checkpoint)
- Port Crew tests to `tests/test_crew_*.py`; add normalize_timestamp/sequence_gap to `gamst.temporal.TemporalAligner`.
- Drop `gamst/crew_bridge.py` + `tests/test_crew_bridge.py`; update `scripts/verify_cross_repo_checksum.py`.
- `docs/crew.md` (dedupe table), README setup section, docs grep for GamstCrew.
- Full test run and final numbers here.

## Adding a registry role (for the agent adding speaker verification, target speech, ASR words, person match, Sapiens2, semantic candidates)
Add a `Role(...)` entry to `ROLES` in `gamst/crew/registry.py` with `_candidate(...)` entries (packages, devices,
RAM/VRAM/disk MiB, resource source, license, verification text, `_weight(name, url, sha256, size, size_source)`).
If apply must turn downloaded files into a worker config, add a branch to `_role_config` in `gamst/crew/setup.py`.
A recipe part then names `role=` / `candidate=` in `gamst/recipes/parts.py`. Nothing else downloads weights.

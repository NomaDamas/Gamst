# gamst.crew: setup and runtime orchestration inside GAMST

GamstCrew used to be a separate package (`gamstcrew`) that `gamst setup` shelled out to. It is now
the `gamst.crew` subpackage (owner decision, 2026-11-25: "흡수해. 분리하는 건 의미없다. 세팅까지
gamst에서."). There is nothing extra to install; Crew is stdlib, plus numpy for the shared media
rings only.

## One setup path

| Entry point | Code |
|---|---|
| `gamst setup plan --goals … [--override role=model] [--json]` / `gamst setup apply PLAN.json` | `gamst.crew.setup.main`, run in-process |
| MCP `plan_setup` / `apply_setup` (`gamst mcp`) | `gamst.mcp._setup_tool` -> `make_plan` / `apply_plan` |
| `gamst recipe setup <recipe> [--isolation venv\|current] [--models DIR] [--profile P] [--dry-run]` | `gamst.recipes.plan` -> `make_plan(roles=…, dependencies=False)` -> `apply_plan` |
| `gamst demo rotation --fetch-models` | `gamst.crew.setup.download_weight` for the registry's `yoloe.26s` and `efficienttam.ti_512` files |

The role registry (`gamst/crew/registry.py`) is the only place that names a weight URL and its
SHA256, and `download_weight` is the only downloader (atomic `.partial` + rename, SHA256 always,
byte size too where it is pinned, an existing file with another digest is refused and never
overwritten). `gamst/demos/rotation.py`'s `MODEL_ASSETS` / `REFINE_ASSETS` are views of the
registry; `gamst/recipes/parts.py` keeps only how to build a Pipeline module, the sampling rule and
the `role` / `candidate` it needs.

### Environments: one per model environment

The registry entry decides where its packages go; a saved plan cannot move them (`_validate_plan`
recomputes the environments and compares `environment`, `python`, `source`, `in_process` and
`provider_profile` with the registry).

| Registry entry | Installed into | Why |
|---|---|---|
| model candidate | `<prefix>/venvs/<environment>` | it runs as an isolated `ProcessProvider` worker there |
| `kind="input"` / `kind="view"` role, `in_process` candidate (`gamst.rotation`) | the running interpreter | the Pipeline process itself imports it (decoding, microphone, HUD drawing, rotation math) |

This is the default for `gamst setup`, MCP and `gamst recipe setup`. `isolation="current"`
(`--isolation current`) is the explicit alternative: everything into the running interpreter, with a
warning when that would mix two OpenCV distributions.

- **Shared environment.** `yoloe.26s` and `efficienttam.ti_512` share `yoloe-efficienttam`, because the
  after-take coin count (`python -m gamst.takes refine|refine-server`) loads both in one process.
  `gamst recipe run coin-flip` starts that process with the environment's Python from the profile, and
  asks for setup when the two entries point at different interpreters.
- **GAMST in a worker environment.** From a source checkout it is installed in place
  (`pip install -e <checkout>`; the plan records the path and the git commit, and a plan made for another
  path or commit is refused). From an installed release it is `gamst-sdk==<version>`, from a git install the
  same commit.
- **OpenCV.** mediapipe pulls in opencv-contrib-python and ultralytics opencv-python, which overwrite each
  other's `cv2`. `registry.cv2_distributions` knows both, and a test asserts that no environment holds two.

`make_plan(roles={role: model}, dependencies=False)` plans exactly a recipe's roles: a recipe composes its own
modules, so `vision.rotation` does not pull in `vision.detect` for the fan recipe.

### Weights: one root

`~/.cache/gamst/models/<model>/<file>`, `GAMST_MODELS` overrides the root. `gamst setup`,
`gamst recipe setup`, `gamst demo rotation` and `gamst.takes` all use it
(`setup.models_root` / `setup.model_directory`). Nothing is written flat, because names such as `config.json`
and `model.safetensors` collide across models.

`--models DIR` replaces the root. An existing flat directory (for example one filled by an older version) still
works for reading: it is used for a model only when it holds all of that model's files and `DIR/<model>` does not;
apply also requires their SHA256 to match. It is never moved or written to. Weights from the former default
`~/.cache/gamst/models/yoloe` can be read with `--models ~/.cache/gamst/models/yoloe`.

Apply writes `~/.config/gamst/profile.json` (`$XDG_CONFIG_HOME` honoured, the same path
`gamst.profile` reads). `gamst recipe run` reads its roles' entries (weight paths, source checkout,
worker Python) from that profile and says `run: gamst recipe setup <recipe>` when one is missing.

## One runtime

`ProfileRuntime` (`gamst mcp --profile`, `gamst dashboard --profile`) now builds a
`gamst.pipeline.Pipeline` named `profile`: host-fed `camera` / `microphone` inputs, fed by a
`PipelineIntake` engine plugin, so `push_video` / `push_audio`, MCP `start_capture` and the
dashboard reach it unchanged. Each role becomes a module through
`gamst.crew.modules.role_module`, the same helper recipes use:

- a worker role -> `WorkerModule` (a `ProcessProvider` in the role's interpreter, frames bounded to
  about 1 MP) added with `queue=1`, a bounded Pipeline lane so a slow model never blocks capture
  (this replaced `QueuedPlugin` + `WorkerPlugin`);
- a registry candidate with `in_process` (`gamst.rotation`) -> that class on the caller's runtime,
  added after the workers because it reads the records they commit;
- input roles and roles without a worker are reported as `skipped` with the reason.

Because the Pipeline is `human.pipeline`, MCP `get_pipeline` / `configure_pipeline_module` now
show and switch profile roles too.

## What was kept, replaced and dropped

| GamstCrew module | Now | Why |
|---|---|---|
| `registry.py` | kept: `gamst.crew.registry` | GAMST had no role/candidate registry. Added `mediapipe.pose_full`, role `vision.video_segmentation` (`efficienttam.ti_512`), input roles `input.video` / `input.microphone`, `Role.kind`, `RoleCandidate.in_process`, `candidate()`, `weight_assets()`. Crew's fixed `gamst-sdk==0.5.0` base is now the caller's own GAMST (above). Role `view.hud` (Pillow and OpenCV for the HUD). |
| `setup.py` | kept: `gamst.crew.setup` | the one plan/apply; additions above, `verified_file`, EfficientTAM in `_role_config`. |
| `mcp.py` (standalone stdio server) | replaced by `gamst.mcp` tools `plan_setup` / `apply_setup` | one MCP server; the tools call the same functions. |
| `os_profiles.py`, `diagnostics.py`, `resources.py`, `host_limits.py` | kept unchanged | GAMST lacked them. |
| `manager.py`, `process_runtime.py`, `launcher.py` | kept | supervised packet runtimes for any argv (e.g. a container command), restarts, reservations, host limits; GAMST's `ProcessProvider` only runs GAMST provider workers. |
| `packet.py` | kept, trimmed: `gamst.crew.packet` | checksum, canonical JSON, `PacketIntegrityCounter` and `verify_packet_checksum` were copies of `gamst.temporal`; now imported from it. Only the one-event `EventPacket`, `RuntimeDescriptor`, `HealthSnapshot` remain (GAMST's `TemporalPacket` is a multi-event batch). |
| `temporal.py` (`TemporalAligner`) | replaced by `gamst.temporal.TemporalAligner` | same clock fit, revisions and relation candidates. Crew's two extra helpers, `normalize_timestamp` and `sequence_gap`, were moved onto GAMST's aligner. |
| `core_provider.py` (`CoreProviderRuntime`) | replaced by `gamst.ProcessProvider` + `gamst.crew.modules.WorkerModule` | it only wrapped `ProcessProvider` start/stop/health; a worker is now a Pipeline module. |
| `transport.py` (`JsonlUnixServer`, `jsonl_call`) | kept: `gamst.crew.transport` | bounded local socket seam GAMST lacked; `jsonl_call` is the generic client. |
| `shared_memory_transport.py` | kept | GAMST lacked it. |
| `media_bus.py`, `media_source.py` | kept | cross-process shared-memory rings on one media clock; GAMST's `media.MediaRouter` / `streams.SensorStreams` are in-process queues, not a replacement. |
| `parallel.py`, `parallel_worker.py` | kept | the one-GPU all-models run; worker module path is now `gamst.crew.parallel_worker`, and `ParallelCrew(crew_root=…)` is gone (`gamst_root` defaults to this checkout). |
| `runtime-profiles.example.json`, `locks/` | kept: `gamst/crew/profiles/` | example launch manifests and their lock files. |
| GAMST `gamst/crew_bridge.py` (`CrewBridge`) | dropped | it sent packets over a Unix socket to an out-of-process Crew; Crew is in-process now, and `gamst.crew.transport.jsonl_call` covers a generic socket client. |
| GAMST `recipes.setup` pip/`_fetch`/`_checkout`, `parts.POSE_ASSETS`, `Part.extras/pip/models/sources` | dropped | duplicated Crew's job; recipe setup is Crew's plan/apply. |
| GAMST `demos.rotation` own downloader | dropped | `fetch_models` calls `download_weight`. |

Crew's other docs moved to [docs/crew/](crew/): [agentic setup](crew/agentic-setup.md),
[runtime manager](crew/runtime-manager.md), [OS profiles](crew/os-profiles.md),
[packet contract](crew/packet-contract.md), [resource allocation](crew/resource-allocation.md).

## Not verified locally

- `ParallelCrew` / `parallel_worker` need `nvidia-smi` and one CUDA GPU; not run here. The code is
  the Crew code with the module path changed.
- Linux cgroup host limits, container (`docker run`) launches and the `linux-cuda` profile are
  exercised only by the ported unit tests (fake cgroup roots, plain subprocesses).
- `isolation="venv"` apply against the network (venv creation + pip + downloads) is not run by
  the tests; apply is tested with installs patched and with local `file://` weights.
- The refine process and the MediaPipe worker were not started in their new environments: no environment
  was created, so the dependency sets are declared, not resolved by pip here.
- `mediapipe.pose_full` / `pose_heavy`, `yoloe.26s`, MobileCLIP2 and EfficientTAM sizes and digests were
  measured on files already on disk.

## Adding a role

Add a `Role(...)` to `ROLES` in `gamst/crew/registry.py` with `_candidate(...)` entries
(packages, devices, RAM/VRAM/disk in MiB with their source, license, verification text, and
`_weight(name, url, sha256, size, size_source)`); mark `automatic=False` when apply must refuse
it. If apply has to turn downloaded files into a worker config, add a branch to `_role_config` in
`gamst/crew/setup.py`. A recipe part then names `role=` / `candidate=` in `gamst/recipes/parts.py`.
Nothing else downloads weights.

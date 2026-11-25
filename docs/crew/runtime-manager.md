# Runtime manager

`RuntimeManager` registers explicit adapters with descriptors, starts them, queues bounded packets, drops oldest work with visible counters, isolates adapter errors and exposes health. Provider environments are not merged into one virtualenv.

## What belongs to GAMST and what belongs to Crew

The boundary is intentional. GAMST owns the provider-facing contract:
capability validation, provider wire framing, inference timeouts and
observation validation. Crew owns deployment orchestration: selecting a
profile, starting and stopping a worker, bounding queues and transport,
capturing health, and accounting for declared resources. A provider worker
chosen by setup is a GAMST `ProcessProvider` wrapped as a Pipeline module
(`gamst.crew.modules.WorkerModule`), so provider semantics stay in GAMST.

This makes Crew a runtime supervisor, not a second provider SDK. A successful
preflight or `ready` descriptor says that metadata, compatibility and declared
local hashes passed their checks. It does not establish model quality, weight
availability or acceptance on the target host.

## Subprocess is the default boundary

`ProcessRuntime` starts the exact `argv` supplied by the caller with
`asyncio.create_subprocess_exec()`. The child has an independent interpreter,
import graph, stdout/stderr pipes and crash boundary. Crew validates readiness,
passes bounded `EventPacket` messages, drains stderr, enforces line limits and
records failures. A host subprocess shares the host kernel and filesystem, so
it is process isolation rather than a container security boundary.

The same path can supervise a containerized worker when the supplied command is
the deployment's container launcher (for example, a `docker run` command).
Crew does not install Docker, build images, download model weights or invent
that command; the selected deployment profile and operator provide it.

## When Docker is appropriate

The declarative profile catalog marks `linux-cuda` as
`docker+nvidia-toolkit+triton`. Use a container for that profile when the
provider needs a pinned Linux userspace, CUDA/NVIDIA/Triton integration,
system libraries or image-level reproducibility that the host Python runtime
does not provide. The container still presents the same readiness and packet
contract to Crew.


On macOS, Docker Desktop adds a Linux virtual machine because containers need
a Linux kernel. The VM's RAM/CPU allocation, image startup and host filesystem
sharing add overhead, and camera/microphone access needs a host bridge or
supported integration. Apple GPU/MPS is generally unavailable inside the Linux
container; it is not a general passthrough path. For a small Apple Silicon
perception worker, a
native host subprocess is therefore the normal profile. On native Linux there
is no Docker Desktop Linux VM; Docker's value is instead the pinned userspace
and NVIDIA Container Toolkit/CUDA boundary used by the `linux-cuda` profile.
The manager keeps these as deployment facts rather than treating Docker as a
universal performance rule.

The macOS Apple Silicon, Linux CPU, Windows CPU/DirectML and Jetson profiles
describe their native runtime families. They may be launched as host
subprocesses or by a deployment-specific wrapper when that wrapper is part of
the tested profile. `preflight_profile()` checks declared profile metadata and
artifacts; it does not claim that Docker or any accelerator is installed.

## Why providers are not merged into one container

One universal image would couple providers with incompatible Python versions,
native ABIs, CUDA/TensorRT revisions, CPU-only or GPU-only assumptions, and
different artifact or license locks. It would also create one large failure
and upgrade domain: an image change or native-library crash could affect
unrelated providers. Separate runtimes keep queue pressure, restart policy,
resource reservations and provenance attributable to the provider that owns
them.

Providers can share an image when they have the same tested dependency lock,
accelerator contract and lifecycle. The default is one provider runtime, or a
deliberately grouped compatible set, per profile. The packet and transport
contracts stay stable whether those runtimes are host processes or containers.

Submission failures are recorded in bounded error history and exposed as
`crash_count`. Deployments may opt into a bounded restart policy with
`RuntimeManager(restart_limit=N, restart_backoff_s=...)`; the default is zero
automatic restarts so a failed provider remains visible for operator action.

Applications that own more than one provider can use `await manager.close_all()`
at shutdown, or use `RuntimeManager` as an async context manager.  Both paths
close each registered runtime through the same cancellation and reservation
release path as `close(name)`, and leave the manager registry empty.  This is
the lifecycle boundary used by launchers and test fixtures; it does not claim
that a host-specific provider process passed hardware acceptance.
Subprocess stderr is drained continuously and its bounded tail is included in
process health diagnostics, preventing a noisy provider from blocking its IPC
pipe. These signals are diagnostics, not host acceptance evidence.

Pass a `ResourceAllocator` to the manager to reserve declared CPU cores, RAM,
GPU count and VRAM before a provider starts. A registration that would overcommit
any dimension is rejected atomically; startup failure and normal close release
the exact reservation token. `resource_telemetry()` exposes bounded reservation
and rejection counters. This is accounting evidence supplied by deployment
discovery. Host enforcement is separate and explicit: pass a
`HostLimitController` to `register_process`; the default `accounting` mode is
non-mutating, Linux cgroup-v2 strict mode fails closed when it cannot attach
CPU/RAM limits, and unsupported macOS/Windows/GPU cases are surfaced in the
health report rather than claimed as enforced. Real host acceptance remains a
separate gate.

## Launcher preflight

`gamst.crew.launcher.preflight_profile(profiles_path, name)` returns a bounded,
JSON-safe report before a child process is started. The report includes the
selected profile name, the host descriptor used for compatibility, each
declared lock/artifact check (`verified`, `missing`, `mismatch` or
`unmeasured`), a `ready`/`blocked` status and an actionable reason. Callers can
show this report in install or run diagnostics without spawning a provider.

`launch_profile()` uses the same preflight path and fails closed on a blocked
report. Unknown profiles still raise `KeyError`; compatibility, malformed
metadata and declared-file failures are retained in the report and surfaced as
`RuntimeError` before manager registration. A `ready` report means only that
metadata, host compatibility and declared local hashes passed; it does not
claim model weights, benchmark acceptance or target-host evidence.

## Provider workers as Pipeline modules

GamstCrew's `CoreProviderRuntime` (start/stop/health around a `ProcessProvider`) is gone:
`gamst.crew.modules.role_module` turns a profile role into a `WorkerModule` (or, for an
`in_process` registry candidate, the named class on the caller's runtime), and the Pipeline runs
it in a bounded lane (`add_module(queue=N)`). `ProfileRuntime` builds exactly that Pipeline.
GAMST retains capability validation, provider wire framing, timeouts and observation validation.

## Unix transport

`gamst.crew.transport.JsonlUnixServer` provides a bounded newline-delimited JSON
Unix-domain socket seam. Every request receives a bounded `request_id`, and
the server echoes it on successful responses and handler errors; a mismatched
ID is rejected before the caller accepts the response. Duplicate JSON keys and
non-finite values are rejected. Requests and responses are size-checked;
provider payloads remain validated by the packet contract. Handler and caller
timeouts (`handler_timeout_s` and `timeout_s`) bound stalled providers. Each
timeout must be a finite numeric value strictly greater than zero and no larger
than 600 seconds (`MAX_TIMEOUT_S`); non-finite values, booleans, values that
underflow to zero, and arbitrarily large integers are rejected before a socket
or provider is started. The 600-second ceiling applies to the configured
handler/caller timeout only. Socket close waits are a separate lifecycle
operation and are not covered by this input bound.
The configured line bound is validated (minimum 256 bytes so a correlated
protocol error remains representable), and error responses are bounded too.
`SharedPacket`
provides an optional bounded local CPU transport with descriptor and packet
checksum verification before dispatch; the JSONL socket path remains an
explicit fallback. GPU/CUDA zero-copy and target-host performance are not
claimed by this contract.

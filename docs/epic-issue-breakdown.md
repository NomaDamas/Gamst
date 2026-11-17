# GAMST Epic and sub-issue execution map

This document is the issue-level execution contract for `docs/approved-plan.md`.
It is written before the remaining implementation work. A sub-issue is complete
only when its code, focused tests, runtime evidence, documentation, and required
review are attached to the issue. A parent epic cannot be closed while any child
is pending or blocked.

The complete requirement-to-issue traceability table is maintained in
[`docs/requirements-traceability.md`](requirements-traceability.md). This map is
the execution status view; the traceability document is the completeness gate.

The three repositories remain separate:

| Repository | Responsibility | Existing branch |
|---|---|---|
| GAMST | Core context, temporal fusion, physics, SDK and cross-repository contract | `feat/temporal-contract` |
| GamstCrew | Isolated runtime manager, lifecycle, IPC and OS profiles | `feat/runtime-pr` |
| Gamst-bench | Frozen manifests, adapters, measurements and scoring | `feat/temporal-metrics` |

## GitHub hierarchy status

The documented crosswalk is now also represented by GitHub's native
parent/sub-issue metadata wherever the issues live in the same repository.
GAMST #294 is the top-level parent for the GAMST runtime/transport/temporal,
memory, benchmark and exact-SHA children; the nested implementation children
are attached under #296, #297, #298, #299, #300, #315 and #318. GamstCrew uses
#11 as its repository-local Epic parent for Crew #2–#10 because the previously
referenced Crew #1 is a pull request and cannot be a GitHub sub-issue parent.
Gamst-bench #1 is the parent for bench #3–#9. Cross-repository relationships
(GAMST↔Crew↔bench) remain explicit links in the issue bodies and the
requirements crosswalk because GitHub does not allow a parent issue in another
repository. The hierarchy is tracking structure only; an open child still
requires code, focused tests, runtime evidence and review before closure.

## Epic A — GamstCrew runtime platform

Parent: `A-runtime-platform` (GAMST issue umbrella #294; implementation in
GamstCrew). The missing resource child is tracked in GAMST #315 and Crew #8.

| Child | Scope | Acceptance gate | Status |
|---|---|---|---|
| A1 lifecycle | start/stop/readiness/health, owned process reap | repeated start/stop, readiness timeout and clean reap tests | implemented and focused-tested; production/host stress evidence remains open |
| A2 crash isolation | child failure, cancellation and restart policy | failed provider cannot corrupt siblings; restart releases handles and queues | bounded crash telemetry and opt-in restart policy implemented/tested; host stress evidence remains open |
| A3 registry and locks | provider descriptor, runtime lock/artifact/model/license hashes | launch refuses mismatched Python, lock, artifact or license metadata | metadata/Python/hash validation implemented/tested; real artifact/weight/license acceptance remains open |
| A4 resource allocation | CPU/RAM/VRAM/GPU reservation, admission, fairness and telemetry | overcommit is rejected; release after crash; telemetry is in benchmark ledger | GAMST #315; accounting foundation exists; host enforcement and stress evidence open (#329) |
| A5 provider integration | real GAMST Core provider bridge and revision pin | Crew CI runs the same pinned Core contract without optional-package skip | Core bridge and revision metadata implemented/tested; manager-wide close lifecycle and full manager-owned bench acceptance remain open (`f8aa970`, `08dedcd`) |
| A6 launcher | `gamst install`/`gamst run`, diagnostics and profile selection | deterministic local launch and actionable preflight errors | explicit source-checkout dispatch implemented and focused-tested (`86c9e0f`); Crew profile selection/preflight is implemented on `feat/runtime-pr`; target-host launch and profile acceptance remain open |
| A8 Playground/BYOK | OpenAI-compatible endpoint, model/key configuration, non-stream/SSE response and compact Context/evidence inspection | user-provided MyProxy endpoint/model/key; raw media remains excluded; errors are bounded and visible; no credentials or successful call are fabricated | GAMST #311; endpoint/credentials blocker remains |
| A9 Playground capture lifecycle | operation-aware local start/stop snapshots and media ownership | stale paused snapshot cannot stop a current capture; genuine current stop still releases tracks | GAMST #332 under #311; implemented with revision/transition guards and focused browser helper regression |
| A7 motion-provider compatibility | preserve temporal clock metadata at OpenEB, separation-media, PG physical and SAM visual native boundaries | legacy eight-key and canonical clock-aware SensorMetadata normalize safely; unknown fields reject; hosted motion-provider job passes all required suites | GAMST #326; current `c950a57` runs `43898470715` and `43899185401` both passed the motion-provider job and the Python 3.10–3.13 matrix. The historical single timeout remains recorded as diagnosed CI timing evidence; no assertion or production timeout relaxation. |

## Epic B — context transport

Parent: `B-context-transport` (GAMST #295/#299). The checksum child is GAMST
#316 and the Crew transport hardening child is Crew #9.

| Child | Scope | Acceptance gate | Status |
|---|---|---|---|
| B1 packet contract | bounded packet, capture/receive/process times, source clock, evidence and revisions | schema round-trip and unknown-field rejection across all three repositories | foundation |
| B2 Unix transport | bounded JSONL or MessagePack request/response | size checked before allocation, correlation ID and timeout enforced | bounded JSONL, size checks, duplicate/nonfinite rejection, bounded protocol errors, request correlation and handler/client/connect/close timeouts implemented/tested (Crew `6bc27e8`, `b803a41`, `85c8fff`; Core bridge `5b39250`, `471e8f8`, `1ec1bbe`); full cross-repository execution/review open |
| B3 backpressure | bounded queues, drop accounting and cancellation | no unbounded queue; every drop has a reason and sequence range | foundation |
| B4 checksum | canonical bytes, algorithm/version and SHA-256 integrity | tampered packet rejected and counted; legacy version behavior explicit | Core/Crew/bench verification, exact-byte probe, and live Crew→Core→bench adapter proof pass; host enforcement and independent production review remain open (#331) |
| B5 shared memory | Linux/CUDA optional zero-copy path | same packet digest/timestamps as socket path; fallback remains valid | bounded `SharedPacket` transport with exact length/SHA-256 verification and close/unlink ownership implemented and portable-tested in Crew `e44ce77`; CUDA zero-copy and host performance acceptance remain open (#327) |

## Epic C — temporal and multimodal fusion

Parent: `C-temporal-fusion` (GAMST #297/#306/#312). Cross-modal relation
materialization is GAMST #317.

| Child | Scope | Acceptance gate | Status |
|---|---|---|---|
| C1 clocks | source/monotonic/wall conversion, offset and drift calibration | calibration fixture reports sample count, residual and uncertainty | Core calibrated offset/drift engine and Crew `normalize_timestamp()` implemented/tested (`8ad9788`, `fdf5322`); cross-runtime calibration remains open |
| C2 intervals | event start/end, overlap and AV offset compensation | known offset/drift fixtures reproduce expected relations | interval/overlap and provenance-bearing relation candidates implemented/tested in Core and Crew; cross-runtime metrics remain open |
| C3 revisions | current, last-seen, stale/fresh and revision lineage | late revision cannot overwrite newer state; evidence lineage remains intact | revision-aware current/last-seen state implemented/tested |
| C4 internal cutoff invariant | query-time evidence consistency only | post-cutoff evidence cannot support an answer; this is not user-input truncation | internal evidence watermark implemented/tested; bench integration evidence remains open |
| C5 contradictions | cross-modal disagreement and abstention | contradiction is emitted with both evidence IDs and uncertainty | GAMST #328; Core candidate/abstention foundation `68cfb0d` and Crew abstaining candidate `fdf5322`; bench metrics and independent review open |
| C6 relations | overlap/causal/association relation materialization and metrics | relation schema preserves source clocks, calibration state and lineage | Core `TemporalContextEngine` materialization + calibrated overlap implemented and focused-tested; cross-runtime metrics/review remain open |

## Epic D — memory and context

Parent: `D-memory-context` (GAMST #318).

| Child | Scope | Acceptance gate | Status |
|---|---|---|---|
| D1 state layers | current state, recent timeline and episodic memory | bounded retention, expiry and provenance are independently inspectable | bounded layers, evidence index, byte eviction, and normalized Core envelope integration implemented/tested (`17e0d30`); long-running/provider acceptance remains open |
| D2 compression | loss-bounded history compression and delta updates | delta replay equals full snapshot within declared loss/byte bound | bounded delta generation/replay implemented and focused-tested; production compression policy remains open |
| D3 retrieval | evidence and source retrieval by time/entity/evidence ID | retrieval never returns unowned or post-cutoff evidence | bounded evidence retrieval with cutoff/entity/time filters implemented and focused-tested |
| D4 session resume | durable checkpoint, restart and clear/delete semantics | resumed state has identical lineage and no stale cross-session leakage | versioned lineage-checked checkpoint/resume plus bounded atomic `CheckpointStore` load/save/clear and Core envelope wiring implemented and focused-tested (`17e0d30`); cross-process/long-running acceptance remains open |
| D5 deterministic physics | bounded kinematics, trajectory hypotheses and physical uncertainty | held-out trajectory/contact evidence; calibrated claims only; synthetic futures isolated | deterministic `PhysicalState` provider emits bounded kinematics, covariance, trajectory hypotheses, contact/support/impact candidates and abstentions without calibration; focused provider tests pass; held-out/real-device acceptance remains open (GAMST #330) |

## Epic E — OS adaptation

Parent: `E-os-adaptation` (GAMST #298/#301/#302/#309/#314; Crew #10).

| Child | Scope | Acceptance gate | Status |
|---|---|---|---|
| E1 macOS Apple Silicon | MLX/MPS/ONNX/llama.cpp and CPU fallback | one-command lightweight profile with pinned revisions on target Mac | pending real run |
| E2 Linux CUDA | containers, CUDA, TensorRT/ORT/vLLM/Triton/Ray candidates | same packet contract on real target hardware | pending host |
| E3 Linux CPU | no-CUDA profile and native dependency diagnostics | profile refuses unsupported acceleration and completes CPU smoke | declarative profile and diagnostics implemented/focused-tested; target-host CPU smoke remains open (#323) |
| E4 Windows | CPU/ORT/DirectML/llama.cpp profile | clean install and provider compatibility report on Windows host | declarative profile and compatibility reporting implemented; Windows acceptance remains open (#324) |
| E5 Jetson/ARM | TensorRT/ORT/GStreamer/V4L2 edge profile | real Jetson acceptance; not inferred from macOS/Linux | declarative profile implemented; Jetson acceptance remains open (#325); Alpamayo2 Edge follow-up is tracked in #333 with children #334/#335/#336/#337/#338 |
| E6 compatibility matrix | available/installed/runtime-compatible/weight-ready/benchmark-accepted states | every provider reports all five states and license gate | explicit five-state catalog and install/native diagnostics implemented/tested (`8651d62`); real host/weight/license acceptance remains open |

## Epic F — benchmark and default selection

Parent: `F-benchmark` (Gamst-bench #1/#3–#9 and GAMST #293/#300/#307/#308).

| Child | Scope | Acceptance gate | Status |
|---|---|---|---|
| F1 manifest binding | frozen runtime/provider/model/license/OS/dependency/input manifest | measured run refuses missing or mismatched manifest digest | frozen digest/provenance and complete temporal-field declaration validation implemented/tested in bench (`e1976e1`); real measured manifests remain open |
| F2 temporal metrics | capture time, age, detection, cutoff, stale, drops, recovery and escalation | collector emits metrics and scorer consumes them | collector/scorer integration is implemented and focused-tested; matched real-run denominators remain open |
| F3 matched B/C/D/E | same held-out inputs, consumer, prompt and context contract; distinct E checkpoint | all arms have complete scored traces and denominators | unmeasured smoke only |
| F4 CostBench | Static, Fast Event, Cross-modal, Semantic Change and Resilience | event P/R/F1, association, latency, calls, resource, energy, cost/hour and Pareto report | exact-decimal aggregation and unknown-cost handling implemented in bench `4a489a0`, focused tests 5/5; measured host/resource/energy rows remain open (Gamst-bench #6) |
| F5 default selection | Mac measured comparison and rejected-candidate report | no default promotion without target-Mac evidence; CUDA cannot promote Mac | Gamst-bench #7; pending real host |
| F6 Linux acceptance | real Linux/CUDA matched runs | no synthetic or Mac substitution | pending real host |
| F7 runtime trace handoff | real Crew Core provider execution through the bench JSONL adapter | live manager-owned lifecycle, temporal packet/checksum/provenance retention and explicit failure categories | implemented by the live Crew→CoreProvider→GAMST Core→bench JSONL adapter; lifecycle/failure diagnostics and focused integration tests 7/7; target-host acceptance and independent production review remain open (Gamst-bench #9 under #1) |

## Exhaustive issue crosswalk

The following inventory prevents an older parent issue from becoming an
untracked implementation request. Parent issues remain open until their child
acceptance gates close; duplicate issue names are intentional cross-repository
facets of the same epic and are linked in the issue bodies.

| Epic | GAMST issue(s) | GamstCrew issue(s) | Gamst-bench issue(s) | Current gate |
|---|---|---|---|---|
| A runtime platform | #294, #296, #304, #305, #313–#315, #326, #329, #331 | #11, #8 | #1 | foundation exists; host enforcement, exact-SHA and production review open |
| B transport | #295, #299, #316, #327 | #11, #9 | #1 | packet foundation; checksum/shared-memory/transport hardening open |
| C temporal/fusion | #297, #306, #312, #317, #328 | #11 | #1, #3 | clocks/relation foundation; contradiction and cross-runtime evidence open |
| D memory/physics | #318–#322, #330 | #11 | #1 | memory and measured physics acceptance open |
| E OS adaptation | #298, #301, #302, #309, #314, #323–#325, #333–#338 | #11, #2–#7, #10, #13 | #1, #5–#7, #10 | real Mac/Linux/Windows/Jetson hosts open; Alpamayo2 Edge remains optional and unaccepted |
| F benchmark/default | #293, #300, #307, #308 | #11 | #1, #3–#9 | live adapter and CostBench code slices implemented; matched real B/C/D/E, resource/energy measurements, Mac/Linux comparison and default promotion remain open |

Optional research issues #285–#292 remain linked from README and are not used
to satisfy a required acceptance gate. They cover intermediate fusion,
retrieval embeddings, event-gated streaming, streaming arm E, physics/world
models, embodied handoff and optional provider acceptance.

The Alpamayo2 Edge provider follow-up is tracked as GAMST #333 with the
discovery/license child #337, Jetson profile child #334, temporal/physics
provenance child #336, Crew runtime child #338, and benchmark child #335.
These issues extend the approved edge/physics plan; they do not close the
existing Jetson or production acceptance gates without a real target host.
The corresponding repository-local integration issues are GamstCrew #13 and
Gamst-bench #10.

## Cross-epic order and blockers

1. Freeze this map and the packet/manifest revisions.
2. Resolve the GAMST motion-provider CI failure and record the child failure
   category before changing assertions. The OpenEB metadata compatibility child
   is tracked separately as A7; it must preserve the clock fields added by the
   temporal contract without pretending that OpenEB itself owns a worker clock.
3. Implement A/B/C contract slices in parallel only where revisions are already
   frozen; integrate them sequentially at the cross-repository boundary.
4. Implement D and E profiles after the transport contract is stable.
5. Run F only on real target hosts with held-out inputs and an independent
   assessment. The existing one-task Qwen3-VL smoke remains execution evidence,
   not quality, cost, readiness or default-selection evidence.
6. Required independent review remains a merge gate. If the approved reviewer
   model is unavailable, the epic remains blocked and no substitute approval is
   claimed.

## Explicit non-goals and invariants

- `query_cutoff` is an internal evidence-consistency invariant. It never truncates
  the user's query or silently removes context from a user-facing request.
- Two-dimensional uncalibrated video cannot claim numeric force, mass or precise
  3-D trajectory. Physics output must retain uncertainty and evidence lineage.
- Generated futures from a world model are labeled synthetic and never enter the
  real evidence store or benchmark ground truth.
- A fixture, smoke, synthetic profile or unmeasured metric cannot close a host,
  quality, cost, default or production-readiness gate.

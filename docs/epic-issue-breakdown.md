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

## Epic A — GamstCrew runtime platform

Parent: `A-runtime-platform` (GAMST issue umbrella #294; implementation in
GamstCrew). The missing resource child is tracked in GAMST #315 and Crew #8.

| Child | Scope | Acceptance gate | Status |
|---|---|---|---|
| A1 lifecycle | start/stop/readiness/health, owned process reap | repeated start/stop, readiness timeout and clean reap tests | foundation; pending production integration |
| A2 crash isolation | child failure, cancellation and restart policy | failed provider cannot corrupt siblings; restart releases handles and queues | pending |
| A3 registry and locks | provider descriptor, runtime lock/artifact/model/license hashes | launch refuses mismatched Python, lock, artifact or license metadata | partial |
| A4 resource allocation | CPU/RAM/VRAM/GPU reservation, admission, fairness and telemetry | overcommit is rejected; release after crash; telemetry is in benchmark ledger | GAMST #315; accounting foundation exists; host enforcement and stress evidence open (#329) |
| A5 provider integration | real GAMST Core provider bridge and revision pin | Crew CI runs the same pinned Core contract without optional-package skip | partial |
| A6 launcher | `gamst install`/`gamst run`, diagnostics and profile selection | deterministic local launch and actionable preflight errors | foundation |
| A7 motion-provider compatibility | preserve temporal clock metadata at OpenEB, separation-media, PG physical and SAM visual native boundaries | legacy eight-key and canonical clock-aware SensorMetadata normalize safely; unknown fields reject; hosted motion-provider job passes all required suites | GAMST #326; OpenEB `73b9e80`, PG `f8c35fa`, SAM/separation follow-up pending hosted rerun |

## Epic B — context transport

Parent: `B-context-transport` (GAMST #295/#299). The checksum child is GAMST
#316 and the Crew transport hardening child is Crew #9.

| Child | Scope | Acceptance gate | Status |
|---|---|---|---|
| B1 packet contract | bounded packet, capture/receive/process times, source clock, evidence and revisions | schema round-trip and unknown-field rejection across all three repositories | foundation |
| B2 Unix transport | bounded JSONL or MessagePack request/response | size checked before allocation, correlation ID and timeout enforced | partial; oversized-line hardening pending |
| B3 backpressure | bounded queues, drop accounting and cancellation | no unbounded queue; every drop has a reason and sequence range | foundation |
| B4 checksum | canonical bytes, algorithm/version and SHA-256 integrity | tampered packet rejected and counted; legacy version behavior explicit | missing |
| B5 shared memory | Linux/CUDA optional zero-copy path | same packet digest/timestamps as socket path; fallback remains valid | optional/pending; GAMST #327 |

## Epic C — temporal and multimodal fusion

Parent: `C-temporal-fusion` (GAMST #297/#306/#312). Cross-modal relation
materialization is GAMST #317.

| Child | Scope | Acceptance gate | Status |
|---|---|---|---|
| C1 clocks | source/monotonic/wall conversion, offset and drift calibration | calibration fixture reports sample count, residual and uncertainty | foundation; cross-runtime pending |
| C2 intervals | event start/end, overlap and AV offset compensation | known offset/drift fixtures reproduce expected relations | partial |
| C3 revisions | current, last-seen, stale/fresh and revision lineage | late revision cannot overwrite newer state; evidence lineage remains intact | partial |
| C4 internal cutoff invariant | query-time evidence consistency only | post-cutoff evidence cannot support an answer; this is not user-input truncation | partial; bench integration pending |
| C5 contradictions | cross-modal disagreement and abstention | contradiction is emitted with both evidence IDs and uncertainty | pending; GAMST #328 |
| C6 relations | overlap/causal/association relation materialization and metrics | relation schema preserves source clocks, calibration state and lineage | missing |

## Epic D — memory and context

Parent: `D-memory-context` (GAMST #318).

| Child | Scope | Acceptance gate | Status |
|---|---|---|---|
| D1 state layers | current state, recent timeline and episodic memory | bounded retention, expiry and provenance are independently inspectable | GAMST #319; partial |
| D2 compression | loss-bounded history compression and delta updates | delta replay equals full snapshot within declared loss/byte bound | GAMST #320; missing as epic-level gate |
| D3 retrieval | evidence and source retrieval by time/entity/evidence ID | retrieval never returns unowned or post-cutoff evidence | GAMST #321; partial |
| D4 session resume | durable checkpoint, restart and clear/delete semantics | resumed state has identical lineage and no stale cross-session leakage | GAMST #322; missing |
| D5 deterministic physics | bounded kinematics, trajectory hypotheses and physical uncertainty | held-out trajectory/contact evidence; calibrated claims only; synthetic futures isolated | GAMST #330; partial implementation, acceptance open |

## Epic E — OS adaptation

Parent: `E-os-adaptation` (GAMST #298/#301/#302/#309/#314; Crew #10).

| Child | Scope | Acceptance gate | Status |
|---|---|---|---|
| E1 macOS Apple Silicon | MLX/MPS/ONNX/llama.cpp and CPU fallback | one-command lightweight profile with pinned revisions on target Mac | pending real run |
| E2 Linux CUDA | containers, CUDA, TensorRT/ORT/vLLM/Triton/Ray candidates | same packet contract on real target hardware | pending host |
| E3 Linux CPU | no-CUDA profile and native dependency diagnostics | profile refuses unsupported acceleration and completes CPU smoke | GAMST #323; missing |
| E4 Windows | CPU/ORT/DirectML/llama.cpp profile | clean install and provider compatibility report on Windows host | GAMST #324; missing |
| E5 Jetson/ARM | TensorRT/ORT/GStreamer/V4L2 edge profile | real Jetson acceptance; not inferred from macOS/Linux | GAMST #325; missing |
| E6 compatibility matrix | available/installed/runtime-compatible/weight-ready/benchmark-accepted states | every provider reports all five states and license gate | partial |

## Epic F — benchmark and default selection

Parent: `F-benchmark` (Gamst-bench #1/#2/#6/#7 and GAMST #293/#300/#307/#308).

| Child | Scope | Acceptance gate | Status |
|---|---|---|---|
| F1 manifest binding | frozen runtime/provider/model/license/OS/dependency/input manifest | measured run refuses missing or mismatched manifest digest | partial; integration pending |
| F2 temporal metrics | capture time, age, detection, cutoff, stale, drops, recovery and escalation | collector emits metrics and scorer consumes them | partial; scorer integration pending |
| F3 matched B/C/D/E | same held-out inputs, consumer, prompt and context contract; distinct E checkpoint | all arms have complete scored traces and denominators | unmeasured smoke only |
| F4 CostBench | Static, Fast Event, Cross-modal, Semantic Change and Resilience | event P/R/F1, association, latency, calls, resource, energy, cost/hour and Pareto report | Gamst-bench #6; missing |
| F5 default selection | Mac measured comparison and rejected-candidate report | no default promotion without target-Mac evidence; CUDA cannot promote Mac | Gamst-bench #7; pending real host |
| F6 Linux acceptance | real Linux/CUDA matched runs | no synthetic or Mac substitution | pending real host |

## Exhaustive issue crosswalk

The following inventory prevents an older parent issue from becoming an
untracked implementation request. Parent issues remain open until their child
acceptance gates close; duplicate issue names are intentional cross-repository
facets of the same epic and are linked in the issue bodies.

| Epic | GAMST issue(s) | GamstCrew issue(s) | Gamst-bench issue(s) | Current gate |
|---|---|---|---|---|
| A runtime platform | #294, #296, #304, #305, #313–#315, #326, #329, #331 | #1, #8 | #1 | foundation exists; host enforcement, exact-SHA and production review open |
| B transport | #295, #299, #316, #327 | #9 | #1 | packet foundation; checksum/shared-memory/transport hardening open |
| C temporal/fusion | #297, #306, #312, #317, #328 | #1 | #1, #3 | clocks/relation foundation; contradiction and cross-runtime evidence open |
| D memory/physics | #318–#322, #330 | #1 | #1 | memory and measured physics acceptance open |
| E OS adaptation | #298, #301, #302, #309, #314, #323–#325 | #2–#7, #10 | #1, #5–#7 | real Mac/Linux/Windows/Jetson hosts open |
| F benchmark/default | #293, #300, #307, #308 | #1 | #1, #3–#8 | synthetic report only; matched real B/C/D/E and CostBench open |

Optional research issues #285–#292 remain linked from README and are not used
to satisfy a required acceptance gate. They cover intermediate fusion,
retrieval embeddings, event-gated streaming, streaming arm E, physics/world
models, embodied handoff and optional provider acceptance.

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

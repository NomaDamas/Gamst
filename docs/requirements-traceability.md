# Approved-plan requirements traceability

This is the implementation gate for the owner-approved GAMST plan. It is
intentionally separate from the implementation status: a documented requirement
is not a completed capability. Every row has an owning epic, a concrete issue
tree, an acceptance artifact, and a current status. New implementation work may
start only after the row and its acceptance gate are present here and in the
linked issue.

## Contract and product requirements

| Requirement | Contract/document | Issue tree | Acceptance evidence | Status |
|---|---|---|---|---|
| Seven-layer product model | `docs/approved-plan.md` § owner-approved seven-layer model | GAMST #294, #295, #297, #318 | layer-to-contract review and focused tests | foundation documented; several runtime gates open |
| Isolated runtimes, one user-facing system | `docs/approved-plan.md` § runtime; `docs/epic-issue-breakdown.md` A/E | GAMST #294, #296, #298; Crew #2–#10 | profile lock, readiness, lifecycle and real-host evidence | partial; real-host gates open |
| Canonical packet and provenance | `docs/approved-plan.md` § canonical packet; `docs/PROTOCOL.md` | GAMST #295, #303, #316; Crew #9 | cross-repository schema/digest round trip | temporal foundation; checksum/transport hardening open |
| Unix transport, bounded queues and backpressure | `docs/approved-plan.md` § runtime and transport | GAMST #299; Crew #9 | size-before-allocation, timeout, cancellation, drop and tamper tests | partial |
| Clock identity, offset/drift and AV alignment | `docs/approved-plan.md` § temporal; `docs/epic-issue-breakdown.md` C | GAMST #297, #306, #312, #317 | drift/offset/interval fixtures and calibration uncertainty | foundation; cross-runtime and contradiction gates open |
| Internal evidence cutoff | `docs/approved-plan.md` § canonical temporal packet | GAMST #297, #312, #317; bench #1, #3 | post-cutoff evidence is excluded from answer support | implemented in foundation; integration evidence open |
| Current/recent/episodic context and resume | `docs/approved-plan.md` § context and memory | GAMST #318–#322 | bounded retention, replay, retrieval ownership and restart lineage | open |
| Rules/decision provider routing | `docs/approved-plan.md` § Decision Router | GAMST #285, #294 | calibrated decision traces and abstention behavior | optional provider roadmap |
| Deterministic physical context | `docs/approved-plan.md` § physics; `docs/physical-state.md` | GAMST #290 and the physics provider tests | kinematics, covariance, trajectory hypotheses, contact/impact alignment; abstain without calibration | implementation exists in parts; acceptance evidence open |
| Synthetic world-model separation | `docs/approved-plan.md` § physics | GAMST #290 | generated futures labeled synthetic and excluded from evidence/ground truth | policy documented |
| OS adaptation | `docs/approved-plan.md` § OS matrix | GAMST #298, #301, #302, #309, #314, #323–#325; Crew #2–#10 | five-state compatibility matrix and host evidence per OS | macOS foundation; Linux/Windows/Jetson hosts unavailable |
| Benchmark adapter and frozen manifests | `docs/approved-plan.md` § benchmark; bench `docs/ADAPTERS.md` | GAMST #300, #307; bench #1, #3, #7 | exact manifest digest, same input/consumer/prompt, complete denominators | foundation; real B/C/D/E open |
| CostBench and default selection | `docs/approved-plan.md` § CostBench | bench #6, #7; GAMST #293, #308 | matched held-out quality/latency/resource/cost and independent assessment | unmeasured; no default promotion |
| Open-source integration candidates | `docs/approved-plan.md` § OS/runtime matrix | GAMST #294, #298, #302 | compatibility is recorded per profile; no tool is treated as universal fix | documented |

## Runtime issue tree

The detailed child map is maintained in [`docs/epic-issue-breakdown.md`](epic-issue-breakdown.md).
The tracked issue roots are:

- **Epic A — GamstCrew runtime platform:** GAMST #294, #296, #304, #305,
  #313–#315, #326, #329; Crew #8.
- **Epic B — context transport:** GAMST #295, #299, #316, #327; Crew #9.
- **Epic C — temporal and multimodal fusion:** GAMST #297, #306, #312, #317,
  #328.
- **Epic D — memory/context:** GAMST #318–#322.
- **Epic E — OS adaptation:** GAMST #298, #301, #302, #309, #314,
  #323–#325; Crew #2–#7 and #10.
- **Epic F — benchmark/default selection:** GAMST #300, #307, #308;
  Gamst-bench #1, #3–#7.

Older roadmap issues (#285–#292) remain explicitly optional research work and
cannot close a required acceptance gate by themselves.

## Review and merge gate

The required independent review was run against exact current handoff commits
through Myproxy `gpt-6.1-sol` with high reasoning and returned **HOLD**. The
review evidence is recorded in `docs/completion-gap-report.md`. The HOLD remains
correct because resource limits are accounting-only, exact-SHA cross-repository
evidence is incomplete, B/C/D/E are unmeasured, and Linux/CUDA, Windows and
Jetson acceptance hosts are unavailable. No PR may merge until those gates and
the hosted motion-provider job pass.

## Temporal cutoff wording

`query_cutoff_ms` is an internal evidence-consistency watermark. It does not
truncate, rewrite or remove the user's query. It prevents an observation
captured after the observation time being used as if it were already known.
The API name may evolve, but this causality invariant is part of the approved
temporal contract.

## Current stop condition

The foundation implementation can continue in parallel across GAMST,
GamstCrew and Gamst-bench only for rows whose contract and issue acceptance are
already frozen. Production acceptance, default promotion and merge remain
blocked until the open rows have implementation, focused tests, real runtime
evidence and independent approval.

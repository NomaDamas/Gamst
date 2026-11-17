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
| Canonical packet and provenance | `docs/approved-plan.md` § canonical packet; [Gamst-bench protocol](https://github.com/NomaDamas/Gamst-bench/blob/feat/temporal-metrics/docs/PROTOCOL.md) | GAMST #295, #299, #316, #331; Crew #9; bench #1 | cross-repository schema/digest round trip plus exact-byte probe | canonical wire/checksum probe and live Crew→Core→bench adapter proof pass; host enforcement and independent production review remain open |
| Unix transport, bounded queues and backpressure | `docs/approved-plan.md` § runtime and transport | GAMST #299; Crew #9; GAMST #327 | size-before-allocation, timeout, cancellation, drop and tamper tests | bounded queues, drop accounting, size checks, handler/client timeouts and portable shared-memory packet fallback implemented/tested; CUDA zero-copy and host stress remain open |
| Clock identity, offset/drift and AV alignment | `docs/approved-plan.md` § temporal; `docs/epic-issue-breakdown.md` C | GAMST #297, #306, #312, #317 | drift/offset/interval fixtures and calibration uncertainty | calibrated `TemporalContextEngine` relation foundation and focused fixtures implemented (`8ad9788`); cross-runtime and contradiction metrics gates open |
| Internal evidence cutoff | `docs/approved-plan.md` § canonical temporal packet | GAMST #297, #312, #317; bench #1, #3 | post-cutoff evidence is excluded from answer support | implemented in foundation; integration evidence open |
| Current/recent/episodic context and resume | `docs/approved-plan.md` § context and memory | GAMST #318–#322 | bounded retention, replay, retrieval ownership and restart lineage | `ContextMemory` state layers, idempotent delta replay, cutoff retrieval, lineage checkpoint/resume, bounded atomic `CheckpointStore`, and Core envelope integration implemented and focused-tested (`17e0d30`); real long-running/provider acceptance remains open |
| Rules/decision provider routing | `docs/approved-plan.md` § Decision Router | GAMST #285, #294 | calibrated decision traces and abstention behavior | optional provider roadmap |
| Deterministic physical context | `docs/approved-plan.md` § physics; `docs/physical-state.md` | GAMST #290, #330 and the physics provider tests | kinematics, covariance, trajectory hypotheses, contact/impact alignment; abstain without calibration | bounded deterministic `PhysicalState` implementation and focused tests exist; held-out/device acceptance evidence remains open |
| Synthetic world-model separation | `docs/approved-plan.md` § physics | GAMST #290 | generated futures labeled synthetic and excluded from evidence/ground truth | policy documented |
| OS adaptation | `docs/approved-plan.md` § OS matrix | GAMST #298, #301, #302, #309, #314, #323–#325, #333–#338; Crew #2–#10/#13; bench #10 | five-state compatibility matrix and host evidence per OS | declarative macOS/Linux/Windows/Jetson catalogs, five-state provider status and install/native diagnostics implemented/tested (`8651d62`); Alpamayo2 Edge is tracked as an optional Jetson/physics provider follow-up; real host acceptance remains unavailable |
| Benchmark adapter and frozen manifests | `docs/approved-plan.md` § benchmark; bench `docs/ADAPTERS.md` | GAMST #300, #307; bench #1, #3, #7 | exact manifest digest, same input/consumer/prompt, complete denominators | manifest binding, temporal scorer and live Crew adapter are implemented/focused-tested; real B/C/D/E remain open |
| CostBench and default selection | `docs/approved-plan.md` § CostBench | bench #6, #7; GAMST #293, #308 | matched held-out quality/latency/resource/cost and independent assessment | exact-decimal aggregation including absent/empty component attribution and conservative default gate implemented/tested (`4a489a0`); matched evidence remains unmeasured and no default promotion is allowed |
| Open-source integration candidates | `docs/approved-plan.md` § OS/runtime matrix | GAMST #294, #298, #302 | compatibility is recorded per profile; no tool is treated as universal fix | documented |
| Playground / BYOK MyProxy Terra Low integration | GAMST Playground and OpenAI-compatible BYOK contract | GAMST #311; bench #1/#9 for runtime trace handoff | user-configured endpoint/model/key, non-stream and SSE response, compact Context/evidence inspection, raw-media exclusion and bounded error report | endpoint and credentials are user-supplied; no successful MyProxy call is claimed in this session |
| Playground capture lifecycle | `docs/playground-capture-lifecycle.md` | GAMST #311 → #332 | stale pre-start paused snapshot cannot stop current media; current post-stop snapshot still releases media | implemented and covered by dashboard/browser helper tests; hardware coverage remains outside this issue |

## Runtime issue tree

The detailed child map is maintained in [`docs/epic-issue-breakdown.md`](epic-issue-breakdown.md).
The tracked issue roots are:

- **Epic A — GamstCrew runtime platform:** GAMST #294, #296, #304, #305,
  #311, #313–#315, #326, #329, #331; Crew #11/#8.
- **Epic B — context transport:** GAMST #295, #299, #316, #327; Crew #11/#9.
- **Epic C — temporal and multimodal fusion:** GAMST #297, #306, #312, #317,
  #328; Crew #11.
- **Epic D — memory/context:** GAMST #318–#322, #330; Crew #11.
- **Epic E — OS adaptation:** GAMST #298, #301, #302, #309, #314,
  #323–#325, #333–#338; Crew #11/#2–#7/#10/#13; bench #10.
- **Epic F — benchmark/default selection:** GAMST #300, #307, #308, #311;
  GamstCrew #11; Gamst-bench #1, #3–#9.

GitHub native hierarchy is also present: GAMST #294 is the top-level parent
with nested children listed in the epic map; GamstCrew #11 is the repository
local parent for Crew #2–#10 (Crew #1 is a pull request, so it cannot be a
sub-issue parent); and Gamst-bench #1 owns #3–#9. Cross-repository links remain
in issue bodies and this crosswalk because parent/sub-issue metadata cannot span
repositories.

Older roadmap issues (#285–#292) remain explicitly optional research work and
cannot close a required acceptance gate by themselves.

## Review and merge gate

The required independent Astra-medium review of the implementation handoff
`915f18c`/`d98723d`/`715f8b8` returned **APPROVE** after the exact-head metadata
correction. It covers the implementation handoff and focused evidence; later
GAMST commits are documentation-only. The earlier Epic 0 approval at `c950a57`
remains separately scoped. The repository-wide production merge gate remains
open because resource limits lack real stress evidence, full runtime-manager→
bench execution is open, B/C/D/E are unmeasured, and Linux/CUDA, Windows and
Jetson acceptance hosts are unavailable. No PR may merge on the basis of these
implementation approvals alone.

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

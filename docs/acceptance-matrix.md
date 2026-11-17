# GAMST implementation acceptance matrix

This matrix is subordinate to `docs/approved-plan.md`. A row is not complete until implementation, focused tests, documentation, runtime evidence and review are all present.

| ID | Workstream | Repository | Acceptance | Status |
|---|---|---|---|---|
| P0-1 | Approved plan | GAMST | Plan is committed verbatim and linked from issues | done |
| P0-2 | Temporal packet | GAMST + GamstCrew | Same bounded packet fields, JSON limits, evidence and cutoff semantics | Core→Crew→bench exact-byte probe and live adapter pass; host enforcement/review open |
| P0-3 | Temporal clocks | GAMST + GamstCrew | Minimum-sample offset/drift, residual, overlap, revisions and contradiction candidates | calibrated Core engine and Crew packet fields implemented/tested; cross-runtime calibration evidence open |
| P0-4 | Query cutoff | GAMST + Gamst-bench | No post-query evidence can support an answer | internal evidence watermark and fail-closed retrieval implemented/tested; measured adapter integration open |
| P0-5 | Runtime lifecycle | GamstCrew | Start/stop/health/readiness, bounded queue, drops, crash isolation, provider metadata | manager/process lifecycle, diagnostics and serialization implemented/focused-tested; target-host stress open |
| P0-6 | Local transport | GamstCrew | Bounded Unix JSONL request/response with size rejection and correlation | bounded Unix, checksum and shared-memory fallback implemented/focused-tested; CUDA performance open |
| P1-1 | OS profiles | GamstCrew + GAMST | macOS, Linux CUDA, Linux CPU, Windows, Jetson/ARM diagnostics | five-state catalogs and install diagnostics implemented; real host runs pending |
| P1-2 | Alpamayo2 Edge provider | GAMST #333–#338 + GamstCrew + gamst-bench | isolated Jetson/ARM physics runtime, temporal provenance, frozen edge manifest and held-out metrics | issue tree documented; upstream/license, implementation, target-host acceptance and independent review pending |
| P1-2 | Runtime isolation | GamstCrew | Python/Torch/Transformers/native environments remain isolated | registry/lock/artifact/architecture validation and launcher seams implemented; container/host acceptance open |
| P1-3 | Benchmark manifest | Gamst-bench | Frozen runtime/provider/model/license/hardware/dependency/input manifest | digest/provenance validation implemented/focused-tested; measured manifests pending |
| P1-4 | Temporal metrics | Gamst-bench | age, detection, p50/p95, cutoff, evidence, stale, drops/recovery, escalation | collector/scorer integration implemented/focused-tested; matched measured runs pending |
| P1-5 | Default Mac profile | Gamst + Gamst-bench | Lightweight stack runs one-command on target Mac with reproducible revisions | pending real profile run |
| P1-6 | Linux CUDA profile | GamstCrew + Gamst-bench | Isolated SOTA runtimes run through same packet contract | pending |
| P2-1 | B/C/D/E | Gamst-bench | Frozen matched runs, same consumer/prompt/input availability | pending |
| P2-2 | Temporal fusion | GAMST | AV offset/drift, interval relations, state/revisions, cutoff and contradiction metrics | calibrated relation/revision/contradiction foundations implemented and focused-tested; cross-runtime metrics/host evidence open |
| P2-3 | Physics context | GAMST | Kinematics, trajectories, contacts, cross-modal impacts, uncertainty | deterministic provider implemented/focused-tested with calibration abstention; held-out/device acceptance and runtime wiring open |
| P2-4 | Review | All | PR review by approved reviewer, findings fixed, then merge | Prior Astra-medium approval covers `50bc7f9`/`d98723d`/`715f8b8`; fresh review is required for current checkout heads `2015c44`/`fdf5322`/`ee2ba10`. Host, matched measurement and broader production gates remain open |
| P3-1 | Completeness | All | Compare code/docs/issues to approved plan; zero unexplained gaps | implementation/documentation audit refreshed; host/reviewer/quality gates remain explicitly tracked |

## Evidence rules

- A passing fixture test is not a real-weight or hardware acceptance.
- CUDA/H100 results do not select a Mac default.
- Synthetic benchmark protocol checks are not measured quality/cost results.
- Optional providers remain opt-in until real-weight, license, hardware and benchmark evidence exists.
- Any unmeasured metric is `null`/unknown, never invented.

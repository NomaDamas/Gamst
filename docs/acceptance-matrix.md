# GAMST implementation acceptance matrix

This matrix is subordinate to `docs/approved-plan.md`. A row is not complete until implementation, focused tests, documentation, runtime evidence and review are all present.

| ID | Workstream | Repository | Acceptance | Status |
|---|---|---|---|---|
| P0-1 | Approved plan | GAMST | Plan is committed verbatim and linked from issues | done |
| P0-2 | Temporal packet | GAMST + GamstCrew | Same bounded packet fields, JSON limits, evidence and cutoff semantics | foundation slices implemented; cross-repo integration pending |
| P0-3 | Temporal clocks | GAMST + GamstCrew | Minimum-sample offset/drift, residual, overlap, revisions and contradiction candidates | foundation slices implemented; cross-runtime calibration pending |
| P0-4 | Query cutoff | GAMST + Gamst-bench | No post-query evidence can support an answer | GAMST core tests; benchmark integration pending |
| P0-5 | Runtime lifecycle | GamstCrew | Start/stop/health/readiness, bounded queue, drops, crash isolation, provider metadata | foundation PR; resource reservation foundation `f4b135c`; provider subprocess lifecycle pending |
| P0-6 | Local transport | GamstCrew | Bounded Unix JSONL request/response with size rejection and correlation | foundation PR; integration pending |
| P1-1 | OS profiles | GamstCrew + GAMST | macOS, Linux CUDA, Linux CPU, Windows, Jetson/ARM diagnostics | discovery foundation; complete profiles pending |
| P1-2 | Runtime isolation | GamstCrew | Python/Torch/Transformers/native environments remain isolated | design documented; manager/container integration pending |
| P1-3 | Benchmark manifest | Gamst-bench | Frozen runtime/provider/model/license/hardware/dependency/input manifest | pending |
| P1-4 | Temporal metrics | Gamst-bench | age, detection, p50/p95, cutoff, evidence, stale, drops/recovery, escalation | foundation integrated in Gamst-bench `3649389`; real measured runs pending |
| P1-5 | Default Mac profile | Gamst + Gamst-bench | Lightweight stack runs one-command on target Mac with reproducible revisions | pending real profile run |
| P1-6 | Linux CUDA profile | GamstCrew + Gamst-bench | Isolated SOTA runtimes run through same packet contract | pending |
| P2-1 | B/C/D/E | Gamst-bench | Frozen matched runs, same consumer/prompt/input availability | pending |
| P2-2 | Temporal fusion | GAMST | AV offset/drift, interval relations, state/revisions, cutoff and contradiction metrics | relation foundation in GAMST `4fea924`; full calibration/metrics engine pending |
| P2-3 | Physics context | GAMST | Kinematics, trajectories, contacts, cross-modal impacts, uncertainty | provider implemented; runtime integration pending |
| P2-4 | Review | All | PR review by approved reviewer, findings fixed, then merge | pending; Astra 6 medium unavailable, alternative reviewer not yet run |
| P3-1 | Completeness | All | Compare code/docs/issues to approved plan; zero unexplained gaps | pending |

## Evidence rules

- A passing fixture test is not a real-weight or hardware acceptance.
- CUDA/H100 results do not select a Mac default.
- Synthetic benchmark protocol checks are not measured quality/cost results.
- Optional providers remain opt-in until real-weight, license, hardware and benchmark evidence exists.
- Any unmeasured metric is `null`/unknown, never invented.

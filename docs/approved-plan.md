# GAMST approved implementation plan

Status: approved by project owner on 2026-11-16.

## Product identity

GAMST is a general-purpose Perception SDK that gives existing LLMs continuous perception through lightweight, event-driven multimodal processing.

GAMST is not a new VLM. It is a sensory middleware/runtime in front of existing LLMs and VLMs.

Core hypothesis:

> Lightweight continuous perception plus selective foundation-model inference can preserve grounded downstream understanding at lower total cost than uniform VLM streaming.

## Non-negotiable principles

- Training-free by default: no new foundation-model training in the default path. Calibration and task-specific adapters are allowed when explicit and measured.
- Model-agnostic: connect OpenAI-compatible, local, LLM, VLM and agent consumers through stable contracts.
- Event-driven: do not send every frame to an expensive model.
- Asynchronous: observation and reasoning continue independently.
- Temporal: preserve capture time, source clock, event intervals, revisions and query cutoffs.
- Evidence-aware: retain evidence IDs and uncertainty, not only summaries.
- Local-first: raw media does not leave the machine without explicit provider configuration and consent.
- Bounded: queues, packets, memory, workers, context windows and output are bounded; drops are visible.
- No fake fallback: unavailable providers report errors or abstain; they never fabricate observations or answers.
- Observations are measurements, not inner states, diagnosis, deception or intent claims.

## Approved architecture

```text
Gamst Core
├── Input Layer: camera/video, microphone/audio, text/events, tactile, RF/radar, thermal, physical sensors
├── Lightweight Perception Layer
├── Event Normalizer
├── Temporal & Multimodal Fusion
├── Decision Router
├── Context & Memory Engine
└── LLM/VLM/Agent integrations
```

The system is one product but not one Python virtualenv.

```text
gamst-core       Python 3.11; context, timestamp, evidence, routing
gamst-vision     PyTorch / ONNX / MPS / CUDA
gamst-audio      PyTorch / ONNX
gamst-sam31      isolated Transformers/SAM runtime
gamst-vlm        llama.cpp / vLLM / MLX
gamst-tactile    tactile provider runtime
gamst-physics    deterministic physics and optional world-model runtime
gamst-bench      independent evaluator and frozen experiment manifests
```

The user-facing experience may be one-command (`gamst install`, `gamst run`), while provider runtimes remain isolated. `GamstCrew` owns asynchronous runtime orchestration. `Gamst-bench` is the benchmark workspace and uses one stable adapter contract.

## Runtime and transport requirements

- GamstCrew must manage provider lifecycle, process isolation, health/readiness, model revision, resource allocation, crash isolation and restart policy.
- macOS/Apple Silicon profile: MLX, MPS, ONNX Runtime, llama.cpp and CPU fallback; lightweight defaults first.
- Linux/NVIDIA profile: CUDA, NVIDIA Container Toolkit, PyTorch CUDA, TensorRT, ONNX Runtime GPU, vLLM and optional Triton/Ray serving.
- Windows profile: CPU/ONNX/DirectML/llama.cpp where supported.
- Linux ARM/Jetson profile: TensorRT/ONNX/GStreamer/V4L2 where supported.
- Runtime profiles must declare OS, architecture, Python, Torch, Transformers, accelerator, native ABI, model revision, license, weights, readiness and benchmark status.
- Prefer Unix-domain sockets for local IPC; support protobuf or MessagePack packets. Shared memory/zero-copy is an optional Linux/CUDA optimization.
- Every queue is bounded and exposes drops, age, latency and backpressure.
- A runtime failure is isolated and reported; it must not silently change provider behavior.

## Canonical temporal packet contract

The owner-approved temporal requirement includes keeping event timestamps, relationships and evidence valid for the question's observation time. `query_cutoff_ms` is the concrete implementation field for that requirement; it is not an unrelated product feature. The field/API name may evolve, but the invariant must remain: evidence captured after the query cutoff cannot support the answer.

Every event/observation packet must preserve:

- event ID
- stream ID and subject/entity ID
- modality and event type
- source clock and monotonic clock
- capture timestamp
- receive timestamp
- processing start/end timestamps
- event start/end interval
- sequence and gap information
- revision and superseded event IDs
- provider and model revision
- confidence/uncertainty semantics
- freshness/staleness
- evidence IDs and source spans
- query cutoff eligibility

The temporal engine must support:

- clock offset and drift calibration
- audio/video cross-modal offset compensation
- event interval overlap
- modality latency compensation
- current state vs last-seen state
- event revisions
- temporal contradiction detection
- event onset/offset
- query-time evidence cutoff
- temporal memory and delta updates

No evidence captured after a query may support that query's answer.

## Fusion policy

Default: event-level and temporal state fusion. Each modality may run independently; normalized events are related by time, entity and evidence.

Optional: bounded intermediate fusion for task-specific synchronized AV/physical tasks (MBT/Perceiver-like provider).

Optional: full multimodal VLM/streaming VLM only where benchmark evidence justifies it.

Late fusion is not the only technique. It is the modular default; intermediate and full fusion are measured alternatives.

## Decision Router

Default: deterministic rules, zero dependency.

Optional learned providers: Intern-Decision-2B and Intern-Decision-4B.

Fixed actions may include:

- IGNORE
- UPDATE
- STORE
- INSPECT
- ESCALATE
- EMIT
- request fresh sensor
- abstain / insufficient evidence

The host owns evidence IDs, freshness, watermark, timeout and schema validation. Model confidence is not calibrated truth until calibrated on GAMST traces.

## Physics context

Implement deterministic, bounded physical context before learned world models:

- uncertainty-carrying kinematic state
- velocity/acceleration/jerk
- free-fall, drag and grounded-friction trajectory hypotheses
- covariance and uncertainty envelopes
- support/containment relations
- collision/contact candidates
- force only with calibrated mass/force evidence
- cross-modal impact alignment using audio, visual and tactile signals

## Owner-approved seven-layer product model

The product is implemented as seven explicit layers:

1. **Multimodal Input Layer**: camera/video, microphone/system audio, text/events, tactile, RF/radar, thermal and physical sensors.
2. **Lightweight Sensory Layer**: motion, OCR, objects, masks, VAD, streaming ASR and audio events. Cheap signals gate expensive perception; not every model runs continuously.
3. **Event Normalizer**: one event schema with timestamp, modality, entity, confidence/uncertainty and evidence.
4. **Temporal & Multimodal Fusion**: event relations, time alignment, state changes, deduplication, contradiction candidates and evidence retrieval. It includes event-level fusion, temporal-state fusion and selective intermediate fusion; it is not limited to late fusion.
5. **Decision Router**: rules and optional Jev-style decisions select ignore/update/store/inspect/escalate/emit/abstain actions. The router is not called on every frame.
6. **Context & Memory Engine**: current state, recent event timeline, episodic memory, compressed history, delta updates, session resumption, evidence references and uncertainty.
7. **LLM/VLM/Agent Integration**: OpenAI-compatible, local and agent/robotics consumers through stable contracts.

The core output distinguishes **Live Context** (current state, recent events, changes and immediate uncertainty) from **Historical Context** (past events, repeated patterns and prior state relations). A context delta may be delivered instead of repeating the entire snapshot.

## Mathematical and physical context

The physics layer is a required bounded capability, not a decorative future claim. It computes position, velocity, acceleration, direction, jerk, covariance, trajectory envelopes, support/containment, collision/contact candidates, impact timing and audio/visual/tactile alignment. It must abstain from numeric force, mass or precise 3D trajectory when calibration or dimensional evidence is absent. Free-fall, drag and grounded-friction hypotheses are explicit alternatives with uncertainty.

Learned world models (Cosmos, V-JEPA 2, Genesis) are optional labeled prediction/simulation paths only. Generated futures never enter the raw evidence store or benchmark ground truth.

## OS and runtime matrix

The supported profiles are explicit:

- macOS/Apple Silicon: Python 3.11, MLX, MPS, ONNX Runtime, CoreML-capable providers, llama.cpp and CPU fallback.
- Linux/NVIDIA: Python 3.11, CUDA, NVIDIA Container Toolkit, PyTorch CUDA, TensorRT, ONNX Runtime GPU, vLLM, Triton and Ray where compatible.
- Windows: CPU, ONNX Runtime, DirectML-capable providers and llama.cpp first; CUDA is a separate profile.
- Linux ARM/Jetson: TensorRT, ONNX Runtime, CUDA, GStreamer and V4L2 as a separate edge profile.

Each profile reports available/installed/runtime-compatible/weight-ready/benchmark-accepted. Triton, Ray, BentoML, KServe, ONNX Runtime, ExecuTorch, MLX, llama.cpp, uv, pixi, conda-lock, Docker/Podman and Unix/shared-memory IPC are integration candidates, not automatic compatibility guarantees.

## Benchmark selection policy

Gamst-bench remains the reference repository. The first benchmark suite includes FastBench, OmniPro Online, VSAS-Bench and GAMST CostBench. SPOT-Bench is a later silence/intervention evaluation, not removed from the research roadmap.

CostBench scenarios are Static, Fast Event, Cross-modal, Semantic Change and Resilience. Comparisons include VLM 1FPS, VLM 2FPS, lightweight event text, full GAMST and an Oracle Router upper bound. Metrics include event precision/recall/F1, time-to-detect, p50/p95, cross-modal association, VLM calls, cost/hour, CPU/GPU/RAM, miss/recovery rate and downstream LLM QA accuracy.

Default models remain lightweight until measured Mac results select a replacement. No CUDA/H100 result promotes a Mac default.

## Approved implementation epics

- **Epic A — GamstCrew Runtime Platform**: runtime API, process lifecycle, health/readiness, crash isolation, provider registry, resource allocation, OS adapter, runtime locks and local launcher.
- **Epic B — Context Transport**: packet schema, Unix socket, shared memory, bounded queue, backpressure, drop accounting, packet checksum and provider revision.
- **Epic C — Temporal Fusion**: source/monotonic clocks, drift, AV offset, interval overlap, revisions, query cutoff, stale/current/last-seen, contradictions and cross-modal relations.
- **Epic D — Memory and Context**: current state, recent timeline, episodic memory, compression, delta update, evidence retrieval and session resumption.
- **Epic E — OS Adaptation**: macOS, Linux CUDA, Linux CPU, Windows, Jetson/ARM, compatibility matrix, install diagnostics and native dependency checks.
- **Epic F — Benchmark**: adapter, frozen manifests, Mac/Linux profiles, B/C/D/E, CostBench, temporal metrics and default selection.

Each epic is decomposed into implementation sub-issues. An issue is not complete because its documentation exists; its code, focused tests, runtime evidence and review must pass.

Cosmos, V-JEPA 2 and Genesis are optional labeled prediction/simulation paths. Generated futures are never raw evidence.

## Default model policy

Initial defaults remain lightweight and local:

- Qwen3-ASR 0.6B
- Silero VAD
- MediaPipe
- RF-DETR Nano
- IoU tracker or OC-SORT
- Depth Anything V2 Small
- EfficientTAM
- Qwen3-VL 2B
- deterministic multimodal decision
- physical-state precompute

Optional providers include Moonshine, TEN-VAD, CED, SAM3.1 video, Video Depth Anything, Intern-Decision 2B/4B, SAM-Audio and Sparsh.

The default combination changes only after real Gamst-bench results on the target hardware. CUDA/H100 results do not automatically select a Mac default.

## Benchmark contract and acceptance

Reference benchmark repository:

```text
https://github.com/NomaDamas/Gamst-bench.git
```

Required arms:

- B: lightweight perception + consumer LLM
- C: GAMST + selective VLM + consumer LLM
- D: periodic VLM control
- E: small streaming VLM

Frozen manifest fields:

- runtime profile
- provider/model revisions and hashes
- license
- Python and dependency lock
- OS, architecture and hardware
- consumer model and prompt revision
- context contract revision
- input availability and timestamps

Required metrics:

- grounded on-time success
- critical-fact recall
- unsupported claims
- stale-state errors
- abstention precision
- evidence-ID correctness
- event precision/recall/F1
- cross-modal association accuracy
- time-to-detect
- TTFT and p50/p95 response latency
- packet age
- VLM calls and escalation fraction
- CPU/GPU/RAM/VRAM/energy
- dropped inputs and recovery time
- cost/hour and quality-cost Pareto frontier

## Approved implementation phases

1. Baseline: freeze this plan, align cross-repository epics/sub-issues, freeze contracts and acceptance matrix.
2. Parallel runtime: build GamstCrew runtime manager/provider lifecycle, GAMST temporal packet transport, OS profiles/diagnostics, and Gamst-bench manifests/temporal metrics in parallel.
3. Integration: connect GAMST core to GamstCrew, connect benchmark B/C/D/E profiles, run Mac default smoke and benchmark.
4. Review: targeted checks, PRs, code review, fixes, merge only after review approval.
5. Completeness: compare implementation/docs/issues to this plan, close every gap, correct inaccurate documentation, and report evidence.

Each phase is decomposed into independently reviewable issues. A parent issue is not complete until all child acceptance criteria pass.

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

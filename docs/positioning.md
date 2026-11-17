# Why GAMST exists

Status: owner-stated positioning, recorded 2026-11-18 after the first public
presentation. This document is the reference answer to "why not an omni model"
and "isn't this just LiteLLM". Update it when the position changes; do not let
feature inventories replace it.

## One sentence

GAMST is not a model. It is the always-on, cheap perception layer in front of
expensive models: it decides **when** to look, **which moment** to send,
**when it happened**, and **what evidence** supports the answer.

Omni models solve "show me and I understand". GAMST solves "out of 24 hours,
which seconds should I show, and what was true at that second".

## Project scope (owner statement, 2026-11-18)

GAMST delivers context so that an LLM or VLM does **not** have to run
inference repeatedly to perceive, see and decide in real time. Perception is
continuous and cheap; model inference is occasional and grounded. Omni models,
streaming VLMs and commercial realtime APIs (Gemini Live, OpenAI Realtime) are
consumers in the LLM position, never the perception layer.


## The three things that do not disappear as models improve

| Axis | Omni / end-to-end model | GAMST | Evidence in this repository |
|---|---|---|---|
| Frequency | Samples video at ~1 fps (Vertex AI Live API input spec: JPEG 1 FPS). Anything between samples is never seen. | 30 fps lightweight signals (motion, VAD, landmarks, person count) detect onset; only that frame is escalated. | Qwen3-VL-2B four-frame answer 10–20 s on Apple Silicon (bench smoke 2026-11-16); MediaPipe/OpenCV per-frame 50–80 ms (`docs/provider-processes.md`). |
| Time | Model answers carry no capture time; a stale frame is answered as if current. | Every packet carries capture timestamp, clock epoch, fresh/stale, revision and query cutoff. | `docs/temporal-packet-checksum.md`, `gamst/temporal.py`, PR #303. |
| Evidence | Answer only. | Every answer references `evidence_id`s that resolve to the observation and its time. | Playground "Evidence sent" panel; `gamst/temporal_context.py`. |

Cost follows from frequency: no change, no call. Better models raise the
price per call and therefore raise the value of deciding when to call.

## What GAMST is not

- Not a competitor to omni models. Omni, streaming VLMs and small local VLMs
  are tools GAMST calls; the benchmark arms A/D/E exist to measure when calling
  them continuously is better.
- Not a modality inventory. The number of supported sensors is not the point;
  the default seven (face, pose, VAD, ASR, voice, motion, VLM) are enough to
  prove the pattern. Additional providers are opt-in extras.
- Not LiteLLM. See below.

## GAMST versus LiteLLM (and other model routers)

LiteLLM routes **requests you already decided to make** across providers:
same prompt in, provider-agnostic completion out. It has no sensors, no clock,
no notion of "nothing happened for the last 40 seconds".

| Question | LiteLLM | GAMST |
|---|---|---|
| Which provider serves this prompt? | Yes | Delegates (OpenAI-compatible endpoint, local llama.cpp, BYOK) |
| Should a prompt be made at all right now? | No | Decision Router: IGNORE / UPDATE / STORE / INSPECT / ESCALATE / EMIT / abstain |
| Which frame/audio window goes into the prompt? | No | Onset detection + bounded temporal context + query cutoff |
| What time was the evidence captured, and is it stale? | No | Temporal packet contract |
| Mixing modalities in real time without an omni model | No | Event-level fusion of independent lightweight providers, each in its own process/venv |
| Composing System-1 perception stacks per domain | No | Provider profiles (`ProcessProvider`, GamstCrew) assemble vision/audio/sensor workers per use case without a retraining or a new venv per project |

A router sits after the decision. GAMST sits before it. They compose: GAMST
can emit through LiteLLM.

## Why realtime multimodal mixing without omni matters

- Omni is over-engineering when the task is "notice that something changed and
  describe it once". Continuous observation does not need continuous inference.
- Situational sampling: a workshop camera, a meeting room and a robot teleop
  loop each need a different cheap stack (person count + sound; speaker change +
  slide change; contact + object appearance). GAMST composes these from
  isolated providers; the consumer LLM/VLM stays the same.
- "I could build it myself" is true for every one of these; the cost is
  rebuilding the timestamp/evidence/escalation plumbing and the venv isolation
  each time. GAMST is that plumbing, once.

## Halo demonstration pattern: Moment Catcher

One screen, three columns, no model talk:

1. Live camera at 30 fps with a 1 fps "omni sampling grid" blinking underneath.
2. Moments GAMST caught: onset frames frozen as thumbnails with capture time
   and a badge "1 fps would have missed / seen this".
3. What the LLM saw: VLM call counter versus frame counter, each answer pinned
   to its thumbnail's time with its evidence id.

Demonstration actions are domain-free sub-second events: coin flip, pen drop,
hand raise, person enters/leaves, thirty seconds of nothing (counter stays
still). Speed capture uses existing kinematics providers and freezes the peak
velocity frame; report px/s unless calibrated.

## Delivery to the LLM: event deltas, not embeddings

GAMST does not hand embeddings to the LLM. It streams bounded text events
`{event, capture_ms, now_ms, fresh, evidence_id, value}` and lets the router
decide when a completion is worth it. Three attachment modes:

1. Realtime session injection (OpenAI Realtime, Gemini Live): deltas enter the
   session as system/tool messages while the user talks. Adapter not yet
   implemented.
2. Router-triggered completion (current Playground).
3. Agent pull through MCP (`get_context`, `get_state`).

## Deployment stance

- macOS/Apple Silicon: one-line install, provider isolation by process and
  virtualenv. Docker on macOS runs a Linux VM without Metal/MPS, so a
  docker-only Mac demo would lose GPU acceleration.
- Linux/NVIDIA: one container per provider under GamstCrew, each shipping its
  own CUDA runtime and framework versions; the host supplies only the driver.
  `ResourceAllocator` reserves VRAM/RAM before start. This is the planned
  `docker compose up` path; it requires a Linux/CUDA host for acceptance.

## Measurement promise

No cost or latency saving is claimed until Gamst-bench CostBench reports it on
the target hardware with arms A (continuous VLM), B (perception + text LLM),
C (GAMST selective VLM), D (periodic VLM) and E (small streaming VLM) using the
same VLM, prompt and consumer model. The 2026-11-16 smoke only proves the
execution path.

## How GAMST can win on existing benchmarks (and where it cannot)

Official leaderboards rank *models*. GAMST enters each benchmark as the
*system around the same model*. The comparison that matters is therefore
"model alone at fixed sampling" versus "same model fed by GAMST", read as a
quality–cost–latency Pareto curve, not a single score. Expected direction of
advantage per benchmark, with the mechanism that produces it:

| Benchmark | What it scores | GAMST mechanism | Expected effect | Risk |
|---|---|---|---|---|
| FastBench (timed QA, proactive frame) | Answer correct **and in time** after a brief event | Onset detection at sensor rate pins the event frame; VLM is called on that frame instead of the next scheduled sample | Lower time-to-detect, same or fewer VLM calls | Trigger miss → no frame → wrong/absent answer |
| OmniPro Online (proactive, ±3 s tolerance) | F1 of *when* the system speaks | The Decision Router's EMIT/ESCALATE is exactly the proactive trigger; audio VAD + visual change gate it | Higher Online F1 at a fraction of calls versus 1 FPS polling | False alerts/hour if thresholds are loose |
| SPOT-Bench (when to speak) | Abstain vs intervene timing | `insufficient_evidence` / abstain path, freshness gating | Better abstention precision | Needs Jev/rules tuned on real traces |
| OVO-Bench (current / last-seen / future) | State at query time vs past evidence | Temporal contract: current vs last_seen, stale flags, query cutoff | Fewer stale-state errors with identical VLM | Backward-tracing questions may need more frames than GAMST retained |
| StreamArena continuous (memory, tool use) | Long-horizon recall and proactive answers | Bounded temporal memory + evidence IDs replace re-watching | Lower cost per question; recall parity | Memory compaction can drop needed detail |
| StreamingBench proactive / Inf-Streams | Long sessions, resource stability | Bounded queues, drops visible, no KV growth | Stable RSS/VRAM over hours | Quality ceiling is the backbone model's |
| VSAS-Bench | Streaming accuracy/latency/consistency | Same as above | — | Official runner not released |

The honest claim shape is: *with the same backbone, GAMST reaches the model's
published score at N× lower whole-pipeline cost and lower time-to-detect, and
never exceeds the backbone's accuracy ceiling.* Where GAMST loses, it loses on
recall when the cheap trigger misses; that number must be reported.

## Halo product: what to show next week

"Moment Catcher" (above) plus one twist: run the **same camera feed** into a
commercial realtime session (Gemini Live, 1 FPS by spec) and into GAMST +
Gemini Flash-Lite side by side. Show three counters live: frames seen, model
calls, dollars. Drop a pen. One side missed it; one side has a thumbnail with a
timestamp and an evidence id. That is the whole pitch, and it is measurable.

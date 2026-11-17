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

# Integrated perception design: who, what, words (owner discussion, 2026-11-27)

This document collects the owner's discussion and approvals from 2026-11-27: the philosophy, the integrated structure, how each thing is specified, how realtime and post-processing differ, the answers to the owner's questions, and the sports track. Decisions are numbered in [decision-log-2026-11-26.md](decision-log-2026-11-26.md) (35-50), lessons in [lessons-2026-11-26.md](lessons-2026-11-26.md), and numbers in [measurements/testbed-2026-11-26.md](measurements/testbed-2026-11-26.md).

## 요약 (한국어)

- **철학 (사장님 말):** 고빈도 모달리티를 LLM·VLM이 일일이 추론하지 않는다. 가벼운 모델과 물리·수학·공학 계산으로 context를 만들어 넘긴다. 그래서 LLM이 실시간으로 보고 있다는 것을 보여 준다. GAMST는 LLM에 perception을 달아 준다.
- **구조:** 입력 → 사용자 지정 참조(few-shot/zero-shot) → Recipe → Pipeline 레인 → 연결(누구) → 찾기·세기(계산) → Context → MCP → LLM.
- **누구 / 무엇 / 말**은 따로 쓸 수도 있고 합쳐 쓸 수도 있다. "특정인의 특정 행동"은 둘을 합친 것이다. 행동은 모든 사람(track)마다 세고, 누구인지는 나중에 붙인다(소급).
- **few-shot 필수:** 인물(누구)은 반드시 예시가 있어야 한다. 사진, 영상 구간, 카메라 앞 몇 초, 목소리 중 하나 이상이다. 없거나 품질 검사를 통과하지 못하면 시작을 거부하고 무엇을 넣으면 되는지 안내한다. 행동은 글 묘사(zero-shot)나 예시 영상(few-shot) 중 하나다.
- **인물은 얼굴만 보지 않고** 전체 모습(몸·옷)과 목소리까지 본다.
- **동작 세기:** 가장 정확한 모델 하나가 모든 프레임을 직접 센다. 검증 레인, 애매함, 건너뛰기는 없다.
  - 실시간 모드: 밀린 구간은 같은 모델의 L2가 병렬로 보정한다. 앞 맥락은 상태만 만들고, 동작이 끝난 시각을 담당하는 레인만 센다.
  - 후처리 모드: 보정이 없다.
  - 모델이 너무 무거워 못 따라가면 MediaPipe full을 쓰고, 그 사실을 알린다.
- **단어 세기**는 지금 구조를 유지한다.
- **recipe 없이 요청하면** 에이전트가 MCP `compose_recipe`로 바로 구성한다.
- **semantic VLM**은 꺼 둔다. 쓸 때는 앞뒤 인과를 잇는 상태 사슬이어야 한다.
- **스포츠:** Ultralytics의 주장을 recipe 사례로 만들고, 다른 SOTA 오픈소스도 recipe로 만든다. 격리된 worktree, 브랜치, PR에서 진행한다(`feat/sports-recipes`).

## 1. Philosophy

The owner's statement (2026-11-27):

> 철학은 high frequency의 모달리티들을 일일히 inference할필요없이 lightmodel들과 물리 수학 공학적 계산 방식으로 context로 넘겨줘서 llm과 vlm이 실시간으로 보고있다는걸 보여주면서 perception들을 달아주는거야.

What this means in practice:

- Frames and audio samples never go to the LLM.
- Light perception models (pose, speech, identity) and computation (rules, rates, angles, timing) produce timestamped records.
- The LLM sets instructions and asks questions over MCP. It receives alarms from Watches, so it does not have to poll.

How the current state measures up against this philosophy:

| philosophy | state on 2026-11-27 |
|---|---|
| LLM/VLM never looks at frames | yes: only Pipeline lanes see frames; MCP returns records, never raw media |
| LLM changes *what* is counted in words | yes: `set_instruction`, `propose_movement`, `set_count_mode` |
| "when did it happen" is indexed | yes: `count_events`, `get_timeline` (start/end), `search_retained` (past records, no model run) |
| realtime answers through GAMST | yes: MCP queries; `create_watch` pushes alarms |
| LLM sets *who* | yes (PR `feat/person-identity`): MCP `enroll_person` (photos, a video range or camera seconds, consent, the same check) and `who` in `set_instruction`, `count_events`, `get_timeline`; refused without a checked few-shot reference ([person-identity.md](person-identity.md)) |
| conversational UX | **not yet**: demos so far are CLI and HUD. Planned: a demo where the LLM answers "how many shotguns, when?" only through GAMST MCP |

## 2. Integrated structure (owner-approved)

```
┌─ Inputs (each enters on its own; user-defined inputs allowed) ──────┐
│  camera │ video file │ microphone │ sensors (IMU, radar, ...) │ user │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌─ User-defined references (few-shot / zero-shot) ─────────────────────┐
│  person P = face · whole-body appearance · voice   ← few-shot REQUIRED │
│             (photo / video range / camera seconds / recording; consent;│
│              refused with guidance when missing or below quality)      │
│  action A = text (the LLM writes a rule) │ example clips │ rule file    │
│  words W  = terms and spellings (current structure kept)               │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  Recipe: chosen, customised, or composed (compose_recipe)
┌─ Pipeline (linked by `consumes`; models chosen for the hardware) ─────┐
│  perception lanes (high frequency, no LLM)                            │
│   pose + track : RTMW-l │ MediaPipe full │ Sapiens2 (opt-in) → tracks  │
│                  (one model per run, chosen by measured throughput)    │
│   identity     : face (LVFace) + whole-body appearance → per track     │
│   speech       : Scribe → words → voice per word (CAM++) [kept]        │
│   realtime: L1 in order + L2 correction (same model, backlog intervals)│
│   post    : every frame, batches + chunk parallelism, no correction    │
│  binding (new: person_binding)                                         │
│   track ↔ P, speaker ↔ P → P timeline (seen / speaking), retroactive   │
│  matchers (computation only)                                           │
│   actions per track │ words per speaker → events {who, what, start, end}│
│  ┄ optional, OFF: semantic (Qwen VLM) → causal state chain ┄           │
└──────────────────────────────┬──────────────────────────────────────┘
┌─ Context (Store) ─────────────────────────────────────────────────────┐
│  {who: P│unknown, what, start, end, state, lane, decided_ms}           │
└──────────────────────────────┬──────────────────────────────────────┘
┌─ MCP → LLM (records only, never frames) ──────────────────────────────┐
│  set_instruction {who?, what, count│search, realtime│post}             │
│  enroll_person (new) · compose_recipe (new)                            │
│  count_events · get_timeline (filter by who) · search_retained ·       │
│  create_watch                                                          │
└───────────────────────────────────────────────────────────────────────┘
```

What this changes in the existing structure:

1. Matchers run for every track and every speaker, not only for one locked person. As a result, a person confirmed later gets their earlier events (retroactive), and "who did X?" can be answered.
2. The person reference holds face, appearance and voice together.
3. The new module `person_binding` and two new MCP tools, `enroll_person` and `compose_recipe`, are added.

The structure answers these questions:

| question | parts used |
|---|---|
| when was 감스트 on screen | person + binding |
| who did a shotgun / how many | action only; `who` attached to each event |
| how many shotguns by 감스트, when | person + action |
| how many times 감스트 said 뽀또 | person (voice) + words |
| 뽀또 said while doing a shotgun | person + words + action, joined on time |
| long video, many people | post mode; P re-identified across track breaks by the identity score |

## 3. How each thing is specified

| what | zero-shot (text) | few-shot (examples) | state |
|---|---|---|---|
| **who** (person) | no: a name or description does not identify a person | **required**: photo, video range ("this person at 0:12"), camera seconds | photo, video range and camera enrollment exist and are checked (refused with guidance); LVFace-B and OSNet-AIN (appearance) are on this Mac; identity consistency measured on two testbed clips, accuracy unmeasured (no owner labels yet) |
| **voice** | no | same-microphone recording, file, range of the same recording, automatic main speaker | exists; four owner clips 30/33 right (v8) |
| **action** | yes: text → rule written by the LLM over MCP (`propose_movement`), or VLM proposals confirmed by the owner. GAMST has no hidden LLM | yes: 3+ example clips (+ negatives) | shotgun text → rule 16/16; example clips 15/16 offline (MediaPipe) |
| **words** | yes: terms, spellings, `word:W`, `word:W*` | – | exists (kept) |

Enforcement (built, `gamst/person_reference.py`): an instruction with `who` and no person reference, or with a reference that fails the quality check, is refused. The refusal says which few-shot is missing and how to provide it, in the same words on the CLI, in MCP and on the HUD. A person reference needs a consent record.

Identity looks at the whole person, not the face only. The score combines:

- the face (LVFace);
- whole-body appearance (body and clothes);
- the voice, when one was enrolled.

Clothes change between broadcasts, so appearance is trusted within one video only. In a new video the person is confirmed again by face or voice.

The appearance model registered today, SOLIDER, is a problem. It is on Google Drive only, has no digest, has a CUDA-only adapter, and its training-data terms need review. It will be replaced by a model chosen by these criteria: official release, permissive licence, published digest, 500 MB or less (decision 44).
Chosen (2026-11-27): OSNet-AIN x1.0 trained on MSMT17, from the authors' own Hugging Face repository (`kaiyangzhou/osnet`,
MIT model card, sha256 published, 17.3 MB), converted once to ONNX. It is the only candidate that met every criterion; the
more accurate ones (SOLIDER, CLIP-ReID, TransReID, PASS: Google Drive only, no digest; PersonViT: weights "cc" without a
variant and LUPerson non-commercial pretraining; fast-reid: no published digest) did not. Its published number is
cross-domain only (MSMT17 -> Market-1501 R1 70.1 / mAP 43.3), well below SOLIDER's same-domain figures. Details:
[measurements/person-identity-2026-11-27.md](measurements/person-identity-2026-11-27.md).

## 4. Movement counting: one model, realtime and post modes

Owner (2026-11-27):

> 검증레인은 없애고 성능이 더 좋은걸로 바로 세면 되지 굳이 멍청한 모델이거나 그저 그런 모델이 세거나 검증할 이유가 없어

> 건너뛴 프레임이 너무 많은데?

> 애매함도 없애. 그냥 세는걸로하고

Rules:

- One pose model per run counts every frame directly. There is no verification lane, no "uncertain" state for movements, and no skipping.
- The default model is RTMW-l 384x288 (CoreML). MediaPipe full stays selectable with `--pose`.
- At start, a short benchmark measures the model's ms per frame and what live plus L2 can process within the memory budget.
  - If that keeps up with the input, the SOTA model is used.
  - If it cannot ("너무 무거우면"), MediaPipe full is used. The report, the HUD and `get_pipeline` say so, with the numbers.
- Two models are never mixed in one run.

### Realtime mode: L1 + L2 correction

```
camera ─▶ L1 (one model, in order)
            │ frames not reached by their deadline
            ▼
         video memory (RAM, then disk; never dropped)
            │ backlog intervals
            ▼
         L2 × N (same model, intervals in parallel)
            ▼
         corrections: capture time + decided time
```

Late counts are corrections (보정). They exist only in realtime mode.

How an interval is handled:

1. **Pre-roll.** An interval starts at `cut - pre_roll`, where `pre_roll` = the rule's windows (sum of `within_ms`) + 3 × the rate window + a few warm-up frames. Pre-roll frames build state only: the rule, the pose model's temporal state and the person lock. They never emit events.
2. **Ownership.** An event belongs to the one lane whose owned range [start, end) contains the event's end time. Owned ranges do not overlap and cover the whole timeline, so an occurrence that straddles a cut is counted once.

   ```
   time →   10.0s        12.0s          15.0s
   L1    ──owned──────────┤ (behind)      ├──owned────
   L2               [pre-roll][──owned───]
                    11.5s    12.0s      15.0s
   ```

   Safety net: the same person's same action recorded twice with end times within ±0.1 s is merged when it is stored.
3. **The L1 resume is a cut too.** The same pre-roll and ownership rules apply there.
4. **Identity handover.** At a cut, the person state is snapshotted: track, box, last embeddings and lock age. L2 re-associates by box overlap during the pre-roll. If it cannot, it confirms by face inside the interval, and events before that confirmation stay `who: unknown`. It does not guess.
5. **Memory budget.** Each run has a memory budget: video memory in RAM, a spill bound on disk, and a number of L2 replicas chosen from free memory at start and checked again during the run. A replica is never started when it would take free memory below the guard floor.
6. **Throughput limit.** On one Mac GPU a second replica may add little; another compute unit (Neural Engine or CPU) is measured.

### Post mode

Post mode processes a finished file: every frame, the largest batches that help, chunk parallelism within the same memory budget, and no corrections. It returns the count and each occurrence's start and end. Both modes must give the same final count (shotgun: 16 at 60 fps and at 30 fps). Implementation and measurements: PR #373 and `measurements/testbed-2026-11-26.md`.

## 5. Answers to the owner's questions (2026-11-27)

**What does "running the person detector less often" mean?** RTMW is top-down: YOLOX-m finds the person's box (~27 ms), then RTMW-l puts keypoints in the crop (~29 ms). With the detector every 500 ms, frames in between take the box from the last keypoints. Keypoints are still computed on every frame. Measured on the shotgun clip: 16/16 both ways, 85 → 30 ms per frame. The cost is that a person who newly enters the frame is found up to 0.5 s late.

**Why is 60 fps "harder" when MediaPipe misses at 30 fps?** 60 fps is harder for speed: there are twice the frames, 16.7 ms each instead of 33 ms. It is easier for accuracy, because fast motion is sampled more often.

The 30 fps misses (MediaPipe full 13/16: 2.05, 4.16, 5.03 s) were first blamed on MediaPipe's tracking. **That guess was wrong.** A direct check (`scratchpad/mp_modes.py`, 2026-11-27) found:

- VIDEO mode (tracking) and IMAGE mode (independent frames) give identical keypoints.
- Wrist-to-shoulder heights at 30 and 60 fps differ by at most 0.07 shoulder widths. Only the measured fall speed is about halved, because frames are further apart.
- A re-simulation of the rule on those keypoints counted 16/16 at 30 fps.

So the misses come from GAMST's side: person choice, the shoulder-width definition, or the rate window. Agent work on PR #373 investigates; this is open.

**Is MediaPipe, made by Google, inaccurate?** That cannot be concluded. It is built for light, fast mobile use, and it tied RTMW at 60 fps on this clip (16/16). It has 33 body points and few hand points, against RTMW's 133 points, including 21 per hand.

| | MediaPipe Pose Landmarker | RTMW-l (+ YOLOX-m) | Sapiens2 1B |
|---|---|---|---|
| method | finds the person once, then follows the previous position | detects the person, then keypoints in a large crop | top-down, large ViT |
| keypoints | 33 | 133 | 308 |
| on disk | 9 MB | 219 MB + 97 MB | 5.7 GB |
| ms per frame on this Mac | GPU 9-18, CPU 33-45 | ~30 (CoreML, detector every 500 ms) | being measured |
| shotgun 16 (measured) | 60 fps 16, 30 fps 13 | 60 fps 16, 30 fps 16 | being measured |
| licence | Apache-2.0 | Apache-2.0 | Sapiens2 License (no biometric processing or surveillance; Meta audit rights): opt-in only, never bundled |

**Did GAMST assemble MediaPipe's model nodes?** No. GAMST uses Google's published `pose_landmarker_full.task` through the official Tasks API (mediapipe 0.10.35, sha256-pinned). The `.task` bundle holds Google's own detector and landmark model. GAMST only converts RGB to RGBA for the GPU, sets `num_poses=4`, and names the 33 landmarks. RTMW likewise uses rtmlib's published ONNX files unchanged.

**Is it one person only?** No. Both models give up to 4 people per frame. Which person is "감스트" is decided by `target_person` (face LVFace + body re-ID), which is not built into either model: they find *where* people are, not *who* they are. It was found today that `person_match.py` aligns faces with MediaPipe landmark indices (0, 2, 5, 9, 10) and ignores `keypoint_set`, so with RTMW (`coco_wholebody133`) it would use wrong points. This is being fixed. Identity thresholds are unmeasured on real people.

**Can models be swapped?** Yes: `--pose` selects MediaPipe or RTMW (Sapiens2 being added), and the same rule file works. The exception is target-person face alignment (above).

**Does the Pipeline wire models by itself?** Yes, once a recipe names its parts: `consumes` links them, and `plan_setup` picks models for the hardware. Choosing or composing the recipe for a request is the agent's job. There are 6 recipes today (coin-flip, person-action-count, person-term-count, person-voice-style-count, rotation, scene). Recipes are used as they are or customised: `set_instruction`, `--with`/`--without`, `configure_pipeline_module`, `--set`. Owner requirement: a request without a recipe is composed by the agent at once, through the new MCP tool `compose_recipe`.

**What is the semantic module?** `semantic` (Qwen3.5 VLM endpoint) reads the last 4 s every 500 ms and writes a scene state. It is OFF by default in all three recipes that have it, and stays off (owner: not needed). Owner requirement: to tell a story it must know cause and effect before and after, so states form a **causal state chain**. Each state links the previous state, what changed, and the events that caused it. Independent 4 s snapshots are not enough.

## 6. Ultralytics sports and the sports track

The owner shared https://www.ultralytics.com/solutions/computer-vision-in-sport to discuss what to adopt.

What the page claims, read on 2026-11-27:

- realtime ball tracking (padel, tennis, billiards);
- player detection and tracking;
- heat maps and movement profiles;
- pose-based form analysis against a reference;
- inference under 5 ms.

What it does not claim:

- identifying a named person;
- counting actions the user specifies;
- connecting to an LLM.

Its models (YOLO26, YOLO11, YOLOE) are AGPL-3.0 or enterprise-licensed. GAMST is Apache-2.0, so they can only be opt-in extras, like `gamst-sdk[yoloe]` today.

| Ultralytics capability | adopt in GAMST | why |
|---|---|---|
| ball and object tracking | yes, as an optional lane | actions involving objects (kick, pick up); generalise the coin demo's YOLOE lane |
| form compared with a reference | the idea | add a per-occurrence similarity and intensity score to counts |
| heat maps and movement paths | yes (computation only) | position timeline per person from existing tracks |
| YOLO26-pose | measure | one-stage, multi-person, fast (cited COCO 71.6 mAP, T4 12.2 ms) but 17 body points, no hands; adopted only if at least as accurate |
| CoreML / TensorRT export | reference | already used for RTMW |
| fine-tuning on hundreds of frames | not by default | GAMST specifies by examples and text, without training |

Ultralytics is a parts supplier for GAMST's perception lanes, not a competitor. Its outputs stop at detections, tracks, keypoints and heat maps; GAMST hands "who, what, when" to the LLM.

The owner's request: build the sports analysis Ultralytics claims as a recipe case. Separately, bring in other state-of-the-art or other-sport open-source analysis (e.g. Roboflow Sports, MIT) as recipes. This runs in an isolated worktree `.local-run/wt-sports`, on branch `feat/sports-recipes` and its own PR, in tmux window `gamst-sports`, building on the earlier sports discussion (session `0b3de200…`). The owner works with that session directly.

## 7. Status and what is left (2026-11-27)

| item | state |
|---|---|
| CI on PR #373 | fixed: shared frame memory imported numpy for every frame; the core CI job has no numpy. After the fix the failures equal Main's (32), plus motion-provider, which also fails on Main (no tests ran) |
| one-model movement counting, no skips, L2 correction, realtime / post, memory budget | in progress, PR #373 |
| Sapiens2 1B opt-in | downloaded (5.7 GB); measured runs in progress (an fp32 profile took free memory to 1-6% and the guard stopped it) |
| person few-shot enforcement, face + appearance, photo enrollment, LVFace-B (455 MB), `person_binding`, `enroll_person` | built (Draft PR #374 on `feat/person-identity`, against PR #373). The binding runs with nobody enrolled, so MCP can enroll mid-run. Faces are scored beside the pose lane, never in it. Tracker swaps are continued by box overlap. Realtime keeps up with 0 pose frames lost. Accuracy waits for the owner's labelled multi-person clips; the appearance threshold 0.8 is provisional. |
| `compose_recipe`, semantic causal state chain | next |
| conversational demo (LLM answers only through GAMST MCP) | after the above |
| sports recipes | separate PR (`feat/sports-recipes`) |
| owner-labelled multi-person clips | needed from the owner to measure identity |
| merge of PR #373 | after the owner sees its results |

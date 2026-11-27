# Compose a recipe from a request (`compose_recipe`, `gamst recipe compose`)

Owner decision 46: a request without a recipe is composed by the agent at once. GAMST has no hidden LLM, so the request
is **structured**: the calling LLM (or the owner) fills it in, GAMST picks the lanes, links them by `consumes`,
validates the graph and returns a recipe that can be saved and run again, with the reason for every choice. A missing
piece comes back as one sentence that says what to provide. Code: `gamst/recipes/compose.py`. Design:
[design-2026-11-27-integrated-perception.md](design-2026-11-27-integrated-perception.md) sections 2 and 5. Issue #369.

## The request

```json
{"name": "shotgun-gamst",
 "inputs": {"video": "b.mp4"},
 "who": {"name": "감스트", "person": "gamst.person.json"},
 "what": {"movement": {"text": "샷건: 주먹을 어깨 위로 들었다가 빠르게 아래로 내리친다", "rule": {"channels": "vision.pose_channels", "steps": ["..."]}}},
 "mode": "count", "timing": "post", "start": true, "wait": true}
```

| key | what | |
|---|---|---|
| `inputs` | `camera` (index) or `video` (file: its pictures, and its sound for words); `microphone` (true) or `audio` (WAV); `sensors` `[{id, file, modality, columns, time}]` | each optional, at least one |
| `who` | `{name, text, photos, consent}` or `{name, text, person}`; `voice` (an enrolled voice) when words are counted for the person. **Both a text (`name` or a description in `text`) and a photo few-shot (`photos`, enrolled at once, or `person`, a file enrolled before) are required** (owner decision 59) | null: everyone, who = unknown |
| `what.movement` | `{text, rule}` (zero-shot: the rule is the JSON of `gamst/providers/movement_rule.py`, written by the caller from the text), `{examples, negatives}` (few-shot clips, each `{file, start_s, end_s}` or `"FILE@START-END"`), **both** (`{text, rule, examples, negatives, combine}`, owner decision 58), or `{exemplar}` (an enrolled movement file) | examples and photos are enrolled at once, into files beside the recipe |
| `what.words` | a list of words (`re:`, `word:W`, `word:W*` allowed) or a terms file | |
| `what.rhythm` | `{input, channel, band}`: the rate of one sensor column (`heart`, `breath`, `tremor`) | |
| `mode` | `count` (default) or `search` | |
| `timing` | `realtime` or `post`; default realtime for a camera or microphone, else post | |
| `recipe` | a built-in recipe used as it is, or customised with `with`, `without`, `set`, `pose`, `asr` | instead of composing |
| `consent` | `{capture, remote_asr}`: needed before a camera or microphone opens, or sound goes to a paid recogniser | |
| `start`, `wait` | run it at once (MCP: in this session; CLI: `--start`); wait for a finished file's run | |

`who` alone (no `what`) answers "when is the person on screen": pose and the person binding only.

### Text, clips, or both (owner decision 58)

| given | what counts | evidence on every event |
|---|---|---|
| `text` + `rule` | the rule (an ordered list of steps over joints, `movement_rule.py`) | `evidence.rule`: `satisfied`, `steps_ms` (when each step was reached) |
| `examples` (+ `negatives`) | a copy of an example (subsequence DTW over pose channels), thresholds from the examples (`movement_match.derive`) | `evidence.example`: `index` of the nearest example, its `distance`, the `enter` it was held against, `negative_distance` |
| `text` + `rule` + `examples` | the rule is the skeleton (which joints, which comparisons, in which order); the clips set its numbers (`movement_rule.calibrate`: positives only widen a comparison they do not pass, only a negative narrows one; a clip trimmed after a step does not teach that step) and are compared too. **count mode: the rule and an example (`combine: all`); search mode: either (`any`), one event when both find it** | both, and `combine` |

Each described unit is one event with its own start and end: one 관제탑 cycle (fold → spread → fold) or one punch;
repetitions are never merged (decision 60). Measured on the owner's labelled clips:
[measurements/action-spec-2026-11-28.md](measurements/action-spec-2026-11-28.md).

A request with everything at once (the 관제탑 by 감스트):

```json
{"name": "gwan-gamst", "inputs": {"video": "감스트-관제탑 모음집.mp4"},
 "who": {"name": "감스트", "text": "탈색 짧은 머리, 검은 뿔테", "photos": ["gamst1.jpg", "gamst2.jpg", "gamst3.jpg"],
         "consent": "owner, for counting on his own clips, 2026-11-28"},
 "what": {"movement": {"text": "관제탑: 두 팔을 앞으로 오므렸다가 양옆으로 폈다가 다시 접는다 (1회씩 따로)",
                       "rule": {"steps": ["... docs/movements/gwanjetap.rule.json ..."]},
                       "examples": ["감스트-관제탑 모음집.mp4@62.051-63.951", {"file": "감스트-관제탑 모음집.mp4", "start_s": 64.127, "end_s": 64.76}]}},
 "mode": "count", "timing": "post", "start": true, "wait": true}
```

Refusals (one sentence each, the same as `set_instruction` and the command line): `who` with a name and no photo
few-shot ("a text alone never identifies a person: add who.photos (with who.consent) or who.person"); photos without a
name or description ("say who it is in words as well"); photos without `consent`; a movement as text only ("GAMST has
no hidden LLM ... write it as a rule, give example clips, both, or an enrolled movement file"); `negatives` without
`examples`.

## What it composes

Each capability takes the lane of the built-in recipe it was measured in, so a composed plan is the hand-assembled one
(issue #369's acceptance; `tests/test_compose_recipe.py` checks that the modules equal those of the built-in recipe):

| capability | lane | from |
|---|---|---|
| movement for every track, with who | `pose` (RTMW-l), `person_binding`, `target_person` (off), `pose_channels`, `exemplar`, `repetition` | `person-action-count` |
| who alone | `pose`, `person_binding` | `person-action-count` |
| words | every module of `person-term-count` unchanged (decision 48); without `who` the voice parts stay out | `person-term-count` |
| semantic | `semantic`, **OFF** in every composed recipe with pictures (decision 47) | `person-action-count` |
| sensor rhythm | `rhythm` on one column | `gamst/providers/rhythm.py` |

- **Pose model** (decision 38): RTMW-l 384x288. Realtime: a short benchmark at the start keeps it when live + one L2
  worker reach the input frame rate, else MediaPipe full counts and the report says so. Post: every frame, nothing late,
  so the most accurate model counts. `pose` names another `vision.pose` candidate.
- **Who**: a text (name or description) and a photo few-shot are both required (decision 59). Photos are enrolled
  first (`gamst recipe enroll person-action-count --kind person --image ...`, embeddings only, beside the recipe as
  `NAME.person.json`, the description kept as `description`); the person file must pass the check (consent, one person
  per sample, agreeing face crops or a voice); a name given must be the enrolled one. The description is kept with the
  reference and shown back; it does not identify anyone (no model reads it). The refusal is `refusal()`'s sentence, the same as on the CLI, HUD and in MCP.
- **Validation**: `gamst.recipes.validate` (what `load` does for a file): every input is read from a source it supports,
  every `consumes` names an input or an earlier module, every part is in `PARTS`, the view exists; every reference the
  chosen recipe needs is given.
- **Words**: the default recogniser (Scribe v3 Live) is hosted and paid. Composing never sends anything; `before_run`
  says `consent.remote_asr` is needed, and `start` refuses without it.

## The answer

```json
{"ok": true, "name": "shotgun-gamst",
 "reasons": [{"capability": "pose model", "chose": "rtmw.l_384x288", "why": "decision 38: ... post: a finished file is counted on every frame ..."},
             {"capability": "movement", "lanes": ["pose", "person_binding", "target_person", "pose_channels", "exemplar", "repetition"], "recipe": "person-action-count", "why": "..."},
             {"capability": "who", "chose": "감스트", "why": "few-shot person reference checked ...: identified by face; ..."},
             {"capability": "semantic", "chose": "off", "why": "... (decision 47) ..."}, "..."],
 "run": ["gamst", "recipe", "run", ".../recipes/shotgun-gamst.json", "--video", ".../b.mp4",
         "--exemplar", ".../recipes/shotgun-gamst.movement.json", "--person", ".../gamst.person.json", "--who", "감스트",
         "--timing", "post", "--headless"],
 "saved": {"recipe": ".../recipes/shotgun-gamst.json", "movement": ".../recipes/shotgun-gamst.movement.json"},
 "before_run": [], "inputs": ["camera"], "modules": ["pose", "person_binding", "pose_channels", "exemplar", "repetition"],
 "left_out": ["target_person", "semantic"], "records": ["temporal.exemplar_events", "temporal.repetition_count", "vision.person_binding"],
 "setup_roles": {"vision.pose": "rtmw.l_384x288", "vision.face_identity": "lvface.b", "vision.body_identity": "osnet.ain_x1_0_msmt17", "...": "..."},
 "report": {"ok": true, "frames": 714, "counts": {"temporal.exemplar_events": {"exemplar.matched": 16}, "...": {}}, "...": "..."}}
```

The saved recipe holds `composed: {request, reasons, run}`; `gamst recipe run <saved file> ...` (the `run` command)
runs it again. Missing pieces:

```json
{"ok": false, "missing": ["what.movement: GAMST has no hidden LLM to read a movement in words: write it as a rule (what.movement.rule, ...) or give an enrolled movement file (what.movement.exemplar)."]}
```

## Command line

```bash
gamst recipe compose request.json                   # print the composition, write nothing
gamst recipe compose request.json --out shotgun.json  # write shotgun.json (and shotgun.movement.json for a rule)
gamst recipe compose request.json --out shotgun.json --start     # and run it, as gamst recipe run
```

On the command line paths are the owner's: any file. A `"start": true` in the file is the same as `--start`.

## MCP

```bash
gamst mcp --clips DIR [--run-profile PROFILE.json] [--store run.sqlite --max-records 100000]
```

- Files are named by their path inside the `--clips` folders (the owner's); anything else is refused. The recipe (and a
  rule's movement file) is saved in the first folder under `recipes/`.
- `count_events` (`who`, `event: exemplar.matched`) returns each occurrence's `start_ms`, `end_ms` and its `evidence`
  (rule steps, nearest example); `get_timeline` shows the same events as intervals, and `input_starts` turns capture
  times into seconds from the start of the video.
- `start: true` runs the recipe in this session, on the server's own runtime, so `count_events`, `get_timeline` (`who`),
  `get_pipeline`, `set_instruction` and `enroll_person` read and change that run. Its JSON lines go to stderr, never to
  the protocol's stdout. One run at a time; `wait: true` waits for a finished file's run and returns its report.
  After the run `get_pipeline` returns `{running: false, last_run}`.
- A run with a camera or microphone needs `consent.capture`; a hosted recogniser `consent.remote_asr`.
- Inside a running recipe (`gamst recipe run --serve`) `compose_recipe` composes with the run's `--clips` folders; it
  cannot start a second run.

## The semantic lane as a causal state chain (decision 47)

The semantic module stays OFF in every recipe (`--with semantic` turns it on; it needs a VLM endpoint set up). When it
runs, each state record (`vision.streaming_semantic`) carries `chain` (`gamst/state_chain.py`):

```json
{"state_id": "ss:6f1c…", "previous_state_id": "ss:83b4…",
 "changed": {"set": {}, "changed": {"action": {"from": "none", "to": "shotgun"}}, "removed": [], "truncated": false},
 "caused_by": [{"record_id": "…", "capability": "temporal.exemplar_events", "event": "exemplar.matched",
                "label": "slam", "start_ms": 1795…, "end_ms": 1795…, "who": "감스트", "track_id": "…"}],
 "since_ms": 1795…, "attribution": "interval"}
```

- `previous_state_id` links the state before it; a new chain starts when the model's own state starts again (reset, or
  a gap inside a segment).
- `changed` is a field diff (dotted paths, three levels, at most 32 entries).
- `caused_by` holds the records of the same or earlier time that ended since the previous state: the recipe's event
  records (movements, words, person bindings) and utterances, read from the Store by the module (also what was not given
  to the model). It is co-occurrence in that interval (`attribution: "interval"`), not a proven cause; a state that did
  not change names none.
- MCP `explain_state {state_id | at_ms, steps <= 32, max_bytes}` walks the chain backwards and says why it stopped
  (`start`, `steps`, `not_retained`, `bytes`); each cause says its `who` as resolved now (`who_now` when a later binding
  changed it). `get_semantic_state` (the short-term chain of `push_semantic_state`) carries the same `chain`.
- No hosted VLM was called for this: the tests use a stand-in model.

## Conversational demo

`docs/demo/conversation/conversation.py` is a scripted MCP client: it calls `MCPServer.call_tool` as an LLM's tool calls
would arrive, prints each call and its bounded JSON answer, then an answer sentence assembled from the results. The
sentence is a template that **stands in for the LLM**. Run (heavy: pose, face and appearance models):
`try-it/demo/conversation/run.sh` (through `guard.sh`). The transcript of one run is
[demo/conversation/transcript-2026-11-27.txt](demo/conversation/transcript-2026-11-27.txt). Excerpt (the whole demo took
32 s of wall time; the shotgun run itself `run_s` 16.09 for 714 frames, RTMW-l + LVFace-B + OSNet-AIN on this Mac):

```
USER: 감스트가 샷건을 몇 번 쳤어? 언제?
>>> LLM calls compose_recipe   {name, inputs: {video: b.mp4}, who: {name: 감스트, person: gamst.person.json},
                                what: {movement: {text, rule}}, mode: count, timing: post, start: true, wait: true}
<<< compose_recipe: 9454 bytes   ok, reasons [...], report {frames: 714, run_s: 16.09, counts: {exemplar.matched: 16}, complete: true}
>>> LLM calls count_events     {capability_id: temporal.exemplar_events, event: exemplar.matched, who: 감스트}
<<< count_events: 9113 bytes     count 16, evidence [{event: {start_ms, end_ms, who: 감스트, track_id}}, ...]
>>> LLM calls count_events     {..., group_by: who}            groups {감스트: 16}
>>> LLM calls get_timeline     {start_ms, end_ms, step_ms: 2000, who: 감스트}   who.seen [{start_ms, end_ms}]
ANSWER: 감스트는 샷건을 16번 쳤어요. 영상 시작부터 각 동작이 끝난 때는 2.07초, 2.48초, ... 6.83초입니다. 사람별로는 감스트 16번.
        감스트가 화면에 보인 구간: 0.00초-11.88초.

USER: 뽀또라고 몇 번 말했어?
>>> LLM calls compose_recipe   {inputs: {audio: a.wav}, who: {..., voice: voice.json}, what: {words: [뽀또]}, with: [verification, vote, tse]}
<<< compose_recipe               ok, before_run: [hosted, paid recognisers need consent.remote_asr; diarization needs consent]
NOT STARTED (hosted ASR costs money): the stored records of run m1 (2026-11-26) are read instead
>>> LLM calls count_events     {capability_id: speech.term_events, event: term.spoken, window: 60d}   count 8 (final 8), uncertain 2
ANSWER: 감스트는 '뽀또'를 8번 말했어요 (확정 8, 잠정 0). 녹음 시작부터 11.56초, ... 47.56초에 말했어요 (...). 목소리가 겹쳐 아직
        판단하지 못한 것이 2개 있고, 세지 않았지만 '말하지 않았다'는 뜻은 아니에요.
```

Against the owner's 16 slams (`testbed/truth_hit.json`, at the end of each, tolerance 0.2 s) the 16 end times printed
(rounded to 0.01 s) are all within 0.02 s (mean 0.007 s), one to one in order. Who = 감스트 for all 16 rests on the
person binding (the only person in the clip, not owner-labelled; docs/measurements/person-identity-2026-11-27.md
section 6). The 8 with 2 uncertain is m1's own result (owner labels: 10 said, 7 of the 8 right;
docs/measurements/tse-2026-11-26.md).

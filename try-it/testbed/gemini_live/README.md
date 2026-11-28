# GAMST vs Gemini Live vs Claude on the owner's truth: design, cost, commands

Status 2026-11-28: **harness and estimate only. Nothing has been sent to Google or run on Claude.** The Gemini arm is
blocked: billing is disabled on the GCP projects meant for it (see "Blockers").

## 요약 (한국어)

- **무엇을 비교하나:** 같은 영상, 같은 정답(v2), 같은 채점기로 GAMST, Gemini Live, Claude Code를 비교합니다.
  - 영상: 관제탑(522초), 맨까송(67초), 샷건 슬램(11.9초).
  - 정답: 관제탑 감스트 46개(예시 7개 구간 빼면 39개) / 다른 사람 22개, 맨까송 감스트 펀치 16개, 슬램 16개.
- **Gemini Live 모델:** Vertex AI Live API의 `gemini-3.8-live`(GA, 권장 모델). Gemini 3.1 Pro는 쓰지 않습니다(하네스가 막음).
- **Gemini가 실제로 보는 것:** 초당 1장(Live API 최대). 768×768 이하로 줄인 JPEG와 16 kHz 소리입니다.
- **비용 추정 (2번씩, 한국어 전부 + 싼 영어 조건):** Gemini Live **$6.94**, Claude **$12.59**, 합계 **$19.53** (≤ $20).
  - 관제탑 실시간 한 번이 $2.44로 가장 큽니다. Live API는 매 응답마다 세션 안의 지난 영상 토큰을 **다시 청구**하기 때문입니다.
- **막힌 것:**
  1. GCP 결제가 꺼져 있습니다(`optimum-zone`, `gamst-bench` 모두 `billingEnabled: false`).
  2. 소리+영상 세션은 압축 없이 **2분**까지입니다. 그래서 관제탑은 110초씩 5개 세션으로 나눕니다(경계에 걸친 정답 동작은 0개).
  3. 초당 1장이라 0.3–0.7초짜리 동작(맨까송 펀치, 슬램)은 대부분 프레임 사이에 있습니다. 이것이 측정 대상입니다.

## The model and its limits (sources)

| item | value | source |
|---|---|---|
| model | `gemini-3.8-live`, generally available, "Recommended" on Vertex AI (also `gemini-live-2.5-flash-native-audio`, GA) | https://docs.cloud.google.com/vertex-ai/generative-ai/docs/live-api |
| video input | JPEG, **1 FPS**, standard input 768 x 768 | same page; https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-8-live.md.txt |
| output | text and audio (TEXT response requested) | 3-8-live model page |
| max input | 128K tokens; thinking not supported | 3-8-live model page |
| regions | `us`/`eu` multi-region, `us-central1` (Standard PayGo) | 3-8-live model page |
| session length | audio-only 15 min, **audio+video 2 min** without context compression; connection ~10 min | https://aistudio.google.com/docs/live-api/session-management |
| price, $ / 1M tokens | text in 0.75, **video/image in 1.00**, **audio in 3.00**, text out 4.50, audio out 12.00 (Standard PayGo, non-global) | https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing ("Gemini 3.8 Live API") |
| billing rule | "tokens consumption is calculated per turn ... Users are charged for all tokens present in the Session Context Window during that turn ... Tokens from past turns are re-processed and billed in every new turn"; audio 25 tokens/s; image 258 tokens per image | same pricing page, "Live API Billing Details" |

The repository's earlier coin-toss comparison (README, "Gemini Live takes video at 1 fps; it answered 'Zero'") left no
harness in the repository or in `try-it/` (searched for `gemini live`, `bidi`, `live.connect`,
`send_realtime_input`). The only Live code is `examples/companion/gemini.py`, a text-only companion; this harness reuses
its connection pattern (`client.aio.live.connect`, TEXT responses) and adds video, audio and activity control.

## Experiment design

**Inputs, the same for every arm:**
- the clip (TestBed, read-only);
- the five reference photos of 감스트 `<try-it>/testbed/identity/photos/a_{5.0,6.0,7.5,8.5,9.5}.jpg`. These are the
  photo form of the identity reference of #379/#381 (`감스트.person.json`, embeddings only, enrolled from the same shop
  clip a.mp4);
- the action text exactly as the owner defined it (ko: the ground-truth README; en: decisions 62/63). The slam clip
  has no Korean owner text on Main: the English is the owner's definition in `testbed-2026-11-26.md`, the Korean is a
  translation made here (`live_eval.CLIPS`).
- The truth is never given. `tests/` checks that no truth time appears in any prompt.

**Arm 1, Gemini Live (`harness.py`):**
- **rt (realtime):** the clip at its own pace: 1 frame/s, each frame preceded by `[t=12.0s]`, plus 16 kHz audio.
  - Automatic voice activity detection is off, so the clip's own speech does not trigger answers. Every `poll_s`
    seconds (5 s; slam 2 s) the window is closed with a cue, and the model lists the actions by 감스트 that ended
    since its last answer. `NO_INTERRUPTION`: streaming goes on while it answers.
  - Sessions cover at most 110 s of clip (limit 120 s). The next session connects 6 s early and gets the photos
    again. No truth occurrence straddles the 관제탑 session boundaries (110/220/330/440 s).
- **after-the-fact:** when a session's stream ends, it is asked for the full list of its part (start, end, who: 감스트
  or other). In `rt` this is the last turn of each session. In `post` it is the only turn (no cues; a cheaper,
  cleaner after-the-fact number).
- **Measured:** count, precision, recall and F1 (감스트; everyone on the multi-person clips); start and end error;
  latency = the clip time an answer line arrived − the truth occurrence's end (matched pairs, realtime only); frames
  actually sent per clip second; wall time; tokens and USD from the API's own `usage_metadata` per turn; the
  estimate beside it.

**Arm 2, Claude Code (`claude_arm.py`):** a fresh subagent with no GAMST output.
- Input: 4 x 4 contact sheets (384 px tiles, each tile labelled with its clip time, 1536 x 936 per sheet) at 2 and
  4 fps for 관제탑/맨까송 and the native 60 fps for the slam clip; the same photos and action text (`prompt.md`).
- Output: JSON lines (start, end, who), parsed and scored like Gemini's after-the-fact list.
- Tokens from the agent's reported usage; USD at Anthropic's list price for `claude-opus-5-5` ($4 in / $20 out per
  1M; cache write $5, cache read $0.20).
- No realtime mode: the agent sees the whole clip first.

**Arm 3, GAMST:** the numbers already on Main (no new run needed for accuracy), API cost 0 (local compute):

| clip | GAMST setting (Main doc) | 감스트 matched / said / truth | F1 | start / end error | time |
|---|---|---|---|---|---|
| 관제탑 | text rule as written, no tuning (action-spec, v1 = v2) | 20 / 21 / 39 | 0.667 | 0.47 / 0.28 s | post run 10.0–10.6 min for 522 s (identity-recall) |
| 관제탑 | best text rule in-sample / held-out odd→even | 31 / 33 / 39 | 0.861 / 0.727 | 0.50 / 0.31 s | same |
| 관제탑 | text + 7 examples, search, v2 (in-sample) | 36 / 42 / 39 | 0.889 | | same |
| 맨까송 | punch rule as written | 8 / 34 / 16 | 0.320 | 0.32 / 0.50 s | |
| 맨까송 | identity-recall chosen setting (v2) | 9 / 13–14 / 16 | 0.600–0.621 | | |
| slam | text rule, every frame | 16 / 16 / 16 | 1.0 (0.2 s) | 9 ms | 28.5 s for 11.9 s (post); live L1 latency p50 26–40 ms |

- The 관제탑/맨까송 numbers exclude the example ranges (39 of 46), as Gemini's are scored by default (`score(...,
  exclude_examples=True)`; the full 46 is also reported).
- **Gap:** GAMST realtime latency is measured on Main only for the slam clip. For 관제탑/맨까송 a `--realtime` run of
  the same recipe is needed (local, free) before the latency row can be filled.

**Fairness:** Korean prompts everywhere, English where cheap (all but 관제탑 realtime, +$4.88 for 2 runs); Gemini's
frames per second stated from what was sent; every hosted condition run twice (lesson 22); same scorer and tolerance
as GAMST (0.5 s on the multi-person clips; slam: by end time within 0.2 s, the testbed rule).

## Cost estimate (2 runs each)

Formula (Gemini, per run): sessions of ≤ 110 s; in a session, turn k bills the whole context after w_k clip seconds:

    prompt_k = 450 text + 5 photos x 258 + w_k x (258 video + 8 text + 25 audio) [+ earlier cue/answer text]
    USD = Σ_k (text x 0.75 + video/image x 1.00 + audio x 3.00) / 1e6 + output x 4.50 / 1e6

| condition | sessions | turns | input tokens billed / run | USD / run | USD x 2 |
|---|---:|---:|---:|---:|---:|
| gemini 관제탑 rt ko | 5 | 110 | 2,147,822 | 2.442 | 4.884 |
| gemini 관제탑 post ko | 5 | 5 | 161,178 | 0.201 | 0.403 |
| gemini 맨까송 rt ko | 1 | 15 | 209,887 | 0.241 | 0.482 |
| gemini 맨까송 post ko | 1 | 1 | 21,401 | 0.029 | 0.058 |
| gemini slam rt ko | 1 | 7 | 31,197 | 0.036 | 0.072 |
| gemini slam post ko | 1 | 1 | 5,292 | 0.008 | 0.015 |
| gemini 관제탑 post en | 5 | 5 | 161,178 | 0.201 | 0.403 |
| gemini 맨까송 rt en / post en | 1 | 15 / 1 | 209,887 / 21,401 | 0.241 / 0.029 | 0.540 |
| gemini slam rt en / post en | 1 | 7 / 1 | 31,197 / 5,292 | 0.036 / 0.008 | 0.087 |
| **Gemini total** | | | | | **6.94** |
| claude 관제탑 2 fps / 4 fps | 66 / 131 sheets | | 153,737 / 278,342 new | 1.390 / 2.413 | 7.604 |
| claude 맨까송 2 fps / 4 fps | 9 / 17 sheets | | 44,468 / 59,804 new | 0.654 / 0.749 | 2.805 |
| claude slam 60 fps | 45 sheets | | 113,480 new | 1.091 | 2.183 |
| **Claude total** | | | | | **12.59** |
| **All** | | | | | **19.53** (≈ 27,300 KRW at 1,400 KRW/USD) |

- Uncertain inputs: the text token counts (Korean), whether the server bills 258 tokens per frame for a 768 x 432
  frame (the stated conversion), and the Claude agent's overhead (25K tokens assumed) and output (20K). Every run
  records the real usage; `--max-usd` refuses a run whose estimate is above a cap.
- For one hour of video, Gemini realtime at 5 s cues and 110 s sessions: 3600/522 x 2.44 = **≈ $16.8/h**; after-the-fact
  only: ≈ $1.39/h.

## Blockers

1. **GCP billing is disabled** on the projects meant for this (`gcloud billing projects describe`):
   `optimum-zone` (gcloud default) `billingEnabled: false`, `gamst-bench` `false`. The only projects with billing on
   are `songsuro` and `ward-gps-alarm`, which are other projects, so they are not used. Auth itself is fine:
   account active, application-default credentials valid, `aiplatform.googleapis.com` enabled on `optimum-zone`.
   **The owner fixes:** link a billing account (with the credits) to the chosen project, e.g.
   `gcloud billing projects link gamst-bench --billing-account=<ID>`, and enable `aiplatform.googleapis.com` there.
2. **Session length:** audio+video sessions end at 2 minutes without compression; handled by 110 s sessions (5 for
   관제탑). With compression the reference photos would slide out of context (system instructions are text only).
3. **1 frame per second** is the Live API's maximum; Gemini cannot see more. Stated in every result
   (`frames_per_clip_s`).
4. `gcloud` is not on `PATH` (it is at `~/google-cloud-sdk/bin/gcloud`; the harness finds it).
5. `google-genai` is not installed in the GAMST venvs (`pip install 'gamst-sdk[gemini]'` or `pip install google-genai`).
6. No Gemini/Google key in `~/.config/gamst/keys.env` (not needed for Vertex: it uses ADC).

## Commands

```bash
cd <repo>            # the worktree of exp/gemini-live-vs-gamst
P=<billing-enabled GCP project>
python try-it/testbed/gemini_live/harness.py --plan                      # this table
python try-it/testbed/gemini_live/harness.py --check --project $P        # read-only auth + billing check
# one condition (add --go to send; without it only the estimate is printed); RUN = 1 or 2
python try-it/testbed/gemini_live/harness.py --clip gwan --mode rt   --lang ko --run RUN --project $P --go
python try-it/testbed/gemini_live/harness.py --clip gwan --mode post --lang ko --run RUN --project $P --go
python try-it/testbed/gemini_live/harness.py --clip man  --mode rt   --lang ko --run RUN --project $P --go
python try-it/testbed/gemini_live/harness.py --clip man  --mode post --lang ko --run RUN --project $P --go
python try-it/testbed/gemini_live/harness.py --clip shot --mode rt   --lang ko --run RUN --project $P --go
python try-it/testbed/gemini_live/harness.py --clip shot --mode post --lang ko --run RUN --project $P --go
python try-it/testbed/gemini_live/harness.py --clip gwan --mode post --lang en --run RUN --project $P --go
python try-it/testbed/gemini_live/harness.py --clip man  --mode rt   --lang en --run RUN --project $P --go
python try-it/testbed/gemini_live/harness.py --clip man  --mode post --lang en --run RUN --project $P --go
python try-it/testbed/gemini_live/harness.py --clip shot --mode rt   --lang en --run RUN --project $P --go
python try-it/testbed/gemini_live/harness.py --clip shot --mode post --lang en --run RUN --project $P --go
# Claude arm: build sheets + prompt (local), give prompt.md to a fresh subagent, then score its answer
python try-it/testbed/gemini_live/claude_arm.py build --clip gwan --fps 2       # also: gwan 4, man 2, man 4, shot 60
python try-it/testbed/gemini_live/claude_arm.py score --clip gwan --fps 2 --run 1 --answer answer.txt \
    --input-tokens N --output-tokens N --cache-read-tokens N --cache-write-tokens N --wall-s S
# offline tests (no network)
python -m unittest discover -s try-it/testbed/gemini_live/tests -v
```

Results go to `<try-it>/testbed/gemini_live/runs/<clip>.<mode>.<lang>.r<run>/` (`result.json`, `messages.jsonl`) and
`<try-it>/testbed/gemini_live/claude/<clip>@<fps>/result.r<run>.json`, outside the repository.

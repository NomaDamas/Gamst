# HUD demo: counting one person's words, live with a microphone and as a video

Two screens show the same revision states (`gamst/counting.py`): the **live window** of `gamst recipe run`
(`gamst/views.py`, `CountView`) while you speak, and the **video HUD** (`python -m gamst.count_video`) drawn afterwards
from a run's report on a recording. Both read only committed records; neither computes a count of its own.

`gamst recipe hud` opens the live window with a **setup step** first (section 0): what to count, of whom, from which
input, then the run in the same window.

## 0. The setup step (`gamst recipe hud`, 2026-11-28)

The user-defined input module of the perception input, before the run starts. Code: `gamst/hud_setup.py`. It fills in
a `compose_recipe` request ([compose-recipe.md](compose-recipe.md)); validation is
`gamst.recipes.compose.validate_request`, the first step of `compose` itself, so the setup step refuses exactly what
`compose_recipe` and the command line refuse. While something is missing, Start is off and one Korean line says what to
add. Start composes (`compose`, which enrolls the example clips) and runs the composed recipe in the same window.

```bash
gamst recipe hud                                   # everything chosen in the window
gamst recipe hud --input "감스트-관제탑 모음집.mp4" --who 감스트 --photo a.jpg --photo b.jpg --photo c.jpg \
  --action "관제탑: 두 팔을 앞으로 오므렸다가 양옆으로 폈다가 다시 접는다" --rule docs/movements/gwanjetap.rule.json \
  --example "감스트-관제탑 모음집.mp4@62.051-63.951" --timing post \
  --profile PROFILE.json --out-dir gamst-hud -- --report run.json        # after --: options of gamst recipe run
```

Every option only fills in the window; `--start` starts at once when nothing is missing. `--input` takes a file or a
device and the kind is detected (`hud_setup.detect_input`), so nobody picks it: a video file (`.mp4 .mov .m4v .mkv .webm
.avi`, its sound for words), a WAV file, a sensor declaration (`.json`), a camera number (`0`, `camera 1`, `카메라`) or
the microphone (`mic`, `마이크`). Repeat it for several inputs (one picture source, one sound source). The older
`--video`, `--camera`, `--microphone`, `--sensor` still work and are read the same way. The composed recipe, the
enrolled person (`NAME.person.json`, embeddings only), the movement file and grabbed photos are saved in `--out-dir`
(default `./gamst-hud`). A video file with timing realtime is replayed at its own pace (`--realtime`): the camera
stand-in.

| part | in the window | rule (compose) |
|---|---|---|
| **1 입력** (input) | `파일` (the system's file dialog; video, WAV or a sensor declaration `{id, file, modality, columns, time}`, columns and time declared, never guessed; then `채널` / `대역` for the rhythm), `장치` (a camera number or `마이크`), `비우기`. The kind is detected from the file or device; there is no button per kind | at least one; one picture source, one sound source |
| **2 누구** (who, optional) | `이름`, `설명` (the system's text dialog, Korean input works); `사진 파일 추가`, `지금 화면을 사진으로` (grabs the picture shown, from the camera or the video), `사진 비우기` | a text (name or description) **and** photos (decision 59). The person file keeps a consent record, filled in as `GAMST 설치·사용자 동의`; nothing is asked |
| photo check | `사진 5장 · 얼굴 5개 일치 0.71 (기준 0.5) · 사진마다 한 사람` (green), or `검사 실패: ...` (red). The photos are enrolled in the background as soon as a name (or description) and photos are there (`compose.photo_reference`, the same function compose uses), then `person_reference.check`: faces agree, exactly one person per photo | the request then carries the enrolled file (`who.person`), so compose does not enroll again |
| **3 무엇** (what) | `동작 글` (the action text), `규칙 파일` (a rule JSON), example clips trimmed on the input video, `단어` (words, the existing word counting) | text alone is refused (no hidden LLM): text + rule, clips, or both; negatives need examples |
| trimming | on the trim bar under the picture: `[` start, `]` end, `e` save as an example (green), `n` save as a negative (red); a click on the picture names the person the clip is learnt from (`at: [x, y]`, a cross); with a camera input, `예시 영상 고르기` picks the video to trim | one movement per clip, at most 30 s |
| **4 방식** | `세기` / `찾기` (count / search), `실시간` / `나중에(파일 전체)` (realtime / post) | post needs a file |

The bottom line (red while missing, green when ready) is the first missing sentence of `validate_request`, in Korean,
e.g. `사람 사진을 추가하세요: 이름만으로는 사람을 알아볼 수 없습니다(한 장에 한 사람).`, `동작을 글로만 줄 수는 없습니다:
규칙 파일을 고르거나 예시 구간을 저장하세요.`, `카메라와 마이크는 실시간만 됩니다: 때를 실시간으로 바꾸세요.`; then
`사진 ... 검사하는 중`, and a cost notice before a paid hosted call: `비용 안내: 단어 인식기는 유료 원격 서비스라 돈이
듭니다. 비용을 확인했으면 --allow-remote-asr 로 시작하세요.`

Keys of the setup step (listed under the picture): `스페이스` play / pause, `,` `.` 0.1 s, `<` `>` 1 s, `[` `]` start /
end, `e` / `n` save the span, `Enter` Start, `q` / `Esc` quit. Everything else is a button.

### Run on the owner's clips (2026-11-28)

Driven by a script (`try-it/testbed/hud_setup/drive.py`, local): the setup step, `compose` and the run are the real
ones; `cv2.imshow` was captured to PNG files and the clicks and keys were scripted, so no window opened. The clips were
read in place (read only); screenshots, grabbed photos and enrolled files stay in `try-it/testbed/hud_setup/`.

| run | setup | result |
|---|---|---|
| 관제탑 clip, post, first 2700 frames (90 s) | name 감스트 + 5 photo files (from `a.mp4`): `얼굴 4개 일치 0.60`; action text + `gwanjetap.rule.json` + 3 of the owner's example clips + one marked with a point; before the photos: `사람 사진을 추가하세요 ...`, Start off | 9 occurrences, all 감스트, equal to the run's report; each overlaps one of the owner's 14 labelled 감스트 occurrences in those 90 s (9 right, 5 missed); 105 s for 2700 frames with the window drawn |
| `b.mp4` as a camera stand-in (realtime, replayed at its pace) | 3 photos grabbed from the video in the window: `얼굴 3개 일치 0.53`; shotgun text + `rt_rule.json` | pose model measured at the start: `rtmw.l_384x288` 34.7 + 38.5 fps for a 60 fps input, kept; 16 occurrences, all 감스트 (the owner's count is 16), no frame lost, none LATE |

No L2 correction happened in these runs, so LATE was seen only in the unit tests (`tests/test_hud_setup.py`). The point
of the marked clip was the middle of the frame (the face finder of the script found no face), not a click of the owner.

Consent and cost (owner, 2026-11-28): the setup step asks no consent for the camera, the microphone or speech
recognition. GAMST is open source and installing it is the consent; `compose` no longer waits for `consent.capture`
(still accepted, it changes nothing). Cost is not consent: sound goes to a hosted, paid recogniser only after the cost
notice, with `--allow-remote-asr` (`consent.remote_asr` in the request); hosted diarization likewise with
`--allow-remote-diarization`. The person check (faces agree, one person per photo) is unchanged.

| state | meaning | colour (both screens) |
|---|---|---|
| heard (들림) | the live recogniser's partial hypothesis held the word steadily: counted at once, times estimated | pale beige |
| provisional (잠정) | the live recogniser's final has it (L1): counted, a later lane may still change it | orange |
| final (확정) | the verification lane confirmed it (or it was added there) | green |
| revoked (취소) | taken out of the count, with the reason (another voice, the recogniser withdrew it, nobody spoke) | red |
| uncertain (불확실) | overlapped voices, neither surely the person's nor another's: shown, not counted (count mode) | violet |
| candidate (후보) | search mode only: counted although it would have been uncertain or revoked; the reason is shown | green, "후보" |

## 1. Live demo with a microphone

Once, in the environment GAMST is installed in:

```bash
gamst recipe setup person-term-count          # installs only what the recipe uses (CAM++ speaker model, aligner)
```

Hosted recognisers need their keys in the environment (`ELEVENLABS_API_KEY` for the live recogniser; for the
verification lane also `OPENAI_API_KEY`, `SPEECHMATICS_API_KEY`, `PYANNOTEAI_API_KEY`).

### a) The live lane only (heard / provisional, another voice excluded)

```bash
gamst recipe run person-term-count --microphone --terms 뽀또 --language ko \
  --allow-remote-asr \
  --record-voice-first 10 \
  --voice-consent "<name> consents to GAMST keeping a sample of their voice on this computer for this demo" \
  --voice-out me.voice.json \
  --report live.json
```

### b) Every state, with corrections when they are decided (the configuration of the promo videos)

```bash
gamst recipe run person-term-count --microphone --terms 뽀또 --language ko \
  --allow-remote-asr --allow-remote-diarization \
  --with verification --with vote \
  --record-voice-first 10 \
  --voice-consent "<name> consents to GAMST keeping a sample of their voice on this computer for this demo" \
  --voice-out me.voice.json \
  --serve 8765 --report live.json
```

Add `--mode search` to show search mode (owner decision 33: never miss the person; unclear words count as
candidates). A second run can reuse the sample: replace `--record-voice-first 10 --voice-consent ... --voice-out ...`
with `--voice me.voice.json`.

What each flag is for:

- `--allow-remote-asr`: consent that the sound goes to the hosted recognisers (ElevenLabs Scribe v3 Live; with
  `--with vote` also Speechmatics, with `--with verification` also OpenAI gpt-live-transcribe). Each prints its list
  price before it starts. `--allow-remote-diarization`: consent for pyannoteAI Precision-3 (verification windows).
- `--record-voice-first 10` + `--voice-consent "..."`: the voice sample step (`gamst/recipes/voice_sample.py`),
  required for a live run without `--voice FILE`. The consent text is stored with the sample: say who consented, to
  what. `--voice-out FILE` keeps the sample (owner-readable only); without it a temporary file is removed after the run.
- `--serve 8765`: an agent can read the run and switch it over MCP (`gamst-mcp-live --url ws://127.0.0.1:8765`),
  e.g. `set_count_mode {"mode": "search", "actor": "user"}` or `set_instruction` for other words.
- `--report live.json`: everything the HUD showed, with each event's history, for the video HUD or for scoring.

### The voice sample step (in the terminal, before the window opens)

1. The terminal prints the consent line, then `세려는 사람만 10초 동안 말해 주세요` (only the person to be counted
   speaks for 10 s) and counts down `3... 2... 1...`.
2. While it records, a level meter: `  4.0/10 s [##########..................] -32.1 dBFS`. Speak normally. If the
   loudest moment stays under -45 dBFS it says `very quiet: move closer to the microphone`.
3. `다시 녹음할까요? [y/N]` (record again?): `y` repeats steps 1-2, Enter keeps the sample.
4. If the sample's pieces disagree (mean pairwise similarity under 0.5), the run says
   `note: 목소리 기준 품질 낮음 ...` and the window shows the voice-quality warning (below). Nothing is counted during
   the sample: the run's counting starts after it.

### Keys (the window must have focus; listed at the bottom of the panel)

- `q` or `Esc`: stop the run (the report is written). After `gamst recipe hud`, the last picture stays after the input
  ends so the timeline can still be clicked; `q` closes it.
- `r`: reset every module (switched off and on: no evidence is joined across the reset).
- `a`: boxes of every person (otherwise only the selected person's box is drawn, the others faint).
- Click: an occurrence on the timeline opens its evidence card (a click on the card closes it); a box selects that person.
- `Ctrl-C` in the terminal also stops it.

### What the live window shows (`CountView`)

Korean words when Pillow and the Korean font are there (the video HUD's font), else the English words.

With a microphone only, the window has no camera picture: the left half says `소리만: 카메라 없음`; the panel is redrawn
at most every 100 ms. With a camera input (`--camera 0`, or a video file) the left half is the picture, shown at most
720 px tall.

On the picture: the box of the **selected person** (the named person at first, green, with the name), the other persons
faint (thin grey); `a` draws every box with its name or person id (`P2`). Boxes are those of the pose records
(`scene.poses`) and the person binding (`vision.person_binding`) of the same moment.

Right panel, top to bottom:

1. **Title line** (yellow): `GAMST  person-action-count   모드: 세기   때: 나중에` (search: `모드: 찾기`, then an orange
   `찾기: 애매한 것도 후보로 셉니다`).
2. **How it was set up**: `자세 모델 rtmw.l_384x288 (나중에: 모든 프레임, 가장 정확한 모델)` in post mode; in realtime the
   start benchmark (`gamst.recipes.choose_pose`): `자세 모델 rtmw.l_384x288: 측정 31+12 fps, 입력 30`, red with
   `(rtmw.l_384x288 너무 느림)` when MediaPipe full counts instead. `뺀 모듈: target_person, semantic` (modules of the recipe
   this run does not run) and `빠진 모델: body_identity: ...` (what a module left out while it was built: the report's
   `setup_notes`). A refused `who` (`gamst/person_reference.py`) and a low voice reference in red.
3. **Per capability**: the label and capability (`관제탑  temporal.exemplar_events`), `셈 N` (the module's own count), the
   **count per person**: the selected person in green, `다른 사람 N (P2 3 · P3 1)` in blue (who as the binding says now,
   retroactive within a track, one action on two tracks once: the same rule as MCP `count_events`); for words,
   `다른 목소리 N (세지 않음)`. Then the state tally `들림 N  잠정 N  확정 N  취소 N  불확실 N`, each in its colour.
4. **Newest events and corrections**, newest first: `+ 확정  관제탑  감스트`, `-> 취소  뽀또  (other_speaker)`,
   `-> 확정  뽀또  후보  (uncertain_overlap)`, `  다른 목소리  뽀또  목소리 0.10`. An L2 correction (realtime: frames the live
   lane could not reach, counted late) starts with **`LATE`**, in yellow.
5. `불완전: N 프레임 잃음` (red) when a module lost input.
6. **Modules**: `모듈 (r: 초기화, q: 끝내기)`, each `name: 켬|끔  latency ms`; the keys at the bottom.

**Timeline strip** (under the picture and the panel): the run from its first frame to now; each occurrence as a bar over
its span (green: the selected person; blue: others; red revoked, violet uncertain, an orange edge while provisional; a
**yellow mark above it: LATE**, an L2 correction); an orange cursor at now. A click on a bar opens the **evidence card**
over the picture:

```
#5 관제탑 · 감스트 (P1)
71.40초 - 72.10초 (0.70초) · 확정
셈: 규칙 그리고 예시
규칙: 맞음
  1단계 71.40초 · 2단계 71.53초 · 3단계 72.10초
가장 가까운 예시: 3번, 거리 0.0514 (기준 0.122)
  반례 거리: 없음  · 예시 구간 71.43초 - 72.10초
트랙 local-user:body:person-1
```

(the card of the 관제탑 run below, as drawn). `모듈 셈` is the module's own count, with the person as the binding knew it
when each event was committed; the count per person resolves the person now (retroactive), like MCP `count_events` and
the run's report, so the two can differ (8 and 9 in the run below).

Times are seconds from the first frame of the run (for a file: from the start of the video when it is run from 0).

The terminal prints every committed record as a JSON line too (the same as `--headless`).

### Live HUD status versus the video HUD (2026-11-28)

| video HUD element | live window |
|---|---|
| count, heard / provisional / final / revoked / uncertain tally, state colours | yes (Korean state words) |
| corrections at the moment they are decided, with the reason | yes, as text lines (`-> 취소 ... (reason)`); L2 corrections say `LATE` |
| candidate (search mode) | yes |
| voice-quality warning | yes |
| voice sample step | in the terminal before the window opens (countdown, level meter, record again) |
| timeline bar | yes: each occurrence with its span, clickable for its evidence (movements; words with a span) |
| per-person counts, the selected person's box | yes (2026-11-28) |
| numbered words (`#4`), struck-out number on a drop, flashing / dashed border | no (listed below) |

Not done here (larger than a fix): a big counter with the struck-out number when a count drops, the flashing border,
and the voice sample step inside the window instead of the terminal. The video HUD's drawing code
(`gamst/count_video.py`, `_draw_hud`) works on a whole report; a live version needs the same per-event history kept
incrementally.

## 2. The video HUD from a recording

Run the recipe on a recording as if it were live, keep the report, then draw it on the video:

```bash
gamst recipe run person-term-count --as-live --video clip.mp4 --terms 뽀또 --language ko \
  --allow-remote-asr --allow-remote-diarization --with verification --with vote \
  --voice-range 31.0-38.5 --voice-consent "..." --report run.json --headless

python -m gamst.count_video --report run.json --video clip.mp4 --out counted.mp4 --title 뽀또 \
  --count speech.term_events:term.spoken --other speech.term_events:term.other_voice \
  --count-label "감스트가 말한 뽀또" --other-label "다른 사람" --hold-settled
```

Options: `--lang en` (English HUD words), `--at-spoken` (presentation replay: each word's final result drawn when it
was said; the video says so; not live timing), `--slow 4` (four times slower, no sound), `--note "..."` (bottom caption).

### Every element of the video HUD (1920x1080)

- **Top-left title**: `GAMST · 뽀또` (`--title`).
- **Top, right of the video**: speed (`원본 속도` original speed, `4배 느리게` 4x slower) or, after the video ends
  with `--hold-settled`, `영상 끝 · 검증이 마지막 말을 정하는 중` (orange: the last words are still being verified).
- **Left: the video** (1376x774 area), with the original sound.
  - When a word enters the count: the video's border flashes green (provisional, from L1) or is drawn **dashed** pale
    (heard in a partial hypothesis), and a large count appears in the video's top-right corner (`+1 들림` under it for
    a heard word).
  - When a correction is decided: the border flashes in the correction's colour (red revoked, violet uncertain, green
    added/final).
- **Top-right panel: the counter**: `지금까지` (count so far) and the video time (`12.34초`); the subtitle
  `들림 즉시 · 잠정 L1 · 정정은 결정된 순간에` (heard at once · L1 provisional · corrections when decided) or, with
  `--at-spoken`, `말한 순간 기준 · 최종 결과`; the **big number** (green while it rises; when it drops, the old number
  in red struck out `→` the new one); the count label (`--count-label`, green); the **state tally** in colour:
  `확정 N  잠정 N  들림 N  취소 N  불확실 N` (heard and uncertain only when there are any).
- **Right panel: corrections** (newest first, up to three, fading out): headline and reason, e.g. `-1 취소 #9` /
  `다른 사람 목소리` (red), `-1 보류 #3` / `불확실: 목소리 겹침` (violet), `+1 추가 #12` / `다시 들어서 찾음` (green),
  `확정 #4` / `검증 통과` (green), `확정 #4 후보` / `찾기 모드: 목소리 겹침` (search mode); right-aligned
  `12.48초에 한 말` (when the word was said); `외 N건` when more.
- **Right panel: the word list** (newest first): state word in its colour (`들림` with a dotted underline, `잠정`,
  `확정`, `취소` with the time struck out, `불확실`, `제외` grey for another person's word), the time it was said, the
  word number `#n` and details (spelling heard, slot score, reason, flags such as `숫자일 수 있음`).
- **Bottom: the timeline** across the whole clip: a tick per word in its state's colour (dashed for heard; short grey
  ticks for another voice), and an orange cursor at the current time.
- **Bottom captions**: `--note` (grey), and when the voice sample was of low quality, `주의: ...` (orange) with the
  voice-quality line.

Colours: green final / counted, orange provisional, pale beige heard, red revoked, violet uncertain, grey another
person's word or not yet known.

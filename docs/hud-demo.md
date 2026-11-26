# HUD demo: counting one person's words, live with a microphone and as a video

Two screens show the same revision states (`gamst/counting.py`): the **live window** of `gamst recipe run`
(`gamst/views.py`, `CountView`) while you speak, and the **video HUD** (`python -m gamst.count_video`) drawn afterwards
from a run's report on a recording. Both read only committed records; neither computes a count of its own.

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

### Keys (the window must have focus)

- `q` or `Esc`: stop the run (the report is written).
- `r`: reset every module (switched off and on: no evidence is joined across the reset).
- `Ctrl-C` in the terminal also stops it.

### What the live window shows (`CountView`)

With a microphone only, the window has no camera picture: the left half says `sound only: no camera`; the panel is
redrawn at most every 100 ms. With a camera input (`--camera 0`, or a video file) the left half is the picture.

Right panel, top to bottom:

1. **Title line** (yellow): `GAMST  person-term-count   mode: count` (or `mode: search`, then an orange line
   `SEARCH: unclear words count as candidates`).
2. **Voice-quality warning** (red, only when low): `VOICE REFERENCE QUALITY LOW 0.42 -> replaced` (the sample's
   self-consistency and what was used instead: `replaced` by a `--voice-fallback`, or `combined`).
3. **Per capability** (`speech.term_events`): `count N` (the module's own count: heard + provisional + final), then the
   state tally `heard N  provisional N  final N  revoked N  uncertain N`, each in its colour.
4. **Newest events and corrections**, newest first, in the colour of their state: `+ heard 뽀또  voice 0.41` (a word
   entered), `-> final 뽀또` (a correction, shown the moment it was decided), `-> revoked 뽀또 (partial_withdrawn)`,
   `-> final 뽀또  candidate  (uncertain_overlap)` (search mode), `  other voice 뽀또  voice 0.10` (another person's
   word, not counted).
5. `INCOMPLETE: N frames lost` (red) when a module lost input.
6. **Modules**: `modules (r: reset, q: quit)`, each `name: on|off  latency ms`.

The terminal prints every committed record as a JSON line too (the same as `--headless`).

### Live HUD status versus the video HUD (2026-11-27)

| video HUD element | live window |
|---|---|
| count, heard / provisional / final / revoked / uncertain tally, state colours | yes |
| corrections at the moment they are decided, with the reason | yes, as text lines (`-> revoked ... (reason)`) |
| candidate (search mode) | yes |
| voice-quality warning | yes |
| voice sample step | in the terminal before the window opens (countdown, level meter, record again) |
| numbered words (`#4`), struck-out number on a drop, flashing / dashed border, timeline bar | no (listed below) |
| Korean correction headlines (`-1 취소 #9`, `+1 추가`) | no: English state words, Korean terms |

Not done here (larger than a fix): a big counter with the struck-out number when a count drops, word numbers, the
bottom timeline with a tick per word, the flashing border, and the voice sample step inside the window instead of the
terminal. The video HUD's drawing code (`gamst/count_video.py`, `_draw_hud`) works on a whole report; a live version
needs the same per-event history kept incrementally.

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

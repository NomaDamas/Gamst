# Wrap-up: counting one person's words and movements in real time (2026-11-25 – 27)

## 요약 (한국어)

- **동작:** 동작을 글로 설명하면(zero-shot, LLM이 규칙으로 바꿈) 또는 예시 영상(few-shot)으로 주면, GAMST가 **몇 번 했는지와 각 동작의 구간(시작·끝 시각)**을 실시간으로 찾는다. 감스트 샷건: 16/16, 시각 오차 평균 9 ms.
- **단어:** 지정한 단어를 **몇 번 말했는지와 각 단어의 구간**을 실시간으로 찾는다(말한 뒤 약 0.7초에 카운트). 철자 변형·낱말 단위·반복 수(달달달 = 3)를 지원한다.
- **누가 말했나:** 지정한 사람의 목소리 샘플(few-shot: 같은 마이크로 미리 녹음 / 파일 / 같은 녹음의 구간 / 자동 주인공)로 그 사람이 말한 것만 센다. 기준 품질을 검사한다. 사장님 라벨 4개 영상: 33개 중 30개 맞음, 틀림 3개.
- **모드:** 세기(count, 기본) / 찾기(search, 놓치지 않게 넉넉히 세고 후보와 이유를 함께 줌).
- **구조:** 입력 → Recipe → Pipeline 레인(L1 실시간 · L2 따라잡기 · L3 TSE · L4 검증) → Context(들림·잠정·확정·취소·애매함) → MCP → LLM.
  잠정은 바로, 고침은 결정된 순간에. 밀려도 소리를 버리지 않는다.
- 찾기 모드 측정(뽀또): 찾은 비율 78% → 89%, 센 것 중 맞은 비율 78% → 53% (놓침을 줄이는 대신 다른 사람 말이 후보로 들어옴).
- HUD: 영상 HUD 구성과 라이브 실행 명령은 `docs/hud-demo.md`. 데모 영상: 이슈 #371.
- 결정 1–34: `decision-log-2026-11-26.md` · 교훈 1–27: `lessons-2026-11-26.md` · 측정: `measurements/*-2026-11-26.md`.

## What GAMST can do now (all measured on owner clips, Mac CPU + hosted recognisers)

| capability | how it is specified | what comes back | measured |
|---|---|---|---|
| Count a movement and find each occurrence | text (zero-shot: an LLM writes a rule over pose joints), example clips (few-shot), or a rule file | count, each occurrence's start/end capture time, its evidence | shotgun clip: 16/16, mean time error 9 ms (`testbed-2026-11-26.md`) |
| Count a word and find each occurrence | words, spellings, `re:` patterns, whole-word `word:W` / repetitions `word:W*`, flags (`#may_be_number`) | count, each word's capture time, which recognisers heard it, state | words heard 45/46 on four clips; word → HUD ~0.7 s p50 (partials) |
| Only one person's words | voice few-shot: `--record-voice-first N` (same microphone, live), `--voice-wav`, `--voice-range`, automatic main speaker; quality check with fallback | each word's speaker decision and reason | four owner-labelled clips: 30/33 right, 3 wrong (v8) |
| Search instead of count | `--mode search`: recall first, candidates with reasons | candidates incl. uncertain ones, each with its reason | 뽀또: recall 78 % → 89 %, precision 78 % → 53 %; four clips 31 right (count mode 30) |
| Realtime with corrections | L1 instant, L4 later; HUD shows corrections when decided | states heard/provisional/final/revoked/uncertain/pending | catch-up under overload: 0 audio dropped, identical finals |

## Experiments in order (word counting, 뽀또 clip unless stated)

| version | change | result vs owner labels | lesson |
|---|---|---|---|
| v1 | Whisper turbo + voice from another recording; video drew counts at speech time after offline processing | 8 counted / 6 right | post-processing looked instant (lesson 25) |
| – | larger voice model, PixIT separation, lips (Light-ASD), Qwen3-ASR | no gain | reference > model (6) |
| catch-up B2 | Scribe v3 Live + verification (gpt-live, Precision-3) | 6/6, precision 1.00 | – |
| vote | 3-recogniser majority | 7/7 | later removed true words (27) |
| v4 (4 clips) | same, four clips | attributed 8/45 | the voice reference was the bottleneck |
| v5 | same-recording automatic voice, instant HUD | 28/34; 뽀또 worse | automatic reference can be mixed (7) |
| v6 | L1 counts every heard word; reference quality check | 뽀또 6/7 | instant must be at L1 (13) |
| v7 | counts from partial hypotheses | word → HUD 0.73 s p50 | – |
| v8 | SOTA first: a word only the best recogniser heard counts | **30/33** (시발 7/7, 달 13/13) | – |

TSE (PS4, Apache-2.0) changed no count and added 10-21 s on this Mac: opt-in. Live diarization mixed people: not used
for attribution. Remaining errors: an invented word counted (뽀또 24.16), another voice given to the person by diarization
(뽀또 27.42, 몇기야 one word), a word no recogniser heard (뽀또 14.05), a word left out of Scribe's final (몇기야 39.24).

## Demo videos

GitHub issue NomaDamas/Gamst#371 (table with playable videos): shotgun count,
뽀또 (Korean / English promo, live timing), 시발, 달, 몇기야. Promo videos are presentation replays (final results drawn when
each word was said, captioned); the live timing video shows what a live run shows.

## Next improvements (owner to schedule)

1. Another voice given to the person (뽀또 27.42, 몇기야): decide on long speaker segments with the person's sample recorded on
   the same microphone; compare with the other speakers present; anchor diarization to each word.
2. Words nobody heard (뽀또 14.05): speech enhancement before recognition, measured on the clip set.
3. Live demo: the voice-sample step in the HUD (`docs/hud-demo.md`), and the person's own microphone track when available.

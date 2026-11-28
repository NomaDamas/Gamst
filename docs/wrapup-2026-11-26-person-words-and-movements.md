# Wrap-up: counting one person's words and movements in real time (2026-11-25 – 27)

## 요약 (한국어)

- **동작 (일반화: 영상 속 인물이 특정 행동을 반복한 횟수와 구간 찾기):** 행동을 지정하는 방법은 세 가지다.
  ① 글 묘사(zero-shot): LLM(에이전트)이 관절 규칙으로 바꾸거나, VLM이 구간을 제안하고 사장님이 확인한다. GAMST 안에 숨은 LLM은 없다.
  ② 예시 영상(few-shot): 그 행동만 담은 짧은 영상 3개 이상(+ 아닌 예시)을 넣으면 임계값이 자동으로 정해진다.
  ③ 규칙 파일.
  결과는 횟수와 **행동마다 시작·끝 시각**, 근거다. 감스트 샷건(①, 글 묘사 → 규칙): 16/16, 시각 오차 평균 9 ms.
  **실시간(PR #373, 사장님 결정 37~42):** 가장 정확한 모델 하나(RTMW-l)가 모든 프레임을 직접 센다. 검증 레인·애매함·건너뛰기는 없앴다.
  배치(최대 8프레임)·고정 크기 CoreML 세션·예열로 60 fps 720p에서 프레임당 14 ms(실시간 계수 0.93). `--as-live`/`--realtime`
  60 fps 7회·30 fps 6회 모두 16/16, 건너뛴 프레임 0, 잃은 프레임 0, 지연 p50 60 fps 0.19~0.24초 / 30 fps 0.02~0.03초.
  후처리(`--timing post`) 60·30 fps 16/16. 일부러 느리게 만들면(밀림) 같은 모델의 L2가 밀린 구간을 보정한다(늦게 센 것은
  correction 표시): 고친 뒤 5회 모두 16/16, 중복 0, 보정 지연 p50 1.6~4.7초. 시작할 때 속도를 재서 못 따라가면 MediaPipe full로
  바꾸고 알린다. few-shot(RTMW로 등록) 14/16(2.05, 2.47 놓침). 측정: `measurements/testbed-2026-11-26.md` "One model, every frame".
- **단어:** 지정한 단어를 **몇 번 말했는지와 각 단어의 구간**을 실시간으로 찾는다(말한 뒤 약 0.7초에 카운트). 철자 변형·낱말 단위·반복 수(달달달 = 3)를 지원한다.
- **누가 말했나:** 지정한 사람의 목소리 샘플(few-shot: 같은 마이크로 미리 녹음 / 파일 / 같은 녹음의 구간 / 자동 주인공)로 그 사람이 말한 것만 센다. 기준 품질을 검사한다. 사장님 라벨 4개 영상: 33개 중 30개 맞음, 틀림 3개.
- **모드:** 세기(count, 기본) / 찾기(search, 놓치지 않게 넉넉히 세고 후보와 이유를 함께 줌).
- **구조:** 입력 → Recipe → Pipeline 레인(L1 실시간 · L2 따라잡기 · L3 TSE · L4 검증) → Context(들림·잠정·확정·취소·애매함) → MCP → LLM.
  잠정은 바로, 고침은 결정된 순간에. 밀려도 소리를 버리지 않는다.
- 찾기 모드 측정(뽀또): 찾은 비율 78% → 89%, 센 것 중 맞은 비율 78% → 53% (놓침을 줄이는 대신 다른 사람 말이 후보로 들어옴).
- HUD: 영상 HUD 구성과 라이브 실행 명령은 `docs/hud-demo.md`. 데모 영상은 저작권 때문에 올리지 않는다(로컬에만 있음).
- 통합 설계(누구·무엇·말, 사장님 승인 2026-11-27): `design-2026-11-27-integrated-perception.md`
- 결정 1–50: `decision-log-2026-11-26.md` · 교훈 1–37: `lessons-2026-11-26.md` · 측정: `measurements/*-2026-11-26.md`.

## What GAMST can do now (all measured on owner clips, Mac CPU + hosted recognisers)

| capability | how it is specified | what comes back | measured |
|---|---|---|---|
| Count a movement and find each occurrence | text (zero-shot: an LLM writes a rule over pose joints), example clips (few-shot), or a rule file | count, each occurrence's start/end capture time, its evidence | shotgun clip (text → rule): 16/16, mean time error 9 ms. One model (RTMW-l) over every frame (PR #373): post 16/16 at 60 and 30 fps; live and realtime 16/16 in 13 of 13 runs (60 and 30 fps), 0 skipped, 0 lost, latency p50 0.19-0.24 s (60 fps) / 24-26 ms (30 fps), 14 ms per frame at 60 fps (real-time factor 0.93); forced backlog: L2 corrections, 16/16 in 5 of 5 after a fix; few-shot 14/16 (`testbed-2026-11-26.md`) |
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

Not published: the clips are broadcast footage (copyright), so the videos (shotgun, 뽀또 Korean / English promo and live
timing, 시발, 달, 몇기야) stay local. Promo videos are presentation replays (final results drawn when each word was said,
captioned); the live timing video shows what a live run shows.

## Next improvements (owner to schedule)

1. Another voice given to the person (뽀또 27.42, 몇기야): decide on long speaker segments with the person's sample recorded on
   the same microphone; compare with the other speakers present; anchor diarization to each word.
2. Words nobody heard (뽀또 14.05): speech enhancement before recognition, measured on the clip set.
3. Live demo: the voice-sample step in the HUD (`docs/hud-demo.md`), and the person's own microphone track when available.

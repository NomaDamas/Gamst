# GAMST vs Gemini Live vs Claude Code on the owner's truth (2026-11-28)

Same clips, same truth (`docs/ground-truth/2026-11-28/v2/`, slam: `<try-it>/testbed/truth_hit.json`), same scorer
(`gamst.eval.multi_person`, 0.5 s; slam: end time within 0.2 s), same action text and the same five reference photos
of 감스트. Every number below is measured; costs are from each provider's own usage records. Harness, tests and
scripts: `try-it/testbed/gemini_live/` (README there: design, sources, commands).

## 요약 (한국어)

- **무엇을:** 관제탑(522초), 맨까송(67초), 샷건 슬램(11.9초)에서 "감스트의 동작"을 세 방법으로 찾게 하고, 같은 정답(v2)과 같은 채점기로 쟀습니다. 사진 5장과 사장님 동작 문구도 똑같이 줬습니다.
- **정확도 (감스트 F1, 회당 평균):**

  | 영상 | GAMST post (Main) / 실시간 (이번) | Gemini Live 실시간 / 사후 | Claude Code (장표) |
  |---|---|---|---|
  | 관제탑 | **0.89** (글+예시) / 0.61 (같은 설정, 프레임 43 % 잃음) | 0.26 / 0.30 | **0.88** (2 fps) · 0.85 (4 fps) |
  | 맨까송 펀치 | **0.60** / 0.38 | 0.11 / 0.12 | 0.17 (2 fps) · 0.12 (4 fps) |
  | 샷건 슬램 | **1.00** (16/16, 60 fps) / Main 실시간 16/16 | 0.06 / 0.00 | 0.00 (60 fps) |

- **한 줄로:** 관제탑에서 Claude Code 2 fps는 F1 0.88로 GAMST(0.89)와 거의 같았습니다. 그러나 영상 1시간에 $19가 들고, 영상을 끝까지 다 본 뒤에야 답합니다. 빠른 동작(펀치 0.17초, 슬램)에서는 두 호스팅 모델(Gemini Live, Claude Code) 모두 실패했습니다.

- **보는 장수:** Gemini Live는 **초당 1장**이 최대입니다(실제로 보낸 것 0.996–1.008장/초). GAMST는 모든 프레임(30/60 fps)을 봅니다. Claude는 우리가 준 장수(2·4·60 fps)를 봅니다.
- **비용 (1시간 영상 기준, 관제탑):** GAMST **0원**(로컬). Gemini Live 실시간 **$7.38(약 1만 원)**, 사후만 $0.91. Claude Code 2 fps **$19.00(약 2만 7천 원)**, 4 fps $30.60.
  - 맞힌 1개당: Gemini 실시간 $0.22, Claude 2 fps $0.09, GAMST 0원.
- **속도:**
  - Gemini 실시간 보고는 동작이 끝나고 **p50 19초**(관제탑), 10초(맨까송), 2초(슬램) 뒤에 왔습니다. 5초마다 물어봤지만 여러 질문에 한 번에 답하거나 건너뛰었습니다.
  - Claude는 영상을 다 본 뒤에만 답합니다(관제탑 2 fps 260초, 4 fps 407초).
  - GAMST 실시간(이번 실행, `--realtime`, Main 설정): 동작 끝 → 개수까지 관제탑 **p50 0.55초, 최대 25.4초**, 맨까송 **p50 2.5초, 최대 4.8초**입니다.
    - 바로 센 것(L1)은 p50 0.37초 / 0.36초입니다. 실시간 포즈가 입력보다 느려 뒤처진 구간은 L2 작업자가 나중에 셉니다(관제탑 76번 넘김, 약 2,400 프레임; 맨까송 29번, 약 1,000 프레임). 그 경우 p50 22.5초 / 4.6초입니다.
    - 관제탑은 여전히 디스크가 모자랐습니다. 남은 6.6 GB에서 바닥선 5 GB를 빼면 spill이 0.8 GB뿐이라 15,670 중 **6,777 프레임을 잃었고**, F1이 post(0.89)보다 낮은 0.61입니다. 맨까송은 잃은 프레임 0입니다.
    - 슬램은 Main의 실시간 측정이 p50 26–40 ms입니다.
- **실제로 쓴 돈:**
  - Gemini Live(GCP `songsuro`, Vertex): **$3.91**(28번 기록, 실패·재실행 포함, 상한 $20). 중간에 멈춘 1회는 사용량이 기록되지 않았습니다(≤ $0.01 추정).
  - Claude Code 비교군: 정가로 **$17.69**(10회). Claude Code 안에서 돌았으므로 실제 청구는 플랜에 따릅니다.
- **해석 (측정된 것만):**
  - 빠른 동작(펀치 0.17초, 슬램 0.3초 간격)은 초당 1장으로 볼 수 없습니다. Gemini는 슬램을 거의 못 셌습니다.
  - Claude는 슬램을 60 fps 장표로 다 봤어도 0개를 맞혔습니다(회당 1개만 말함). 느린 큰 동작(관제탑)은 2 fps 장표로 GAMST 최고 설정과 비슷했습니다(0.88 vs 0.89). 다만 1시간에 $19가 들고, 실시간이 아닙니다.
  - Gemini Live의 실시간 비용은 "질문 횟수 × 세션에 쌓인 영상"으로 늘어납니다. 사후 질문만 하면 8배 쌉니다.

## 1시간 영상 기준 (한눈에)

**관제탑** (522초 영상을 1시간 분량으로 환산; 감스트 정답 39개 = 1시간에 269개)

| 방법 | F1 | 1시간에 맞힌 수 | 비용 USD / KRW | 1시간 처리 시간 | 동작 끝 → 보고 | 초당 본 장수 |
|---|---|---|---|---|---|---|
| GAMST post (Main, 글+예시) | █████████░ 0.89 | 248 | 0 (로컬) | 71분 | 영상 끝난 뒤 (post) | 30 |
| GAMST 실시간 (같은 설정) | ██████░░░░ 0.61 | 124 | 0 (로컬) | 65분 | p50 0.55초, 최대 25.4초 (L1 p50 0.37초) | 17 |
| Gemini Live 실시간 | ███░░░░░░░ 0.26 | 41 | $7.38 / 10,328원 | 62분 | p50 19.2초 | 0.999 |
| Gemini Live 사후 | ███░░░░░░░ 0.30 | 48 | $0.91 / 1,272원 | 62분 | 영상 끝난 뒤 | 0.999 |
| Claude Code 2 fps | █████████░ 0.88 | 210 | $19.00 / 26,599원 | 30분 | 영상을 다 본 뒤 | 2 |
| Claude Code 4 fps | ████████░░ 0.85 | 210 | $30.60 / 42,845원 | 47분 | 영상을 다 본 뒤 | 4 |

**맨까송** (67초 영상을 1시간 분량으로 환산; 감스트 정답 16개 = 1시간에 856개)

| 방법 | F1 | 1시간에 맞힌 수 | 비용 USD / KRW | 1시간 처리 시간 | 동작 끝 → 보고 | 초당 본 장수 |
|---|---|---|---|---|---|---|
| GAMST post (Main, 고른 설정) | ██████░░░░ 0.60 | 482 | 0 (로컬) | — | 영상 끝난 뒤 (post) | 29.97 |
| GAMST 실시간 (같은 설정) | ████░░░░░░ 0.38 | 214 | 0 (로컬) | 120분 | p50 2.51초, 최대 4.8초 (L1 p50 0.36초) | 29.9 |
| Gemini Live 실시간 | █░░░░░░░░░ 0.11 | 54 | $9.31 / 13,038원 | 78분 | p50 10.3초 | 0.996 |
| Gemini Live 사후 | █░░░░░░░░░ 0.12 | 54 | $1.02 / 1,423원 | 73분 | 영상 끝난 뒤 | 0.996 |
| Claude Code 2 fps | ██░░░░░░░░ 0.17 | 80 | $14.47 / 20,263원 | 26분 | 영상을 다 본 뒤 | 2 |
| Claude Code 4 fps | █░░░░░░░░░ 0.12 | 54 | $19.61 / 27,454원 | 26분 | 영상을 다 본 뒤 | 4 |

**샷건 (슬램)** (12초 영상을 1시간 분량으로 환산; 감스트 정답 16개 = 1시간에 4840개)

| 방법 | F1 | 1시간에 맞힌 수 | 비용 USD / KRW | 1시간 처리 시간 | 동작 끝 → 보고 | 초당 본 장수 |
|---|---|---|---|---|---|---|
| GAMST (Main) | ██████████ 1.00 | 4840 | 0 (로컬) | 144분 | p50 26–40 ms (Main 실시간 실측) | 60 |
| Gemini Live 실시간 | █░░░░░░░░░ 0.06 | 151 | $5.40 / 7,555원 | 153분 | p50 2.2초 | 1.008 |
| Gemini Live 사후 | ░░░░░░░░░░ 0.00 | 0 | $1.67 / 2,340원 | 130분 | 영상 끝난 뒤 | 1.008 |
| Claude Code 60 fps | ░░░░░░░░░░ 0.00 | 0 | $305.98 / 428,379원 | 391분 | 영상을 다 본 뒤 | 60 |

- "1시간 처리 시간" = 이 실행의 걸린 시간 × (3600 / 영상 길이). Gemini는 세션 연결 6초와 마지막 답 대기가 짧은 영상에서 크게 보입니다. GAMST post는 Main의 실행 시간입니다(관제탑 10.0–10.6분, 슬램 28.5초). GAMST 실시간은 모델 적재(약 25초)를 포함합니다.
- GAMST 실시간 "초당 본 장수" = 잃지 않고 센 프레임 / 영상 길이. 관제탑은 디스크 spill 한도가 0.8 GB뿐이라(남은 6.6 GB − 바닥선 5 GB) 15,670 중 6,777 프레임을 잃었습니다.
- 막대는 F1 0.1당 한 칸입니다.

## What each arm got

| | GAMST | Gemini Live | Claude Code |
|---|---|---|---|
| model | local: RTMW-l 384x288 + YOLOX-m, LVFace + PersonViT person binding (Main, post mode; realtime run here) | `gemini-3.8-live`, Vertex AI Live API (GA), project `songsuro`, us-central1, ADC | `claude-opus-5-5` (Claude Code subagent, general-purpose) |
| frames | every frame (30 / 29.97 / 60 fps) | **1 frame/s** (the Live API maximum), ≤ 768 px, with `[t=..s]` before each, + 16 kHz audio | 4 x 4 contact sheets, 384 px tiles labelled with their time: 2 and 4 fps (관제탑, 맨까송), 60 fps (slam) |
| person | 23 enrolled faces/bodies (`감스트.person.json`) | the 5 photos, once per session | the 5 photos |
| action | rule/examples from the owner's text (Main) | the owner's text (ko; en variant) | the owner's text (ko) |
| answer | events with start/end/who, as frames arrive | `report` function calls; realtime: every 5 s (slam 2 s) cue; after-the-fact: one question at the end of each ≤110 s session | one JSON list after reading all sheets |
| cost | local compute, $0 API | Vertex usage metadata x list price | Anthropic list price x the transcript's usage |

## Results by condition (mean of the runs; each run listed after)

| 영상 | 팔 | 조건 | 회 | F1 (회별) | 맞음/말함/정답 (평균) | 시작/끝 오차(초) | 지연 p50(초) | 걸린 시간(초) | 초당 장수 | 토큰 입력 / 출력 | USD/회 | KRW/회 | USD/영상 1시간 | KRW/영상 1시간 | USD/맞힌 1개 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 관제탑 | GAMST | Main post: 글 규칙 그대로 | 1 | 0.667 (0.667) | 20.0/21.0/39 | 0.47 / 0.28 | — | 615 | 30.0 | 0 / 0 | 0.0000 | 0 | 0.00 | 0 | 0.0000 |
| 관제탑 | GAMST | Main post: 글+예시 search (v2, 최고) | 1 | 0.889 (0.889) | 36.0/42.0/39 | — / — | — | 615 | 30.0 | 0 / 0 | 0.0000 | 0 | 0.00 | 0 | 0.0000 |
| 관제탑 | GAMST | 실시간 (글+예시 search; 잃은 프레임 6,777/15,670) | 1 | 0.610 (0.610) | 18.0/20.0/39 | 0.41 / 0.46 | 0.55 | 563 | 17.0 | 0 / 0 | 0.0000 | 0 | 0.00 | 0 | 0.0000 |
| 관제탑 | Gemini Live | post.en 사후 | 2 | 0.280 (0.333, 0.227) | 6.5/7.0/39 | 1.87 / 1.38 | — | 537 | 1.0 | 88,346 / 3,830 | 0.1301 | 182 | 0.90 | 1,255 | 0.0213 |
| 관제탑 | Gemini Live | post.ko 사후 | 2 | 0.299 (0.222, 0.375) | 7.0/7.5/39 | 2.18 / 2.13 | — | 538 | 1.0 | 88,569 / 4,100 | 0.1319 | 185 | 0.91 | 1,272 | 0.0203 |
| 관제탑 | Gemini Live | rt.ko 사후 | 2 | 0.224 (0.304, 0.143) | 5.0/5.0/39 | 1.90 / 2.07 | — | 542 | 1.0 | 851,978 / 15,373 | — | — | — | — | — |
| 관제탑 | Gemini Live | rt.ko 실시간 | 2 | 0.261 (0.340, 0.182) | 6.0/6.5/39 | 1.88 / 1.83 | 19.24 | 542 | 1.0 | 851,978 / 15,373 | 1.0706 | 1,499 | 7.38 | 10,328 | 0.2157 |
| 관제탑 | Claude Code | 2 fps | 2 | 0.878 (0.886, 0.870) | 30.5/30.5/39 | 0.41 / 0.21 | — | 260 | 2.0 | 6,182,218 / 24,505 | 2.7572 | 3,860 | 19.00 | 26,599 | 0.0903 |
| 관제탑 | Claude Code | 4 fps | 2 | 0.847 (0.833, 0.861) | 30.5/33.0/39 | 0.32 / 0.24 | — | 407 | 4.0 | 9,814,161 / 39,878 | 4.4412 | 6,218 | 30.60 | 42,845 | 0.1457 |
| 맨까송 | GAMST | Main post: 펀치 고른 설정 (v2) | 1 | 0.600 (0.600) | 9.0/14.0/16 | — / — | — | — | 30.0 | 0 / 0 | 0.0000 | 0 | 0.00 | 0 | 0.0000 |
| 맨까송 | GAMST | Main post: 펀치 규칙 그대로 | 1 | 0.320 (0.320) | 8.0/34.0/16 | 0.32 / 0.50 | — | — | 30.0 | 0 / 0 | 0.0000 | 0 | 0.00 | 0 | 0.0000 |
| 맨까송 | GAMST | 실시간 (펀치 고른 설정; 잃은 프레임 0/2,013) | 1 | 0.381 (0.381) | 4.0/5.0/16 | 0.38 / 0.33 | 2.51 | 135 | 29.9 | 0 / 0 | 0.0000 | 0 | 0.00 | 0 | 0.0000 |
| 맨까송 | Gemini Live | post.en 사후 | 2 | 0.118 (0.118, 0.118) | 1.0/1.0/16 | 0.00 / 1.37 | — | 81 | 1.0 | 14,174 / 322 | 0.0187 | 26 | 1.00 | 1,405 | 0.0187 |
| 맨까송 | Gemini Live | post.ko 사후 | 2 | 0.118 (0.118, 0.118) | 1.0/1.0/16 | 1.32 / 1.48 | — | 82 | 1.0 | 14,216 / 348 | 0.0190 | 27 | 1.02 | 1,423 | 0.0190 |
| 맨까송 | Gemini Live | rt.en 사후 | 2 | 0.111 (0.118, 0.105) | 1.0/2.0/16 | 0.50 / 0.87 | — | 86 | 1.0 | 241,272 / 4,154 | — | — | — | — | — |
| 맨까송 | Gemini Live | rt.en 실시간 | 2 | 0.109 (0.118, 0.100) | 1.0/2.5/16 | 0.50 / 0.87 | 4.91 | 86 | 1.0 | 241,272 / 4,154 | 0.2847 | 399 | 15.24 | 21,334 | 0.2847 |
| 맨까송 | Gemini Live | rt.ko 사후 | 2 | 0.111 (0.111, 0.111) | 1.0/2.0/16 | 1.36 / 1.43 | — | 87 | 1.0 | 143,132 / 2,254 | — | — | — | — | — |
| 맨까송 | Gemini Live | rt.ko 실시간 | 2 | 0.111 (0.111, 0.111) | 1.0/2.0/16 | 1.36 / 1.43 | 10.33 | 87 | 1.0 | 143,132 / 2,254 | 0.1740 | 244 | 9.31 | 13,038 | 0.1740 |
| 맨까송 | Claude Code | 2 fps | 2 | 0.170 (0.118, 0.222) | 1.5/1.5/16 | 4.73 / 0.29 | — | 29 | 2.0 | 308,846 / 2,587 | 0.2704 | 379 | 14.47 | 20,263 | 0.2266 |
| 맨까송 | Claude Code | 4 fps | 2 | 0.118 (0.118, 0.118) | 1.0/1.0/16 | 2.64 / 0.97 | — | 29 | 4.0 | 334,252 / 2,930 | 0.3664 | 513 | 19.61 | 27,454 | 0.3664 |
| 샷건 (슬램) | GAMST | Main post: 글 규칙 (60 fps 전부) | 1 | 1.000 (1.000) | 16.0/16.0/16 | 0.01 / 0.01 | — | 28 | 60.0 | 0 / 0 | 0.0000 | 0 | 0.00 | 0 | 0.0000 |
| 샷건 (슬램) | Gemini Live | post.en 사후 | 2 | 0.000 (0.000, 0.000) | 0.0/3.0/16 | — / — | — | 26 | 1.0 | 3,660 / 340 | 0.0057 | 8 | 1.71 | 2,399 | — |
| 샷건 (슬램) | Gemini Live | post.ko 사후 | 2 | 0.000 (0.000, 0.000) | 0.0/0.0/16 | — / — | — | 26 | 1.0 | 3,706 / 280 | 0.0055 | 8 | 1.67 | 2,340 | — |
| 샷건 (슬램) | Gemini Live | rt.en 사후 | 2 | 0.056 (0.111, 0.000) | 0.5/2.0/16 | 0.14 / 0.14 | — | 31 | 1.0 | 13,334 / 926 | — | — | — | — | — |
| 샷건 (슬램) | Gemini Live | rt.en 실시간 | 2 | 0.056 (0.111, 0.000) | 0.5/2.5/16 | 0.14 / 0.14 | 1.16 | 31 | 1.0 | 13,334 / 926 | 0.0185 | 26 | 5.59 | 7,824 | 0.0249 |
| 샷건 (슬램) | Gemini Live | rt.ko 사후 | 2 | 0.059 (0.000, 0.118) | 0.5/1.0/16 | 0.14 / 0.14 | — | 30 | 1.0 | 13,516 / 736 | — | — | — | — | — |
| 샷건 (슬램) | Gemini Live | rt.ko 실시간 | 2 | 0.056 (0.000, 0.111) | 0.5/1.5/16 | 0.14 / 0.14 | 2.16 | 30 | 1.0 | 13,516 / 736 | 0.0178 | 25 | 5.40 | 7,555 | 0.0163 |
| 샷건 (슬램) | Claude Code | 60 fps | 2 | 0.000 (0.000, 0.000) | 0.0/1.0/16 | — / — | — | 78 | 60.0 | 1,336,128 / 7,871 | 1.0114 | 1,416 | 305.98 | 428,379 | — |

- 지연 = 보고가 도착한 영상 시각 − 정답 동작의 끝(맞은 짝만). GAMST는 Store가 개수를 안 시각(`committed_ms`)입니다. Gemini "사후" 줄의 토큰·비용은 같은 실행의 "실시간" 줄에 들어 있습니다.
- Claude 토큰 입력 = 새 입력 + 캐시 쓰기 + 캐시 읽기(대부분 캐시 읽기).

### Per run

| 팔 | 영상 | 조건 | 회 | 맞음/말함/정답 | 정밀 | 재현 | F1 | 시작/끝 오차(초) | 지연 p50 / 최대(초) | 걸린 시간(초) | 초당 장수 | 토큰 입력 / 출력 | USD | KRW | USD/분 | USD/시간 | USD/맞힌 1개 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Claude Code | 관제탑 | 2 fps | 1 | 31/31/39 | 1.00 | 0.79 | 0.886 | 0.42 / 0.22 | — / — | 265.3 | 2.0 | 6,957,542 / 24,798 | 2.9152 | 4,081 | 0.3348 | 20.09 | 0.0940 |
| Claude Code | 관제탑 | 2 fps | 2 | 30/30/39 | 1.00 | 0.77 | 0.870 | 0.40 / 0.21 | — / — | 254.6 | 2.0 | 5,406,893 / 24,212 | 2.5992 | 3,639 | 0.2985 | 17.91 | 0.0866 |
| Claude Code | 관제탑 | 4 fps | 1 | 30/33/39 | 0.91 | 0.77 | 0.833 | 0.34 / 0.24 | — / — | 403.8 | 4.0 | 10,367,311 / 39,084 | 4.4559 | 6,238 | 0.5117 | 30.70 | 0.1485 |
| Claude Code | 관제탑 | 4 fps | 2 | 31/33/39 | 0.94 | 0.79 | 0.861 | 0.31 / 0.24 | — / — | 410.9 | 4.0 | 9,261,011 / 40,673 | 4.4265 | 6,197 | 0.5084 | 30.50 | 0.1428 |
| GAMST | 관제탑 | Main post: 글 규칙 그대로 | 1 | 20/21/39 | 0.95 | 0.51 | 0.667 | 0.47 / 0.28 | — / — | 615.0 | 30.0 | 0 / 0 | 0.0000 | 0 | 0.0000 | 0.00 | 0.0000 |
| GAMST | 관제탑 | Main post: 글+예시 search (v2, 최고) | 1 | 36/42/39 | 0.86 | 0.92 | 0.889 | — / — | — / — | 615.0 | 30.0 | 0 / 0 | 0.0000 | 0 | 0.0000 | 0.00 | 0.0000 |
| GAMST | 관제탑 | 실시간 (글+예시 search; 잃은 프레임 6,777/15,670) | 1 | 18/20/39 | 0.90 | 0.46 | 0.610 | 0.41 / 0.46 | 0.55 / 25.40 | 562.7 | 17.0 | 0 / 0 | 0.0000 | 0 | 0.0000 | 0.00 | 0.0000 |
| Gemini Live | 관제탑 | post.en 사후 | 1 | 8/9/39 | 0.89 | 0.21 | 0.333 | 1.70 / 0.46 | — / — | 536.4 | 1.0 | 88,347 / 2,918 | 0.1260 | 176 | 0.0145 | 0.87 | 0.0158 |
| Gemini Live | 관제탑 | post.en 사후 | 2 | 5/5/39 | 1.00 | 0.13 | 0.227 | 2.05 / 2.30 | — / — | 538.4 | 1.0 | 88,346 / 4,741 | 0.1342 | 188 | 0.0154 | 0.92 | 0.0268 |
| Gemini Live | 관제탑 | post.ko 사후 | 1 | 5/6/39 | 0.83 | 0.13 | 0.222 | 2.00 / 2.80 | — / — | 536.9 | 1.0 | 88,571 / 3,254 | 0.1280 | 179 | 0.0147 | 0.88 | 0.0256 |
| Gemini Live | 관제탑 | post.ko 사후 | 2 | 9/9/39 | 1.00 | 0.23 | 0.375 | 2.36 / 1.46 | — / — | 539.7 | 1.0 | 88,567 / 4,945 | 0.1358 | 190 | 0.0156 | 0.94 | 0.0151 |
| Gemini Live | 관제탑 | rt.ko 사후 | 1 | 7/7/39 | 1.00 | 0.18 | 0.304 | 2.22 / 2.18 | — / — | 541.7 | 1.0 | 650,519 / 14,026 | — | — | — | — | — |
| Gemini Live | 관제탑 | rt.ko 사후 | 2 | 3/3/39 | 1.00 | 0.08 | 0.143 | 1.57 / 1.97 | — / — | 543.3 | 1.0 | 1,053,437 / 16,720 | — | — | — | — | — |
| Gemini Live | 관제탑 | rt.ko 실시간 | 1 | 8/8/39 | 1.00 | 0.21 | 0.340 | 2.20 / 2.06 | 32.44 / 48.39 | 541.7 | 1.0 | 650,519 / 14,026 | 0.8311 | 1,163 | 0.0954 | 5.73 | 0.1039 |
| Gemini Live | 관제탑 | rt.ko 실시간 | 2 | 4/5/39 | 0.80 | 0.10 | 0.182 | 1.56 / 1.59 | 6.05 / 30.92 | 543.3 | 1.0 | 1,053,437 / 16,720 | 1.3101 | 1,834 | 0.1505 | 9.03 | 0.3275 |
| Claude Code | 맨까송 | 2 fps | 1 | 1/1/16 | 1.00 | 0.06 | 0.118 | 7.64 / 0.09 | — / — | 36.7 | 2.0 | 373,231 / 3,225 | 0.3657 | 512 | 0.3263 | 19.58 | 0.3657 |
| Claude Code | 맨까송 | 2 fps | 2 | 2/2/16 | 1.00 | 0.12 | 0.222 | 1.82 / 0.48 | — / — | 20.6 | 2.0 | 244,461 / 1,949 | 0.1751 | 245 | 0.1562 | 9.37 | 0.0876 |
| Claude Code | 맨까송 | 4 fps | 1 | 1/1/16 | 1.00 | 0.06 | 0.118 | 2.64 / 1.09 | — / — | 28.5 | 4.0 | 334,299 / 2,854 | 0.4032 | 564 | 0.3597 | 21.58 | 0.4032 |
| Claude Code | 맨까송 | 4 fps | 2 | 1/1/16 | 1.00 | 0.06 | 0.118 | 2.64 / 0.84 | — / — | 29.6 | 4.0 | 334,206 / 3,005 | 0.3295 | 461 | 0.2940 | 17.64 | 0.3295 |
| GAMST | 맨까송 | Main post: 펀치 고른 설정 (v2) | 1 | 9/14/16 | 0.64 | 0.56 | 0.600 | — / — | — / — | — | 30.0 | 0 / 0 | 0.0000 | 0 | 0.0000 | 0.00 | 0.0000 |
| GAMST | 맨까송 | Main post: 펀치 규칙 그대로 | 1 | 8/34/16 | 0.24 | 0.50 | 0.320 | 0.32 / 0.50 | — / — | — | 30.0 | 0 / 0 | 0.0000 | 0 | 0.0000 | 0.00 | 0.0000 |
| GAMST | 맨까송 | 실시간 (펀치 고른 설정; 잃은 프레임 0/2,013) | 1 | 4/5/16 | 0.80 | 0.25 | 0.381 | 0.38 / 0.33 | 2.51 / 4.83 | 134.7 | 29.9 | 0 / 0 | 0.0000 | 0 | 0.0000 | 0.00 | 0.0000 |
| Gemini Live | 맨까송 | post.en 사후 | 1 | 1/1/16 | 1.00 | 0.06 | 0.118 | 0.00 / 0.87 | — / — | 81.7 | 1.0 | 14,173 / 464 | 0.0194 | 27 | 0.0173 | 1.04 | 0.0194 |
| Gemini Live | 맨까송 | post.en 사후 | 2 | 1/1/16 | 1.00 | 0.06 | 0.118 | 0.00 / 1.87 | — / — | 81.2 | 1.0 | 14,174 / 181 | 0.0181 | 25 | 0.0162 | 0.97 | 0.0181 |
| Gemini Live | 맨까송 | post.ko 사후 | 1 | 1/1/16 | 1.00 | 0.06 | 0.118 | 2.64 / 1.09 | — / — | 81.7 | 1.0 | 14,216 / 389 | 0.0191 | 27 | 0.0170 | 1.02 | 0.0191 |
| Gemini Live | 맨까송 | post.ko 사후 | 2 | 1/1/16 | 1.00 | 0.06 | 0.118 | 0.00 / 1.87 | — / — | 82.0 | 1.0 | 14,216 / 307 | 0.0189 | 26 | 0.0169 | 1.01 | 0.0189 |
| Gemini Live | 맨까송 | rt.en 사후 | 1 | 1/1/16 | 1.00 | 0.06 | 0.118 | 0.00 / 0.87 | — / — | 85.5 | 1.0 | 241,681 / 3,949 | — | — | — | — | — |
| Gemini Live | 맨까송 | rt.en 사후 | 2 | 1/3/16 | 0.33 | 0.06 | 0.105 | 1.00 / 0.87 | — / — | 86.8 | 1.0 | 240,862 / 4,358 | — | — | — | — | — |
| Gemini Live | 맨까송 | rt.en 실시간 | 1 | 1/1/16 | 1.00 | 0.06 | 0.118 | 0.00 / 0.87 | 4.98 / 4.98 | 85.5 | 1.0 | 241,681 / 3,949 | 0.2868 | 402 | 0.2559 | 15.35 | 0.2868 |
| Gemini Live | 맨까송 | rt.en 실시간 | 2 | 1/4/16 | 0.25 | 0.06 | 0.100 | 1.00 / 0.87 | 4.83 / 4.83 | 86.8 | 1.0 | 240,862 / 4,358 | 0.2825 | 396 | 0.2521 | 15.12 | 0.2825 |
| Gemini Live | 맨까송 | rt.ko 사후 | 1 | 1/2/16 | 0.50 | 0.06 | 0.111 | 0.07 / 1.76 | — / — | 86.2 | 1.0 | 119,854 / 2,469 | — | — | — | — | — |
| Gemini Live | 맨까송 | rt.ko 사후 | 2 | 1/2/16 | 0.50 | 0.06 | 0.111 | 2.64 / 1.09 | — / — | 88.1 | 1.0 | 166,409 / 2,039 | — | — | — | — | — |
| Gemini Live | 맨까송 | rt.ko 실시간 | 1 | 1/2/16 | 0.50 | 0.06 | 0.111 | 0.07 / 1.76 | 1.59 / 1.59 | 86.2 | 1.0 | 119,854 / 2,469 | 0.1482 | 207 | 0.1322 | 7.93 | 0.1482 |
| Gemini Live | 맨까송 | rt.ko 실시간 | 2 | 1/2/16 | 0.50 | 0.06 | 0.111 | 2.64 / 1.09 | 19.08 / 19.08 | 88.1 | 1.0 | 166,409 / 2,039 | 0.1998 | 280 | 0.1783 | 10.70 | 0.1998 |
| Claude Code | 샷건 (슬램) | 60 fps | 1 | 0/1/16 | 0.00 | 0.00 | 0.000 | — / — | — / — | 66.9 | 60.0 | 1,196,891 / 7,007 | 0.9625 | 1,348 | 4.8529 | 291.18 | — |
| Claude Code | 샷건 (슬램) | 60 fps | 2 | 0/1/16 | 0.00 | 0.00 | 0.000 | — / — | — / — | 88.1 | 60.0 | 1,475,364 / 8,735 | 1.0604 | 1,485 | 5.3466 | 320.79 | — |
| GAMST | 샷건 (슬램) | Main post: 글 규칙 (60 fps 전부) | 1 | 16/16/16 | 1.00 | 1.00 | 1.000 | 0.01 / 0.01 | — / — | 28.5 | 60.0 | 0 / 0 | 0.0000 | 0 | 0.0000 | 0.00 | 0.0000 |
| Gemini Live | 샷건 (슬램) | post.en 사후 | 1 | 0/1/16 | 0.00 | 0.00 | 0.000 | — / — | — / — | 25.4 | 1.0 | 3,660 / 290 | 0.0054 | 8 | 0.0274 | 1.65 | — |
| Gemini Live | 샷건 (슬램) | post.en 사후 | 2 | 0/5/16 | 0.00 | 0.00 | 0.000 | — / — | — / — | 25.6 | 1.0 | 3,660 / 389 | 0.0059 | 8 | 0.0297 | 1.78 | — |
| Gemini Live | 샷건 (슬램) | post.ko 사후 | 1 | 0/0/16 | — | 0.00 | 0.000 | — / — | — / — | 26.1 | 1.0 | 3,706 / 292 | 0.0057 | 8 | 0.0286 | 1.72 | — |
| Gemini Live | 샷건 (슬램) | post.ko 사후 | 2 | 0/0/16 | — | 0.00 | 0.000 | — / — | — / — | 25.6 | 1.0 | 3,706 / 268 | 0.0054 | 8 | 0.0271 | 1.63 | — |
| Gemini Live | 샷건 (슬램) | rt.en 사후 | 1 | 1/2/16 | 0.50 | 0.06 | 0.111 | 0.14 / 0.14 | — / — | 30.2 | 1.0 | 18,960 / 1,075 | — | — | — | — | — |
| Gemini Live | 샷건 (슬램) | rt.en 사후 | 2 | 0/2/16 | 0.00 | 0.00 | 0.000 | — / — | — / — | 31.0 | 1.0 | 7,708 / 778 | — | — | — | — | — |
| Gemini Live | 샷건 (슬램) | rt.en 실시간 | 1 | 1/2/16 | 0.50 | 0.06 | 0.111 | 0.14 / 0.14 | 1.16 / 1.16 | 30.2 | 1.0 | 18,960 / 1,075 | 0.0249 | 35 | 0.1256 | 7.54 | 0.0249 |
| Gemini Live | 샷건 (슬램) | rt.en 실시간 | 2 | 0/3/16 | 0.00 | 0.00 | 0.000 | — / — | — / — | 31.0 | 1.0 | 7,708 / 778 | 0.0120 | 17 | 0.0606 | 3.64 | — |
| Gemini Live | 샷건 (슬램) | rt.ko 사후 | 1 | 0/1/16 | 0.00 | 0.00 | 0.000 | — / — | — / — | 30.2 | 1.0 | 14,733 / 758 | — | — | — | — | — |
| Gemini Live | 샷건 (슬램) | rt.ko 사후 | 2 | 1/1/16 | 1.00 | 0.06 | 0.118 | 0.14 / 0.14 | — / — | 30.5 | 1.0 | 12,299 / 714 | — | — | — | — | — |
| Gemini Live | 샷건 (슬램) | rt.ko 실시간 | 1 | 0/1/16 | 0.00 | 0.00 | 0.000 | — / — | — / — | 30.2 | 1.0 | 14,733 / 758 | 0.0194 | 27 | 0.0979 | 5.87 | — |
| Gemini Live | 샷건 (슬램) | rt.ko 실시간 | 2 | 1/2/16 | 0.50 | 0.06 | 0.111 | 0.14 / 0.14 | 2.16 / 2.16 | 30.5 | 1.0 | 12,299 / 714 | 0.0163 | 23 | 0.0820 | 4.92 | 0.0163 |

## Method notes

- **Gemini Live.**
  - Vertex refuses TEXT responses for this model ("Unsupported modality"), so it answers by a `report` function
    call (declared `NON_BLOCKING`, answered `SILENT`: no second turn re-bills the context).
  - Sessions cover ≤ 110 s of clip (audio+video sessions are limited to 2 minutes); 관제탑 = 5 sessions. The next
    session connects 6 s early and gets the photos again. No truth occurrence straddles a session boundary.
  - Automatic voice detection is off; a window is closed every 5 s (slam 2 s) with the cue. The model answered
    several cues in one turn and skipped others (lesson 105); answers are filed by arrival time.
  - Realtime latency = the clip time the `report` call arrived − the truth occurrence's end, on matched pairs.
  - "사후" in an `rt` run is the full-list answer at the end of each session of that same run (same cost); `post`
    runs stream the same media without cues and ask only that question.
  - Run 1 of three realtime conditions failed and was run again (slam ko/en: the final answer was misfiled before
    the time-based filing; 맨까송 ko: "No audio present in the turn" for the final question, now sent with 0.2 s of
    silence). One 관제탑 realtime run was stopped after 15 s of clip to apply that fix (usage not recorded, ≤ $0.01).
    Their spend is in the ledger below. In the re-run 관제탑 realtime run 1, session 1 never answered the final
    question within 90 s (the model went silent): that session has no after-the-fact list, and the run is kept.
- **Claude Code.** 10 fresh subagents (2 at a time), each given `prompt.md` only. Tokens, wall time and files read
  are from each subagent's transcript; every Read and Bash call was checked: only its photos, its sheets and zoom
  crops it cut from its own sheets. One 관제탑 4 fps run read 128 of 131 sheets. Two hand-back messages that match
  no transcript were discarded (lesson 108).
- **GAMST.** "post" rows are Main's measured post-mode runs (no new tuning). The realtime rows are this
  experiment's runs: `gamst_rt.sh`, `--realtime`, Main's configuration (the identity-recall chosen settings; 관제탑
  the rule + 7 examples, search mode; 맨까송 the chosen punch rule), through `guard.sh`, `GAMST_SPILL_FLOOR_GB=5`,
  after the disk was freed (6.6 GB free at start; the script now refuses to start below the floor).

  | realtime run | F1 (matched / said / truth) | lost frames | handed to L2 (times, ~frames) | events L1 / L2 | latency action end → count, p50 / max | L1 only p50 / max | L2 only p50 / max | wall (RTF) |
  |---|---|---|---|---|---|---|---|---|
  | 관제탑, search | 0.610 (18 / 20 / 39) | **6,777 / 15,670** (spill bound 0.8 GB) | 76, ~2,400 | 23 / 37 | 0.55 / 25.4 s (n 9) | 0.37 / 0.55 s | 22.5 / 25.4 s | 563 s (1.08) |
  | 맨까송, punch | 0.381 (4 / 5 / 16) | 0 / 2,013 | 29, ~1,030 | 10 / 15 | 2.51 / 4.83 s (n 4) | 0.36 / 0.59 s | 4.63 / 4.83 s | 135 s (2.0) |

  - Latency is measured on matched 감스트 events that have a commit time (9 of 18 on 관제탑, 4 of 4 on 맨까송).
  - Wall time includes model loading (~25 s) and, on 맨까송, the L2 worker finishing after the clip ended.
  - The first realtime attempts (0.2-3.2 GB free, spill bound 0 GB; lesson 107) lost 15,573 of 15,670 and 326 of
    2,013 frames. They are kept under `gamst/diskfull/` and are not in the tables.
  - The slam clip was not re-run: Main has its realtime measurement (16/16 final, L1 latency p50 26-40 ms,
    `testbed-2026-11-26.md`).
- **KRW** at 1,400 KRW/USD (a stated rate, not a quote). "USD/영상 1시간" = USD per run / clip length x 3600 s.
  "USD/맞힌 1개" = USD per run / 감스트 occurrences matched.

## Spend

Gemini Live, from `runs/ledger.jsonl` (every hosted call, from `usage_metadata`; Standard PayGo list price, non-global):

| condition | runs billed | USD | input tokens | output tokens (incl. thinking) |
|---|---:|---:|---:|---:|
| smoke (TEXT refused at setup; then a blocking tool) | 2 | 0.0120 | 7,470 | 720 |
| slam post ko / en | 2 / 2 | 0.0111 / 0.0113 | 7,412 / 7,320 | 560 / 679 |
| 맨까송 post ko / en | 2 / 2 | 0.0380 / 0.0375 | 28,432 / 28,347 | 696 / 645 |
| slam rt ko / en (1 failed run each, re-run) | 3 / 3 | 0.0457 / 0.0562 | 34,078 / 41,974 | 1,999 / 2,527 |
| 관제탑 post ko / en | 2 / 2 | 0.2638 / 0.2602 | 177,138 / 176,693 | 8,199 / 7,659 |
| 맨까송 rt ko (1 failed, re-run) / en | 3 / 2 | 0.4613 / 0.5694 | 378,752 / 482,543 | 6,064 / 8,307 |
| 관제탑 rt ko (1 stopped after 15 s, re-run) | 3 | 2.1412 | 1,703,956 | 30,746 |
| **total** | **28** | **3.9075** (≈ 5,471 KRW) | 3,074,115 | 68,801 |

Claude Code arm (10 subagent runs, `claude-opus-5-5`, list price): **$17.69** (≈ 24,770 KRW); 35,951,209 input tokens
(mostly cache reads) and 155,542 output tokens; 1,605 s of agent time in all.

## Limits

- Two runs per hosted condition; the run-to-run spread is in the per-run table.
- 관제탑 scores exclude the 7 owner example ranges (39 of 46 감스트 occurrences), as every GAMST number on Main.
  Gemini and Claude were given no examples; the exclusion only keeps the denominators equal.
- The slam clip's Korean action text is a translation made for this experiment (no Korean owner text on Main).
- GAMST's 관제탑/맨까송 numbers were chosen while looking at these labels (in-sample, as Main says); its held-out
  numbers are in `action-spec-2026-11-28.md`. Gemini and Claude were not tuned at all.

# jev decision 모델 선정 (딥리서치, 2026-11-20)

## jev가 하는 일 (PLAN.md 보고 2 §1)
- 앱 쪽 판단 에이전트입니다. GAMST MCP Context(시간 타임라인, 사건, 근거 ID)를 읽고, MCP 도구로 Watch를 등록·수정합니다.
- 오알람 피드백을 받으면 조건 수정안을 **제안**합니다.
- 필요한 능력:
  1. 정확한 도구 호출(MCP)
  2. 긴 Context에서 지시 따르기
  3. 멀티모달 근거(필요할 때 프레임) 이해
  4. 실시간에 가까운 응답

## 후보와 공개 근거

| 후보 | 공개 근거 (출처: 각 모델 카드 / 논문) | 판단 |
|---|---|---|
| **Qwen3.8-27B (FP8)** — 2026-08, Apache-2.0, 멀티모달, dense 27.8B | IFBench **79.5** (Qwen3.6-27B 69.1, Qwen3.7-Plus 79.1).<br>Multimodal tool use ClawEval-MM Pass@3 **57.4** (Qwen3.7-Plus와 같음, Opus4.6 Max 52.5).<br>Agents' Last Exam 42.9 (Qwen3.6-27B 27.3).<br>Terminal Bench 2.1 73.0. | **선정.** 공개 가중치 중 멀티모달 도구 사용과 지시 따르기가 가장 높고, FP8 31GB라 96GB GPU에 다른 모델과 함께 올라감 |
| Qwen3.6-35B-A3B — 2026-04 | MCP-Atlas 62.8, TAU3 67.2, MCPMark 37.0. 활성 3B라 빠름 | 차선. 지시 따르기와 멀티모달 도구 사용 근거가 3.8-27B보다 약함 |
| Qwen3.5-9B — 2026-02 | BFCL-V4 66.1, TAU2 79.1 | 스트리밍 VLM으로 이미 사용 중. jev로 겸용하면 메모리는 아끼지만 성능은 낮음 |
| GLM-4.7-Flash, Gemma 4 31B, gpt-oss | τ²/Tau2 76.9~79.5 (벤더마다 측정 harness가 달라 직접 비교 불가) | 멀티모달 도구 사용 근거 부족 |
| Qwen3.8-Flash-Next — 125B MoE (6B 활성) | FP8 185GB | 96GB GPU에 올라가지 않음 |
| SyncPro-RL (2026-10), Em-Garde-7B (2026-03) | 영상 프레임에서 "언제 응답할지"를 학습한 proactive 스트리밍 모델 | jev의 역할(MCP Context 기반 조건 관리)과 다름. 영상 기반 트리거는 GAMST Watch가 맡음 |
| BLIP-Live (2026-11), StreamPro (2026-05) | proactive 벤치마크 SOTA 주장 | 가중치 미공개 |

참고 자료:
- 모델 카드: Qwen/Qwen3.8-27B, Qwen/Qwen3.6-35B-A3B, Qwen/Qwen3.5-9B
- vdf.ai "Best Local LLM for Agents and Tool Calling (2026-10-06)"
- arXiv 2611.21553 (BLIP-Live), 2605.16381 (StreamPro), 2610.32132 (SyncPro)

## 고정
- 모델: `Qwen/Qwen3.8-27B-FP8`, revision `c3a462cc3d38cf351f3b26cbd763469eeb74ef81`
- 서빙: vLLM 0.34, 도구 호출 파서는 카드 지정값(`qwen3_coder`)
- 역할 경계: jev는 Watch 등록·수정·제안만 합니다. 실제 알람 판정은 GAMST가 결정론적으로 계산합니다. 숨은 트리거는 없습니다.

# 인수인계서: parallel GAMST experiment

다른 모델이나 다른 사람이 이 작업을 이어받을 때 **이 파일을 가장 먼저 읽습니다.**

이 파일은 큰 방향과 규칙을 설명합니다. 세부 진행 상태는 `RESUME.md`(체크리스트, 결정 기록)에 있습니다. 원문 계획은 `PLAN.md`에 있으며 **수정하면 안 됩니다.**

- 마지막 갱신: 2026-11-20 (Claude 작업분)

---

## 1. 처음에 하려던 것 (사용자 의도 요약, 원문은 PLAN.md)

### 1.1 핵심 질문
- 이 실험이 확인하려는 것은 하나입니다. **"GAMST가 실시간 센서 모델들을 계속 돌려 만든 Context가, 같은 VLM/LLM 혼자 보는 것보다 나은가?"**
- 과거 비교 결과:
  - StreamArena 88문항: Gemini 3.1 Pro 단독 33/88, GAMST 15/88
  - 이전 GAMST는 가벼운 모델 위주였고, 한 번에 하나씩 순차로 돌렸습니다.
- 사용자는 이렇게 지시했습니다. **"실시간 SOTA를 붙이고, 성능을 최대한 끌어올리고, 모두 동시에 돌려라. 타협하지 마라."**

### 1.2 할 일 (7개 축)

**1. 모델**

"가벼운 것" 말고 실시간으로 돌아가는 SOTA를 붙입니다.

| 영역 | 모델 |
|---|---|
| 스트리밍 VLM | Qwen3.5-9B |
| 분할·추적 | SAM 3.1 |
| 객체 검출 | RF-DETR, YOLOE |
| 깊이 | Video Depth Anything |
| 자세·손 | 자세·손 모델 |
| 심박 | rPPG |
| 오디오 | CED, Qwen3-ASR, VAD, 화자 분리, 말하는 사람 찾기 |

**2. 병렬 실행**

- 위 모델을 RTX PRO 6000 96GB(안 되면 H100) **한 장에서 동시에** 실행합니다.
- 실행 관리는 GamstCrew가 맡습니다.
  - 모델별로 독립된 프로세스
  - GPU 메모리 예산
  - 모든 모델이 같은 프레임과 같은 시계를 공유
  - 실시간 감시
- 한 번에 하나씩 순차로 돌리는 방식은 **금지**입니다.

**3. 실제 가중치 검증**

각 모델마다 다음을 확인합니다.
- sha256
- 실제 영상으로 실행
- 공식 데모와 결과 비교
- 실시간 배율 측정

검증에 실패하면 공식 패키지나 상용 API로 바꿉니다. 이때 **비용을 표시하고 사용자에게 먼저 보고**합니다.

**4. MCP Watch**

- 도구: `create_watch`, `update_watch`, `delete_watch`, `list_watches`
- 주인: 사용자(자연어로 말한 규칙) 또는 **jev**(앱 쪽 판단 모델)
- 알람이 울리면 사건과 시간 구간 Context를 받는 쪽에 전달합니다.
- 쿨다운과 분당 최대 횟수는 주인이 정합니다.
- 규칙 버전은 해시로 남기고, 알람마다 근거별 기여도를 기록합니다.
- jev는 **수정안 제안만** 합니다. 실제 변경은 명시적인 update 호출로만 하고 로그에 남깁니다.
- **GAMST에는 숨은 트리거가 없습니다.**

**5. jev 모델**

딥리서치로 SOTA 판단 모델을 고릅니다.

**6. 평가**

| 팔 | 구성 |
|---|---|
| B | Qwen3.5-9B 단독 (최근 프레임 슬라이딩 윈도우) |
| C-local | Qwen3.5-9B + GAMST MCP Context |
| C-Gemini | `gemini-3.8-flash` + GAMST MCP Context |

- 데이터 순서: MOT17(확률 구름) → StreamArena(말과 기억) → RepCount(횟수) → MCD-rPPG(심박)
- A 팔(Gemini 3.1 Pro 단독)은 **기존 결과를 재사용**합니다. 재실행 금지.
- 점프와 "사람 + 소리" 확률 구름은 사용자가 직접 촬영해야 합니다.

**7. 마무리**

1. 모든 벤치마크 완료
2. 결과 보고
3. 사용자가 직접 써보는 데모
4. VM 정지
5. PR 머지

### 1.3 설계 원칙 (사용자와 합의, 바꾸지 말 것)
- **폴백 체인은 없습니다.** GAMST는 모든 모델을 계속 돌립니다.
- 소비자(LLM/VLM)는 **사용자 질문 1번당 1번** 호출합니다. 또는 Watch가 울릴 때 호출합니다.
- "답이 부족하니 더 큰 모델로 넘기는" 단계적 전환은 하지 않습니다.
- 역할 분담:
  - 충돌이 "일어났는가?"는 느린 VLM 추론의 몫입니다.
  - GAMST는 물리량, 흔적이 남지 않는 사건, 즉각 반응을 맡습니다.
- 과장 금지:
  - 확인하지 않은 것은 확인했다고 말하지 않습니다.
  - 실패와 무효는 그대로 적습니다. 예: StreamingVLM 0/88은 설정 오류로 **무효**입니다.

---

## 2. 반드시 지킬 규칙
1. **답변은 한국어로, 쉽고 짧게 씁니다.** 과장하면 안 됩니다(과거에 과장 보고로 신뢰를 잃었습니다).
2. **돈이 드는 일은 하기 전에 보고합니다.** 쿼터 업그레이드, 결제, 상용 API가 해당합니다.
3. **Gemini 3.1 Pro 호출 금지**(토큰 소진). C-Gemini는 `gemini-3.8-flash`를 씁니다.
4. **합성 데이터 금지.** 실제 데이터만 씁니다.
5. 모든 작업은 GAMST PR, 이슈, 문서로 남깁니다. `RESUME.md`와 이 파일을 계속 갱신합니다.
6. 원문과 다르게 결정한 것은 `RESUME.md`의 "결정 기록"에 적습니다.
7. **VM이 놀면 즉시 정지합니다.** 예산 가드는 `gamst-bench-guard` $300입니다.
8. 사용자의 커밋되지 않은 GAMST 변경은 건드리지 않습니다.
   - 수정된 파일: `docs/multimodal-decision.md`, `docs/positioning.md`, `docs/realtime.md`, `docs/research-roadmap.md`, `site/app.js`
   - 추적 안 되는 파일: `.DS_Store`, `.gjc/`, `docs/product-direction.md`, `runtime-finance-*`
   - 테스트를 돌리면 `runtime-finance-<랜덤>/` 폴더가 생깁니다. 커밋하지 않습니다.

---

## 3. 저장소와 브랜치

| 저장소 | 로컬 경로 | 작업 브랜치 | 내용 |
|---|---|---|---|
| GAMST (NomaDamas/Gamst) | `~/Desktop/catch-your-mind` | `feat/parallel-gamst-experiment` (base `feat/temporal-contract`) | 이슈 #351, draft PR #352. Watch 구현, 어댑터 수정, 실험 문서 |
| GamstCrew | `~/Desktop/catch-your-mind/.local-run/GamstCrew` | 현재 `feat/runtime-locks`에서 새 파일 작성 중. **새 브랜치 `feat/parallel-crew`로 옮겨 PR 필요** | 병렬 실행 관리 |
| Gamst-bench | `~/Desktop/gamst-bench` | `feat/parallel-gamst-experiment` (base `origin/feat/gamst-extensions`) | 실험 스크립트와 원시 결과: `experiments/parallel-gamst-2026-11-20/` |
| GamstCapture | `~/Desktop/GamstCapture` | (이번 실험에서 변경 없음) | |

커밋 방법:
- eval의 `commit` 도우미를 씁니다(`d = await commit.draft(repo=...)`; `await d.only("파일",...)`; `await d.commit("제목", ["- 본문"])`).
- 출력이 매우 길므로 stdout을 숨기고 씁니다.
- 그다음 `git push`, `gh pr`를 실행합니다.

---

## 4. 클라우드 자원 (2026-11-20 현재)

**gcloud**
- 경로: `/opt/homebrew/share/google-cloud-sdk/bin/gcloud`
- 프로젝트 `optimum-zone`, 계정 jinminsung0722@gmail.com

**실험 VM: `gamst-par`**
- 위치: asia-northeast3-a
- 사양: g4-standard-48, **RTX PRO 6000 Blackwell 96GB**, **Spot**
- 선점되면 STOP됩니다. `max-run-duration`은 12시간입니다.
- 부팅 디스크: hyperdisk 150GB
- 데이터 디스크: `gamst-par-data` hyperdisk 450GB → `/mnt/ssd`(= `/srv`, fstab 등록)
  - 이 디스크는 스냅샷 `gamst-hf-data-2026-11-20`에서 만들었습니다.
- **원래 디스크 `gamst-hf-data`(asia-northeast3-b)는 삭제했고 스냅샷만 남아 있습니다.**

**쿼터 처리 내역 (모두 무료 쿼터)**
- RTX PRO 6000 온디맨드와 H100: 거절
- RTX PRO 6000 Spot: 1 승인
- SSD_TOTAL_GB, HDB_TOTAL_GB(asia-northeast3): 1000으로 승인

**옛 VM**
- `gamst-hf-bench`(asia-northeast3-b, L4): TERMINATED, 데이터 디스크 분리됨
- `gamst-bench-gpu-usa`, `gamst-bench-gpu-us`: TERMINATED

**Vertex**
- 로컬에서 REST로 부를 때 `x-goog-user-project: optimum-zone` 헤더가 필요합니다.
- 엔드포인트: `.../locations/global/publishers/google/models/gemini-3.8-flash:generateContent`

**HF 토큰**
- 로컬 `~/.cache/huggingface/token`에 있습니다.
- VM에는 다운로드할 때만 잠깐 올리고, 끝나면 지웁니다.

**데이터**
- GCS `gs://gamst-bench-datasets-optimum-zone/parallel-2026-11-20/data/{mot,repcount,rppg}`: 로컬 `/tmp/caplab`에서 올린 것
- VM `/srv/hf/clips`: StreamArena 119 클립

### VM 안 구성 (`/srv`)

**모델 (`/srv/models/`, revision 고정)**

| 모델 | revision |
|---|---|
| `Qwen3.5-9B` | `c202236…` |
| `Qwen3.8-27B-FP8` | `c3a462c…` |
| `Qwen3-ASR-1.7B` | `7278e1e…` |
| `sam3.1` | `daa6319…`, `sam3.1_multiplex.pt` |
| `Video-Depth-Anything-Large` | `7aafbcb…` |

- `fetch_weights.sh`로 받는 것: rfdetr, yoloe, mediapipe, ced, talknet, nemotron3-diar

**공식 소스 (`/srv/src/`, 어댑터가 고정한 revision)**
- sam3 `f664253`
- Video-Depth-Anything `4f5ae23`
- CED `4b2149d`
- ten-vad `9a3f4b5`
- TalkNet-ASD `6d68214`
- rPPG-Toolbox `b7500b8`

**모델별 실행 환경 (`/srv/envs/`)**

| 환경 | 구성 |
|---|---|
| `sam31` | torch 2.10 cu128 |
| `vda` | torch 2.8 |
| `det` | rfdetr 1.12.1, ultralytics 8.4.212 |
| `torch28` | transformers 5.23, TalkNet / diarization / EfficientPhys 용 |
| `asr` | qwen-asr 0.0.6, vllm 0.14.0 |
| `mp` | mediapipe 0.10.35 |
| `audio` | onnxruntime-gpu, CED / TEN-VAD 용 |
| `/srv/venv-vllm34` | vllm 0.34, Qwen3.5-9B / jev 서버 |

- uv python 3.12는 `~/.local/share/uv/python`에 있습니다.

**GAMST 소스**
- `/srv/gamst`: 로컬 작업 트리를 tar로 복사한 것입니다. 수정하면 다시 복사해야 합니다.

**LLM 서버** (`/tmp/serve_llms.sh`, 로그 `/srv/par/logs/`)
- VLM: `:8000` `qwen3.5-9b`, gpu-mem 0.24, 약 22GB
- jev: `:8001` `jev`, gpu-mem 0.40, 약 38GB
- 둘이 **동시에 기동하는 것을 확인**했습니다.
- jev의 도구 호출(`qwen3_coder` 파서)이 동작합니다.

---

## 5. 지금까지 한 것 (검증된 사실만)
- GPU 확보: Spot RTX PRO 6000 VM 가동 중입니다.
- **jev = Qwen3.8-27B-FP8**로 정했습니다(`JEV_DECISION_MODEL.md`).
- **MCP Watch 구현 완료** (GAMST 커밋 `ee2bf22`, push됨)
  - 파일: `gamst/watch.py`, `gamst/mcp.py`, `gamst/engine.py`, `tests/test_watch.py`
  - 테스트 4개 통과
  - 기능:
    - 도구 6개
    - 규칙 sha256 버전
    - 주인만 수정 가능
    - jev 제안은 자동 적용하지 않음
    - 감사 로그
    - 근거별 기여도
    - 쿨다운과 분당 상한
    - 시간 구간 Context
    - `notifications/gamst/watch_alarm` 푸시
- 전체 테스트 스위트에는 **기존 실패**가 있습니다. 예: smolvlm 프로필이 빠져 있음.
  - 내 변경으로 새로 생긴 실패인지 base와 비교하는 중입니다(`/tmp/fail_base.txt` vs `/tmp/fail_new.txt`).
- **GamstCrew 병렬 실행 코드 작성** (아직 VM에서 실행 안 함, 커밋 전)
  - `gamstcrew/media_bus.py`: 공유메모리 프레임·오디오 링, seqlock
  - `gamstcrew/media_source.py`: 한 번 디코딩해서 실시간 속도로 게시
  - `gamstcrew/parallel_worker.py`: 모델 환경 안에서 GAMST `provider_worker._construct`로 어댑터를 생성. GPU 예산 적용
  - `gamstcrew/parallel.py`: 예산 예약, 서버·워커 기동, 1Hz 감시, 요약
- Gamst-bench 실험 폴더:
  - `run_crew.py`: 스택 JSON으로 여러 모델 동시 실행
  - `fetch_weights.sh`
- 어댑터 수정 (커밋 전): RF-DETR에 variant 지원 추가(`rfdetr.large` 프로필)
  - 파일: `gamst/providers/detection.py`, `provider_profiles.py`, `provider_worker.py`

## 5.5 현재까지의 실제 검증 결과

- `VERIFY.md`에 solo 결과를 기록했습니다.
- 성공: RF-DETR Large, YOLOE-26x, MediaPipe pose/hand/face, CED-Base, TEN-VAD, Nemotron diarization, Qwen3-ASR.
- 제한/실패:
  - SAM 3.1: 공식 16-object multiplex 구조로 로드되지만 실제 tracker에서 `vision_feat=[5184,0,256]` 오류가 반복됩니다. 실패로 기록합니다.
  - VDA-L: 89.7ms/frame, 53.5% drop. 실행은 되지만 1배 realtime 전체 프레임 처리는 안 됩니다.
  - FactorizePhys: 공식 checkpoint/strict=False graph는 로드됐지만 Haar face window가 부족해 pulse 결과 확정 전입니다.
  - Qwen3.5 semantic: JSON schema 출력은 정상. 6프레임/3초 cadence solo에서 drop 2.5%, 호출 p95 4.86초로 실시간 응답 자체는 느립니다.
- MOT17 평가 완료. 상세: Gamst-bench `MOT17_RESULTS.md`.
  - MOT17-09: GAMST 0.649/0.963, B 0.703/0.712, C-local 0.687/0.721, C-Gemini 0.649/0.963 (precision/recall).
  - MOT17-04: GAMST 0.742/0.972, B 0.782/0.936, C-local 0.742/0.972, C-Gemini 0.742/0.972.
  - jev가 첫 Watch에 잘못된 subject를 만들었고, 명시적 검증 후 재시도하여 최종 결과를 냈습니다.
 - 남은 범위: 사용자 직접 촬영이 필요한 점프/사람+소리 확률 구름, SAM3.1 내부 오류 해결, 공식 패키지 직접 비교 일부.
- 최종 평가 완료: MOT17 2 sequences, StreamArena 88개 B/C arms, RepCountA 151개, MCD-rPPG 40개. 수치는 Gamst-bench `FINAL_REPORT.md`와 `REMAINING_RESULTS.md`.
- hands-on MCP Watch demo 완료: alarm 1건, evidence contribution 1.0 확인.
- VM `gamst-par`는 최종 작업 후 TERMINATED.
- PR은 draft 상태로 남겨 실패 모델과 blocker를 검토할 수 있게 했습니다. 자동 merge하지 않았습니다.



## 6. 다음 할 일 (순서대로)

**1. 가중치 받기**
- VM에서 `fetch_weights.sh` 결과를 확인합니다.
- sha256을 `stack.json`과 `RESUME.md`에 기록합니다.

**2. 어댑터 능력 상향** (결정 기록에 남길 것)
- VDA: Small 전용 → Large 허용
  - Large는 CC-BY-NC입니다. 연구용으로만 쓴다고 표시합니다.
  - VDA 어댑터가 torch 2.1.1을 요구하지만 Blackwell(sm_120)에서는 돌지 않습니다. torch 2.8로 고정을 바꿔야 합니다.
- CED: tiny/mini → base
  - worker 프로필(`ced.onnx`)을 추가합니다.
  - 공식 `export_onnx.py`로 ONNX를 만듭니다.
- TEN-VAD: worker 프로필을 추가합니다.
- YOLOE: `yoloe-26x-seg` + `mobileclip2_b`
  - 어댑터의 파라미터 상한은 80M입니다. 넘으면 26l을 씁니다.
- MediaPipe: heavy 자세 모델과 손·얼굴 모델
- rPPG: EfficientPhys 대신 **FactorizePhys**(NeurIPS 2024, rPPG-Toolbox `final_model_release`)
  - 지금 어댑터는 EfficientPhys 전용이라 FactorizePhys 지원을 추가해야 합니다.
  - CVPR 2026 PHASE-Net은 가중치 공개 여부를 확인하지 못했습니다.
- 말하는 사람 찾기:
  - 현재 TalkNet 어댑터를 먼저 검증합니다.
  - 그다음 LR-ASD(IJCV 2025, MIT, 가중치 공개)로 교체할지 판단합니다.
  - 최신 CAST-ASD와 ROAM-ASD는 가중치 공개 여부를 확인해야 합니다.

**3. 모델별 단독 검증**
- `run_crew.py --only <모델>`로 모델 하나씩 실행합니다.
- 공식 패키지 직접 호출 결과와 비교합니다.
- 실시간 배율, 지연, 프레임 드롭을 기록합니다.

**4. 전체 동시 실행**
- 모든 모델 + VLM 서버 + jev 서버를 함께 띄웁니다.
- GPU 예산 합계가 96GB 이하인지 확인합니다.
- 모델별 실시간 배율을 측정합니다.

**5. GAMST 연결**
- crew의 `results.jsonl`을 GAMST 엔진(`push_observation`)으로 넣습니다.
- MCP Context와 Watch를 연결합니다.
- jev가 MCP 도구로 Watch를 등록하는 루프를 만듭니다.

**6. 평가**
- B / C-local / C-Gemini 팔로 MOT17 → StreamArena → RepCount → MCD-rPPG 순서로 실행합니다.
- 재개 가능하게 만듭니다: JSONL 이어쓰기, 이미 끝난 id는 건너뛰기.

**7. 마무리**
- StreamingVLM 0/88을 문서에 무효로 표시합니다(Gamst-bench `docs/results/gcp-2026-11-19-hf.md`).
- 보고, 데모, VM 정지, PR 머지.

## 7. 함정 (겪은 것)

**셸**
- bash 도구가 zsh 위에서 돕니다. 반복문은 `bash -lc`로 감쌉니다.
- 원격에서 프로세스를 끌 때 `pkill -f`를 쓰지 않습니다(자기 자신이 매칭됨).
- 원격 장기 작업은 `setsid nohup … < /dev/null &`로 띄웁니다.
- ssh 명령이 끝나지 않고 매달리면 `timeout`으로 감쌉니다.

**VM 환경**
- 옛 venv는 uv python 경로에 묶여 있습니다. 새 VM에서는 `uv python install 3.12`가 필요했습니다.
- `/srv`가 이미 디렉터리로 있으면 `ln -sfn`이 안쪽에 링크를 만듭니다. 먼저 `rmdir /srv`를 합니다.
- g4 머신은 hyperdisk만 됩니다. 디스크는 SSD_TOTAL과 HDB_TOTAL 쿼터를 모두 씁니다.

**모델 서빙**
- Qwen3.5와 Qwen3.8 서빙에는 `--reasoning-parser qwen3`을 씁니다.
- 상태 생성에는 `chat_template_kwargs.enable_thinking=false`를 씁니다.

**GAMST 어댑터**
- 어댑터는 버전을 엄격하게 고정합니다. 예: RF-DETR 1.12.1, torch 2.8.0, mediapipe 0.10.35, qwen-asr 0.0.6 / vllm 0.14.0 / transformers 4.57.6, SAM 3.1 torch 2.10.0.
- 어댑터는 `GAMST_PROVIDER_WORKER=1`일 때만 로드됩니다. 영상 입력은 uint8 **BGR**입니다.
- RF-DETR은 `sensor.coordinate_frame`이 필요합니다. worker 스펙에 넣어야 합니다(**아직 미구현**).

**기타**
- Gemini Live는 us-central1의 `gemini-live-2.5-flash-native-audio`에서만 동작합니다. Context 압축이 필요합니다.

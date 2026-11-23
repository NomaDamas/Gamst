# RESUME: parallel GAMST experiment (2026-11-20 ~)

이 파일은 중단되었다가 다시 시작할 때 읽는 파일입니다. 작업 단계마다 갱신합니다. **처음 읽는 사람(다른 모델 포함)은 `HANDOFF.md`를 먼저 읽으세요.**

- **원문 계획:** `PLAN.md` (수정 금지)
- **브랜치:** `feat/parallel-gamst-experiment` (GAMST, base `feat/temporal-contract`)

## 0. 재개 방법
1. 이 파일의 "체크리스트"에서 첫 번째 미완료 항목부터 진행합니다.
2. "현재 자원"에서 GCP VM 상태를 확인합니다(`gcloud compute instances list`). 작업이 없으면 VM은 반드시 정지합니다.
3. 결과물은 "산출물 위치"에 적힌 경로에만 둡니다.

## 1. 체크리스트

| # | 항목 | 상태 | 메모 |
|---|---|---|---|
| 1 | 계획 원문 문서화 (`PLAN.md`) | ✅ | |
| 2 | GitHub 이슈 생성 + draft PR | ✅ | 이슈 #351, PR #352 |
| 3 | GPU 할당 신청: RTX PRO 6000 1장 (안 되면 H100 1장) | ✅ | 온디맨드 RTX·H100 거절, **Spot RTX PRO 6000 1장 승인** |
| 4 | jev decision 모델 딥리서치 → SOTA 선정·고정 | ✅ | Qwen3.8-27B-FP8 (`JEV_DECISION_MODEL.md`) |
| 5 | 모델별 SOTA 재확인 (rPPG, 말하는 사람 찾기 등) 및 버전 고정 | 🔶 | ASR=Qwen3-ASR-1.7B@7278e1e, SAM3.1@daa6319, VDA-L@7aafbcb, RF-DETR Large, YOLOE-26x, rPPG=FactorizePhys 후보, ASD=TalkNet 검증 후 LR-ASD 판단 |
| 6 | 실제 가중치 검증: Qwen3.5-9B, Qwen3-ASR, SAM 3.1, RF-DETR, Video Depth Anything, rPPG, CED, 화자 분리, 말하는 사람 찾기, 자세·손 | 🔶 | `VERIFY.md`: RF-DETR/YOLOE/pose/hand/face/CED/TEN-VAD/diarization/ASR 완료. SAM 실패(SAM3 내부 tracker 오류), VDA 실행 완료(53.5% drop), FactorizePhys pulse output 미확정. 공식 패키지 비교 일부 남음 |
| 7 | GAMST MCP Watch 도구 구현: create, update, delete, list + 알림 + 시간 Context | ✅ | 커밋 ee2bf22, tests/test_watch.py 4개 통과 |
| 8 | jev 연결: Watch 등록, 조정 제안 | ✅ | jev tool call 확인. MOT17에서 첫 subject 오류를 수정하고 재실행: Watch 생성·C-local/C-Gemini 경로 동작 |
| 9 | GamstCrew 병렬 구성: 모델별 격리, GPU 메모리 예산, 공유 프레임과 시계, 실시간 감시 | ✅ | PR #23. full concurrent smoke에서 shared rings/clock/1Hz monitor/worker isolation 확인. 실패 worker는 결과에 남김 |
| 10 | 큰 GPU에서 모든 모델 동시 기동 + 실시간 계수 측정 | 🔶 | `PARALLEL_SMOKE.md`: 60초 clip, wall RTF 1.0438. RF-DETR/audio/pose/hand/face/diar 성공; SAM/ASR/FactorizePhys 실패 또는 미완, VDA/YOLOE/semantic drop 큼 |
| 11 | 평가 B / C-local / C-Gemini(3.8 flash): MOT17 확률 구름 | ✅ | `MOT17_RESULTS.md`; 두 시퀀스 실행 완료 |
| 12 | 평가: StreamArena (말과 기억) | ✅ | 88개 admission. B 7/88, C-local 0/88, C-Gemini 3/88. Judge=`gemini-3.8-flash`, Pro 재호출 없음 |
| 13 | 평가: RepCount (횟수) | ✅ | 151개 complete real-data artifact: pose native MAE 4.404/OBO 0.437, pose 1Hz MAE 10.868/OBO 0.172 |
| 14 | 평가: MCD-rPPG (심박) | ✅ | 40개: POS MAE 19.09 bpm, ±5 bpm 0.45; 1Hz MAE 86.06 bpm |
| 15 | 점프, "사람 + 소리" 확률 구름: 직접 촬영 필요 | ⏳ | 사용자 촬영 대기 |
| 16 | 결과 보고 + 직접 써보기(데모) + VM 정지 + PR 머지 | 🔶 | demo 완료, VM `gamst-par` TERMINATED. 최종 보고서와 PR ready/merge 정리 중 |

## 2. 현재 자원 (마지막 갱신: 2026-11-20)

**GCP**
- 프로젝트 `optimum-zone`
- 예산 가드: `gamst-bench-guard` $300
- `GPUS_ALL_REGIONS` 한도 = 1

**VM** (자세한 내용은 HANDOFF.md §4)
- `gamst-par` (asia-northeast3-a, g4-standard-48, RTX PRO 6000 96GB, **Spot**): RUNNING (2026-11-20 11:5x UTC부터)
- 데이터 디스크: `gamst-par-data` → `/srv` (스냅샷 `gamst-hf-data-2026-11-20`에서 복원)
- `gamst-hf-bench` (L4): TERMINATED. 원래 디스크 `gamst-hf-data`는 삭제(스냅샷만 남김)
- 모델별 환경: `/srv/envs/{sam31,vda,det,torch28,asr,mp,audio}`, LLM 서버용 `/srv/venv-vllm34`

**Vertex 모델**
- C-Gemini: `gemini-3.8-flash` (GA, global; 2026-11-20 호출 확인)
- Gemini 3.1 Pro는 사용 금지 (토큰 보존)

**데이터**

| 데이터 | 위치 |
|---|---|
| MOT17-09 / 04 | 로컬 `/tmp/caplab/mot` (HF `Morrison1025/MOT17@18fea9c`) |
| RepCountA test | 로컬 `/tmp/caplab/repcount` (HF `lmms-lab-eval/repcounta-lance@ae2e0b0`) |
| MCD-rPPG | 로컬 `/tmp/caplab/rppg` (HF `kyegorov/mcd_rppg@bf59b3c`) |
| StreamArena 클립 119개 | VM `/srv/hf/clips`, GCS `gs://gamst-bench-datasets-optimum-zone/gamst-hf/2026-11-19/clips`, HF `minsing-jin/gamst-hf-bench`(private) |
| 기존 A 결과 (Gemini 3.1 Pro 단독) | Gamst-bench `data/campaigns/gcp-2026-11-19-hf/`, `capability_lab/results/gemini_count.jsonl` |

## 3. 결정 기록 (원문 계획과 다른 점, 확정된 세부)
- 2026-11-20: Qwen3.5-9B를 스트리밍 VLM으로 확정했습니다. Qwen 공식 카드 기준 VideoMME(자막 없음) 78.4로 Qwen3-VL-30B-A3B(73.3)보다 높습니다.
- 2026-11-20: C-Gemini 모델은 `gemini-3.8-flash`입니다(사용자 지정, Pro 아님).
- 2026-11-20 GPU 할당 결과:
  - RTX PRO 6000 온디맨드(asia-northeast3): 자동 거절
  - H100(us-central1): 자동 거절
  - **RTX PRO 6000 Spot(asia-northeast3) 1장: 승인**
  - 결정: 승인된 Spot RTX PRO 6000 96GB 1장(g4-standard-48, asia-northeast3-a)으로 모든 모델을 병렬 실행합니다. Spot은 선점될 수 있으므로 모든 작업은 재개 가능하게(JSONL append, done-id skip) 만듭니다. 성능(모델 구성) 타협은 없습니다.
- 2026-11-20 jev decision 모델 딥리서치 결과 → **Qwen3.8-27B-FP8** (`Qwen/Qwen3.8-27B-FP8@c3a462c`, Apache-2.0, 2026-08 출시, 멀티모달). 근거는 `JEV_DECISION_MODEL.md`에 있습니다.
- 2026-11-20 어댑터 능력 상향/수정 (모두 GAMST `feat/parallel-gamst-experiment`):
  - RF-DETR: Nano 전용 → variant 지원, 실험은 **Large** (`rfdetr.large`). 단독 실측: 24fps 실시간, 평균 7.9ms/프레임, p95 지연 9.9ms, GPU 910MiB, 드롭 0.4%(시작 직후).
  - Video Depth Anything: Small 전용 → **Large** 허용. Large 가중치는 CC-BY-NC-4.0(비상업 연구용)이며 capability license에 표기. torch 2.1.1 고정은 Blackwell(sm_120)에서 돌지 않아 torch 2.8.0/cu128로 변경. fp16(공식 데모 기본값) 옵션 추가.
  - CED: tiny/mini → **base** 허용. Zenodo 체크포인트 403 → 저자 HF `mispeech/ced-base@db3e14a` ONNX(waveform 입력) 사용. 저자 PyTorch 모델과 같은 실제 음성 5초 비교: 상위 2개 라벨 동일, 최대 확률 차이 0.031.
  - Qwen3-ASR: 공식 HF 리비전이 2개 shard로 배포되어 번들 검증기가 shard+index를 받도록 확장. vLLM 메모리 비율 하드코딩(0.5) → 설정값(병렬용 0.07).
  - StreamingVLM(semantic) 어댑터: `interpretation="untrusted_semantic_hypothesis"`가 GAMST 스키마에 없는 값이라 생성 자체가 실패하던 기존 버그 → `semantic_description`으로 수정. Qwen3.5용 요청 옵션(thinking 끔, JSON 출력) 추가.
  - rPPG: **FactorizePhys**(NeurIPS 2024) provider 신규(`factorizephys_context.py`). 공식 rPPG-Toolbox 모델·추론 설정·후처리를 그대로 호출. 차이: 얼굴 검출은 공식 옵션 중 "HC"(Haar) 사용(공식 설정의 YOLO5Face 가중치가 배포되지 않음).
  - CED, TEN-VAD, FactorizePhys를 worker 프로필로 등록(`ced.onnx`, `ten_vad.native`, `factorizephys.local`).
- 2026-11-20 말하는 사람 찾기(ASD): 기존 TalkNet 어댑터는 얼굴 crop+음성 짝 입력(`gtav001-paired-input`)이 필요해 범용 worker로 바로 못 붙음. 얼굴 crop 공급 경로를 만든 뒤 TalkNet 검증 → LR-ASD 교체 여부 판단 (미완).
- 2026-11-20 VM 환경 보강: mediapipe용 `libgles2 libegl1`, sam31 `setuptools<81`(pkg_resources), vda `easydict xformers 0.0.32.post2`, det `ultralytics/CLIP@7ffa84b`.
- 2026-11-20 단독 검증 측정은 Gamst-bench `VERIFY.md`에 기록했습니다. 확인된 실측:
  - RF-DETR Large: 24fps 실시간, 7.9ms/프레임, p95 9.9ms, 드롭 0.4%, GPU 910MiB.
  - YOLOE-26x: 12.0ms/프레임, p95 16.9ms, 드롭 0.5%, GPU 1.71GiB.
  - Pose heavy: 32.3ms/프레임, p95 91.6ms, CPU, 드롭 8.3%; Hand 9.8ms/프레임; Face 5.7ms/프레임.
  - CED-Base, TEN-VAD, Nemotron diarization: 각각 0% 드롭; CED 4.0ms/100ms block, TEN-VAD 0.27ms/32ms, diarization 1.5ms/80ms.
  - Qwen3.5 semantic은 8프레임/2초 설정에서 평균 호출 약 2.6초라 20% 드롭 → 6프레임/3초 설정으로 변경. 이는 **모델을 순차로 돌린 폴백이 아니라**, 병렬 스트림에서 VLM 자체의 cadence를 낮춘 설정입니다.
  - SAM3.1은 checkpoint가 16-object multiplex 구조라 `multiplex_count=16`으로 수정. solo에서 8.7GiB 이상 사용 → stack 예산 12GiB.
  - VDA-L은 Blackwell xformers flash 커널 오류로 xformers 제거. solo 약 99ms/프레임, stack 예산 8GiB.
  - FactorizePhys는 공식 inference의 strict=False(FSAM train-only keys)와 Haar face backend를 사용. ASR는 official 2-shard bundle + max_model_len=8192.
- 2026-11-20 데이터 업로드/준비: MOT17, RepCount, MCD-rPPG가 GCS `parallel-2026-11-20/data/`에 있고 VM `/srv/data`로 복사 완료. StreamArena admission 88개, 합계 약 95.4분.
- 2026-11-20 MOT17 평가 완료:
  - 결과 문서: Gamst-bench `MOT17_RESULTS.md`.
  - MOT17-09: GAMST RiskField P/R 0.649/0.963, B 0.703/0.712, C-local 0.687/0.721, C-Gemini 0.649/0.963.
  - MOT17-04: GAMST RiskField 0.742/0.972, B 0.782/0.936, C-local 0.742/0.972, C-Gemini 0.742/0.972.
  - C-local/C-Gemini 모두 실제 Watch alarm + MCP timeline을 사용했습니다. 첫 jev subject 오류는 explicit validation 후 재실행했습니다.
- 2026-11-20 SAM 3.1 재검증 확정 실패: 공식 checkpoint 구조(`multiplex_count=16`)로 로드·propagation 시작은 되지만 실제 프레임에서 SAM3 내부 `vision_feat`가 `[5184, 0, 256]`이 되어 tracker가 실패합니다. 공식 코드/가중치와 Blackwell에서 재현되었습니다. 결과는 실패로 보고합니다.
- 2026-11-22 추가 범위 결정: RelateAnything은 현재 도입하지 않습니다. RF-DETR/YOLOE box+track+GAMST temporal state로 가능한 관계/zone 조건은 기존 perception에서 처리하고, holding/riding/on 등 필요 시 Issue #353의 별도 provider로 검토합니다.
- 2026-11-22 추가 구현: `audio.context` provider와 기본 disabled realtime voice backend metadata 설정. full-duplex 자체는 이번 범위가 아니며, raw audio 대신 VAD/silence/AEC/CED/diarization/DOA 가능값/prosody/ASR를 Context로 전달합니다. 관련 Issue #354.
- Coin Halo (#355, 2026-11-23): 사용자 승인으로 동전 데모를 첫 Halo로 확정했습니다. 둥근 윤곽 추적은 폐기했고, YOLOE-26s(ProcessProvider)가 이름을 붙이고 `ShapeTracks`가 매 프레임 추적합니다. 회전수는 `ObjectSpinCounter`(면 밖 회전)와 `AngularMotionCounter`(면 안 회전)가 셉니다. 실제 영상에서 동전 관측 프레임은 344/375와 54/54(비행 구간)였고, 30fps·25fps 영상은 모두 `undersampled`로 판정해 회전수를 내지 않았습니다. 회전수 정확도는 사용자가 촬영할 240fps 던지기 10회로 검증해야 합니다(통과 기준: 8회 이상 정확히 일치). 결과는 `docs/halo-coin.md`에 있습니다. GCP 디스크·스냅샷 삭제는 결제 비활성으로 거부된 상태입니다.
- 2026-11-23: 240fps iPhone 실시간 앱은 사용자 결정으로 후순위입니다. GitHub star는 동의를 받은 경우에만 누릅니다(`gamst` 첫 대화형 실행 시 질문하고, 에이전트는 `AGENTS.md`를 따릅니다). PyPI trusted publisher(gamst-sdk, NomaDamas/Gamst, publish.yml, environment pypi)는 Aside에서 등록했습니다.

## 4. 산출물 위치
- **설계·결과 문서:** 이 폴더
- **실행 스크립트·원시 결과:** Gamst-bench `experiments/parallel-gamst-2026-11-20/` (브랜치 `feat/parallel-gamst-experiment`)
- **병렬 실행 코드:** GamstCrew `gamstcrew/parallel*.py`, `media_*.py`

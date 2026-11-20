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
| 6 | 실제 가중치 검증: Qwen3.5-9B, Qwen3-ASR, SAM 3.1, RF-DETR, Video Depth Anything, rPPG, CED, 화자 분리, 말하는 사람 찾기, 자세·손 | ⏳ | 실패하면 공식 패키지나 상용 API로 대체하고 비용을 표시. VLM·jev 서버 동시 기동은 확인 |
| 7 | GAMST MCP Watch 도구 구현: create, update, delete, list + 알림 + 시간 Context | ✅ | 커밋 ee2bf22, tests/test_watch.py 4개 통과 |
| 8 | jev 연결: Watch 등록, 조정 제안 | ⏳ | jev 서버(:8001) 도구 호출 확인. 루프는 미구현 |
| 9 | GamstCrew 병렬 구성: 모델별 격리, GPU 메모리 예산, 공유 프레임과 시계, 실시간 감시 | 🔶 | 코드 작성(media_bus, media_source, parallel, parallel_worker). VM 실행 전, 커밋 전 |
| 10 | 큰 GPU에서 모든 모델 동시 기동 + 실시간 계수 측정 | ⏳ | |
| 11 | 평가 B / C-local / C-Gemini(3.8 flash): MOT17 확률 구름 | ⏳ | |
| 12 | 평가: StreamArena (말과 기억) | ⏳ | |
| 13 | 평가: RepCount (횟수) | ⏳ | |
| 14 | 평가: MCD-rPPG (심박) | ⏳ | |
| 15 | 점프, "사람 + 소리" 확률 구름: 직접 촬영 필요 | ⏳ | 사용자 촬영 대기 |
| 16 | 결과 보고 + 직접 써보기(데모) + VM 정지 + PR 머지 | ⏳ | |

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

## 4. 산출물 위치
- **설계·결과 문서:** 이 폴더
- **실행 스크립트·원시 결과:** Gamst-bench `experiments/parallel-gamst-2026-11-20/` (브랜치 `feat/parallel-gamst-experiment`)
- **병렬 실행 코드:** GamstCrew `gamstcrew/parallel*.py`, `media_*.py`

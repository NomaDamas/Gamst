# HANDOFF feat/sota-modules (서브에이전트 D)

워크트리 `.local-run/wt-sota-modules`. 통합 브랜치 `96e886e`(crew, iPhone, MediaPipe 병합 뒤)까지 합쳤습니다. push와 PR은 하지 않았습니다.

## 1. 상태 요약

- 코드와 계산 테스트는 끝났습니다. **실제 모델로 정확도를 잰 것은 하나도 없습니다.**
- 실제 가중치로 돌린 것은 CAM++ 하나뿐입니다. macOS 한국어 TTS 목소리 2개(Yuna, Eddy), 문장 3개씩.
  - 같은 목소리 코사인 0.910, 다른 목소리 0.277 (M2 Pro CPU, sherpa-onnx 2.0.3, 비교 1쌍).
  - 이 실행은 sherpa-onnx를 같은 프로세스에 둔 경로였고, 지금 레시피가 쓰는 워커 경로로는 돌리지 않았습니다.
  - **사람 목소리 아님.** 두 사람 목소리를 가려내는 정확도, 겹치는 말, 소음, 게임 소리, 임계값 0.5는 확인 안 함.
- LVFace-T 가중치(76.7MB)는 받아서 sha256만 확인했고 돌리지 않았습니다.
- 그 뒤 owner 지시로 다운로드와 실모델 실행을 멈췄습니다.

| 모듈 (파일) | 코드 | 계산 테스트 | 실모델 실행 |
|---|---|---|---|
| `audio_ring` 링 버퍼, 홉 시계, 리샘플 | 있음 | 있음 | 모델 없음 |
| `speaker_verify` 등록 평균, 코사인, 히스테리시스, 발화 재확인 | 있음 | 있음 | CAM++만(TTS 목소리). WeSpeaker w2v-BERT는 코드만 |
| `target_speech` SpeakerGate | 있음 | 있음 | 모델 없음 |
| `target_speech` WeSep, USEF, SAM-Audio span | 있음 | 블록·앵커 계산만 | 확인 안 함 |
| `asr_words` AlignedASR + Qwen3-ForcedAligner | 있음 | 홉·공백·단어 시각 계산 | 확인 안 함 |
| `asr_words` WhisperWords, `FasterWhisperPlugin(word_timestamps)` | 있음 | 끝 창 채우기만 | 확인 안 함 |
| `person_match` LVFace, SOLIDER, TargetPerson | 있음 | 정렬, 잠금, 해제 | 확인 안 함 |
| `sapiens2_pose` | 있음 | NMS, 308점 → 포즈 채널 | 확인 안 함 |
| semantic lane (`semantic-scene` part) | 있음 | 가짜 로컬 HTTP 서버와 실제 워커 프로세스로 1 fps, 끄기 | 실제 VLM 확인 안 함 |
| 말투 채널(`voice_style`) + exemplar(audio) | 있음 | 있음(합성음 등록) | 모델 없음. 임계값 .12/.18은 포즈 기본값이고 목소리에는 확인 안 함 |
| `model_workers` 워커 프로토콜 | 있음 | 실제 하위 프로세스 + 대역 모델 | 확인 안 함 |
| 레시피 2개와 `person-action-count`, enroll, `--truth` | 있음 | 대역 모델로 전 구간 | 확인 안 함 |

대역(stand-in) 모델로 나온 개수는 배선 확인일 뿐이고 정확도가 아닙니다.

## 2. 구조

- **모델마다 자기 환경, 자기 프로세스.** 무거운 모델은 전부 `gamst/providers/model_workers.py`의 워커로 돕니다. 레시피는 워커만 씁니다.
  - 워커 프로필: `speaker.embedding`, `face.embedding`, `body.embedding`, `target_speech.local`, `qwen3.forced_aligner`, `faster_whisper.words`, `sapiens2.pose`. 파이썬은 설치 프로필의 `python`(환경별 venv)입니다.
  - semantic lane은 기존 `streamingvlm.local` 워커, Qwen3-ASR은 기존 `qwen3.asr.endpoint` 워커를 그대로 씁니다.
  - 이 워커들은 **Pipeline 모듈이 아닙니다.** 임베딩은 생체 정보이고 분리된 음성은 원본 매체라서 레코드가 되면 안 됩니다. `role_module`은 이 프로필을 거절하고, 한 모듈의 질문에만 답합니다.
  - 음성을 돌려주는 프로필은 `target_speech.local` 하나뿐입니다. 길이가 요청과 같아야 하고 레코드로 검증되지 않습니다(`process_provider.py`).
  - 워커가 아닌 것(계산만): SpeakerGate, VoiceChannels, ExemplarEvents, TermEvents, TargetPerson의 판단, 물리 모듈. `prosody`도 지금은 같은 프로세스입니다.
- `Derived.timestamp_ms`: 파생 입력이 자기 캡처 시각을 가집니다. 입력 이름에 `:`를 쓸 수 있습니다(`voice:me`).
- 선택 모듈: 레시피 JSON `"optional": "on"|"off"`, CLI `--with` / `--without`, `consumes_without`, 레퍼런스 `for`.
- `recipes/__init__.py`의 `plan`/`setup`과 `DEFAULT_MODELS`는 고치지 않았습니다.
- 레퍼런스 하나에 예시 하나를 넣는 구조를 새로 강제하지 않았습니다(few-shot/텍스트 동작 지정은 설계 중).

## 3. 레지스트리에 넣은 것 (`gamst/crew/registry.py`)

sha256은 배포처 메타데이터(GitHub digest, HF LFS oid, ModelScope API)에서 읽은 값입니다. "받아서 확인"이라고 쓴 것만 바이트를 직접 해시했습니다. RAM/VRAM/디스크는 추정(estimated)입니다.

| 역할 / 후보 | 자동 설치 | 가중치 | 라이선스, 비고 |
|---|---|---|---|
| `audio.speaker_verify` / `sherpa.campplus_zh_en` (기본) | O | 28.3MB, 받아서 확인 `53cab20b…` | 코드 Apache-2.0, 가중치 라이선스 확인 안 함 |
| 〃 / `wespeaker.w2vbert2_mfa_vox` | X(수동) | 2.35GB `26d3ac4a…` + 백본 2.32GB `eb890c96…` | CC BY 4.0(WeSpeaker 기준) |
| 〃 / `wespeaker.w2vbert2_mfa_voxblink2` | X(opt-in) | 2.35GB `4c9ace77…` | VoxBlink2 라이선스 확인 안 함 |
| `audio.target_speech` / `wesep.bsrnn_ecapa` (기본) | X | 261.6MB, sha256 미공개 | 코드 Apache-2.0, 가중치 라이선스 표기 없음 |
| 〃 / `usef.tfgridnet` | X(opt-in) | 188MB `74737c8e…`(wham!) | **CC BY-NC 4.0**, `allow_noncommercial`, 8 kHz |
| 〃 / `sam_audio.span` | X | 기존 SAM-Audio | SAM License, 게이트 |
| `audio.word_alignment` / `qwen3.forced_aligner_0_6b` | X | 1.84GB `47831d0e…` | Apache-2.0, CUDA |
| `vision.face_identity` / `lvface.b` (기본) | O | 455.5MB `9d834ed8…` | MIT, Glint360K 조건 검토 필요 |
| 〃 / `lvface.t` | O | 76.7MB, 받아서 확인 `bf8da0e1…` | 〃 |
| `vision.body_identity` / `solider.swin_b_msmt17` | X | Google Drive, 크기와 sha256 모름 | MIT/Apache-2.0, MSMT17 조건 검토 필요, CUDA |
| `vision.pose` / `sapiens2.1b`, `sapiens2.5b` | X(opt-in) | 6.08GB `2dab7014…`, 20.5GB `b4848da8…` | **Sapiens2 License: 생체 처리와 재식별 금지** |
| `vision.semantic_scene` / `qwen3.8-flash.hosted` | X | 없음(호스팅) | `DASHSCOPE_API_KEY`, `allow_remote`, 유료: 견적 먼저 |
| 〃 / `qwen3.8-flash-next.endpoint`, `qwen3.6-35b-a3b.endpoint`, `qwen3.5-4b.endpoint`, `qwen3.5-2b.endpoint` | O(엔드포인트만) | 직접 호스팅 | 리비전 고정, 실행 확인 안 함 |

- 수동 후보는 레시피 항목의 `"config"`(로컬 경로, sha256)와 `"python"`으로 씁니다.
- **고친 것:** `qwen3.asr_1_7b`가 `transformers==5.23.0`을 고정하고 `qwen-asr`가 빠져 있어 로딩이 항상 실패했습니다. 이제 `qwen_asr.py`가 검사하는 `qwen-asr==0.0.6, vllm==0.14.0, transformers==4.57.6`과 같습니다. vllm 0.14.0 메타데이터는 torch 2.9.1을 요구해서 `torch==2.8.0` 고정도 뺐습니다. 상태는 "not run"입니다.
- `tests/test_model_workers.py`가 프로바이더 소스에서 버전 검사를 읽어 레지스트리와 비교합니다(PG는 매니페스트로 검사하므로 제외, YOLOE의 CLIP 커밋은 별도).

## 4. 남은 일

1. 실모델 실행(owner 결정 후). `tests/test_sota_real_models.py`, 환경변수로 로컬 파일을 지정합니다. 없으면 skip입니다.
2. 정답 테이크: 두 사람 이상이 말하는 한국어 녹음과 단어 횟수, 등록 인물 영상. `--truth`로 점수를 냅니다. 화자·얼굴·말투 임계값은 그 자료로 정해야 합니다.
3. Qwen3.8-Flash 호스팅 견적. 공개 가격은 싱가포르 기준 입력 $0.15 / 출력 $0.47 per 1M 토큰(2026-11-25 알리바바 문서). 실제 엔드포인트와 이미지 입력은 확인 안 함.
4. Sapiens2 라이선스가 "지정 인물" 카운트와 충돌하는지 owner 확인. 코드는 Sapiens2 점으로 얼굴을 찾지 않습니다.
5. 설치 환경을 역할별 venv로 바꾸는 것(코디네이터 작업). 지금 `plan`은 `isolation='current'`라서 워커도 같은 파이썬을 씁니다. `faster_whisper.small`이 여러 파일을 한 폴더(`models_dir`)에 풀면 다른 가중치와 섞이고, `FasterWhisperPlugin`이 폴더 전체를 해시합니다. 역할별 폴더가 필요합니다.
6. MOTIP(정확도 옵션 추적기)은 만들지 않았습니다. 추적은 기존 OC-SORT입니다.
7. SOLIDER-REID는 2023년 코드라 shim 3개로 감쌌고 실제 로딩은 확인 안 함. Sapiens2, WeSep, USEF, WeSpeaker, ForcedAligner 어댑터도 공식 소스를 읽고 썼지만 로딩은 확인 안 함.
8. SpeakerVerify 기본 설정은 0.5초마다 임베딩 한 번과 발화마다 한 번을 더 합니다. 속도는 재지 않았습니다.

## 5. 테스트

- 전체 1329개, 오류 15개(원래 있던 stereo 13, sam_audio_native 2), skip 47개. 작업 전 기준은 1166개, 오류 15개, skip 40개였습니다(그 사이 crew, iPhone, MediaPipe 병합 포함).
- 새 테스트: `test_speaker_verify`, `test_target_speech`, `test_asr_words`, `test_person_match`, `test_model_workers`, `test_perception_recipes`, `test_sota_real_models`(전부 skip).

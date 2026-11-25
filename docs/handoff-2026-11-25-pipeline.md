# HANDOFF 2026-11-25: Pipeline / Crew 흡수 / MediaPipe Framework 흡수 / 지정 인물 인식 레시피

다음 세션(사람이든 에이전트든)이 이 문서 하나로 바로 이어 할 수 있도록 정리한 것입니다.
owner 규칙: 한국어로 짧고 쉽게 답하기, 측정한 값만 말하기("확인 안 함" 표시), 범용 모듈만 만들기,
중복 구현 금지, Gemini 3.1 Pro 사용 금지, 비용이 드는 일은 먼저 보고하기, main 체크아웃의
미커밋 파일(docs/*.md, site/app.js, .gjc/)은 건드리지 않기.

## 1. 저장소, 브랜치, 워크트리

| 워크트리 | 브랜치 | 용도 |
|---|---|---|
| `.local-run/wt-speaker-pose` | `feat/target-speaker-keyword-and-pose-count` | **통합 브랜치**, Draft PR #366 (base `feat/humansense-20261112-154922` = Main) |
| `.local-run/wt-crew` | `feat/absorb-gamstcrew` | 서브에이전트 A: GamstCrew를 `gamst/crew/`로 흡수 (30b705f에서 분기) |
| `.local-run/wt-iphone-input` | `feat/iphone240-input` | 서브에이전트 B: iPhone 240 수신기를 입력 모듈로, coin-flip을 Pipeline으로 (30b705f에서 분기) |
| `.local-run/wt-mediapipe` | `feat/mediapipe-graph` | 서브에이전트 C: MediaPipe Framework 흡수 (아래 4절) |
| `.local-run/wt-sota-modules` | `feat/sota-modules` | 서브에이전트 D: SOTA 모델 어댑터, 레시피 (아래 5절) |

- 각 서브에이전트는 자기 워크트리에 `docs/handoff-<branch>.md`를 쓰고 커밋합니다. 다음 세션은 그 문서와 `git log`로 상태를 확인합니다.
- 통합 순서: A(crew) → B(iphone) → C(mediapipe) → D(sota) 순서로 통합 브랜치에 `git merge`합니다. 충돌이 날 곳은 `gamst/recipes/parts.py`와 `gamst/recipes/__init__.py`입니다. 원칙은 다음과 같습니다.
  - **설치와 가중치는 crew 레지스트리 하나만 씁니다.**
  - parts.py에는 build 함수, 샘플링 규칙, crew 역할 참조만 남깁니다.
- 테스트: `PYTHONPATH=. ~/.cache/gamst/demo/venv/bin/python -m unittest discover -s tests -q`
  - Python 3.11, torch 2.8 MPS, mediapipe 0.10.35, ultralytics.
  - 원래 있던 오류 15개는 Main에서도 같습니다: test_stereo_context 7, test_stereo_metric 6, test_sam_audio_native 2.
- 레포 규칙: `main` 브랜치는 없고 Main은 `feat/humansense-20261112-154922`입니다. 릴리스는 태그로 합니다.
- GamstCrew 저장소(NomaDamas/GamstCrew)는 **흡수 PR이 머지된 뒤 archive**합니다(owner 결정). 아직 하지 않았습니다.

## 2. owner 결정 사항 (확정)

1. GAMST 구조: 입력층(카메라·마이크·센서·사용자 정의 입력) → **Pipeline(조합 계층, 승인됨)** → perception(경량 모델 + 수학·물리) → Context → MCP → LLM.
   - GAMST의 범위는 **context 전달까지**입니다. LLM 쪽 조건은 사용자가 만듭니다.
   - Halo는 개념입니다. HUD는 앱 UI(뷰 모듈)이고, 화면의 숫자는 레코드에서 읽습니다.
2. 레시피 = 조합 프리셋입니다. 모든 레시피는 Pipeline으로 돕니다(coin-flip 포함).
3. GamstCrew를 GAMST로 흡수하고, 설치 경로는 하나로 둡니다(중복 금지). 그 뒤 Crew 저장소는 archive합니다.
4. 모델은 SOTA를 씁니다. 법적 검토는 owner가 허가를 받았습니다.
   - 비상업(CC BY-NC) 가중치는 사용자가 직접 켜는 opt-in으로만 둡니다.
5. semantic lane: 1 fps 장면 설명을 context에 넣습니다. AI 안경(1인칭 영상)을 염두에 둡니다.
   - SOTA는 **Qwen3.8-Flash-Next**(호스팅)입니다.
   - 켜고 끌 수 있어야 합니다.
   - 더 가벼운 실시간 대안도 둡니다. 후보는 Qwen3.6-35B-A3B와 Qwen3.5-4B/2B이고, 비전 성능 비교는 확인 안 함입니다.
6. MediaPipe Framework 흡수(합의함):
   - C++ 그래프 전체를 Pipeline 모듈 하나로 감쌉니다. 별도 프로세스로 PyTorch 모듈들과 **병렬·독립**으로 돕니다.
   - Framework 기능은 GAMST에서 조작할 수 있게 합니다.
   - 겹치는 기능(타임스탬프, 스트림 동기화, 흐름 제어, 그래프 조합)은 GAMST(캡처 시계, Pipeline, SensorStreams)에 연결합니다.
   - 없는 기능(C++ 병렬 스케줄링, GPU 이미지 연산, TFLite 추론, 스무딩·추적 계산기)은 추가합니다.
7. 요구 레시피:
   - (1) 지정 인물 동작·자세 카운트(`person-action-count`)
   - (2) 지정 인물 단어 카운트(`person-term-count`)
   - (3) 말투 카운트: 텍스트 패턴과 목소리 톤 둘 다(`person-voice-style-count`)
   - (4) 지정 인물 인식과 목소리 추출을 범용 기능으로(파생 입력)

## 3. 통합 브랜치에서 끝난 것 (커밋됨)

- `gamst/pipeline.py`: Pipeline. 입력, 모듈, 레코드 링크(`"pose:scene.poses"`), `Derived` 파생 입력, 켜고 끄기, `run`/`finish`, `add_reference`(사람 대상은 동의 기록 필수).
- `gamst/inputs.py`: 카메라, 영상 파일, take, WAV, 마이크.
- `Gamst.count_events` + MCP `count_events`, `get_pipeline`, `configure_pipeline_module`(`gamst/counting.py`).
- `gamst/views.py`: CountView, RotationView(HUD = 앱 UI).
- `gamst/recipes/`: 레시피 = inputs/references/modules/view. rotation은 Pipeline으로 돕니다(합성 선풍기 12 rev/s → 11.9회전). `person-action-count` 레시피와 `gamst recipe enroll`.
- 계산 모듈:
  - `providers/pose_exemplar.py`: 포즈 채널, 시연 매칭(hold, sequence DTW). 모달리티는 범용입니다.
  - `providers/term_events.py`: 한국어 정규화, 표기 변형, 단어 시점, 창 단위 텍스트도 받음.
  - `providers/voice_style.py`: 운율 → 톤 채널.
  - `gamst/scoring.py`: 정답 비교(개수, 시각, 값). CLI 연결은 아직입니다.
- 측정값:
  - MediaPipe Pose full VIDEO 중앙값 15.7ms, heavy 63.5ms. M2 Pro CPU, 480×854, 150프레임.
  - 실제 영상 10.6초를 등록하고 실행했습니다(정답이 없어서 정확도는 확인 안 함).
- 설계 문서: `docs/pipeline-plan.md`. SOTA 리서치: `docs/research/speaker-pose-sota-2026-11/`(노트 7개와 보고서).

## 4. 서브에이전트 C: MediaPipe Framework 흡수 (실행 전 구현까지)

- `gamst/mediapipe_graph/`에 다음을 만듭니다.
  - (a) C++ graph runner 소스: stdin으로 프레임 헤더와 BGR을 받고, `CalculatorGraph`를 실행하고, 출력 패킷을 JSON 줄로 내보냅니다. 입력 타임스탬프는 GAMST 캡처 시각(µs)을 그대로 씁니다.
  - (b) Bazel BUILD/WORKSPACE 스니펫과 빌드 스크립트. MediaPipe 소스 리비전을 고정합니다.
  - (c) 그래프 pbtxt: pose+hand+face holistic, 스무딩, FlowLimiter.
  - (d) Python 모듈 `MediaPipeGraphModule`: 프로세스 수명 관리, 그래프 옵션과 side packet, 계산기 켜고 끄기를 GAMST에서 조작합니다. 출력은 레코드로 커밋합니다.
  - (e) crew 레지스트리 역할 등록(빌드 산출물 SHA).
- 테스트는 가짜 runner 프로세스로 프로토콜과 수명을 검증합니다. 실제 빌드와 실행은 디스크 결정 후에 합니다.
- **차단 요인:** 디스크 여유가 18GB인데, Bazel 빌드에 약 8~10GB(추정), 모델에 약 13GB가 필요합니다. owner에게 (a) 외장 디스크 (b) 공간 정리 (c) GPU 서버 중 무엇으로 할지 물어 둔 상태입니다(미응답).

## 5. 서브에이전트 D: SOTA 모델 모듈과 레시피 (실행 전 구현까지)

새 provider 파일만 만들고, parts.py와 레시피는 crew 통합 후 마지막 커밋으로 넣습니다. 근거 수치는 research 노트에 있습니다.

- `speaker_verify`: w2v-BERT 2.0 LoRA-MFA. WeSpeaker `--w2vbert2_mfa`, 상업 가능한 VoxCeleb 전용 가중치가 기본이고 VoxBlink2 가중치는 opt-in입니다. 경량 옵션은 CAM++(sherpa-onnx). 등록 임베딩 평균, 세그먼트 코사인, 임계값.
- `target_speech`(파생 입력 `Derived('voice:<ref>')`): USEF-TFGridNet(CC BY-NC, opt-in) 또는 WeSep BSRNN(Apache). 대안으로 speaker_match 구간을 기존 SAM-Audio span 프롬프트로 넘기는 방법이 있습니다.
- `asr_words`: Qwen3-ASR-1.7B(기존 `qwen_asr.py`, vLLM CUDA)와 Qwen3-ForcedAligner-0.6B를 붙여 단어 시점을 냅니다. Mac에서는 기존 FasterWhisperPlugin(word_timestamps)을 씁니다.
- `person_match` / `target_person`: LVFace-B(MIT) 얼굴, SOLIDER Swin-B(Apache) 몸 재식별, 추적은 기존 OC-SORT(정확도 옵션 MOTIP). 사람 대상 레퍼런스는 동의 기록이 필수입니다.
- `sapiens2_pose`: Sapiens2 1B/5B, DETR-R101 사람 검출, 308 키포인트를 `pose_channels` 매핑으로 연결합니다.
- semantic lane: 기존 `providers/streaming_semantic.py`(OpenAI 호환 엔드포인트)를 Pipeline part로 씁니다(1 fps, `every_ms=1000`). crew 역할 `vision.semantic_scene`에 Qwen3.8-Flash-Next와 가벼운 후보를 추가합니다. **새 VLM 클라이언트를 만들지 않습니다.**
- 레시피:
  - `person-term-count`: 마이크 → (target_speech) → asr_words → term_events, speaker_verify로 재확인.
  - `person-voice-style-count`: 마이크 → prosody → voice_channels → exemplar(audio). 텍스트 말투 패턴은 term_events 정규식 확장으로.
  - `person-action-count`에 person_match → target_person을 앞에 추가합니다.
- 기타:
  - 물리 모듈(kinematics, trajectories, physical_state …)을 레시피 부품으로 등록합니다.
  - `gamst recipe run --truth truth.json` → `gamst.scoring.score`.
  - `enroll`에 voice/person 등록을 추가합니다(샘플 녹음 → 임베딩만 저장, 동의 기록 필수).
- 각 모델 모듈에는 실제 모델을 호출하는 테스트를 두고, 가중치가 없으면 skip합니다(owner 규칙: stub을 완성으로 치지 않습니다).

## 6. 실행 단계 (owner 결정 후)

1. 디스크/서버 결정 → `gamst recipe setup <recipe>`(crew)로 가중치 설치 → 실모델 테스트.
2. Qwen3.8-Flash-Next 호스팅은 비용이 드므로 **견적을 먼저 보고**합니다.
3. 정답 테이크 측정:
   - coin(이미 있음: `~/Desktop/gamst-takes/take-20261124-140345`)
   - owner에게 샷건 자세 영상과 횟수, 단어 음성과 횟수를 요청합니다.
4. PR #366 정리 → Main 머지 → GamstCrew archive.

## 7. 진행 갱신 (2026-11-25 저녁)

- A(crew, 5624cb1)와 B(iphone, 1f1019c)는 통합 브랜치에 합쳤습니다. 합친 뒤 전체 테스트 1254개, 오류는 원래 있던 15개뿐입니다.
- C와 D는 사용량 한도로 중단됐다가 다시 시작했습니다. 중단 시점 파일은 검토 후 각 브랜치에 WIP로 커밋했습니다. **C++는 한 번도 컴파일하지 않았습니다.**
- 바로잡음:
  - `~/.cache/gamst/models/yoloe`는 이 Mac에 없습니다. YOLOE/EfficientTAM 가중치는 `~/.cache/gamst/demo/models`, 포즈는 `~/.cache/gamst/models/mediapipe`에 있습니다.
  - 데모 환경의 OpenCV 중복은 11/25 18:18에 mediapipe를 설치하면서 생겼습니다(mediapipe → opencv-contrib-python 5.1, ultralytics → opencv-python 4.11).
  - PyPI의 gamst-sdk 최신은 0.6.0이지만, 그 0.6.0에는 이 브랜치의 새 모듈이 없습니다. 머지할 때 버전을 올려야 합니다.
  - "PyTorch는 MediaPipe 노드로 못 넣는다"는 과장입니다. 기본 제공 노드가 없을 뿐, LibTorch를 쓰는 C++ calculator를 직접 만들면 가능합니다.

## 8. 추가 진행 (2026-11-25 밤)

- 코인 회귀: A와 B를 합친 1f1019c에서 같은 take를 다시 돌렸고, `refined.json` 결과 7개가 옮기기 전과 완전히 같았습니다.
- owner 확정: 새 버전으로 PyPI 배포(0.7.0, 머지 때 다시 확인), 가중치 루트 하나, 모델마다 환경 하나(레시피 설치도). 실제 설치·다운로드·실모델 테스트는 **구현이 끝난 뒤**에 합니다.
- E: 서브에이전트 A가 `feat/one-env-per-model`(wt-crew)에서 위 결정을 구현합니다. 합치는 순서는 C → E → D입니다(E와 D는 `gamst/crew/registry.py`, `setup.py`, `recipes/parts.py`가 겹칩니다).
- 발견한 결함(수정 맡김):
  - 레지스트리 `qwen3.asr_1_7b`는 transformers 5.23.0을 설치하지만, `qwen_asr.py`는 qwen-asr 0.0.6 / transformers 4.57.6이 아니면 거부합니다 → D.
  - `pose_channels`가 화면 비율을 보정하지 않아, 가로 영상으로 등록하고 세로 영상에서 세면 각도가 달라집니다 → E.
  - `mediapipe.body.video` worker는 랜드마크를 내보내지 않아 시연 매칭에 쓸 수 없습니다 → E.
- 동작 지정 방식 두 가지(논의 중, 구현하지 않음): (1) 예시 영상 few-shot (2) 텍스트. 제안은 텍스트로 후보를 찾고 → 사용자가 확인하면 → 예시로 등록하고 → 빠른 계산으로 세는 것입니다. 텍스트를 받는 카운터로는 OVRCounter가 있습니다(arXiv 2407.17085). 코드 공개 여부는 확인 안 함입니다. "동작을 지정할 수 있는 건 exemplar 방식뿐"이라는 앞의 설명은 틀렸습니다.
- C(mediapipe, 91cb7ce)를 합쳤습니다. 합친 뒤 전체 테스트 1280개, 오류는 원래 있던 15개뿐입니다. C++ 컴파일과 그래프 실행은 한 번도 하지 않았습니다. 남은 합치기는 E → D입니다.
- D(f7daab0)를 합쳤습니다. 테스트 1329개, 오류는 원래 있던 15개입니다. 실모델 정확도는 아무것도 측정하지 않았습니다.
- **합치기 전에 고쳐야 할 것:** D와 E가 worker 출력 필드를 `landmarks` 대신 `keypoints`로 지어 context 원본 차단 검사(`RAW_KEYS`)를 통과했습니다. 우회로 보고 허용하지 않습니다. E가 `feat/one-env-per-model`에서 비공개 worker 통로로 바꾸고 검사에 `keypoints`도 추가합니다. 이것이 끝나기 전에는 PR #366을 머지하지 않습니다.
- E는 구현을 마쳤고(c5bd814), D를 합치고 위 문제를 고치는 중입니다. 기존 exemplar 파일은 다시 등록해야 합니다(화면 비율 보정).
- 병렬 실행 실측(M2 Pro, 별도 프로세스, 480×854, 각 12초, 다른 작업이 함께 도는 상태): 포즈 단독 59.1 fps / YOLOE(MPS) 단독 51.8 → 포즈 1 + YOLOE 1: 57.5 / 49.8 → YOLOE 2개: 43.2~43.3 → 포즈 2 + YOLOE 2: 포즈 42.9~43.1, YOLOE 38.8. 프로세스 메모리: 포즈 약 320MB, YOLOE 약 610MB. GAMST worker 전송 비용은 포함하지 않은 값입니다.
- owner 결정: 동작 지정 두 방식(예시 영상, 텍스트)을 모두 구현합니다. 예시 영상은 owner가 실제 실행 전에 준비합니다. 사용자 확인 단계는 설명 후 답을 기다립니다.
- 원격 실행: semantic lane만 hosted 엔드포인트(https+키, 명시적 opt-in) 코드가 있고 미검증입니다. 일반 모듈을 원격 GPU 서버에서 돌리는 네트워크 worker는 없습니다(ProcessProvider는 로컬 파이프, crew transport는 유닉스 소켓).
- E(dd09e02)를 합쳤습니다. 테스트 1362개, 오류는 원래 있던 15개입니다. `keypoints` 우회는 비공개 통로(`gamst/private_payload.py`)로 바뀌었습니다.
- 남은 owner 결정: 예전 MediaPipe 얼굴/포즈 플러그인과 MediaPipe 그래프 모듈은 관절·얼굴 좌표를 Store에 남기고, MCP `get_state`/`inspect_interval`로 읽힙니다(`get_context`는 아님). 비공개 통로로 옮기면 대시보드 오버레이 그리기를 바꿔야 합니다.
- F: `feat/movement-spec`(wt-movement). 예시 여러 개 + 텍스트로 동작 지정. 확인 단계는 기본 켬(끌 수 있음), owner 답을 받으면 바꿉니다.

## 9. 실시간과 서버 이전 (2026-11-25, 논의)

- 코인 회귀(E까지 합친 코드, 데모 환경 Python): 7개 결과가 옮기기 전과 완전히 같습니다.
- 동시 실행 한계 실측(M2 Pro, 480×854, 12초, 다른 작업이 함께 도는 상태, GAMST 전달 비용 제외): 포즈2+YOLOE2 = 44/30 fps, 포즈4+YOLOE2 = 40/30, 포즈4+YOLOE4 = 34/16, 포즈8+YOLOE4 = 30/9. 같은 포즈2+YOLOE2가 앞선 측정에서는 43/39였습니다(부하 차이).
- `make_plan(['all'])`: 역할 26개, 환경 19개, RAM 95.6GB / VRAM 49.2GB / 디스크 128.9GB(대부분 추정치). 이 Mac은 RAM 32GB, 디스크 여유 10.5GB. 3개 역할은 CUDA 전용. 전부 동시 실행은 불가능합니다.
- 격리의 실제 범위: venv = 패키지, 별도 프로세스 = 메모리와 크래시. CPU/GPU/RAM 사용량 제한과 보안 격리는 아닙니다. 사용량 제한은 Linux cgroup만 지원합니다(`crew/host_limits.py`).
- 네트워크 실측: 720p JPEG q85 30fps = 19~22 Mbps(인코딩 약 2ms), 360p = 7~8.5 Mbps. 16kHz 16bit 모노 = 0.256 Mbps(계산값).
- 제안(G, F 다음): (1) 모듈별 실시간 목표(초당 횟수, p95 지연, 프레임을 버려도 되는지) (2) 이 컴퓨터에서 재는 `gamst bench` (3) 실행 전 예산 검사와 거부, 우선순위와 자동 낮춤 (4) 묶음(pack): local / speech / person / semantic, 묶음 단위로 서버 배치 (5) 네트워크 worker(TLS, 인증, 캡처 시각 유지, 끊김 처리). 서버 비용은 시작 전에 보고합니다.
- **실제 Pipeline 실측에서 찾은 결함**(M2 Pro, 480×854, 30fps 카메라를 흉내 낸 입력 10초, 다른 작업이 함께 도는 상태):
  - 고침: Pipeline 프레임에 `coordinate_frame='image'`가 없어 YOLOE worker가 모든 프레임을 거부했습니다.
  - 레시피 모듈은 semantic을 빼면 queue가 없어 **차례로** 돕니다. 포즈+YOLOE에서 카메라 300프레임 중 150개, 2+2에서 226개를 읽지 못했고, 이 손실은 어디에도 세지 않습니다.
  - queue=1로 나누면 카메라는 놓치지 않지만 포즈가 2~3/s로 무너집니다. `landmarker_video.py`가 sequence가 하나만 건너뛰어도 검출기를 다시 만들기 때문입니다. 그 조건만 빼고 시험하니 포즈 29/s + YOLOE 23/s(2+2는 25/s씩, 14/s씩)였습니다. 시험 후 되돌렸습니다.
  - `MicrophoneInput`은 오디오 유실(overflow) 표시를 무시합니다.
- G: `feat/realtime-pipeline`(wt-realtime)에서 위 결함을 고칩니다. 새 계층은 만들지 않습니다.
- F(622192a, 동작 지정 두 방식)를 합쳤습니다. 테스트 1405개, 오류는 원래 있던 15개입니다. 정확도는 측정하지 않았습니다(합성 데이터와 가짜 VLM). MCP에서의 확인은 `actor`를 스스로 밝히는 방식이라 에이전트가 user라고 속이면 구분하지 못합니다(Watch와 같음).
- 남은 것: G(`feat/realtime-pipeline`) 합치기 → 설치·다운로드·빌드·실모델 측정. owner 말: "묶음"은 실시간으로 돌리려는 **무거운 모델들의 묶음**입니다.

## 10. 무손실과 시각 정렬 (2026-11-25)

- owner 확정: 프레임을 버리지 않습니다. 모델마다 속도가 달라도 캡처 시각으로 맞춰 LLM/VLM에 줍니다. "묶음"은 병렬로 돌려야 하는 무거운 모델들을 서버에서 돌리고 로컬과 맞추는 패키지입니다.
- 확인: 처리 시간이 5/60/500ms인 가짜 모듈 세 개를 Pipeline에 넣으면 기록 45개 모두 캡처 시각이 유지됩니다. `get_context`는 모듈마다 **최신 값 하나**만 줍니다. "시각 T에 모든 모듈이 무엇이었나, T까지 다 처리됐나"를 주는 조회는 없습니다.
- 실측(YOLOE MPS, 별도 프로세스, GAMST 밖, 각 12초): 1개 43 fps, 2개 합 74, 3개 합 95. 상태 없는 모델은 복제해서 프레임을 나눠 처리할 수 있습니다.
- G에 추가: 무손실 기본, 상태 없는 모듈 복제, 밀린 시간 표시, 모듈별 "여기까지 완료" 시각. H(`feat/aligned-timeline`, wt-timeline): 정렬 조회.
- 그다음 I: 무거운 모델 묶음의 서버 실행(구현만, 배포와 비용은 보고 후).

## 11. 과부하에서도 보기, 과거 상태 고치기 (2026-11-25)

- owner 요구(전체 GAMST에 해당): 계산이 밀린다는 이유로 "못 봄"을 남기지 않습니다. context는 비동기로 갱신되는 상태이고, 과거 상태도 나중에 고쳐집니다.
- 실측(M2 Pro, 720p 영상, 별도 프로세스, 스레드 1개): 움직임 격자 0.75~1.08ms, 160×90 광류 0.83~0.89ms, JPEG 보관 1.8ms(중앙값). 포즈8+YOLOE4를 함께 띄운 상태(부하 평균 21.8)에서도 같은 수준이고 p95는 최대 5.1 / 3.2 / 16.5ms입니다. 보관 크기는 30fps에 분당 145MB입니다. GAMST 안에서는 재지 않았습니다.
- 설계(G, H 다음에 구현, J): (1) 바닥층: 모든 프레임에서 움직임만 계산 (2) 단계 낮추기: 해상도, 짝수/홀수 프레임. 단, 목표 동작의 샘플링 한계(`check`) 아래로는 내리지 않음 (3) 남은 프레임은 보관했다가 여유가 생기면 정식 처리 → provisional을 final로 교체 (4) 사람이 보는 결과에는 어느 수준으로 봤는지 표시.
- 지킬 수 없는 경우(장치가 주지 않은 프레임, 프로세스 종료, 정전, 보관 공간 초과)는 사실대로 남깁니다. 보지 않은 것을 봤다고 기록하지 않습니다.
- 프레임을 메모리에 잠시 두는 것은 현재 정책(처리하는 동안만)에 맞습니다. 디스크 보관은 owner 결정이 필요합니다.

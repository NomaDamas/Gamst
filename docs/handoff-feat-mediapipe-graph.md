# HANDOFF feat/mediapipe-graph (서브에이전트 C: MediaPipe Framework 흡수)

워크트리 `.local-run/wt-mediapipe`, 브랜치 `feat/mediapipe-graph`(07ff17d에서 분기, 통합 커밋 5624cb1을 merge함).
push와 PR은 하지 않았습니다. 설계와 사용법은 [mediapipe-graph.md](mediapipe-graph.md)에 있습니다.

## 상태

- **C++은 컴파일도 실행도 하지 않았습니다.** 소스, BUILD, 그래프, 빌드 스크립트만 있습니다. 그래프의 정확도와 속도는 확인 안 함입니다.
- 테스트한 것은 Python 쪽입니다. 상대는 실제 runner가 아니라 같은 프로토콜을 말하는 가짜 runner(`tests/fake_mediapipe_runner.py`)입니다.
- 고정한 리비전의 소스 파일 일부(API 헤더, 계산기, BUILD)를 `gh api`로 읽어 이름과 시그니처를 대조했습니다. 컴파일을 대신하지는 못합니다.

## 만든 것

- `gamst/mediapipe_graph/`
  - `module.py`: `MediaPipeGraphModule`. 프로세스 수명, 사이드 패킷, 브랜치 스위치, 타임스탬프 대응, 레코드 변환.
  - `runner/`: `gamst_graph_runner.cc`, `gamst_frame_calculator.cc`, `gamst_frame.h`, `BUILD`.
  - `graphs/holistic_cpu.pbtxt`, `holistic_gpu.pbtxt`.
  - `build.sh`, `pin.json`(v0.10.35, f8ef212d), `WORKSPACE.snippet`.
- 공용 코드 변경
  - `gamst/pipeline.py`: 레코드 링크가 레코드 자신의 캡처 시각을 씁니다. `configure()`와 모듈 `state`를 추가했습니다.
  - `gamst/mcp.py`: `configure_pipeline_module`에 `options`(선택)를 추가했습니다. owner가 선언한 브랜치와 숫자/불리언 사이드 패킷만 바꿀 수 있습니다.
  - `gamst/process_provider.py`: 프로세스 종료 코드를 `stop_child`로 뺐습니다(동작 같음, 재사용용).
  - `gamst/crew/registry.py`: 역할 `vision.mediapipe_graph`. `crew/modules.py`와 `profile.py`: 자기 프로세스를 가진 in-process 모듈도 레인(`queue`)에 넣습니다.
  - `gamst/recipes`: 부품 `mediapipe-graph`, 모듈 항목의 `queue`.
- 큐는 Pipeline 레인 하나만 씁니다. 모듈 안에 두 번째 큐는 없고, runner 안 프레임 수를 `max_in_flight`로 제한합니다.

## 테스트 (2026-11-25, `PYTHONPATH=. ~/.cache/gamst/demo/venv/bin/python`)

- `tests.test_mediapipe_graph`: 26개 통과. 핵심 줄 5곳을 일부러 망가뜨리면 해당 테스트가 실패하는 것도 확인했습니다.
- 전체: 1269개 실행, 오류 15개(원래 있던 stereo 13, sam_audio_native 2), skip 38.
- `build.sh --check`: 여기서는 Bazel과 OpenCV가 없다고 정확히 멈춥니다. 실제 빌드 단계는 실행 안 함.

## 다음 단계 (owner 디스크 결정 후)

1. `GAMST_MEDIAPIPE_BUILD_DIR`을 정하고 `brew install bazelisk opencv`. OpenCV가 4.x인지 확인합니다(스크립트는 `include/opencv4`를 요구).
2. `build.sh --check` → `build.sh`. 처음 컴파일이므로 C++/BUILD 오류가 나올 수 있습니다.
3. `MediaPipeGraphModule.from_manifest(...)`로 실제 영상 실행. 먼저 볼 것:
   - 사람이 없는 프레임과 FlowLimiter가 버린 프레임에서도 `done`이 나오는지(`SetProcessTimestampBounds` 동작)
   - `FINISHED` back edge로 두 번째 프레임부터 막히지 않는지
   - 브랜치를 끈 동안 그 브랜치 추론이 정말 없는지, RESET 후 재시작이 되는지
   - BGR→RGB 방향, 손/얼굴 스무딩 값
4. 실제 runner 테스트를 추가합니다(manifest가 없으면 skip). Python `MediaPipePosePlugin`과 같은 영상으로 지연과 좌표를 비교합니다.

## 열린 항목

- owner 결정: 빌드 위치(외장 디스크 / 공간 정리 / GPU 서버).
- MediaPipe 버전: 0.10.35(현재 Python과 같음)로 고정했습니다. 최신은 1.2.3입니다.
- 모델 파일 라이선스(모델 카드): 확인 안 함.
- 레시피 JSON은 바꾸지 않았습니다. `person-action-count`는 아직 `mediapipe-pose`를 씁니다.
- 얼굴 랜드마크는 프레임마다 468~478점을 Store에 씁니다. 필요 없으면 `face` 브랜치를 끄세요.

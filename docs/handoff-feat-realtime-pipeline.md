# HANDOFF feat/realtime-pipeline (2026-11-25)

Pipeline의 실시간 결함을 고쳤습니다. owner 확정("프레임을 버리지 않는다")에 따라 기본이 **무손실**입니다.
통합 브랜치 139934c(feat/movement-spec 포함)까지 합친 상태입니다. push와 PR은 하지 않았습니다.

## 1. 바뀐 규칙 (`gamst/pipeline.py` 한 곳)

- **무손실이 기본.** 모든 모듈이 모든 프레임을 한 번씩, 캡처 순서대로 처리합니다. 느리면 밀립니다(backlog).
  - 자기 프로세스를 가진 모듈(`owns_process = True`)은 `queue='auto'`로 자기 lane을 받습니다(레시피, `ProfileRuntime`). 수학 모듈은 lane이 없습니다.
  - lane을 가진 모듈을 읽는 수학 모듈은 **그 lane에 합류**합니다. pose → target_person → pose_channels → exemplar가 프레임마다 한 번에, 캡처 순서대로 돕니다. `TargetPerson`은 같은 캡처의 카메라 프레임과 관절을 짝지으므로 일부러 pose lane에서 돕니다.
- **표본 추출은 선언한 것만.** `every_ms`가 있는 모듈만 건너뛸 수 있고, `sampled: {every_ms, skipped}`로 보고합니다(유실이 아님). `yoloe` 부품의 기본 100ms 표본 추출은 없앴습니다.
- **한계.** lane당 900프레임, Pipeline 전체 512MB. 같은 프레임을 여러 모듈이 읽으면 배열을 공유하고 한 번만 셉니다. 카메라 입력은 별도로 최대 256MB를 기다리게 합니다.
- **넘칠 때.** 절반을 넘으면 레시피가 `optional`로 선언하고 필수 모듈이 쓰지 않는 모듈을 멈추고(`paused`), 밀린 것이 없어지면 다시 켭니다. 그래도 넘치면 `Pipeline._overflow(lane, bound)` **한 곳**에서 결정합니다. 기본 동작: 가장 오래된 10%를 `backlog` 이유로 선언합니다. 과부하 단계 낮추기(J)는 이 함수를 바꾸면 됩니다.
- **빠진 구간 선언.** 이유를 구분합니다.
  - `input`, `capture_paused`: 데이터가 GAMST에 오지 않음
  - `error`: 모듈이 그 프레임에서 실패
  - `backlog`, `paused`, `disabled`: 캡처했지만 그 모듈이 처리하지 않음
  - `pipeline.gaps` 레코드, `status()`의 `gap_intervals`, 걸친 이벤트의 `gap_frames`, `count_events`의 `complete`/`lost`, `recipe run --report`의 `complete`/`lost`/`errors`, `--truth` 점수의 `complete`, `find` 요약, `export`의 `pipeline.gap` 행에 나옵니다.
- **오디오**는 빠진 구간을 이어 붙이지 않습니다. 파생 입력(voice → ASR)까지 reset이 전달됩니다. 빠진 구간 직전에 모듈이 들고 있던 짧은 오디오(ASR 3초 창의 일부)도 버려집니다.
- **녹화 입력**(`live = False`: 영상 파일, take, WAV)은 lane이 끝날 때까지 기다립니다. 아무것도 버리지 않습니다. 필수 모듈이 실패하면 전처럼 실행이 멈추고, optional 모듈의 실패는 기록만 합니다.
- **복제**(`replicas`): 상태 없는 모듈만. `open_worker(..., replicas=N)`, 레시피 항목 `"replicas": N`(yoloe). 프레임당 한 번, 결과는 캡처 순서로 커밋합니다. `spec.stateful`이면 거부합니다.
  - YOLOE의 계약은 `stateful=True`였지만 실제 상태는 입력 순서 검사뿐이라, 설정 `stateless: true`를 추가했습니다(계약 버전 `.stateless`, 기본은 그대로). 임베딩 모델은 상태 있는 모듈 안에서 차례로 불리므로 복제하지 않았습니다.
- **watermark**: `status()['modules'][이름]['complete_until_ms']`. 그 시각까지의 모든 것이 처리됐거나 빠짐으로 선언됐을 때만 앞으로 갑니다(앞 모듈, 대기 프레임, 처리 중인 프레임, 꺼진 구간, 모듈이 들고 있는 `pending_since_ms` 반영). 창 단위 모듈(예: worker 안의 Whisper 3초 창)은 그 창만큼 이전 시각의 결과가 나중에 더 나올 수 있습니다.
- 그 밖의 상태: `backlog`, `backlog_ms`, `max_backlog_ms`, `latency_ms`·`capture_to_result_ms`의 p50/p95(최근 512개), `newest_result_age_ms`, 입력의 `missed`. 나이 값은 실시간 입력에만 있습니다.

## 2. 입력과 landmarker

- `CameraInput`: 스레드가 드라이버가 주는 모든 프레임을 읽어 기다리게 합니다. 한계를 넘은 것만 `missed`로 셉니다. **드라이버가 준 적 없는 프레임은 셀 수 없습니다.**
- `MicrophoneInput`: 샘플 시계로 시각을 매깁니다(도착 시각에서 장치 버퍼와 보고된 지연을 뺀 값에서 시작). overflow 표시와 250ms 넘는 시계 차이를 빠진 구간으로 선언하고 시계를 다시 시작합니다. 실제 마이크로는 확인하지 않았습니다.
- `mediapipe.*.video`의 `allow_skips`(기본 꺼짐, 켜면 버전 `.skips`): `max_gap_ms` 안에서 건너뛴 프레임은 검출기를 다시 만들지 않습니다. 이유는 `docs/landmarker-video.md`. `mediapipe-pose` 레시피 부품에서 켭니다.

## 3. 실측 (M2 Pro, 480×854, 각 1회, 다른 작업이 함께 도는 상태라 대략적인 값)

같은 스크립트(`pipe.py`, 10초, 30fps, worker 설정 그대로):

| 설정 | 전 (부하 8.1→10.1) | 후 (부하 18.8→9.3) |
|---|---|---|
| 차례로, 포즈+YOLOE | 못 읽음 158/300, 14/s, 14/s | 148/300, 15/s, 15/s |
| lane(queue=1), 포즈+YOLOE | 0, 포즈 2/s, YOLOE 23/s | 0, 포즈 2/s, YOLOE 27/s |
| 차례로, 2+2 | 234/300, 각 7/s | 225/300, 각 8/s |
| lane(queue=1), 2+2 | 0, 포즈 2/s, YOLOE 18/s | 0, 포즈 3/s, YOLOE 21/s |

이 스크립트는 `queue`를 직접 지정하고 `allow_skips`를 켜지 않으므로 개선되지 않는 것이 맞습니다(기본 계약 그대로). 달라진 것은 유실이 빠진 구간으로 선언된다는 점입니다.

레시피가 쓰는 설정(30초, 30fps, 900프레임. `CameraInput`에 30fps 장치를 흉내 낸 것을 연결, `allow_skips`, `queue='auto'`):

| | 포즈 | YOLOE | 최대 밀림(포즈 / YOLOE) | 최대 메모리 | 끝난 시각 |
|---|---|---|---|---|---|
| 기본 엔진, YOLOE 1 (부하 7.1) | 900/900 | 900/900 | 6.7s / 32.6s | 439MB | 62.5s |
| 기본 엔진, YOLOE 2 (부하 7) | 900/900 | 900/900 | 12.1s / 19.8s | 267MB | 49.7s |
| `sensor_inference_interval_ms=1000`, YOLOE 1 (부하 8.6) | 900/900 | 900/900 | 0.48s / 11.2s | 327MB | 41.2s |
| `sensor_inference_interval_ms=1000`, YOLOE 2 (부하 10.2) | 900/900 | 900/900 | 0.78s / 2.1s | 77MB | 31.1s |

- 네 경우 모두 장치 유실 0, 처리 안 한 프레임 0, 오류 0입니다.
- **찾은 결함:** 엔진이 레코드마다 창 전체를 SQL로 다시 집계합니다. 레코드 하나에 처음 600개는 3.1ms, 다음 600개는 8.0ms, 그다음 600개는 19.5ms였습니다. 기존 옵션 `sensor_inference_interval_ms`(기본 0)를 `gamst recipe run`에서 1000으로 켰습니다. 레코드는 모두 커밋되고, 집계만 초당 한 번입니다. **MCP와 ProfileRuntime의 기본값은 그대로 0이라 같은 한계가 남아 있습니다.**
- 카메라를 스레드 없이 흉내 낸 첫 판에서는 이벤트 루프가 늦어 못 읽은 프레임이 있었습니다(YOLOE 1: 2~21, YOLOE 2: 64~231). 그래서 장치 읽기를 스레드로 옮긴 위 값이 실제 경로입니다.

## 4. 테스트

- 전체 1431개, 오류는 원래 있던 15개뿐입니다(stereo 13, sam_audio_native 2). 통합 브랜치는 1405개였습니다.
- 새 테스트 26개: `tests/test_realtime_pipeline.py`(23), `tests/test_landmarker_video.py`(3). 이 중 2개는 **실제 worker**(YOLOE 복제 2개, MediaPipe 포즈 건너뛰기)를 Pipeline으로 돌립니다. 가중치가 없으면 skip하고, 이 Mac에서는 실행됐습니다.
- `tests/test_mediapipe_graph.py`: `queue == 1` 확인을 lane 이름 확인으로 바꿨습니다(무손실 기본).

## 5. 남은 한계

- 이 Mac의 부하 상태에서 포즈 + YOLOE 1개는 YOLOE가 실시간을 따라가지 못합니다(30초 뒤 8.8초 밀림, 이후 모두 처리). 처리량이 계속 모자라면 결국 한계에 닿아 `backlog`로 선언됩니다. 단계 낮추기는 J의 일입니다.
- 1080p 프레임(6.2MB)은 512MB에 82장, 약 2.7초 분량뿐입니다.
- 커밋(SQLite, 검증)은 주 프로세스의 이벤트 루프 하나에서 돕니다. 모듈이 많으면 여기가 병목입니다.
- `q`로 끝내면 밀린 프레임을 다 처리한 뒤 끝납니다.
- 호스트가 직접 넣는 입력(`push_video`/`push_audio`, MCP `start_capture`)의 장치 유실은 세지 않습니다. iPhone의 sequence 누락도 세지 않습니다.
- `rotation`처럼 자기 레코드를 직접 커밋하는 모듈은 이벤트에 `gap_frames`가 붙지 않습니다(개수의 `complete`와 `pipeline.gap` 행에는 나옵니다). `newest_result_age_ms`도 없습니다.
- `exemplar` 시퀀스는 1초 이하로 빠진 구간을 건너 이어질 수 있습니다(`gap_frames`로 표시됩니다). 1초를 넘으면 전처럼 끊습니다.
- `pipeline.reset()`(엔진 일시정지)은 처리 중인 worker 요청을 취소합니다. 취소된 worker는 닫힙니다(전부터 있던 동작, 고치지 않음).
- 실제 카메라, 실제 마이크, 정확도는 확인하지 않았습니다.

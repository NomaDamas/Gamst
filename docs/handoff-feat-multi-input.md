# HANDOFF feat/multi-input (서브에이전트 K)

워크트리 `.local-run/wt-multi-input`, `6804bef`에서 분기하고 `01e3222`(결정 문서)를 합쳤습니다. push와 PR은 하지 않았습니다.

## 1. 상태

- 요구 (1)~(8)을 구현했습니다. **정확도, 실시간 부하, 실제 VLM은 아무것도 측정하지 않았습니다.**
- 실제 모델로 함께 돌려 본 조합: 포즈(MediaPipe full) + YOLOE-26s + 운율 DSP를 역할 부품으로, 합성 영상 4초(96×64, 사람 없음)에서 한 번.
  오류 0, 모든 프레임 처리. 사람이 없는 영상이라 인식 결과는 확인하지 않았습니다. 처리 속도도 재지 않았습니다. 그 밖의 조합은 실제 모델로 돌린 적이 없습니다.
- 실제 카메라와 마이크는 열지 않았습니다(owner 사생활). 실시간 두 입력은 가짜 입력으로만 확인했습니다.
- 테스트: 아래 5절.

## 2. 무엇이 바뀌었나

| 파일 | 내용 |
|---|---|
| `gamst/inputs.py` | `RecordedClock`(녹화 입력들의 한 시계), `media_streams`(ffprobe), `SoundTrackInput`(ffmpeg 파이프, 100ms씩, 디스크에 안 씀), `SensorFileInput`(CSV/JSONL), `AudioFileInput`의 구간 읽기 |
| `gamst/pipeline.py` | 녹화 입력을 캡처 시각 순서로 읽음(입력마다 한 장만 들고 있음), 실시간+녹화 혼합 거부, 입력별 `offset_ms`, 입력 실패 기록(`inputs.X.error`), 먼저 끝난 입력이 다른 입력의 완료 시각을 막지 않음, 말 없는 입력이 있어도 멈춤 |
| `gamst/recipes/__init__.py` | 입력 여러 개(`open_inputs`, `--input`, `--offset-ms`), 부품 묶음(`parts`), `--with-role`, 빠진 등록을 한꺼번에 알리고 빼는 방법을 말함, `roles` 명령 |
| `gamst/crew/{registry,modules}.py` | `Role.modality`, `RoleCandidate.adapter`(일반 부품이 될 수 없는 이유), `refusal`, `role_report` |
| `gamst/providers/streaming_semantic.py`, `provider_worker.py`, `recipes/perception_parts.py` | VLM에 전사·세기를 글 자료로. 새 클라이언트 없음, 기존 `process_regions` 통로 사용(응답이 없는 프레임을 허용하도록 한 줄 바꿈) |
| `gamst/recipes/terms.py`, `gamst/movement_proposals.py`, `gamst/mcp.py` | 단어를 예시로(전사 → 제안 → owner 확인·수정 → 단어 파일) |
| `gamst/recipes/scene.json`, `person-term-count.json`, `person-voice-style-count.json` | 종합 레시피, `--without voice`, `--video`(영상의 소리) |
| `gamst/providers/movement_match.py` | 채널이 4개보다 적은 센서도 비교(그 전에는 4개 미만이면 항상 "모름") |

고친 기존 결함:
- `--camera 0`이 "지정 안 함"으로 처리되던 것(`0 == False`).
- `person-action-count`의 `records`가 `temporal.repetitions`(출력 이름)라서 반복 횟수가 보고서에 **항상 0**이던 것 → `temporal.repetition_count`.
- Pipeline 음성 프레임에 채널 이름이 없어 diarization worker가 모든 프레임을 거부하던 것 → `('mono',)`(실제 모델로는 확인 안 함).
- hand/face 등 MediaPipe 역할이 프레임을 건너뛸 때마다 검출기를 다시 만들던 것 → 레지스트리 설정에 `allow_skips`(이미 설치한 프로필에는 다시 setup 해야 들어감).

## 3. 명령

```bash
gamst recipe run scene --video film.mp4 --exemplar move.json --terms 대박 --without person --without voice
gamst recipe run scene --camera 0 --microphone --without movement --without voice --terms 대박 --offset-ms camera=-80
gamst recipe run scene ... --with semantic [--vlm-context none|transcript|counts]
gamst recipe run person-action-count --video a.mp4 --exemplar move.json --with-role vision.hand --with-role vision.depth@500
gamst recipe roles [--json]
gamst recipe enroll person-term-count --kind terms --video a.mp4@3.1-3.8 --audio b.wav@0-1 --proposals p.json
gamst recipe proposals confirm ID=대박,대애박 --proposals p.json
gamst recipe enroll person-term-count --kind terms --proposals p.json --out words.json
gamst recipe run person-term-count --microphone --without voice --terms words.json
gamst recipe run my-imu.json --sensor file.csv
```

- 영상+소리: 두 입력이 같은 파일이면 시계가 하나이고, 컨테이너의 시작 시각(ffprobe `start_time`)만큼 어긋나게 놓습니다. ffmpeg가 없으면 무엇이 없는지 말하고 멈춥니다.
- ffmpeg에는 로컬 파일만 줍니다(`file:` 경로, `-protocol_whitelist file`, 인자 목록, 셸 없음).
- 서로 다른 녹화 파일은 `--offset-ms ID=MS`를 말해야 합니다. 어떻게 맞는지 추측하지 않습니다.
- 여러 입력에서 보고서의 `frames`와 `--frames`는 영상(첫 입력) 기준이고, 입력별 수는 `input_frames`에 있습니다.
- 한 입력이 실패하면 전체 실행이 멈추고 그 입력 이름을 남깁니다(기존 규칙: "아무것도 세지 않으면서 계속하지 않는다").

## 4. 역할 상태 (`gamst recipe roles`)

"사용 가능"은 설치하면 일반 부품으로 돌 수 있다는 뜻입니다. **실행해 봤다는 뜻이 아닙니다**(무엇을 확인했는지는 `--json`의 `verified`).

| 상태 | 역할 |
|---|---|
| 사용 가능 | vision.detect(rfdetr 2종, yoloe 2종, custom), vision.track, vision.rotation, vision.pose(mediapipe 3종), vision.hand, vision.face, vision.depth(video_depth.large, CUDA), audio.vad, audio.events(yamnet), audio.diarization, audio.prosody, physio.rppg |
| 손으로 준비 | vision.mediapipe_graph(C++ 빌드), vision.depth(depth_anything_v2.small, 파일 해시 없음), audio.events(ced.base), audio.vocal_expression |
| 거절: 등록된 사람·동의가 필요한 비공개 모델 | vision.pose(sapiens2), audio.speaker_verify, audio.target_speech, audio.word_alignment, vision.face_identity, vision.body_identity (전용 부품은 있음) |
| 거절: 전용 부품을 쓸 것 | audio.asr(whisper-words, qwen-words), vision.semantic_scene(semantic-scene) |
| 거절: 입력이 다름 | vision.video_segmentation(정확한 프레임의 상자), vision.physical_concepts(물체 조각 사진과 물을 개념), audio.asr의 moonshine(worker 없음) |
| 거절: 모델이 아님 | input.video, input.microphone, view.hud |

이 브랜치의 역할 부품으로 실제로 실행해 본 것: vision.pose(full), vision.detect(yoloe.26s), vision.track(OpenCV), audio.prosody(DSP). 나머지는 실행한 적이 없습니다.
설치 프로필이 없으면 `gamst recipe setup RECIPE --with-role ROLE`을 하라고 말하고 멈춥니다.

## 5. 테스트

`PYTHONPATH=. ~/.cache/gamst/demo/venv/bin/python -m unittest discover -s tests -q`: 1530개, 오류는 원래 있던 15개(stereo 13, sam_audio_native 2)뿐입니다.

- `tests/test_multi_input.py`: 합성 영상(ffmpeg, 없으면 skip)에서 번쩍임과 딸깍이 레코드와 `get_timeline`에서 한 프레임 안, 시작 시각 차이(mp4/mkv), 소리 구멍, offset, 앞서가지 않음, 끝, gap, 일시정지, 한 입력 실패, 멈춤, 센서.
- `tests/test_scene_recipe.py`: 역할 부품(가중치 없는 실제 worker 2개), VLM 문맥(실제 worker 프로세스와 가짜 서버), 주입 문구, 단어 예시, 인물 없이 세기, `scene` 전체.
- 주입 테스트는 **요청 구조만** 확인합니다(전사는 `untrusted_data` 안, 지시문과 분리). 실제 VLM이 그 말을 따르는지는 확인 안 함.

## 6. 한계

1. 카메라 자체의 지연은 모릅니다. 재지 않았고, 잰 값을 `--offset-ms`로 넣을 수만 있습니다. 두 입력 모두 컴퓨터 시계(벽시계 기준)입니다.
2. 여러 실제 모델을 한 컴퓨터에서 함께 돌릴 때의 부하는 재지 않았습니다.
3. VLM에 주는 것은 구간의 마지막 프레임을 보낼 때 저장소에 있는 것까지입니다. 3초 창 ASR이면 구간의 마지막 말은 빠지기 쉽습니다(기다리지 않음). 입력이 끝날 때 열린 구간은 마지막 프레임 때 알던 것을 씁니다.
4. `counts`는 값에 정수 `count`가 있는 모듈만 줍니다(rotation은 사건만). 10분보다 오래된 마지막 값은 주지 않습니다.
5. `gamst recipe run`은 MCP 서버를 같은 프로세스에 띄우지 않습니다(전부터 그랬음). 부품을 실행 중에 켜고 끄는 MCP 도구는 SDK로 Pipeline과 MCP를 함께 띄운 앱에서만 닿습니다. `--without`으로 뺀 부품은 실행 중에 켤 수 없습니다.
6. 센서: kinematics는 미터 단위 물체 위치 스냅샷과 보정 정보가 필요해 IMU 표본을 받지 못합니다(억지로 넣지 않음). 레지스트리에는 imu 역할이 없습니다. BNO055, 레이더, CSI처럼 원시 바이트를 받는 provider는 이 입력과 맞지 않아 SDK의 SensorStreams로만 씁니다. 초당 1000개까지, 레코드는 만들지 않고 모듈 입력으로만 갑니다.
7. 센서 예시 매칭은 `scale`로 채널을 1 안팎으로 맞춰야 기본 임계값이 뜻이 있습니다. 한 주기가 이어지는 완만한 사인파에서는 기존 DTW가 짝을 못 찾았습니다(동작 사이에 쉼이 있는 신호로 시험).
8. 단어 예시는 클립 전체 전사를 단어로 제안합니다. 공통 부분을 찾지 않으므로 `@시작-끝`으로 자르거나 확인할 때 고쳐야 합니다. 클립마다 ASR을 다시 엽니다.
9. 소리가 중간에 없는 구간은 무음으로 채워집니다(위치는 유지). 원본이 조용한 것과 구별되지 않습니다.
10. 먼저 끝난 입력의 뒤쪽 시간은 `none`(처리했고 아무 일 없음)으로 보입니다. 입력이 끝났다는 것은 `status()['inputs'][id]['ended']`에만 있습니다.

## 7. owner 결정 필요

1. 한 입력이나 필수 부품이 실패하면 `scene` 전체가 멈춥니다. 나머지만 계속할지.
2. 호스팅 VLM에도 전사와 세기가 기본으로 나갑니다(2026-11-25 결정대로). 실행할 때 안내문만 냅니다. 별도 동의를 받게 할지.
3. 전사는 등록된 사람의 것만이 아니라 그 레시피의 ASR이 들은 것입니다(`--without voice`면 모두). 다른 사람의 말도 VLM에 갑니다.
4. MCP로 실행 중인 레시피에 닿게 할지(`gamst recipe run --mcp` 같은 것, 6절 5번).
5. 실제 카메라+마이크의 어긋남 측정(박수 영상)과 실제 모델 동시 실행 부하 측정(GCP).

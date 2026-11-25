# HANDOFF feat/runtime-instructions (서브에이전트 L)

워크트리 `.local-run/wt-runtime`, `2d932d7`에서 분기. push와 PR은 하지 않았습니다. 요구 1~7을 구현했습니다.
**실제 모델, 실제 카메라·마이크, 실제 VLM으로는 아무것도 돌리지 않았습니다.** 정확도도 재지 않았습니다.

## 1. 무엇이 바뀌었나

| 파일 | 내용 |
|---|---|
| `gamst/pipeline.py` | `set_reference` / `remove_reference`(말·동작만, 적용 시각과 revision), 실행 중 모듈 추가, 실패해도 계속(`failures`, 다시 열기, 포기), 느린 줄 경고(`warnings`) |
| `gamst/recipes/instructions.py` (새 파일) | 실행 중 지시: 출처 확인, 폴더 제한, 별도 스레드 등록, 빠진 부품 추가, 목록, 남은 기록에서 찾기 |
| `gamst/providers/pose_exemplar.py`, `term_events.py` | `prepare` / `schedule` / `offline`. 이벤트와 값에 `reference_id`, `reference_revision` (규격 버전 3, 2) |
| `gamst/counting.py`, `engine.py`, `timeline.py` | 옛 지시와 새 지시를 더하지 않음, `instructions`, `not_instructed`, SDK 호출 4개 |
| `gamst/mcp.py` | 도구 4개 |
| `gamst/redact.py` (새 파일), `process_provider.py`, `provider_worker.py` | worker가 시작하지 못한 이유(300자, 비밀값 가림) |
| `gamst/visualization.py`, `plugins.py`, `providers/pose.py`, `mediapipe_graph/module.py`, `capabilities.py` | 좌표를 기록에서 뺌, 화면용 임시 자리(`Drawing`) |
| `gamst/extensions.py`, `inputs.py`(`PluginInput`), `recipes/parts.py` | `"use": "plugin"` 모듈과 입력 |
| `gamst/recipes/perception_parts.py`, `providers/streaming_semantic.py` | hosted VLM에 글 보내기는 따로 동의 |
| `gamst/recipes/__init__.py` | `--clips`, `--proposals`, `--plugin`, `--vlm-remote-context`, `--stop-on-failure`, 보이기만 하는 카메라, 실패·경고 보고 |

## 2. MCP 도구와 호출 예

```bash
gamst recipe run scene --camera 0 --microphone --without movement --without voice --terms 대박 \
    --serve --clips ~/my-clips --proposals p.json --headless
```

| 도구 | 예 | 답 |
|---|---|---|
| `list_instructions` | `{}` | 지금 지시, 언제부터, 누가, 빠진 부품과 필요한 입력, 쓸 수 있는 파일 |
| `set_instruction` | `{"kind":"terms","actor":"jev","terms":["진짜","헐"]}` | `effective_from_ms`, `reference_id`, `reference_revision`, `confirmed` |
| | `{"kind":"terms","actor":"jev","terms":{"진짜":["진짜","찐"]}}` / `"file":"words.json"` / `"proposals":["ID"]` | |
| | `{"kind":"exemplar","actor":"jev","text":"팔을 든다","mode":"sequence","clips":[{"file":"a.mp4","start_s":2,"end_s":3.5},{"file":"b.mp4"}],"negatives":[{"file":"c.mp4"}]}` | 부품이 없으면 `added_modules`, `not_limited_to` |
| | `{"kind":"exemplar","actor":"jev","text":"손목이 어깨 위","rule":{"steps":[{"when":{"measure":{"height":["right_wrist","right_shoulder"]},"op":"gt","value":0.5},"hold_ms":200}]}}` | |
| `remove_instruction` | `{"reference":"exemplar","actor":"user"}` | |
| `search_retained` | `{"reference":"terms","window":"10m"}` | `retrospective: true`, `in_live_counts: false` |

SDK는 같은 이름: `human.set_instruction('terms', terms=[...])`, `list_instructions`, `remove_instruction`, `search_retained`.
레시피 없이 Pipeline만 쓰면 `pipeline.set_reference(name, 'terms', value)`.

## 3. 규칙

- **언제부터**: 아직 받지 않은 첫 캡처 시각부터입니다. 그 전에 받은(밀려 있는 것 포함) 프레임은 옛 지시로 처리합니다. 단어는 말한 시각으로 나눕니다.
  단어 시각이 없는 발화가 바뀌는 시각에 걸치면 옛 단어와 새 단어를 각각 찾고 `across_change: true`를 붙입니다.
- **섞지 않음**: `count_events`는 가장 새 revision만 세고 나머지는 `other_instructions`에 적습니다. 걸쳐 있던 동작은 `cut: instruction_changed`로 끝납니다.
- **지시가 없던 구간**: 시각표에서 `off` / `not_instructed`이고, 누락이 아닙니다.
- **파일**: `--clips` 폴더 안의 실제 파일만(링크를 푼 뒤 확인). 주소, 바깥 경로, 재생 목록 파일(m3u8 등)은 거절합니다. 확인된 제안의 영상도 그 폴더 안에 있어야 합니다.
- **한도**: 단어 64개 × 표기 16개 × 100자, 예시·반례 각 16개, 한 개 30초 이하·파일 2GB 이하, 제안 ID 32개, 폴더 8개, 등록은 한 번에 하나(300초).
  LLM은 정규식(`re:`)을 줄 수 없습니다.
- **사람·목소리 등록은 이 길로 못 합니다.** owner가 시작할 때 준 등록 파일이 있으면 추가된 부품도 그 사람으로 한정하고, 없으면 `not_limited_to`로 말합니다.
- 누가 했는지(`actor`)는 **스스로 밝힌 값**입니다. `jev`가 준 단어·규칙·예시는 `confirmed: false`입니다.
- 장치를 여는 도구는 없습니다. 카메라가 안 열려 있으면 어떤 옵션으로 시작해야 하는지 말하고 거절합니다.

## 4. 실패해도 계속

- 실시간 입력: 0.5, 1, 2, 4, 8, 16초 뒤 여섯 번 다시 엽니다. 돌아오면 `resumed_ms`. 끊긴 동안은 `input_failed` 빈틈이고, 몇 장을 놓쳤는지는 모릅니다(1로 적음).
- 부품: 연속 25번 실패하면 그만두고(`failed`), 다시 켜면(`configure_pipeline_module`) 다시 시도합니다.
- 남은 것이 없으면 실행이 끝납니다. 한 장도 못 읽었으면 명령이 오류로 끝납니다. 시작할 때 부품이 안 뜨는 것은 예전처럼 시작하지 않습니다(이유를 말함).
- 카메라가 `read()`에 실패하면 예전에는 조용히 끝났습니다. 이제 실패입니다.

## 5. 테스트

`PYTHONPATH=. ~/.cache/gamst/demo/venv/bin/python -m unittest discover -s tests -q`: 1571개, 오류는 원래 있던 15개(stereo 13, sam_audio_native 2)뿐입니다. 두 번 돌렸습니다.

- `tests/test_runtime_instructions.py`: 진짜 레시피, 진짜 worker 프로세스(포즈, Whisper), 진짜 MCP 서버. 모델 객체만 가짜입니다.
- `tests/test_failures.py`: 진짜 `CameraInput`, `MicrophoneInput`, 포즈 플러그인, 운율 모듈에 가짜 장치·가짜 검출기. 시작 실패 메시지.
- `tests/test_recipe_plugins.py`: 사용자 모듈과 입력, hosted 동의, 보이기만 하는 카메라.
- `tests/test_visualization.py`: 진짜 얼굴·포즈 플러그인(가짜 검출기)에서 화면은 그려지고 Store와 MCP에는 좌표가 없음.
- 재 본 값(M2 Pro, 부하 평균 약 3, 한 번): 실행 중 예시 2개(25fps, 64×48, 각 1초) 등록과 포즈 부품 추가 동안(약 0.5초) 이벤트 루프를 5ms마다 깨웠을 때
  가장 늦은 깨어남 8.0ms, 카메라와 마이크가 놓친 프레임 0. 가짜 검출기라 실제 포즈 모델이 CPU를 나눠 쓰는 부담은 들어 있지 않습니다.

## 6. 한계

1. 예시 등록은 포즈 worker를 하나 더 띄웁니다. 같은 컴퓨터의 계산을 나눠 쓰므로 실제 모델에서는 그동안 줄이 밀릴 수 있습니다(재지 않음). 디코딩과 채널 계산은 같은 프로세스의 다른 스레드입니다.
2. 지시는 레시피에 적힌 reference 하나씩입니다. 동작 두 개를 동시에 세려면 레시피에 exemplar를 두 개 적어야 합니다.
3. 지시를 지워도 그 앞의 모델 worker는 계속 돕니다(부품 끄기는 `configure_pipeline_module`).
4. 실행 중 추가된 세기는 VLM에 주는 글(`counts`)에는 들어가지 않습니다.
5. `search_retained`는 남은 기록만 봅니다. 포즈 실행은 프레임마다 기록이 여러 개라 기본 5000개는 1분이 안 됩니다(`--max-records`). 관절을 쓰는 규칙은 관절이 저장되지 않아 찾을 수 없습니다.
6. 등록 스레드는 강제로 끝낼 수 없습니다. 300초가 지나면 지시를 바꾸지 않고, 그 스레드가 끝날 때까지 새 등록을 받지 않습니다.
7. 실시간 입력을 다시 여는 일(`cv2.VideoCapture`)은 이벤트 루프에서 합니다. 장치는 자기 스레드로 읽으므로 다른 입력이 잃지는 않지만 처리는 그만큼 늦습니다(재지 않음).
8. 비밀값 가리기는 환경 변수 이름(KEY, TOKEN 등)과 흔한 모양을 봅니다. 모든 비밀을 알아보지는 못합니다. 그래서 시작 메시지에만 쓰고, 실행 중 오류는 예전처럼 종류만 냅니다.
9. 플러그인은 격리하지 않습니다. 녹화형 사용자 입력은 다른 입력과 함께 못 돕니다.
10. 다른 프로그램이 `landmarks` 이름으로 기록을 넣으면 이제 거절됩니다(예전에는 저장). 이름이 다르거나 숫자만 있으면 걸러지지 않습니다.
11. 재생 목록 거절은 파일 첫 바이트(mp4/mov, mkv/webm, avi, wav)로 합니다.

## 7. 이미 저장된 파일

예전에 저장된 store 파일의 좌표는 건드리지 않았습니다. 그 파일을 열면 MCP `get_state` / `inspect_interval`에 그대로 나옵니다.
남은 자리 전체는 `docs/private-payloads.md`에 있습니다.

## 8. owner 결정이 필요한 것

1. `landmarks`를 이름으로 막은 것(외부 플러그인이 쓰고 있었다면 깨짐)이 괜찮은지. 지금은 막습니다.
2. `jev`(LLM)가 준 규칙·예시를 owner 확인 없이 바로 세되 `confirmed: false`로 표시하는 것이 괜찮은지. VLM이 찾은 후보는 여전히 owner 확인이 필요합니다.
3. 실패한 실시간 입력을 다시 여는 횟수와 간격(지금 6번, 합 31.5초), 부품 포기 기준(연속 25번).
4. 예전 store 파일의 좌표를 지우는 도구를 만들지.
5. 실제 확인: `try-it` 환경에서 `--serve --clips`로 한 번 돌려 보기(카메라와 마이크가 필요해서 하지 않았습니다).

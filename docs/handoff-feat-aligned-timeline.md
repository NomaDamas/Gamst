# HANDOFF feat/aligned-timeline (서브에이전트 H)

워크트리 `.local-run/wt-timeline`, `0331eb9`에서 분기, G가 합쳐진 통합 브랜치(`feat/target-speaker-keyword-and-pose-count`)를 합쳤습니다. push와 PR은 하지 않았습니다. 설명은 `docs/aligned-timeline.md`.

## 1. 상태

- `get_timeline`(SDK, MCP)을 구현했습니다. 캡처 시각 행, 상태 8가지, 완료 시각, 두 시각(캡처/알게 된 시각), `as_of_ms`, 수정·철회, 변경분 커서.
- 실제 모델, 네트워크, 다른 컴퓨터의 시계는 시험하지 않았습니다. 가짜 모듈과 합성 저장소로만 확인했습니다.
- 새 테스트 26개(실제 Pipeline 포함). 전체 결과는 6절.

| 파일 | 내용 |
|---|---|
| `gamst/timeline.py` | 행 계산, Pipeline 어댑터 `progress()`, 완료 기록 `History`, 커서 |
| `gamst/revisions.py` | 어떤 기록이 현재 값인지: `supersedes`/`revision`, 철회, 의존 기록, 구간 |
| `gamst/storage.py` | `record_offsets.known_ms`, 검색 조건, 지워진 범위(`lost_until_ms`), `latest_signals`가 대체된 기록을 제외 |
| `gamst/engine.py` | `get_timeline`, `retract_observation`, `add_clock_sample`, `push_*`의 `supersedes`, `count_events(as_of_ms)` |
| `gamst/counting.py` | 증거에 `depends_on` |
| `gamst/streams.py` | 고침: worker의 작업 시계가 소스 시계를 덮어써서 다른 시계가 로컬처럼 보였습니다 |
| `gamst/context.py`, `observation_pages.py`, `mcp.py` | 안전 검사와 커서 서명을 함께 쓰도록 나눔, MCP 도구 |

`get_context`는 수정·철회가 없는 저장소에서 그대로입니다. 있으면 현재 값만 보입니다.

## 2. G와 합친 것

- 충돌 두 곳(`engine.py` `count_events`, `mcp.py` 설명)은 양쪽을 다 남겼습니다. `as_of_ms`가 있으면 빈틈 기록도 그때까지 알던 것만 봅니다.
- 어댑터 `progress()`는 `complete_until_ms`, `gap_intervals`(reason: `disabled`/`paused`/`capture_paused`는 `off`, 나머지는 `gap`), `gaps`, `sampled`, `enabled`, `paused`를 읽습니다.
- Pipeline에 있으면 좋은 것(없어도 동작):
  - `complete_from_ms`: 모듈의 완료 기록이 시작된 캡처 시각. 없으면 처음 완료가 확인된 시각 앞은 `unknown`입니다.
  - `holding_from_ms`: 모듈이 아직 들고 있는 가장 이른 캡처 시각(말하는 중인 발화 등). 없으면 발화와 동작은 끝난 뒤에 앞 행을 바꿉니다(`late_intervals`, 커서가 알려 줌).
  - 시각이 다른 기록을 이어 줄 때 `sensor=None`이라 다른 시계의 `clock_id`가 사라집니다(`_commit`의 link).

## 3. provisional/final

owner가 과부하 처리(J)를 취소해서 만들지 않았습니다. `quality.level`은 준 값을 저장하고 보여 주기만 하며, 아무것도 그 값에 반응하지 않습니다.
수정은 `supersedes`로 직접 지정할 때만 일어납니다.

## 4. I(서버 묶음)에게

서버가 받은 캡처 시각을 그대로 돌려주는 방식을 쓰면 시계 동기화가 필요 없습니다. 서버 시계로 커밋하려면 `SensorMetadata.clock_id`/`clock_epoch`와 `add_clock_sample`(8쌍 이상)이 필요합니다.

## 5. 실측 (합성 저장소, 5 capability, 60초, M2 Pro, 부하 평균 약 8)

| 조회 | 5,000개 | 10,000개 |
|---|---|---|
| 60초를 100ms로, 첫 64KB | 176ms | 430ms |
| 최근 10초를 200ms로(기본) | 58ms | 131ms |
| 그 변경분 | 112ms | 207ms |
| 601행 변경분 | 438ms | 814ms |
| `get_context` 60초 | 53ms | 61ms |

이벤트 루프 안에서 돌므로 그동안 캡처가 기다립니다. `Store.add`는 22.5 → 24~28us, 매번 지우는 상태에서는 30 → 36us입니다.

## 6. 테스트

`PYTHONPATH=. ~/.cache/gamst/demo/venv/bin/python -m unittest discover -s tests -q`: 합친 뒤 1457개, 오류는 원래 있던 15개뿐(stereo 13, sam_audio_native 2), skip 47.

## 7. 한계

- 과거 시점의 완료 상태는 200ms 간격으로 메모리에 약 5분만 남습니다. 재시작하면 없습니다.
- `get_trends`는 모든 수정본을 평균냅니다. Watch는 기록이 고쳐져도 이미 울린 것을 취소하지 않습니다.
- semantic state chain과 `observation.event`는 열로 나오지 않습니다.
- 보간은 제공하지 않습니다.

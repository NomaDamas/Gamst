# HANDOFF feat/aligned-timeline (서브에이전트 H)

워크트리 `.local-run/wt-timeline`, `0331eb9`에서 분기. push와 PR은 하지 않았습니다. 설명은 `docs/aligned-timeline.md`.

## 1. 상태

- `get_timeline`(SDK, MCP)을 구현했습니다. 캡처 시각 행, 상태 8가지, 완료 시각, 두 시각(캡처/알게 된 시각), `as_of_ms`, 수정·철회, provisional/final, 변경분 커서.
- 실제 모델, 네트워크, 다른 컴퓨터의 시계는 시험하지 않았습니다. 가짜 모듈과 합성 저장소로만 확인했습니다.
- 새 테스트 25개. 전체 결과는 6절.

| 파일 | 내용 |
|---|---|
| `gamst/timeline.py` | 행 계산, Pipeline 어댑터 `progress()`, 완료 기록 `History`, 커서 |
| `gamst/revisions.py` | 어떤 기록이 현재 값인지: `supersedes`/`revision`, 철회, provisional/final, 의존 기록 |
| `gamst/storage.py` | `record_offsets.known_ms`, 검색 조건, 지워진 범위(`lost_until_ms`), `latest_signals`가 대체된 기록을 제외 |
| `gamst/engine.py` | `get_timeline`, `retract_observation`, `add_clock_sample`, `push_*`의 `supersedes`와 `quality.level`, `count_events(as_of_ms)` |
| `gamst/counting.py` | `final`/`provisional` 구분, 대체된 이벤트 제외, `depends_on` |
| `gamst/streams.py` | 고침: worker의 작업 시계가 소스 시계를 덮어써서 다른 시계가 로컬처럼 보였습니다 |
| `gamst/context.py`, `observation_pages.py`, `mcp.py` | 안전 검사와 커서 서명을 함께 쓰도록 나눔, MCP 도구 |

`get_context`는 수정·철회가 없는 저장소에서 그대로입니다. 있으면 현재 값만 보입니다.

## 2. G(feat/realtime-pipeline)와 합치기

`1248b74`와 `git merge-tree`로 확인한 충돌은 두 곳뿐입니다.

1. `engine.py` `count_events`: 두 쪽을 다 남깁니다. G의 `gaps`/`complete`/`lost`와 H의 `as_of_ms`/`current(...)`. 빈틈 기록 조회에도 `known_until=as_of_ms`를 넣습니다.
   `as_of_ms`가 있으면 `pipeline.commit_gaps(force=True)`는 부르지 않습니다.
2. `mcp.py` `count_events` 설명: 두 문장을 이어 붙입니다.

G의 Pipeline(`1248b74`)을 겹쳐 실제로 돌려 봤습니다(스크래치패드). `off`, `pending`, `none`, `exact`, 완료 시각이 맞게 나왔습니다.

G에 있으면 좋은 것(없어도 동작):
- `complete_from_ms`: 모듈의 완료 기록이 시작된 캡처 시각. 없으면 처음 완료가 확인된 시각 앞은 `unknown`입니다.
- `holding_from_ms`: 모듈이 아직 들고 있는 가장 이른 캡처 시각(말하는 중인 발화 등). 없으면 발화와 동작은 끝난 뒤에 앞 행을 바꿉니다(`late_intervals`, 커서가 알려 줌).
- Pipeline이 다른 시계의 프레임에서 시각이 다른 기록을 이어 줄 때 `sensor=None`이라 `clock_id`가 사라집니다(`_commit`의 link).

## 3. J(과부하, provisional)에게

- 커밋할 때 `quality=dict(status='sufficient', level='provisional')`. 정식 처리는 같은 capability, source, sensor, 같은 캡처 구간으로 그냥 커밋하면 자동으로 대체합니다.
- 정식 처리 결과가 없으면(`None`) `retract_observation(record_id, reason)`을 불러야 합니다. 빈 값으로 커밋해도 됩니다.
- Pipeline은 아직 `quality`를 넘기지 않습니다. provisional 입력에서 나온 파생 결과의 수준도 넘겨야 합니다(GAMST가 추측하지 않음).
- 빈틈이었던 시간에 나중에 기록이 들어오면 그 행은 `exact`가 되고 `gap` 이유가 함께 남습니다.

## 4. I(서버 묶음)에게

서버가 받은 캡처 시각을 그대로 돌려주는 방식을 쓰면 시계 동기화가 필요 없습니다. 서버 시계로 커밋하려면 `SensorMetadata.clock_id`/`clock_epoch`와 `add_clock_sample`(8쌍 이상)이 필요합니다.

## 5. 실측 (합성 저장소, 5 capability, 60초, 다른 작업이 함께 도는 M2 Pro)

기록 10,000개: 60초를 100ms로 417ms(첫 64KB), 최근 10초를 200ms로 101ms, 그 변경분 154ms, 601행 변경분 583ms. `get_context`는 62ms.
기록 5,000개: 159 / 47 / 80 / 319ms. 이벤트 루프 안에서 돌므로 그동안 캡처가 기다립니다.
저장 비용: 24~25us로 같고, 매번 오래된 기록을 지우는 상태에서는 32 → 약 40us.
provisional과 final을 3,000쌍 커밋: 한 번에 0.22ms(처음) → 0.37ms(마지막).

## 6. 테스트

`PYTHONPATH=. ~/.cache/gamst/demo/venv/bin/python -m unittest discover -s tests -q`: RESULT

## 7. 한계

- 과거 시점의 완료 상태는 200ms 간격으로 메모리에 약 5분만 남습니다. 재시작하면 없습니다.
- `get_trends`는 모든 수정본을 평균냅니다. Watch는 provisional에도 울리고 나중에 취소하지 않습니다.
- semantic state chain과 `observation.event`는 열로 나오지 않습니다.
- 보간은 제공하지 않습니다.

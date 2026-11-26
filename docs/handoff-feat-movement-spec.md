# HANDOFF feat/movement-spec (서브에이전트 F)

워크트리 `.local-run/wt-movement`, `806fd1e`에서 분기. push와 PR은 하지 않았습니다.

## 1. 상태

- 동작 지정 두 방식(예시 영상 여러 개, 텍스트)과 검색(`find`), 내보내기(`export`)를 구현했습니다.
- **정확도는 아무것도 측정하지 않았습니다.** 합성 데이터와 가짜 VLM, 가짜 포즈 모델로 배선과 계산만 확인했습니다.
- 실제 포즈 모델, 실제 VLM은 돌리지 않았습니다. 내려받은 것도 없습니다.
- 테스트: 전체 1405개, 오류는 원래 있던 15개뿐입니다(stereo 13, sam_audio_native 2). 새 테스트는 43개입니다.

| 파일 | 내용 |
|---|---|
| `gamst/providers/movement_match.py` | 여러 예시 매칭(스트리밍 DTW), 부정 예시, 임계값 규칙 |
| `gamst/providers/movement_rule.py` | 규칙 언어(JSON, 크기·깊이·이름 검사, eval 없음) |
| `gamst/providers/pose_exemplar.py` | `ExemplarEvents`: 예시와 규칙을 모두 받는 카운터 하나 |
| `gamst/movement_finder.py` | VLM에게 구간 후보만 묻고 답을 엄격히 검사 |
| `gamst/movement_proposals.py` | 후보 파일, owner만 확인·거절 |
| `gamst/recipes/movement.py` | `enroll`, `propose`, `proposals`, `find`, `export` |
| `gamst/counting.py` | 이벤트를 JSONL/CSV 행으로 |
| `gamst/mcp.py` | `list_movement_proposals`, `propose_movement`, `resolve_movement_proposal` |

## 2. 예시 영상 준비 방법

1. 클립 하나에 동작 하나, 30초 이하. 동작 앞뒤의 가만히 있는 부분은 짧게 자릅니다.
2. 한 사람만 나오게, 실제로 셀 때와 비슷한 카메라 방향으로 찍습니다.
3. 셀 사람이 직접 3개 이상. 빠르기와 거리를 조금씩 다르게 합니다(1개면 임계값을 검증할 수 없고, 파일에 그렇게 적힙니다).
4. 세면 안 되는 비슷한 동작(하다 만 동작, 다른 동작)도 1개 이상을 `--negative`로 넣습니다.
5. 긴 영상은 자르지 않고 `파일@시작초-끝초`로 쓸 수 있습니다. 말투는 반드시 이 방식을 쓰고, 앞에 평소 말이 몇 초 있게 합니다.

## 3. 명령

```bash
gamst recipe setup person-action-count                     # 텍스트 찾기를 쓰면 --with semantic
gamst recipe enroll person-action-count --mode sequence --name "팔 들기" \
    --video a.mp4 --video b.mp4 --video long.mp4@12.0-13.4 --negative half.mp4 --out move.json
gamst recipe propose person-action-count --video long.mp4 --text "양팔을 든다" --proposals p.json
gamst recipe proposals list --proposals p.json
gamst recipe proposals confirm ID앞6자리이상[@시작-끝] --proposals p.json
gamst recipe proposals reject ID --proposals p.json
gamst recipe enroll person-action-count --text "양팔을 든다" --proposals p.json --mode sequence --out move.json
gamst recipe enroll person-action-count --text "오른팔을 든다" --rule rule.json --out move.json
gamst recipe run person-action-count --camera 0 --exemplar move.json --person me.json --store run.db --max-records 200000
gamst recipe find person-action-count --video long.mp4 --exemplar move.json --person me.json --out found.csv --format csv
gamst recipe export --store run.db --event exemplar.matched --format csv --out events.csv
gamst mcp --store run.db --proposals p.json
```

## 4. 임계값 규칙 (튜닝 없음)

- `d_i`: 예시 i를 빼고 나머지로 셀 때, 예시 i가 세지는 가장 작은 거리. `n`: 부정 예시가 세지는 가장 작은 거리.
- `p` = `n`보다 작은 `d_i` 중 최댓값(예시 1개면 0). `base = max(기본값, p)`, 부정 예시가 막아 주지 않으면 기본 exit까지만.
- `enter = min(base, (p+n)/2)`, `exit = min(enter × 기본비율, (enter+n)/2)`.
- 예시는 기본값을 넓히기만 하고, 좁히는 것은 부정 예시뿐입니다. 근거는 `threshold_source`, `threshold_evidence`에 남습니다.
- 부정 예시가 긍정 예시와 같으면 등록을 거절합니다. 실행 중에도 부정 예시가 더 가까우면(같아도) 세지 않습니다.

## 5. 고친 기존 결함

- `--mode sequence`로 등록해도 `person-action-count` 실행이 실패했습니다(모듈이 항상 hold). 이제 파일의 mode를 씁니다.
- 2초보다 긴 동작(30fps 기준)은 검색 창이 짧아 맞지 않았습니다. 이제 예시의 실제 길이를 저장하고 시간으로 찾습니다.
- 기본 임계값에서는 동작의 시작 자세로 가만히 서 있어도 sequence가 맞았습니다(합성 데이터로 확인). 예시 움직임 범위의 절반 이상 움직인
  구간만 인정합니다. 쉬지 않고 이어지는 반복도 겹치지 않는 구간이면 각각 셉니다. **예전 형식 파일에는 길이 정보가 없어 이 검사가 없습니다.**
- `recipe run`의 개수와 `--truth`는 Store에서 읽어서, 레코드가 5000개를 넘으면 앞부분이 빠졌습니다. 이제 커밋될 때 모읍니다. 같은 Store 파일의 이전 실행도
  섞이지 않습니다. Store 자체의 한도는 `--max-records`로 올립니다. `export`는 Store에서 사라진 레코드 수를 알려 줍니다.
- `--video` 등록 기본값이 "앞 1.5초"에서 "클립 전체"로 바뀌었습니다(`--start`/`--seconds`를 주면 예전처럼 그 구간).
- 이벤트: sequence 매칭에도 `start_ms`와 `exemplar.ended`가 생겼습니다. 그래서 `--truth`의 시각 비교 기준이 sequence에서 끝 시각 → 시작 시각으로 바뀝니다.
- `ExemplarEvents` 레코드의 provider 버전이 `1.hold`/`1.sequence` → `2`입니다.

## 6. 계산 비용 (합성 13채널, M2 Pro, 파이썬 프로세스 CPU 시간, 2000프레임, 다른 작업이 함께 도는 상태)

| 예시 | 중앙값 | p95 |
|---|---|---|
| hold 5+3 | 0.020 ms | 0.025 ms |
| sequence 5+3 (30벡터, 2초), 무관한 움직임 | 0.295 ms | 0.320 ms |
| sequence 5+3, 반복 동작 | 0.564 ms | 0.707 ms (최대 4.449) |
| sequence 16+16(상한), 반복 동작 | 2.185 ms | 2.699 ms |

예전 방식(창 전체를 매 프레임 다시 계산)은 예시 1개에 중앙값 2.598 ms였습니다. 새 DTW는 예전 DTW와 무작위 300쌍에서 값이 같습니다.

## 7. 측정하지 않은 것

- 세 방식의 카운트 정확도, VLM이 후보를 찾는 정확도와 비용, 말투 매칭 정확도.
- 실제 포즈 노이즈에서 임계값 규칙이 적절한지, 움직임 범위 절반 규칙이 적절한지.
- 실제 코덱(H.264 등)에서 `@start-end` 탐색 정확도(MJPG AVI만 확인).

## 8. 한계

- 파일 하나에는 hold나 sequence 중 하나만 넣습니다(섞지 않음).
- VLM은 영상만 봅니다. 말투 레시피는 텍스트로 찾을 수 없고 예시나 규칙을 씁니다.
- 거리는 채널 평균이라 한쪽 팔만 조금 움직이는 동작은 "가만히 있음"과 거리가 작습니다. 부정 예시가 필요합니다.
- 후보 파일의 버전은 실수로 고친 것을 막는 검사이지 서명이 아닙니다. 파일을 쓸 수 있는 프로그램은 위조할 수 있습니다.

## 9. owner 결정

1. MCP `resolve_movement_proposal`의 `actor`는 Watch와 같이 **호출하는 쪽이 스스로 밝히는 값**입니다(`user`/`jev`). `jev`는 거절되지만, 에이전트가
   `user`라고 밝히면 막을 수 없습니다. 더 강하게 하려면 MCP의 확인 도구를 없애고 CLI로만 확인하게 할 수 있습니다.
2. 확인 단계는 기본 켬입니다(`--no-confirm`이 명시적 예외). owner 답에 따라 바꿉니다.
3. 예시 영상과 부정 예시, 정답 횟수를 직접 센 영상이 필요합니다(실측용).

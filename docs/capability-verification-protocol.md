# 실제 촬영 검증 프로토콜 (촬영 전에 확정)

이 문서는 `docs/capabilities.md`의 능력(Count, 순간 포착·물리, 리듬, 확률 구름, 기억)을 **직접 찍은 실제 영상과 실측 정답**으로 검증하는 절차입니다. 합격선은 같은 문서의 "실제 검증 합격선"을 그대로 따르며, 촬영 후에 바꾸지 않습니다. 합성 데이터는 쓰지 않습니다.

## 0. 준비물

| 항목 | 용도 |
|---|---|
| 스마트폰 (60fps 이상, 240fps 슬로모션 가능) + 삼각대 | 모든 촬영. 카메라는 고정, 자동 노출·손떨림 보정은 끔 |
| 줄자, 마스킹 테이프, 분필(손끝 표시용) | 점프 최고 도달점 실측, 구역 표시 |
| 가슴 스트랩 심박계 또는 손가락 산소포화도계 | 심박 정답 |
| 스톱워치·박수(또는 손뼉 판) | 시작 동기화 표시 (영상·소리·센서가 같은 순간을 기록) |
| GamstCapture + 모델 | `gamst-capture fetch-models --models models` (고정 URL, sha256 확인) |

동의: 촬영되는 사람 모두에게 목적·보관·삭제를 설명하고 서면 동의를 받습니다. 원본 영상은 공개 저장소에 올리지 않습니다. 공개는 해시와 정답·결과 JSON만 합니다.

## 1. 파일 규칙

```
captures/<YYYYMMDD>/<mode>-<take>.mp4          # 원본 (native fps, 소리 포함)
captures/<YYYYMMDD>/<mode>-<take>.truth.json   # 정답 사이드카 (촬영 직후 작성)
captures/<YYYYMMDD>/<mode>-<take>.report.json  # gamst-capture verify 출력
```

정답 사이드카 스키마 (필요한 키만 채움):

```json
{
  "mode": "count | jump | rhythm | zone",
  "count": 12,
  "subject_height_m": 1.74,
  "jumps": [{"height_m": 0.34}, {"height_m": 0.31}],
  "heart_bpm": 68,
  "breath_per_min": 14,
  "alarms": [[3.2, 7.9], [21.4, 24.0]],
  "annotators": ["A", "B"],
  "notes": "자유 기록"
}
```

정답 작성 규칙: Count와 구역 시각은 **두 사람이 독립적으로** 기록하고 일치한 값만 씁니다(불일치면 그 take 제외). 기계 정답(심박계, 줄자)은 측정값 그대로.

## 2. 능력별 촬영

### Count — 반복 횟수
- 장면: 한 사람, 전신이 보이게 측면 또는 정면. 스쿼트·점핑잭·팔굽혀펴기·버피 등.
- 분량: 동작별 10세트 이상, 세트당 5~30회, 템포를 섞음(느리게/빠르게).
- 정답: `count` (두 사람이 센 값).
- 실행: `gamst-capture verify count count-01.mp4 --truth count-01.truth.json`

### ② 순간 포착 + 물리 — 점프
- 장면: 측면, 발과 골반이 프레임 안, 카메라 높이는 골반 높이. 60fps 이상(가능하면 240fps).
- 시작 1초는 **가만히 서 있기** (기준 발 높이와 키 스케일 측정).
- 정답: 벽에 분필 손끝으로 서서 닿는 높이와 점프 최고 도달 높이를 줄자로 재서 차이를 `jumps[].height_m`에 기록. 가능하면 검증된 점프 앱/포스플레이트 값도 함께.
- `subject_height_m`: 실측 키.
- 품질 게이트: provider가 낸 `fitted_gravity_m_s2`가 7.5~12 m/s² 밖이면 스케일·원근이 틀린 것이므로 그 점프는 제외 사유로 기록합니다(정면에서 찍은 점핑잭 시험에서 3~6 m/s²가 나와 이 게이트를 추가함).
- 편향 보정: 같은 카메라 배치의 **다른 세션** 점프로 `calibration_offset_m`을 정합니다. 시험 대상 take로 보정하지 않습니다.
- 실행: `gamst-capture verify jump jump-01.mp4 --truth jump-01.truth.json`
- 1 FPS 대조: 같은 리포트의 `1fps` lane에서 점프가 검출되지 않아야 합니다.

### ③ 리듬 — 심박 / 호흡
- 심박: 얼굴 정면, 밝고 일정한 조명, 60초 정지. 가슴 스트랩/산소포화도계를 같은 60초 동안 기록해 평균 `heart_bpm`.
- 호흡: 상체 정면 또는 측면, 60초. 두 사람이 호흡 수를 세서 `breath_per_min`.
- 조건: 안정 시 + 가벼운 운동 직후 각각.
- 실행: `gamst-capture verify rhythm hr-01.mp4 --truth hr-01.truth.json`
- 1 FPS 대조: `1fps` lane은 `undersampled`여야 합니다(숫자를 내면 실패).

### ① 확률 구름 — 사용자 규칙
- 장면 예: 부엌. 바닥에 테이프로 "가스레인지 1m" 구역 표시. 소리 조건: 물 끓는 소리/주전자 휘슬(또는 블렌더).
- 시나리오(각 5회 이상): (a) 사람만 구역에 들어옴, (b) 소리만 남, (c) 둘 다 → 알람이어야 함, (d) 둘 다 → 사람이 나감(해제), (e) 빠르게 들어왔다 나감(1초 미만).
- 규칙 파일: `docs/capabilities.md`의 "A이고 B" 작성법대로(항목당 weight < bias < 합). 소리 신호 이름은 리포트에서 `audio.loudness`(녹화 소리의 크기 점수) 또는 YAMNet/CED 사건 점수.
- 정답: 두 사람이 영상에서 "두 조건이 모두 충족된 구간"을 `alarms`에 기록.
- 실행: `gamst-capture verify zone kitchen-01.mp4 --rule stove.rule.json --truth kitchen-01.truth.json`

### 기억
- 모든 모드의 리포트에 `memory`가 포함됩니다. provider 사건을 GAMST 세션에 넣고 **MCP `get_observation_page(events_only)`만으로** 답한 횟수·마지막 시각입니다.
- 판정: `memory.events`가 정답 횟수와 같고, 마지막 시각이 정답 ±0.5초 안.

### 동전 회전 (Halo)
- 240fps 슬로모션, 동전이 화면 중앙, 배경 단색. 스핀 20회 이상.
- 정답: 슬로모션을 프레임 단위로 보며 두 사람이 센 회전 수.
- 실행: 기존 `gamst-capture run spin-01.mp4 --report spin-01.report.json --no-vlm`

## 3. 판정과 공개

1. 모든 take를 실행하고 리포트를 그대로 보관합니다(실패 take도 삭제하지 않음).
2. 능력별 합격선(`docs/capabilities.md`)으로 판정합니다. 품질 게이트로 제외한 take는 사유와 개수를 함께 공개합니다.
3. 결과 표에는 항상 native lane과 1fps lane을 나란히 둡니다.
4. 영상 해시(`video_sha256`), 정답 JSON, 리포트 JSON, 사용한 모델 sha256을 함께 커밋합니다.

## 4. 이미 끝난 사전 점검 (공개 실데이터로 도구 동작 확인)

| 모드 | 사용한 실제 영상 | 결과 (정확도 주장 아님, 도구 동작 확인) |
|---|---|---|
| count | RepCountA `stu10_25` (사람 카운트 7) | native 8 (±1), 1fps 6 (±1), MCP 기억 8건 |
| rhythm | MCD-rPPG 1785 before 60초 (접촉 PPG 62.6 bpm) | native 50.3 bpm (오차 −12.3), 1fps `undersampled` |
| zone | MOT17-09 (사람 박스 기반 방문 3회) | native 알람 1회(3회 중 1회 1초 안), 1fps 3/3, MCP 기억 1건 |
| jump | RepCountA 점핑잭 (키 1.70 m 가정, 높이 정답 없음) | 12건 검출, `fitted_gravity` 3~6 → 품질 게이트 필요 확인, 1fps 0건 |

이 점검에서 드러난 것: 심박(rPPG)과 확률 구름의 파라미터(sigma, gain)는 실제 촬영 전에 더 다듬을 여지가 있습니다. 다만 합격선은 고정이며, 다듬는 데 시험 take를 쓰지 않습니다(별도 연습 take만 사용).

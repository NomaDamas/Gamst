# 새 레시피 두 개: `pose-count`와 `voice-keyword`의 아키텍처

> **대체됨(2026-11-25):** 상위 설계는 [references-and-targeted-perception.md](references-and-targeted-perception.md)입니다. 이 문서의 두 레시피는 그 일반 기능 위의 사용 예이고, 모델 선택도 그 문서의 8절이 우선입니다.

상태: 설계 문서입니다. 아직 구현하지 않았고, 측정한 값도 없습니다.
모델 선택의 근거는 [research/speaker-pose-sota-2026-11/report.md](research/speaker-pose-sota-2026-11/report.md)와 같은 폴더의 `notes/`에 있습니다.
리서치는 2026-11-25에 했습니다.

## 1. 지금의 레시피 구조

`gamst/recipes/__init__.py`와 `<name>.json`이 레시피를 이룹니다.

| 요소 | 지금의 모양 | 회전에 묶인 부분 |
|---|---|---|
| `STEPS` | capture → find → follow → count | 없음. 이 4단계는 일반적입니다. |
| `PARTS` | `(step, use)` 조합마다 extras, 모델, 소스 체크아웃 | 회전 부품만 등록되어 있습니다. |
| `FLOWS` | count 부품마다 허용하는 (find, follow) 조합 | `flip`, `in-plane` 두 가지뿐입니다. |
| `check()` | 샘플링 한계 `fps / (2 × symmetry)` | rev/s 공식이 하드코딩되어 있습니다. |
| `demo_argv()`와 `run` | 모든 레시피를 `gamst demo rotation`으로 넘깁니다 | 회전 데모 하나에만 묶여 있습니다. |
| Halo (`demos/rotation_live.Hud`) | 1920×1080 화면: 영상, 오른쪽 패널, 아래 GAMST CONTEXT 피드 | 패널 문구가 코인 던지기용입니다(LAST THROW, flips). |

결론적으로 4단계 모양과 setup, check, run 흐름은 그대로 둘 수 있습니다. 회전에 묶인 곳은
세 군데(`check`의 공식, `run`의 전달 대상, Halo 패널)뿐이고, 이 세 곳만 레시피가 고를 수
있게 바꾸면 됩니다.

## 2. 새 레시피를 기존 4단계에 대응하기

4단계의 뜻을 조금 넓히면 두 레시피가 그대로 들어갑니다.

- find: 대상이 어디 있는지 찾습니다.
- follow: 그 대상이 같은 대상인지 계속 유지합니다.
- count: 이벤트를 셉니다.

| 단계 | rotation (기존) | **pose-count** (신규) | **voice-keyword** (신규) |
|---|---|---|---|
| capture | camera / video / take | camera / video / take | **microphone / audio / take** (16 kHz mono, 캡처 시계 포함) |
| find | roi / motion / yoloe | **사람 키포인트**: RTMW(기본), Sapiens2(호스팅) | **발화 구간**: Silero VAD (기존 `vad.py`) |
| follow | fixed / shape-tracks / efficienttam | **같은 사람 유지**: 트랙 ID와 One-Euro 필터 | **같은 목소리 유지**: 등록 목소리 화자 검증 (ReDimNet2 또는 CAM++) |
| count | in-plane / flip | **exemplar**(시연 매칭과 subsequence DTW), **period**(기존 `repetition.py`) | **keyword**: Qwen3-ASR-1.7B, Qwen3-ForcedAligner, 정규화 텍스트 매칭 |
| record | `vision.rotation(.refined)` | `temporal.repetitions` (기존 출력 타입 재사용) | `speech.keyword_count` (신규) |

부품 선택은 리서치 결과를 따랐습니다. "SOTA"와 "실제로 쓸 수 있는 SOTA"를 구분합니다.

| 역할 | 정확도 SOTA | 레시피 기본값 | 이유 |
|---|---|---|---|
| 2D 전신 포즈 | Sapiens2-5B (Meta 자체 테스트셋 82.3 mAP) | **RTMW-x** (COCO-WholeBody 70.2 AP, Apache-2.0)와 rtmlib | Sapiens2 라이선스는 "biometric processing"을 금지합니다. 특정 인물을 대상으로 하는 카운팅이 여기에 걸리는지는 법적 검토가 필요하므로, `find: sapiens2`는 호스팅 옵션으로만 둡니다. |
| 반복 카운터 | Count What Repeats/PAMS (RepCount OBO 0.842, 자체 보고) | **exemplar 매칭 + 기존 period 카운터** | PAMS는 코드를 찾지 못했습니다. ESCounts(MIT, OBO 0.563)는 비교 기준선으로만 씁니다. "이 동작"을 지정해서 셀 수 있는 것은 exemplar 매칭과 ESCounts뿐입니다. |
| 화자 검증 | w2v-BERT 2.0 SV (Vox1-O EER 0.12%) | **ReDimNet2-B6** (0.29/0.52/0.99, MIT 코드). 경량 대안은 sherpa-onnx의 **CAM++** | w2v-BERT 2.0 repo에 라이선스가 없습니다. 가중치 라이선스는 학습 데이터에 따라 정해지므로, VoxCeleb2로만 학습한 가중치를 고릅니다. |
| 한국어 ASR | Qwen3-ASR-1.7B (FLEURS-ko CER 2.57) | **Qwen3-ASR-1.7B**와 **Qwen3-ForcedAligner-0.6B** (한국어 AAS 37.2 ms) | SOTA와 기본값이 같습니다. 다만 스트리밍 모드에서는 타임스탬프가 나오지 않으므로, VAD 세그먼트 단위 준실시간으로 처리합니다. |

## 3. 바꿀 곳

### 3.1 레시피 JSON: `view`와 `halo` 블록 추가

```json
{
  "name": "pose-count",
  "summary": "Count how many times one person makes an owner-demonstrated pose or movement.",
  "skill": "skills/gamst-pose-count/SKILL.md",
  "view": "halo",
  "hardware": {"min_fps": 30, "camera": "a fixed camera that sees the whole body", "compute": "CUDA or MPS; Sapiens2 only hosted"},
  "enroll": {"kind": "pose-exemplar", "takes": 1, "store": "local"},
  "pipeline": [
    {"step": "capture", "use": "camera", "sources": ["camera", "video", "take"]},
    {"step": "find", "use": "rtmw"},
    {"step": "follow", "use": "pose-track", "smoothing": "one-euro"},
    {"step": "count", "use": "exemplar", "max_reps_s": 2, "mode": "sequence"}
  ],
  "records": ["temporal.repetitions"],
  "halo": {"capability": "temporal.repetitions", "number": "count", "unit": "reps",
           "detail": ["period_s", "last.distance"], "history": "events"},
  "measured": "not yet"
}
```

```json
{
  "name": "voice-keyword",
  "summary": "Count how many times one enrolled person says an owner-declared word.",
  "view": "halo",
  "enroll": {"kind": "voice", "takes": 5, "store": "local", "consent": true},
  "pipeline": [
    {"step": "capture", "use": "microphone", "sources": ["microphone", "audio", "take"]},
    {"step": "find", "use": "vad"},
    {"step": "follow", "use": "speaker", "model": "redimnet2-b6"},
    {"step": "count", "use": "keyword", "words": {"대박": ["대박", "대-박"]}}
  ],
  "records": ["speech.keyword_count"],
  "halo": {"capability": "speech.keyword_count", "number": "count", "unit": "times",
           "detail": ["keyword", "speaker_similarity"], "history": "events"}
}
```

### 3.2 `gamst/recipes/__init__.py`

- `PARTS`에 새 부품을 등록합니다.
  - `('find','rtmw')`, `('find','sapiens2')`
  - `('follow','pose-track')`
  - `('count','exemplar')`, `('count','period')`
  - `('capture','microphone')`
  - `('find','vad')`, `('follow','speaker')`, `('count','keyword')`
  - 각 부품의 extras와 SHA 고정 가중치를 함께 적습니다.
- `FLOWS`에 `exemplar`와 `period`는 `('rtmw'|'sapiens2', 'pose-track')`, `keyword`는
  `('vad', 'speaker')` 조합을 추가합니다.
- `SOURCES_BY_CAPTURE['microphone'] = ('microphone', 'audio', 'take')`를 추가합니다.
- `check()`는 공식을 count 부품별 함수로 나눕니다. 셋 다 물리나 샘플링 규칙이고, 튜닝값이 아닙니다.
  - rotation: 지금처럼 `fps / (2 × symmetry)`입니다.
  - exemplar, period: 가장 빠른 반복의 2배를 넘는 프레임레이트가 필요합니다(Nyquist).
    여기에 키포인트 모델의 실측 처리 fps가 캡처 fps를 따라가는지도 봅니다.
  - keyword: 샘플레이트가 16 kHz 이상인지, 등록 발화가 몇 초인지 봅니다.
- `enroll` 서브커맨드를 추가합니다(`gamst recipe enroll pose-count --camera 0`).
  - 시연 또는 목소리를 `~/.cache/gamst/enroll/<recipe>/`에 저장합니다.
  - 목소리는 임베딩만 남기고 원본 오디오는 지웁니다.
  - 동의를 묻습니다.
- `run`은 `recipe['view']`로 보낼 곳을 정합니다.
  - `rotation`과 `coin-flip`은 지금처럼 `gamst demo rotation`으로 갑니다.
  - `halo` view는 새로 만드는 `gamst demo count`로 갑니다.

### 3.3 Halo: 화면 틀과 카드로 나누기

`rotation_live.Hud`를 **화면 틀(Shell)**과 **카드(Card)**로 나눕니다.

- Shell은 그대로 재사용합니다: 로고, 영상 영역, 위쪽 상태 표시(REC/STANDBY), 캡처 fps,
  아래쪽 "GAMST CONTEXT `<capability>` → …" 피드.
- `CoinCard`는 지금 있는 `_panel`을 그대로 옮긴 것입니다. 코인 화면은 바뀌지 않습니다.
- `CountCard`는 범용 카드입니다. 레시피의 `halo` 블록이 정한 필드만 읽어서 그립니다.
  - 큰 숫자: `count`
  - 단위
  - 마지막 이벤트의 세부 정보
  - 이번 세션의 이벤트 목록

  카드에는 숫자 계산이 없습니다. 기존 원칙과 같이 "화면에 보이는 숫자는 모두 커밋된 GAMST
  레코드에서 온다"를 지킵니다.
- 영상 영역에는 레시피별 오버레이를 그립니다.
  - pose-count: 스켈레톤, 그리고 등록 자세와의 거리 막대(임계값 선 표시).
  - voice-keyword: 카메라 화면(있으면), 파형, 화자 유사도 띠. 그리고 단어가 잡힐 때
    "등록인물A · 대박 #14" 칩을 띄웁니다.

`gamst demo count`는 캡처와 표시만 맡습니다. 계산은 모두 provider가 합니다. 지금의
`demos/rotation.py`와 같은 역할 분리입니다.

### 3.4 새 provider (부품)

| 파일 | 하는 일 | 모델 |
|---|---|---|
| `providers/rtmw_pose.py` | 전신 133개 키포인트, `pose.py`와 같은 출력 모양 | rtmlib RTMW-x ONNX (SHA 고정) |
| `providers/pose_track.py` | 트랙 ID 유지(기존 `tracking.create_tracker`)와 One-Euro 필터 | 계산만 |
| `providers/pose_exemplar.py` | 정규화(골반 원점, 어깨 폭, 몸통 회전), 관절각 특징, hold 히스테리시스, subsequence DTW | 계산만 |
| `providers/speaker_verify.py` | 등록 임베딩 평균, 세그먼트 코사인, 연속 세그먼트 스무딩 | ReDimNet2-B6 (torch.hub, 커밋 고정) / CAM++ ONNX |
| `providers/forced_align.py` | 세그먼트 텍스트를 단어 시점으로 바꿈 | Qwen3-ForcedAligner-0.6B |
| `providers/keyword_count.py` | 단어 시점, 화자 판정, 키워드 선언을 시간으로 조인. 한국어 정규화(띄어쓰기, 조사)와 `overlapped` 상태 처리 | 계산만 |

모두 범용 부품입니다. 특정 자세나 단어는 owner의 등록 데이터와 레시피 선언으로만 들어갑니다.
MCP와 Watch는 손대지 않아도 붙습니다. 레코드만 커밋하면 `get_observations`로 읽히고, Watch는
`events contains keyword.spoken` 같은 조건으로 걸면 됩니다.

### 3.5 skills

`skills/gamst-pose-count/SKILL.md`와 `skills/gamst-voice-keyword/SKILL.md`를 추가합니다.
기존 `gamst-rotation`과 같은 형식입니다: 레시피 이름, check, enroll, run, MCP로 결과 읽기.

## 4. 측정 게이트 (각 레시피의 `measured` 필드)

| 레시피 | 정답 | 지표 |
|---|---|---|
| pose-count | owner가 직접 센 테이크 2개 이상(샷건 자세 hold, 패스 동작) | 카운트 MAE, OBO, 이벤트 시점 오차, 같은 테이크 재생 시 같은 레코드(replay) |
| voice-keyword | owner 목소리와 다른 사람 목소리가 섞인 한국어 테이크, 단어 시점을 사람이 표시 | 화자 EER와 임계값별 정밀도·재현율, 키워드 카운트 정밀도·재현율, 키워드가 없는 오디오에서의 오탐 |

비교 기준선도 같은 테이크에 돌립니다.
- 포즈: ESCounts, 그리고 Gemini Live(1 fps)나 VLM에 묻는 방식. 기존 원칙대로 Gemini 3.1 Pro는 쓰지 않습니다.
- 음성: 녹음 전체를 오디오 LLM에 한 번에 넣고 묻는 방식.

비용이 드는 호스팅이나 API 호출은 시작하기 전에 보고합니다.

## 5. 구현 순서 제안

1. Halo를 Shell과 Card로 나눕니다. 코인 화면이 바뀌지 않았는지 기존 take로 회귀 확인합니다.
2. 레시피 일반화를 합니다: `view`, `check` 분리, `enroll`. rotation과 coin-flip 테스트가 그대로 통과해야 합니다.
3. pose-count: RTMW 부품, pose-track, exemplar 순서로 만들고 정답 테이크로 측정합니다.
4. voice-keyword: speaker_verify, forced_align, keyword_count 순서로 만들고 정답 테이크로 측정합니다.

1과 2는 기존 레시피에 영향을 주므로 별도 PR로 먼저 올리고, 3과 4는 레시피마다 PR을 하나씩 올리는 것을 제안합니다.

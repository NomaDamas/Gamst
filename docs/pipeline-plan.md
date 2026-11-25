# 최종 기획: 입력 모듈 · 파이프라인 · 레시피로 "지정한 인물" 인식을 일반화

상태: 기획 문서입니다(2026-11-25). 아직 구현하지 않았고, 측정한 값도 없습니다.
모델 근거는 [research/speaker-pose-sota-2026-11/](research/speaker-pose-sota-2026-11/report.md)에 있습니다.

## 0. 전제 (owner 확정)

```text
┌──────── Perception 입력층 ─────────────────────────┐
│ 카메라 · 마이크 · 센서 · 사용자 정의 입력 모듈           │
└──────────────────────────────────────────────────┘
                    ▼
┌──────── GAMST perception (realtime) ──────────────┐
│ 경량 모델 + 수학/물리 계산 여러 개, 지시마다 켜고 끔     │
└──────────────────────────────────────────────────┘
                    ▼
              Context ──(MCP)──▶ LLM
```

- **GAMST의 범위는 Context 전달까지입니다.** LLM과 GAMST 사이에 어떤 조건을 걸지는 사용자가 정합니다.
  기존 Watch도 사용자가 선언한 조건을 저장하고 평가할 뿐입니다. 이번 기획은 LLM 쪽 로직을 추가하지 않습니다.
- **Halo는 개념입니다.** "고주파 관측을 context로 바꾸는 과정을 한 화면에 보여주는 시연"을 뜻합니다(positioning.md).
  코드 계층이 아닙니다.
- **레시피는 조합 프리셋입니다.** 어떤 입력과 어떤 모듈을 켤지만 적습니다.
- 모델은 SOTA로 고릅니다. 법적 검토는 owner가 허가를 받았습니다.

## 1. 지금 코드와 전제가 어긋난 곳

| 항목 | 지금 코드 | 문제 |
|---|---|---|
| 레시피 | `gamst recipe run coin-flip`은 `--iphone 9123 --refine "metal coin" …` 같은 CLI 옵션을 만듭니다. 그다음 `demos/rotation.py`의 `main(argv)`를 부릅니다. | 레시피가 런타임에 모듈을 켜는 게 아니라, **회전 데모 프로그램 하나의 실행 옵션**입니다. 데모 안에는 캡처 루프, `RotationPlugin` 등록, HUD가 고정되어 있습니다. 그래서 포즈나 음성 모듈을 켜는 레시피는 표현할 수 없습니다. `PARTS`, `FLOWS`, `check()`(rev/s 공식)도 회전 부품만 압니다. |
| 캡처 | 카메라, 영상 파일, take, iPhone 240 수신 루프가 데모 안에 있습니다(`_TakeCapture`, `cv2.VideoCapture`, rotation_live 수신기). | 입력이 GAMST 입력층에 있지 않고 데모에 묶여 있어서, 다른 레시피가 다시 쓸 수 없습니다. |
| HUD | `rotation_live.Hud`는 코인 전용 화면입니다. | 레시피와 상관없는 범용 뷰가 없습니다. |
| 켜고 끄기 | MCP `configure_feature`가 있지만, capture를 멈춘 상태에서만 되고 설치된 플러그인 단위로만 됩니다. | "지시마다 조합을 바꾼다"는 전제를 반만 지원합니다. |
| 사용자 정의 입력 | `push_text`, `push_claim`, `push_observation`은 있습니다. | 목소리 샘플, 인물 사진, 시연 동작, 단어 목록처럼 **사용자가 넣는 기준 입력**을 받는 칸이 없습니다. |

## 2. 지금 추상화의 한계 (코드로 확인)

1. **모듈 추상화가 둘입니다.**
   - `PerceptionPlugin`: `human.register(plugin)`, video와 audio만 받습니다. 켜고 끄기는 `runtime_enabled`, 해제 API는 없습니다.
   - CapabilitySpec provider: `SensorStreams.open/finish/clear`로 lane 단위로 엽니다.

   레시피가 "모듈 조합"을 한 가지 방식으로 표현하기 어렵습니다.
2. **모듈 출력이 다른 모듈의 미디어 입력이 될 수 없습니다.**
   - 관찰 값은 `bounded_json(forbid_raw=True)`라서 원본 미디어를 담지 못합니다. 이것은 맞는 규칙입니다.
   - 모듈 간 연결은 Store를 읽는 방식뿐입니다(예: `RotationPlugin`이 검출 관찰을 읽음).
   - 그래서 "A 목소리만 추출 → ASR"이나 "A만 크롭 → 포즈"를 만들 수 없습니다.
3. **"지금 켜진 조합"을 나타내는 런타임 객체가 없습니다.** 레시피가 인스턴스화할 대상이 없고, MCP가 바꿀 대상도 없습니다.
4. **입력은 앱이 갖고 있습니다.** 캡처는 host가 하고 `push_*`만 GAMST로 들어옵니다. 이 방식은 유지하되, 재사용 가능한 입력 모듈을 GAMST가 제공해야 합니다.

## 3. 제안 (승인 필요): 새 계층 1개 + 새 개념 2개

### 3.1 [새 개념 A] 입력 모듈 (Perception 입력층)

입력층에 꽂는 모듈의 공통 규약입니다. 하는 일은 캡처 시계가 붙은 프레임을 내보내는 것 하나입니다.

| 종류 | 예 | 지금 위치 → 옮길 곳 |
|---|---|---|
| 장치 | 카메라, 마이크, iPhone 240, IMU | 데모 안 → `gamst/inputs/` |
| 재생 | 영상 파일, take(`raw.gmst`), 녹음 파일 | 데모 안 → `gamst/inputs/` |
| **사용자 정의 기준 입력** | voice(목소리 샘플), person(얼굴과 몸 사진), exemplar(시연 자세와 동작), terms(단어 목록) | 없음 → 신규 |
| **파생 입력** | 대상 인물 목소리 스트림, 대상 인물 크롭 영상 | 없음 → 신규. 아래 B 참고 |

- 기존 `push_video`와 `push_audio`는 그대로 둡니다. 앱이 직접 캡처하는 방식도 계속 됩니다.
- 사용자 정의 기준 입력은 원본을 남기지 않고 임베딩과 특징만 남깁니다. 동의, 만료, 삭제 규칙은 기존
  retention과 `delete_data` 경로를 탑니다.
- 사용자 정의 입력 모듈도 기존 플러그인처럼 `gamst.inputs` entry point로 꽂을 수 있게 합니다. 서드파티 센서도 같은 방식입니다.

### 3.2 [새 개념 B] 파생 입력 (한계 2 해결)

- 모듈은 원본 미디어를 **context로는 내보내지 못합니다**. 이 규칙은 그대로입니다.
- 대신 **입력층으로는 돌려보낼 수 있게** 합니다. 그러면 그 모듈은 다음 입력을 만드는 입력 모듈이 됩니다.
- 시간축은 원본 lane의 캡처 시계와 sequence를 물려받습니다. 출처(`derived_from`)를 기록합니다.

```text
마이크 ─▶ [목표 화자 추출 모듈] ─▶ 파생 입력 "A의 목소리" ─▶ (입력층) ─▶ ASR, prosody, emotion …
카메라 ─▶ [인물 일치 + 크롭]    ─▶ 파생 입력 "A의 영상"   ─▶ (입력층) ─▶ pose, hand, gaze …
```

이것이 "특정인 목소리 추출"과 "특정 인물 행동 인식"의 일반형입니다. 기존 perception 모듈을 하나도
고치지 않고, 그 사람만 보는 입력을 받게 됩니다.

### 3.3 [새 계층] 파이프라인 (조합 계층, 한계 1·3 해결)

입력층과 perception 사이에 "지금 켜진 조합"을 나타내는 얇은 런타임 객체를 둡니다.

```text
Pipeline = { inputs: [...], modules: [...], links: [모듈 → 파생 입력 → 모듈] }
  · recipe run      → Pipeline을 만들어 런타임에 적용
  · MCP apply/enable/disable_module → 같은 Pipeline을 capture 중에도 바꿈
  · 기존 PerceptionPlugin, SensorStreams provider는 adapter로 둘 다 모듈로 감쌈(재작성 없음)
```

- 이것이 이번 기획에서 유일한 **새 계층**입니다. 필요한 이유는 둘입니다.
  - "레시피 = 조합"과 "지시마다 켜고 끔"이라는 전제를 코드로 표현할 대상이 지금은 없습니다.
  - 두 갈래로 나뉜 모듈 추상화를 하나로 묶을 곳이 필요합니다.
- 실행 엔진은 새로 만들지 않습니다. 실행은 기존 `register`와 `SensorStreams`가 합니다. Pipeline은 그것을 조립하고 상태를 보여주기만 합니다.

### 3.4 캡처 루프와 HUD를 어디에 둘지

- **캡처 루프**는 입력 모듈로 옮깁니다(3.1). 카메라, 파일, take, iPhone 240이 모든 레시피에서 재사용됩니다.
- **HUD**는 owner 지시대로 사용자 정의 모듈로 옮깁니다. 다만 HUD는 perception 결과를 그리는 것이라서,
  입력층이 아니라 **context를 구독하는 쪽**(SDK record stream)에 꽂히는 모듈이 됩니다. MCP와 나란한 위치입니다.
  - 레시피는 어떤 뷰 모듈을 쓸지만 적습니다.
  - 지금의 코인 화면은 "코인 레시피의 뷰 모듈"이 됩니다.
  - 범용 뷰 모듈은 켜진 모듈들의 레코드를 그대로 그립니다.
  - Halo 시연은 레시피와 뷰 모듈의 조합으로 만듭니다.

## 4. 최종 구조

```text
┌──────────────── Perception 입력층 ───────────────────────────────┐
│ 장치: 카메라 · 마이크 · iPhone240 · IMU      재생: 파일 · take          │
│ 사용자 정의: voice · person · exemplar · terms · (서드파티 입력)       │
│ 파생: "A의 목소리" · "A의 영상"  ◀─────────────────────┐             │
└───────────────────────────────────────────────────┼─────────────┘
                     ▼                               │
┌──────────────── Pipeline (조합, 신규 계층) ──────────┼─────────────┐
│ recipe / MCP 지시로 켜고 끔. 기존 모듈은 adapter로 감쌈 │             │
└───────────────────────────────────────────────────┼─────────────┘
                     ▼                               │
┌──────────────── GAMST perception (realtime) ───────┼─────────────┐
│ 경량·SOTA 모델: 화자 일치 · 목표 화자 추출 ───────────┘             │
│                 인물 일치(얼굴·몸·추적) · 포즈 · ASR+단어 시점       │
│ 수학·물리: 시연 매칭(DTW·히스테리시스) · 단어 매칭 · 반복 · 회전 …   │
└───────────────────────────────────────────────────────────────┘
                     ▼
┌──────────────── Context (기존) ──────────────────────────────────┐
│ Store · 레코드 · Watch(사용자가 선언한 조건) · count 질의(신규 도구) │
└───────────────────────────────────────────────────────────────┘
        │ MCP                                │ SDK record stream
        ▼                                    ▼
       LLM (조건과 사용은 사용자 몫)         뷰 모듈(HUD) — Halo 시연
```

## 5. 요구사항 구현

| 요구 | 입력 모듈 | perception 모듈 (SOTA) | context로 나가는 것 |
|---|---|---|---|
| 지정 인물이 단어를 몇 번 말했나 | 마이크, voice, terms | 화자 일치 **w2v-BERT 2.0 LoRA-MFA**, 목표 화자 추출 **USEF-TFGridNet**(상업용은 WeSep), ASR **Qwen3-ASR-1.7B + ForcedAligner**, 단어 매칭(계산) | `speech.term_events`(단어, 시각, 화자 점수), count |
| 지정 인물이 동작/자세를 몇 번 했나 | 카메라, person, exemplar | 인물 일치 **LVFace-B + SOLIDER + MOTIP**(실시간은 기존 OC-SORT), 포즈 **Sapiens2-1B/5B**, 시연 매칭(계산), 학습 백엔드 **ESCounts** 옵션 | `temporal.exemplar_events`, 기존 `temporal.repetitions`, count |
| 일반화 | 같은 입력 모듈과 파생 입력 | 기존 모듈(감정, 말속도, 손, 시선 …)이 파생 입력을 그대로 받음 | 기존 레코드 |

- 수치는 모두 인용값입니다. 한국어 화자 검증, 한국어 TSE, 스트리머 제스처에는 공개 벤치마크가 없습니다.
- CC BY-NC 가중치(USEF, VoxBlink2로 학습한 w2v-BERT)는 기본값이 아니라 opt-in으로 두는 것을 제안합니다.
- 지연을 공개하지 않은 SOTA 모델(Sapiens2, w2v-BERT)은 호스팅 GPU에서 실측한 뒤, 실시간 lane에 쓸지
  재채점용으로만 쓸지 정합니다.

## 6. 레시피 예

```json
{
  "name": "person-term-count",
  "inputs": [{"use": "microphone"}, {"use": "voice", "enroll": true}, {"use": "terms", "values": ["대박"]}],
  "modules": [
    {"use": "target_speech", "model": "usef-tfgridnet", "from": ["microphone", "voice"], "emits": "voice_target"},
    {"use": "asr_words", "model": "qwen3-asr-1.7b+aligner", "from": ["voice_target"]},
    {"use": "speaker_match", "model": "w2v-bert2-mfa", "from": ["microphone", "voice"]},
    {"use": "term_events", "from": ["asr_words", "speaker_match", "terms"]}
  ],
  "view": "count-card",
  "measured": "not yet"
}
```

기존 `coin-flip`과 `rotation`도 같은 모양으로 옮깁니다. `check()`의 샘플링 규칙은 모듈마다 갖게 합니다
(회전 `fps/(2·symmetry)`, 포즈 Nyquist, 음성 16 kHz).

## 7. 구현 순서 (PR 단위)

1. **입력 모듈 규약**: 카메라, 파일, take, iPhone 240을 데모에서 `gamst/inputs/`로 옮깁니다. 회귀 기준은 코인 take에서 같은 레코드가 나오는 것입니다.
2. **Pipeline 계층**: 기존 두 모듈 추상화의 adapter를 만들고, `recipe run`을 Pipeline 적용으로 바꿉니다. MCP에 enable, disable, apply 도구를 추가합니다(capture 중에도 동작). 회귀 기준은 coin-flip과 rotation 레시피가 그대로 도는 것입니다.
3. **뷰 모듈**: HUD를 SDK record stream 구독 모듈로 옮깁니다. 코인 화면과 범용 count 카드를 만듭니다.
4. **사용자 정의 기준 입력과 파생 입력**: voice, person, exemplar, terms를 추가합니다. 파생 입력 lane은 provenance를 담습니다.
5. **perception 모듈**: speaker_match, target_speech, asr_words, person_match, Sapiens2 포즈, exemplar_events, term_events를 추가합니다. 각 모듈마다 실제 모델을 호출하는 테스트를 둡니다.
6. **레시피 2개와 측정**: owner가 직접 센 테이크로 count 오차와 화자 EER을 재고 `measured`에 기록합니다.
   비용이 드는 호스팅은 시작하기 전에 보고합니다.

## 8. 바뀌지 않는 것

Context(Store, 레코드 스키마, Watch), MCP의 기존 도구, SDK `push_*`, 기존 perception provider,
캡처 시계, retention, `delete_data`, "context에는 원본 미디어를 넣지 않는다"는 규칙은 바뀌지 않습니다.

## 9. 구현 상태 (2026-11-25)

| 단계 | 상태 | 근거 |
|---|---|---|
| Pipeline 계층 | **완료** | `gamst/pipeline.py`. 레코드 링크(`"pose:scene.poses"`), 파생 입력(`Derived`), capture 중 켜고 끄기, `finish`. 테스트 `tests/test_pipeline.py` |
| 입력 모듈 | **완료**(카메라, 영상 파일, take, WAV, 마이크) | `gamst/inputs.py`. iPhone 240 실시간 수신기는 아직 halo-coin 앱 안에 있습니다 |
| 사용자 정의 기준 입력 | **완료**(exemplar, terms) / voice·person은 동의 기록 검사만 | `Pipeline.add_reference`, `gamst recipe enroll` |
| count 질의 | **완료** | `Gamst.count_events`, MCP `count_events`, `get_pipeline`, `configure_pipeline_module` |
| 레시피 흡수 | **완료**(rotation, person-action-count) / coin-flip은 halo-coin 뷰가 실행 | `gamst/recipes/parts.py`. rotation을 Pipeline으로 돌려 합성 선풍기 12 rev/s에서 11.9회전을 셈(테스트) |
| 뷰 모듈(HUD) | **완료**(count, rotation) | `gamst/views.py` |
| 계산 모듈 | **완료**: pose channels, exemplar(hold/sequence DTW), term_events | 합성 데이터 테스트만 했고, 정답 테이크로 측정한 값은 없습니다 |
| 실모델 스모크 | MediaPipe Pose로 실제 영상(10.6초, 316프레임)을 등록하고 실행 | 정답 카운트가 없어서 정확도는 확인하지 않았습니다 |
| SOTA 모델 모듈 | **미착수** | 디스크 여유 공간이 15GB뿐이고, Qwen3.8-27B는 호스팅이 필요합니다. 아래 참고 |

### MediaPipe

- MediaPipe **Tasks**(Pose/Hand/Face Landmarker)는 이미 perception 모듈입니다(`providers/pose.py` 등).
  person-action-count 레시피는 MediaPipe Pose를 실시간 기본값으로 씁니다.
  - 2026년 비교 연구에서 jitter가 가장 낮았고, CPU에서 돕니다.
  - Sapiens2는 정확도 SOTA 옵션으로 추가합니다.
- MediaPipe **Framework**(C++ Packet/Graph/Calculator)를 GAMST의 실행 엔진으로 들이지는 않습니다.
  - 그래프와 타임스탬프 동기화가 GAMST의 SensorStreams·Pipeline·캡처 시계와 역할이 겹칩니다.
  - 사용자 정의 calculator는 C++/Bazel 빌드가 필요합니다. 그런데 SOTA 모델(Sapiens2, Qwen3-ASR, w2v-BERT)은 PyTorch라서 그래프 안에 넣기 어렵습니다.
  - 대신 MediaPipe 그래프 하나를 통째로 모듈 하나로 감쌀 수 있습니다(Tasks가 바로 그 형태입니다).

### 저주파 semantic lane

`Qwen/Qwen3.8-27B`(Apache-2.0, image-text-to-text)를 1 fps로 돌려서 장면 설명을 타임스탬프 레코드로
context에 함께 넣습니다. 기존 SmolVLM/StreamingVLM lane과 같은 계약을 씁니다.

- 27.8B 파라미터라 bf16 약 56GB, FP8 약 28GB가 필요합니다. 로컬 M2 Pro(32GB, 디스크 여유 15GB)에서는 돌릴 수 없습니다.
- 그래서 OpenAI 호환 엔드포인트(vLLM 호스팅) 어댑터로 연결합니다. 호스팅 비용은 시작하기 전에 보고합니다.

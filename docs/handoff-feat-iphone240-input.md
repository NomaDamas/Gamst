# 인계: coin-flip을 Pipeline으로 (브랜치 `feat/iphone240-input`)

목표: GAMST 240 iPhone 수신기, take 기록, take 뒤 카운트를 Pipeline 부품으로 옮깁니다. 카운트 방식은 바꾸지 않습니다.
owner 규칙은 `wt-speaker-pose/docs/handoff-2026-11-25-pipeline.md` 2절을 따릅니다.

## 구조 (끝남)

```text
camera (gamst240: IPhone240Input) -> record (TakeRecorder) --take(파생 입력)--> count (TakeFlips) -> vision.rotation.refined
                                     find(yoloe), follow(efficienttam)은 count 안의 단계("in": "count")
view: halo-coin (HaloCoinView). 레코드와 모듈 상태를 읽어 그리기만 합니다.
```

| 파일 | 내용 |
|---|---|
| `gamst/inputs.py` | GAMST 240 wire format을 `rotation_live.py`에서 옮김(`parse_header`, `read_recording`, `decode`, 좌표 변환). `PhoneFrame`(지연 디코드), `PhoneEvent`(연결 끊김), `IPhone240Input`(TCP 수신 또는 `raw.gmst` 재생, `clock()`, `state()`, `command(record/camera/zoom/crop)`) |
| `gamst/takes.py` | `VideoSink`, `Take`(옮김), `TakeRecorder`, `TakeFlips`(옛 `start_refine`/`analyze_last`/`collect_refined`), `refine_take`, refine CLI(옮김, 내용 그대로). `python -m gamst.takes refine|refine-server|render` |
| `gamst/halo.py` | `Text`, `Hud`, `render_refined`(옮김), `HaloCoinView`(키·마우스 → 폰/카운터 명령, `hud.mp4` 기록) |
| `gamst/demos/rotation_live.py` | 옛 이름을 다시 내보내는 얇은 파일. `run(args)`은 `recipes.from_rotation_demo` → `recipes.run_pipeline` |
| `gamst/recipes/__init__.py` | `demo_argv` 삭제, `from_rotation_demo` 추가, `build_pipeline`에 gamst240 입력·`"in"` 단계·컨텍스트 키 추가, `run_pipeline`에 halo 뷰와 `idle` 태스크. `run` 옵션: `--record-dir --refine-show --refine-cached --all-throws` |
| `gamst/recipes/parts.py` | `take-recorder`, `flip` 빌더(`_take_recorder`, `_flip`). setup 코드는 안 건드림 |
| `gamst/recipes/coin-flip.json` | 모듈 `record`, `find`/`follow`(`"in": "count"`), `count` |

동작 차이(의도함):

- 같은 초에 두 take가 생기면(재생) 예전엔 폴더가 덮어써졌습니다. 지금은 `take-…-2`로 둘 다 남습니다.
- 재생이 끝나면 이미 돌고 있는 카운트도 기다립니다(예전 replay 루프와 같음).
- `gamst recipe run coin-flip --headless`가 이제 정말 창 없이 돕니다. 예전엔 `--headless`가 데모로 넘어가지 않았습니다.
- 레코드 시각 매핑의 기준점이 "첫 take 프레임"에서 "입력의 첫 프레임"으로 바뀌었습니다. 식은 같습니다(`base + round((ms/1000 - first)*1000)`).

## 테스트 (2026-11-25 측정)

- `tests/test_gamst240.py` 10개 통과: wire format, 재생, 디코드, 실제 TCP 수신(take 프레임 유지, 끊김 이벤트, crop 명령), 기록+카운트(refine은 mock), stop 모드, 모듈 끄기, `recipe run`+HUD, `demo rotation` 얇은 경로.
- `tests/test_recipes.py`: 옛 `demo_argv` 테스트를 Pipeline 테스트로 바꿈.
- 전체: `python -m unittest discover -s tests -q` → 1174개, errors 15, skipped 40. 오류 15개는 main에도 있는 것(test_stereo_context 7, test_stereo_metric 6, test_sam_audio_native 2)입니다.

## 회귀 (끝남, 2026-11-25, M2 Pro, MPS, 같은 가중치)

take `140345` 복사본을 옛 코드(30b705f)와 새 Pipeline으로 headless 재생했습니다. 원본 폴더는 건드리지 않았습니다.

| 토스 | 옛 경로 (flips) | Pipeline (flips) |
|---:|---:|---:|
| 1 | 8.8 | 8.8 |
| 2 | 5.2 | 5.2 |
| 3 | 14.4 | 14.4 |
| 4 | 8.6 | 8.6 |
| 5 | 11.6 | 11.6 |
| 6 | 2.4 | 2.4 |
| 7 | 10.2 | 10.2 |

- 모든 throw: `refined.json`의 results가 필드 단위로 같습니다(프레임별 타원 포함). 저장된 `vision.rotation.refined` 레코드 7개의 flips도 같습니다.
  - 새 쪽은 `gamst recipe run coin-flip --all-throws`와 `gamst demo rotation --iphone-replay … --refine "metal coin"` 둘 다 확인했습니다.
- 레시피 기본값(throw 하나, 모델 유지): 옛 경로와 새 경로 모두 토스 5를 11.4 flips로 셌고, detections는 둘 다 55개입니다.
- 다시 저장된 take의 `raw.gmst` sha256은 원본과 같습니다(88ef534b…).
- 시간(각 1회): every throw는 옛 6분 50초, 새 7분 37초, demo 5분 54초입니다. 기본값은 옛 1분 28초, 새 1분 39초입니다. 새 실행 중에 테스트 스위트가 같이 돌아서 시간 비교는 못 합니다.
- 다시 해보려면: `scratchpad/reg/run_old.sh`(옛 worktree는 지웠으니 `git worktree add --detach … 30b705f`로 다시 만들어야 함), `run_new.sh`, `compare.py OLD_RUN NEW_RUN`.

## 남은 일

- 실제 iPhone으로 live 실행(창, 키, 마우스 drag crop, ANALYZE)은 확인 안 함입니다. TCP 수신은 테스트에서 소켓으로만 확인했습니다.
- `docs/runtime-and-recipes.md` 47행(replay 상태)은 고치지 않았습니다.

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
- `tests.test_recipes` 수정(옛 `demo_argv` 테스트 → Pipeline 테스트).
- 전체 스위트: 아래 "남은 일" 참고(아직 안 돌림).

## 회귀 (진행 중)

원본 take `~/Desktop/gamst-takes/take-20261124-140345`는 건드리지 않았습니다. 복사본과 스크립트:
`/private/tmp/claude-501/-Users-jinminseong-Desktop-catch-your-mind/8a5839a6-55dd-4971-9dc0-28736ef356d3/scratchpad/reg/`

- 옛 코드: `reg/oldwt`(30b705f, detached worktree), `reg/run_old.sh`. 끝나면 `git worktree remove` 해야 합니다.
  - `old-fast`: 옛 레시피 argv(`--refine-on button --refine-fast`) + `--headless`. 끝남(87 s).
  - `old-full`: 모든 throw 카운트(`docs/halo-coin.md` 측정 경로). 진행 중.
- 새 코드(할 일): 같은 모델(`~/.cache/gamst/demo/models`), MPS.
  - `gamst recipe run coin-flip --take reg/copy-new --headless --models M --record-dir reg/new-fast`
  - `gamst recipe run coin-flip --take reg/copy-new --headless --models M --record-dir reg/new-full --all-throws`
- 비교: 각 `take-*/refined.json`의 `results[*].turns`(flips = 2 × turns)를 throw별로 비교합니다. 정확히 같아야 합니다.

## 남은 일

1. 새 경로 회귀 실행과 표 작성.
2. 전체 테스트(`python -m unittest discover -s tests -q`, 기존 오류 15개는 main에도 있음).
3. `docs/halo-coin.md`, `skills/gamst-coin-flip/SKILL.md`, `docs/pipeline-plan.md` 9절 갱신.
4. 회귀용 detached worktree 제거.

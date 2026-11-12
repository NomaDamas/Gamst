# Local settings and guide QA — 2026-11-13

- Python: 150 tests passed, including disabled synchronous/queued model execution,
  shared dashboard/MCP switches and text request gating.
- JavaScript: dashboard/chart rendering, six-tab keyboard navigation, split sizing,
  readable observation names and guide coordinates/visibility passed.
- Dependencies: `pip check` passed.
- Running local server: English → Chinese → Korean → auto → Korean settings
  applied successfully. All nine runtime feature switches turned off and back on.
- Actual Qwen VLM endpoint: synthetic blue square described in 505 ms. The test
  used no camera/microphone, left capture paused and did not change context history.
- Camera guide rendering tests use coordinate fixtures. This is not a new physical
  camera accuracy test or a body/face identity association evaluation.

Detailed local output is in ignored `.local-run/qa/settings-report.json`.
The local server was restarted for this change with devices off.

Actual MediaPipe Pose fixture QA: one body, 33 landmarks, both wrists above shoulders, 29 ms. No physical camera was used for this test.

Native Chrome UI checks: English Settings/Observability, model and guide switches, Advanced MCP, and a real VLM test (397 ms) passed with collection stopped. Interface and speech language selectors are independent.

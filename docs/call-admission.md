# Call admission diagnostics

Call recommendations, reservation and actual dispatch are separate events. Python `CallController.snapshot()` and Gamst/MCP `get_call_status()` expose:

- `desired_route`: the retained policy demand for the original observation.
- `allowed_route`: an advisory route only for a valid, unexpired reservation without runtime, source, freshness or cooldown blockers. Pending requests report `none` and `reservation_required`.
- `dispatched_route`: the route actually admitted for a dispatched request. This is historical admission, not proof of transport completion or semantic correctness.
- `admission_blockers`: checks at snapshot time. Dispatched requests report `already_dispatched`; atomic `dispatch()` still rechecks. A snapshot is not a dispatch token.
- `execution`: selected provider/model revision and prompt/decision hashes for an actual reservation or dispatch. These bounded hashes are not prompt text or provider authentication.
- `last_admission`: one bounded attempt per request, with stage, outcome, evaluated time, fixed reason codes and candidate provenance. A denied candidate is not an executed model.

`reserve()` and `dispatch()` return `status='blocked'` with structured blockers for expected admission failures such as stale capture, cooldown, runtime unavailability, processing-epoch invalidation, wrong route or reservation expiry. Invalid types and malformed or replayed lifecycle operations raise errors rather than being converted into accepted denials. `admission_denials` counts structured denial reasons without retaining subject IDs.

Source rows report `last_semantic_capture_ms` and current `semantic_audit_due`, including when the request queue is empty. Request rows report overdue semantic coverage and unresolved unknown candidates. Text-only completion does not satisfy a video audit. Unknown-candidate expiry produces an explicitly unresolved history entry rather than successful coverage. Bounded history eviction may remove individual entries; omission counts remain explicit.

Capture/inference and consumer Context delivery remain independent of call admission. Source/sensor/capability IDs link controller summaries to `get_sensor_status`, while Context delivery remains caller-selected. These diagnostics do not infer FPS, adjust cadence, measure end-to-end latency or establish model quality.

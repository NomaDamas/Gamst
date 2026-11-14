# Call admission diagnostics

Call recommendations, budget reservation and actual dispatch are separate events. Python `CallController.snapshot()` and HumanSense/MCP `get_call_status()` now expose:

- `desired_route`: the retained policy demand for the original observation.
- `allowed_route`: an advisory current dispatch route only for a valid, unexpired reservation without runtime/source/freshness/cooldown/ledger blockers. Pending requests always report `none` and `reservation_required`, even if no other blocker exists.
- `dispatched_route`: the route actually admitted for a dispatched request; it is historical admission, not proof of transport completion, semantic correctness or current permission to call again.
- `admission_blockers`: current checks at snapshot time. Dispatched requests report `already_dispatched`. Atomic `dispatch()` still rechecks; a status snapshot is not a dispatch token or spending authorization.
- `execution`: the selected provider/model revision and prompt/decision hashes for an actual reservation/dispatch. These are provenance, not prompt text or authentication.
- `last_admission`: one bounded attempt per request, with stage, outcome, evaluated time, fixed reason codes and proposed candidate provenance. A denied candidate is not an executed model. This historical outcome is separate from current blockers and is replaced on the next admission attempt.

## Explicit denials

`CallController.reserve()` returns `status='blocked'` for expected financial denial, retaining the original event. Stable reasons include `budget_exhausted`, `inflight_limit`, `call_limit`, `ledger_closed`, `reservation_overrun` and `unknown_charge_ceiling`. `maximum_charge=None` explicitly means no verified upper quote: it cannot reserve priced budget. Invalid types, negative amounts and malformed/replayed lifecycle operations still raise errors and are not accepted denials.

Raw `CallBudget` admission raises `BudgetDenied`, a `RuntimeError` subclass with a stable `reason` attribute. The controller translates only that exception to a structured outcome. It does not swallow arbitrary failures or malformed inputs. A ledger overrun blocks dispatch of an already reserved request without silently refunding it; explicit cancellation/release remains available.

Cooldown/staleness/missing-time validity and reservation expiry remain distinct from financial denial. Per-controller `admission_denials` counts reasons across attempts, not unique requests. The map uses fixed reason codes and survives source reset, receipt eviction and data deletion without retaining subject IDs. Each retained receipt preserves its last admission and execution provenance; sensory deletion removes these associated receipts as before. No unbounded denial log is accumulated.

## Coverage and cadence

Source rows report `last_semantic_capture_ms` and current `semantic_audit_due`, including when the request queue is empty. Request rows report overdue semantic coverage and unresolved unknown candidates. Text-only completion does not satisfy a video audit. Unknown-candidate expiry produces an explicitly unresolved receipt rather than successful coverage. Bounded receipt eviction may remove individual history; omission counts remain explicit. These fields report actual application acknowledgements, not model quality or guarantees that all unknown events were detected.

Capture/inference and consumer Context delivery remain independent of call admission. Controller summaries explicitly identify this boundary; source/sensor/capability IDs link to `get_sensor_status`, while Context delivery remains caller-selected. No inferred FPS, automatic cadence adjustment, or combined latency measurement is invented by these diagnostics.

These are the diagnostic portions of #161/#154. The integrated deterministic replay driver and joining its traces to the frozen A/B/C evaluation remain required in #161. Real trained-model #150 and actual quality/cost/latency #57/#135 remain open. No model weights, paid calls or quality-preservation claims are introduced.

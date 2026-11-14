# Optional call budget ledger

`humansense.call_budget.CallBudget` is an opt-in in-memory accounting primitive for #154. It executes no model and does not attach itself to observation delivery. It is not a completed controller, pricing service, authenticated billing feed or durable financial store.

```python
from humansense.call_budget import CallBudget

budget = CallBudget(unit='USD.micro', pricing_revision='provider-prices-v1',
                    ceiling=1000000, max_calls=100, max_inflight=2)
reservation = budget.reserve(
    sequence=1, decision_sha256=decision_hash, evidence_sha256=evidence_hash,
    provider='my.provider', model_revision='model-v1',
    maximum_charge=10000, now_ms=1000, ttl_ms=2000)
budget.dispatch(reservation['token'], now_ms=1001)
# The application performs its separately authorized call here.
# Supply verified full charges, or components=None while charges are unknown.
budget.settle(reservation['token'], components={'input':1500, 'output':2000},
              now_ms=1300)
```

Use exact integer units with a fixed pricing revision or a predeclared local resource proxy. Unknown ceilings are rejected; zero is allowed only when the caller has evidence of a genuinely zero-cost call. `maximum_charge` must cover the complete declared liability, not just expected input tokens. SDK hashes bind caller-provided evidence and model selection, not authenticated provider responses. Price calculation and verification remain application responsibilities.

Each request has a strictly increasing sequence across the entire ledger. Lower/out-of-order sequences are rejected, including after receipt eviction. Concurrent applications should serialize sequence allocation/admission or retry rejected requests with a new sequence. A rejected admission is not dispatched and does not consume the sequence; an accepted then released reservation does consume it and the lifetime call allowance. This conservative accounting bounds all successful reservations, including zero-cost calls. Source epochs do not reset this sequence or financial totals.

Each retry or chained model call requires its own reservation and sequence. Components are `input`, `output`, `cache_read`, `cache_write`, `media`, `text_consumer`, `local_compute`, and `other`. A nonempty component map asserts a **complete verified total**, with omitted categories known to be inapplicable/zero. Partial billing must use `None` until all charges are known. Do not supply a partial known subtotal as final settlement. Aggregate known charges and unresolved dispatched counts are reported separately.

Dispatch must occur before expiry. Release, expiry and close return only undispatched reservations. Dispatched timeout/cancellation/error is not evidence of zero billing: report unknown, retaining the declared liability and active slot until settlement. Repeated unknown reports do not double-count. Close prevents new admission/dispatch but still permits late settlements. There is no general reset/refund operation.

Settlement exceeding the reservation records the full cost and latches an overrun, blocking future admission/dispatch even if the overall ceiling was not crossed. Already dispatched calls can still settle and unused reservations can be released. Financial limits cannot prevent a dishonest/inaccurate provider estimate from producing an overrun; they bound admission against declared ceilings. `available` excludes declared unresolved liabilities and is not a claim that unknown actual charges are bounded in reality.

Configuration properties are read-only. All mutation and snapshots are locked; returned data is copied. Active records are bounded by `max_inflight` (1–1024), completed receipts by `receipt_limit` (1–1024), accepted reservations by `max_calls` (1–1,000,000). Totals and the high-water sequence survive receipt eviction. Times are caller-supplied integer milliseconds on one nondecreasing clock (equal values allowed); clock regression is rejected. Tokens belong to a random ledger scope and retired/foreign tokens are rejected.

The ledger is memory-only: never create a replacement on source reset, reconnect or worker restart and treat outstanding liability as erased. Durable recovery and multi-process budget coordination require application persistence/controller integration; no such guarantee is claimed here. `close()` is irreversible and preserves unresolved liabilities for late accounting. This component is not exposed over MCP yet and does not authorize a recommendation from `CallPolicy` by itself.

Tests cover budget boundaries, concurrent reserve and duplicate settlement, stale/foreign tokens, bounded receipt eviction, unknown charges, expiry, close with outstanding calls, overrun, full charge chains, clock regression and copied outputs. They establish software accounting behavior, not measured VLM savings. #154 controller/queues/API and #57/#135 paired model evaluation remain required.

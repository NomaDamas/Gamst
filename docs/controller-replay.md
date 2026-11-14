# Controller lifecycle replay

This dedicated subprocess exercises actual `HumanSense`, `CallPolicy`,
`CallController`, `CallBudget` and MCP status using synthetic observations.
It verifies lifecycle behavior; it does not invoke models or measure their
quality, cost, streaming latency or cold/warm performance.

```sh
python -m benchmarks.controller_replay < benchmarks/fixtures/controller-replay.json
```

Input is one newline-terminated JSON record, at most 262144 bytes. Only the
first record is consumed. A scenario has exactly `schema_version: 1`, `policy`,
`budget`, `limits`, and `actions`. Policy/budget accept their existing constructor
configuration; limits accept controller source/request/receipt/retention limits.
There are 1–48 actions with nondecreasing integer `at` milliseconds in 0–86400000.
Each action is at most 4096 bytes. Output is at most 65536 bytes; excessive output
fails rather than truncating actions. Split large scenarios explicitly.

Actions are `observe`, `reserve`, `dispatch`, `finish`, `settle`, `cancel`,
`expire`, `pause`, `resume`, `reset`, `delete`, and `close`. See the executable
fixture for fields. Observation rules use the policy's explicit rule inputs.
Observations require name, subject, source, epoch, capture, object value and rules;
optional modality is video/audio/text and `include_visual` defaults to true.
Visual references hash fixture JSON, **not camera bytes**. Audio/text have no
visual reference. Request actions reference an earlier observation name.
`finish.success` and settlement components are supplied synthetic outcomes.
Neither validates an actual provider response or bill.

The driver patches UUID generation and the engine clock process-wide. Never
embed `_replay` in an application or run it alongside real providers. Separate
processes isolate those patches. Virtual time is not latency. Budget units are
fixture units, not automatically currency or energy. The final state is captured
before implicit context-manager cleanup; add an explicit close action to include
close in the trace. Application-owned fixture inputs/record copies remain until
process exit, even after SDK deletion; do not use personal data in fixtures.

Every action, including denied calls and exceptions, contributes to a hash chain.
Trace rows retain action/result hashes, bounded accounting and route summaries,
full controller-state hashes and actual MCP-status hashes. Observation bodies,
prompts and responses are not emitted. Hashes are reproducibility bindings, not
anonymization or proof of model execution. Unknown bills preserve liability;
reset/deletion must not erase financial accounting. An action exception is retained
as an error outcome; an unexpected driver failure exits nonzero.

## Frozen paired collector

The command also accepts the existing `benchmarks.paired_replay` request envelope.
Both frozen system configurations must declare:

```json
{"model":"synthetic-controller-fixture","controller_replay_policy":{"mode":"periodic"}}
```

Include all other required frozen system/runner fields. The system policy replaces
only the scenario policy; the response binds the original input, effective scenario
and complete collector request with separate hashes. Freeze executable, source,
dependencies and configuration as required by the collector protocol.

Collector response `status: success` means the replay driver completed, including
any retained denied/error actions. It never means a model answered correctly.
Measurements explicitly report synthetic evidence, no model execution, null
quality/cost and no quality-preservation proof. Collector wall time measures the
fixture process, not model answer latency. External synthetic assessments can be
joined through `benchmarks.replay_evidence`, but the frozen plan must remain
`evidence: synthetic`; these traces cannot certify measured acceptance.

This is a software bridge, not the complete A/B/C evaluation. Issues #161, #57
and #135 still require all-call/provider accounting, real cold/warm runs, prices
or whole-pipeline resource measurements, independently scored held-out tasks and
paired uncertainty. Never convert synthetic reservations into measured costs or
fixture completion into task-quality scores.

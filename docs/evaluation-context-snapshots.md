# Frozen Context task evidence

`Gamst.create_evaluation_snapshot_publisher(subject_id,
implementation_sha256=...)` issues one publisher and an immutable
`publisher.context_view` for that host, subject and processing generation.
The expected implementation hash must equal
`evaluation_snapshot_implementation_sha256()` from the installed SDK; recording
plans must freeze it before execution.

Pass the view explicitly to `get_context(context_view=...)`. The SDK excludes its
registered task-wrapper capability before selecting observations, models,
evidence, event counts, record counts and trends. This prevents an earlier task's
frozen Context from becoming a later task's sensory evidence. A view from another
host, subject or processing generation is rejected. Pause, reset, deletion,
consent revocation and close invalidate it. The default `context_view=None`
retains normal Context behavior, including visibility of stored wrappers.

`await publisher.publish(envelope)` publishes a registered
`evaluation.context_snapshot.v1` observation. Its value has exactly thirteen
fields: `kind`, `task_id`, `input_sha256`, `task_sha256`,
`source_manifest_sha256`, `mapping_sha256`, `context_sha256`,
`snapshot_timestamp_ms`, `scheduled_offset_ns`, `observed_begin_ns`,
`observed_end_ns`, `watermarks`, and `context`. The kind is
`frozen_context_bundle_not_new_sensor_measurement`. The original Context
timestamp, subject and digest must match; publication preserves that timestamp.
Nested source timestamps, ages and quality remain unchanged. The wrapper is an
estimate about a detached bundle and does not trigger sensor inference.

Context content is limited to 32768 canonical bytes, the complete value to 64KiB,
and the complete registered record to 128KiB. SDK structural limits apply to the
whole record. Each publisher reserves at most 64 unique original task IDs and
4MiB of Context bytes, synchronously before awaiting publication. Failed tasks
cannot reuse their reservation as an implicit retry. Capacity failures reject
the complete bundle; content is never trimmed or republished as fresh.

The returned actual retained observation can be admitted to a `CallController`
and supplied unchanged to `TextContextCalls`. An answer is not reinserted into
the source Context. This API provides evidence isolation and bounded storage;
it does not certify model quality, source authenticity, financial settlement or
the complete recording A/B/C runtime.

Recording execution lineage is explicit per operation. TextContextCalls,
VisualContextCalls, and SmolVLMCalls accept `evaluation_relationship` (`root`,
`retry`, or `escalation`) and `evaluation_predecessor`. The default root call
keeps legacy observer reservation arguments unchanged. Nonroot calls require
an owner-issued RecordingScope and original NativeEntryBinding from the same
recording and scope; the recording session additionally checks the earlier
actual entry and identical retained evidence before reserving. A predecessor
can cross small/strong provider bindings within that scope. No adapter keeps
an ambient previous task or previous entry.

A recording operation ticket exposes `entry_reference()` only after its actual
callback entered. This immutable NativeOperationRef includes exact entry,
worker, source-evidence, and request provenance; direct source callbacks have
no semantic prompt. `semantic_entry_binding()` is available only for actual
semantic entries and can be used for a later explicit retry/escalation. These
references do not certify billing, remote model execution, or output quality.

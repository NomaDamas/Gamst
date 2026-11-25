# Temporal and memory engine contract

`gamst.temporal_context.TemporalContextEngine` is the bounded application
layer over the wire-level `TemporalEvent`/`TemporalPacket` contract. It keeps
the source clock, clock epoch, evidence IDs and revision lineage on every
relation. Clock samples are fit with a bounded linear offset/drift estimate;
uncalibrated clocks remain explicitly uncalibrated and never become facts.

`ContextMemory` separates `current_state`, `recent_timeline`, and
`episodic_memory`. Every retained record is evidence-owned and bounded by
count and byte budgets. Deltas are deterministic and replayable, and
checkpoints include a SHA-256 lineage over the state they contain.

`CheckpointStore` is the optional durable boundary. It writes a bounded,
lineage-checked JSON checkpoint through an fsync-and-rename sequence, scopes
filenames to a hash of the session ID, and exposes explicit load/clear
semantics. It persists context state only; model weights and provider process
state remain runtime-owned.

`query_cutoff_ms` is an internal evidence-causality watermark. It excludes
evidence captured after the query time from support and retrieval. It never
truncates, rewrites, or removes the user's query or user-visible context.

These modules are dependency-free and host-independent. They do not claim
real-device clock accuracy, quality, cost, or production readiness. Those
acceptance gates remain with the cross-runtime integration and the matched
Mac/Linux/Windows/Jetson benchmark plans.

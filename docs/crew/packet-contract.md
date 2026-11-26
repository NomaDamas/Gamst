# Packet contract

`EventPacket` preserves event/stream/subject identity, modality/type, source and monotonic clocks, capture/receive/process times, event intervals, sequence/gap, revision/supersedes, provider/model revision, uncertainty, freshness, evidence IDs/source spans and query cutoff. Packets are JSON-safe and bounded at 128 KiB. Evidence after a query cutoff is rejected.

Serialized packets include a versioned SHA-256 checksum envelope. The digest
covers canonical UTF-8 JSON of the packet body (`ensure_ascii=false`, sorted
keys, compact separators); the `checksum` member is excluded from the covered
body. The checksum, canonical form and `PacketIntegrityCounter` are GAMST's own
(`gamst.temporal`, the same functions `TemporalPacket` uses); `gamst.crew.packet`
only adds the one-event packet shape. `EventPacket.from_bytes()` and `ProcessRuntime.submit()` verify the
envelope before dispatch and reject mismatches. Rejections are bounded in the
runtime health `integrity_rejected` counter. Pre-checksum packets require the
explicit `allow_legacy=True` compatibility switch and are re-emitted with the
current checksum when serialized.

## Optional shared-memory transport

`SharedPacket` provides a bounded local CPU transport for one canonical
`EventPacket`. The control-plane descriptor carries segment name, exact length
and a SHA-256 digest; the receiver checks the digest and packet checksum before
dispatch. Ownership is explicit: the creator closes and unlinks the segment
after acknowledgement or timeout. `make_packet_transfer(...,
prefer_shared_memory=False)` returns the identical packet bytes as a socket
fallback. GPU/CUDA zero-copy and target-host performance remain separate
acceptance work.

## Temporal alignment helpers

GAMST's `gamst.temporal.TemporalAligner` (Crew's copy was dropped; these two
helpers were moved onto it) retains bounded source-clock samples and exposes the
calibration status already carried by the packet contract. Once the minimum
sample count is available, `normalize_timestamp(clock, source_ms)` maps a
source-clock timestamp into the monotonic timeline using the fitted offset and
drift. Before calibration it returns an explicit uncalibrated result instead of
silently treating source time as monotonic.

The aligner also tracks the last accepted sequence per stream. A
`sequence_gap(event)` result reports the expected sequence, observed sequence
and whether the packet's declared `gap` agrees with the observed discontinuity;
it never fabricates missing events. `relation_candidate()` retains both event
clock identities, calibration status and evidence IDs. `contradiction_candidate()`
creates an abstaining candidate with both event values and provenance; it is a
candidate for downstream review, never a truth assertion.

Acceptance is contract-level only: bounded state, deterministic normalization,
explicit uncalibrated status, sequence discontinuity visibility and provenance
preservation. Cross-process clock calibration, contradiction quality and real
host timing remain benchmark/host gates.

# Temporal packet checksum

`gamst.temporal.TemporalPacket` emits a versioned integrity envelope on every
packet. The checksum is operational transport metadata and is never copied into
an event's value, evidence IDs, relation evidence, or answer support.

```json
"checksum": {
  "algorithm": "sha256",
  "version": 1,
  "sha256": "<64 lowercase hexadecimal characters>"
}
```

The digest covers the packet body after canonical JSON encoding: UTF-8,
`ensure_ascii=false`, sorted object keys, compact `,`/`:` separators, and
`allow_nan=false`. The `checksum` member itself is excluded from the covered
body. `TemporalPacket.canonical_bytes()` exposes the covered bytes;
`TemporalPacket.to_bytes()` emits the complete canonical packet.

Receivers call `TemporalPacket.from_bytes()` or `verify_packet_checksum()`
before dispatch. A missing, malformed, unsupported-version, or mismatched
checksum raises `ValueError`. Pass a `PacketIntegrityCounter` to count rejected
packets; its bounded `integrity_rejected` value is operational telemetry and is
not packet evidence. `allow_legacy=True` is an explicit compatibility mode for
pre-checksum packets and returns an unverified result (`False`) from the
verification function. Legacy packets are accepted only when that mode is
selected and are re-emitted with the current checksum when serialized.

The GAMST implementation and focused tests are present here. A reproducible
cross-repository probe is available at
`scripts/verify_cross_repo_checksum.py`. The current probe evidence is recorded
in [`evidence/cross-repo-checksum-2026-11-17.json`](evidence/cross-repo-checksum-2026-11-17.json):
GAMST `20b294f61338411d7f9ed6d49c60d0ff57ff446a`, Crew
`e0d2de1efc17dfa9cacd504657b4867a50a2a170`, and bench
`576494b9dd9bf9469f3145ac7c571b776312195b` all verified the Core-produced
envelope and the shared digest is
`6048cd0cda6b935d1a6a138737ec5c2f011f42fa9333819504475a529384ab63`.
This probe is scoped to these exact implementation heads; later documentation-only
commits do not change the packet implementation. The bench head is the merged
Gamst-bench `main` result for PR #12; the GAMST and Crew heads include the
follow-up browser lifecycle and launcher preflight changes.
It proves canonical-byte agreement and closes the wire-compatibility
portion of #331. Full runtime-manager→bench execution, target-host acceptance,
and independent review remain open; this does not establish model, host,
benchmark, or production readiness.

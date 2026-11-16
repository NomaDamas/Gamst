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

The GAMST implementation and focused tests are present here. Cross-repository
agreement with GamstCrew and Gamst-bench remains an open B4 acceptance item
until their fixtures use the same canonical bytes and digest.

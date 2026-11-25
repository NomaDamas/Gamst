import json
import unittest

from gamst.crew.packet import (
    EventPacket,
    PacketIntegrityCounter,
    canonical_packet_bytes,
    packet_sha256,
    verify_packet_checksum,
)


def packet():
    return EventPacket(
        "e", "s", "u", "vision", "event", "mono", 1, 1, None, None,
        1, 1, 1, False, 0, "fixture", "1", {"text": "한글"},
    )


class PacketIntegrityTests(unittest.TestCase):
    def test_canonical_roundtrip_and_digest(self):
        value = packet().to_dict()
        self.assertEqual(value["checksum"]["sha256"], packet_sha256(value))
        self.assertEqual(EventPacket.from_bytes(packet().to_bytes()), packet())
        self.assertEqual(canonical_packet_bytes(value), canonical_packet_bytes(
            json.loads(packet().to_bytes())
        ))

    def test_tamper_rejected_and_count_bounded(self):
        value = packet().to_dict()
        value["value"]["text"] = "변조"
        counter = PacketIntegrityCounter(limit=1)
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            EventPacket.from_dict(value, counter=counter)
        with self.assertRaises(ValueError):
            EventPacket.from_dict(value, counter=counter)
        self.assertEqual(counter.integrity_rejected, 1)

    def test_checksum_version_and_legacy_are_explicit(self):
        value = packet().to_dict()
        value["checksum"]["version"] = 2
        counter = PacketIntegrityCounter()
        with self.assertRaisesRegex(ValueError, "unsupported temporal packet checksum"):
            EventPacket.from_dict(value, counter=counter)
        self.assertEqual(counter.integrity_rejected, 1)

        legacy = packet().to_dict(include_checksum=False)
        with self.assertRaisesRegex(ValueError, "checksum is required"):
            EventPacket.from_dict(legacy, counter=counter)
        self.assertFalse(verify_packet_checksum(legacy, allow_legacy=True))
        restored = EventPacket.from_dict(legacy, allow_legacy=True)
        self.assertEqual(restored.to_dict()["checksum"]["algorithm"], "sha256")

    def test_noncanonical_current_wire_is_rejected(self):
        value = packet().to_dict()
        raw = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
        self.assertNotEqual(raw, packet().to_bytes())
        with self.assertRaisesRegex(ValueError, "not canonical"):
            EventPacket.from_bytes(raw)


if __name__ == "__main__":
    unittest.main()

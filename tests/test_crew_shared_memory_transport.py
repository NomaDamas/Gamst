import unittest

from gamst.crew import EventPacket
from gamst.crew.shared_memory_transport import (
    make_packet_transfer, read_packet_transfer, read_shared_packet,
)


def packet():
    return EventPacket(
        "evt", "stream", "subject", "vision", "state", "mono", 10, 11,
        None, None, 10, 10, 1, False, 0, "fixture", "rev", {"x": 1},
    )


class SharedMemoryTransport(unittest.TestCase):
    def test_shared_memory_roundtrip_preserves_checksum_and_timestamp(self):
        descriptor, owner = make_packet_transfer(packet())
        self.assertEqual(descriptor["transport"], "shared_memory")
        try:
            restored = read_packet_transfer(descriptor)
            self.assertEqual(restored.to_dict(), packet().to_dict())
        finally:
            owner.close()

    def test_socket_fallback_preserves_identical_packet(self):
        descriptor, owner = make_packet_transfer(packet(), prefer_shared_memory=False)
        self.assertIsNone(owner)
        self.assertEqual(read_packet_transfer(descriptor).to_dict(), packet().to_dict())

    def test_tamper_is_rejected_before_packet_dispatch(self):
        descriptor, owner = make_packet_transfer(packet())
        try:
            descriptor["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                read_shared_packet(descriptor)
        finally:
            owner.close()

    def test_descriptor_rejects_oversized_or_unknown_fields(self):
        with self.assertRaises(ValueError):
            read_packet_transfer({"transport": "shared_memory", "version": 1, "name": "x", "length": 1, "sha256": "0" * 64, "extra": 1})


if __name__ == "__main__":
    unittest.main()

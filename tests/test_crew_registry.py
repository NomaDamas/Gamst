import json
import tempfile
import unittest
from pathlib import Path
from gamst.crew.registry import read_profiles


class Registry(unittest.TestCase):
    def profile(self):
        return {
            "name": "asr",
            "runtime": "darwin",
            "python": "3.11",
            "accelerator": "mps",
            "provider": "qwen-asr",
            "model_revision": "r1",
            "license": "Apache-2.0",
            "lock_sha256": "a" * 64,
        }

    def test_read_and_validate_profile(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "profiles.json"
            p.write_text(json.dumps([self.profile()]))
            profiles = read_profiles(p)
            self.assertEqual(profiles[0].name, "asr")
            self.assertTrue(
                profiles[0].validate_for({"os": "darwin", "accelerator": "mps"})
            )

    def test_duplicate_or_incomplete_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "profiles.json"
            value = self.profile()
            value["license"] = ""
            p.write_text(json.dumps([value]))
            with self.assertRaises(ValueError):
                read_profiles(p)

    def test_python_metadata_and_declared_hash_are_checked(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            lock = root / "lock.txt"
            lock.write_text("locked")
            import hashlib
            value = self.profile()
            value["lockfile"] = "lock.txt"
            value["lock_sha256"] = hashlib.sha256(b"locked").hexdigest()
            p = root / "profiles.json"
            p.write_text(json.dumps([value]))
            profile = read_profiles(p)[0]
            self.assertEqual(profile.verify_artifacts(root), {"lockfile": "verified", "artifact": "unmeasured"})
            value["python"] = "python-three"
            p.write_text(json.dumps([value]))
            with self.assertRaises(ValueError):
                read_profiles(p)[0].validate_for({"os": "darwin", "accelerator": "mps"})

    def test_architecture_metadata_is_enforced(self):
        with tempfile.TemporaryDirectory() as d:
            value = self.profile()
            value["arch"] = "x86_64"
            p = Path(d) / "profiles.json"
            p.write_text(json.dumps([value]))
            descriptor = read_profiles(p)[0]
            with self.assertRaises(RuntimeError):
                descriptor.validate_for({"os": "darwin", "arch": "arm64", "accelerator": "mps"})

    def test_python_metadata_is_enforced_when_host_reports_python(self):
        with tempfile.TemporaryDirectory() as d:
            value = self.profile()
            value["python"] = "3.11"
            p = Path(d) / "profiles.json"
            p.write_text(json.dumps([value]))
            descriptor = read_profiles(p)[0]
            with self.assertRaisesRegex(RuntimeError, "requires Python 3.11"):
                descriptor.validate_for({
                    "os": "darwin", "arch": "arm64", "accelerator": "mps",
                    "python": "3.10.11",
                })


if __name__ == "__main__":
    unittest.main()

import unittest
from gamst.crew.diagnostics import Requirement, check


class Diagnostics(unittest.TestCase):
    def test_current_mac_like_profile_is_incompatible_when_versions_mismatch(self):
        result = check(
            Requirement(arch="arm64", opencv_max_major=4, accelerator="mps"),
            python_version=(3, 10),
            arch="arm64",
            opencv_version="5.0.0",
            accelerator="mps",
        )
        self.assertEqual(result.status, "incompatible")
        self.assertEqual(result.checks["python"], "incompatible")
        self.assertEqual(result.checks["opencv"], "incompatible")

    def test_unmeasured_version_is_unknown(self):
        result = check(Requirement(), python_version=(3, 11), arch="arm64")
        self.assertEqual(result.status, "unknown")


if __name__ == "__main__":
    unittest.main()

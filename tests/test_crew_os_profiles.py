import unittest
from gamst.crew.os_profiles import profile, require_profile, profile_catalog, provider_compatibility, install_diagnostics


class Profiles(unittest.TestCase):
    def test_profile_has_explicit_runtime_fields(self):
        value = profile().descriptor()
        for key in (
            "name",
            "os",
            "arch",
            "accelerator",
            "python",
            "supports_cuda",
            "supports_mps",
        ):
            self.assertIn(key, value)

    def test_requirement_rejects_wrong_environment(self):
        with self.assertRaises(RuntimeError):
            require_profile({"os": "darwin", "accelerator": "mps"}, required_os="linux")

    def test_catalog_covers_supported_profiles_without_acceptance_claims(self):
        catalog = profile_catalog()
        self.assertEqual(set(catalog), {"macos-arm64", "linux-cuda", "linux-cpu", "windows-cpu-directml", "jetson-arm64"})
        self.assertIn("cuda", catalog["linux-cuda"]["accelerators"])

    def test_provider_states_and_preflight_are_explicit(self):
        state = provider_compatibility(provider="onnxruntime", profile_name="linux-cpu", installed=True)
        self.assertEqual(set(state) - {"provider", "profile"}, {"available", "installed", "runtime-compatible", "weight-ready", "benchmark-accepted"})
        diag = install_diagnostics(profile_name="macos-arm64", required_commands=("definitely-missing",), required_modules=("definitely_missing",))
        self.assertFalse(diag["commands"]["definitely-missing"])
        self.assertEqual(diag["host_acceptance"], "unmeasured")


if __name__ == "__main__":
    unittest.main()

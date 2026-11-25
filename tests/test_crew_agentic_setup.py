import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gamst.crew.setup import SetupError, apply_plan, download_weight, make_plan


def host(tier="apple-mps", **changes):
    value = {
        "tier": tier,
        "accelerator": "mps" if tier == "apple-mps" else "cuda" if tier.startswith("cuda") else "cpu",
        "os": "darwin" if tier == "apple-mps" else "linux",
        "arch": "arm64" if tier == "apple-mps" else "x86_64",
        "python": "3.11.9",
        "ram_bytes": 128 * 1024 ** 3,
        "vram_bytes": 96 * 1024 ** 3 if tier == "cuda-ge80gb" else 24 * 1024 ** 3 if tier == "cuda-ge24gb" else 16 * 1024 ** 3 if tier == "cuda-lt24gb" else 0,
        "disk_free_bytes": 512 * 1024 ** 3,
    }
    value.update(changes)
    return value


class Planning(unittest.TestCase):
    def test_mac_rotation_defaults_dependencies_and_totals(self):
        plan = make_plan("rotation", hardware=host())
        self.assertEqual(set(plan["roles"]), {"vision.detect", "vision.rotation"})
        self.assertEqual(plan["roles"]["vision.detect"]["model"], "rfdetr.nano")
        self.assertEqual(plan["roles"]["vision.detect"]["config"]["device"], "mps")
        self.assertGreater(plan["totals"]["disk_bytes"], 0)
        self.assertGreater(plan["totals"]["ram_bytes"], 0)
        self.assertTrue(any(">= 160 fps" in warning for warning in plan["warnings"]))

    def test_override_and_dependency_closure(self):
        plan = make_plan("speech", ["vision.detect=yoloe.26s"], hardware=host())
        self.assertEqual(plan["roles"]["vision.detect"]["model"], "yoloe.26s")
        self.assertIn("audio.vad", plan["roles"])
        self.assertIn("audio.asr", plan["roles"])
        plan = make_plan("speech", ["audio.asr=moonshine.tiny"], hardware=host())
        self.assertEqual(plan["roles"]["audio.asr"]["model"], "moonshine.tiny")

    def test_cuda_tier_defaults(self):
        small = make_plan("scene", hardware=host("cuda-lt24gb"))
        self.assertEqual(small["roles"]["vision.detect"]["model"], "rfdetr.nano")
        self.assertEqual(small["roles"]["vision.semantic_scene"]["model"], "smolvlm.500m")
        large = make_plan("scene,speech", hardware=host("cuda-ge24gb"))
        self.assertEqual(large["roles"]["vision.detect"]["model"], "rfdetr.large")
        self.assertEqual(large["roles"]["vision.semantic_scene"]["model"], "qwen3.5.9b")
        self.assertEqual(large["roles"]["audio.asr"]["model"], "qwen3.asr_1_7b")

    def test_hardware_and_capture_warnings(self):
        plan = make_plan("scene", ["vision.semantic_scene=qwen3.5.9b"], hardware=host())
        self.assertTrue(any("requires CUDA" in warning for warning in plan["warnings"]))
        constrained = host("cuda-lt24gb", vram_bytes=8 * 1024 ** 3)
        plan = make_plan("scene", ["vision.semantic_scene=qwen3.5.9b"], hardware=constrained)
        self.assertTrue(any("at least 24 GiB VRAM" in warning for warning in plan["warnings"]))

    def test_custom_detector_is_hashed_at_plan_time(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "fine-tuned.pth"
            checkpoint.write_bytes(b"local unit-test checkpoint")
            plan = make_plan("rotation", ["vision.detect=custom:" + str(checkpoint)], hardware=host())
            weight = plan["roles"]["vision.detect"]["weights"][0]
            self.assertEqual(weight["sha256"], hashlib.sha256(b"local unit-test checkpoint").hexdigest())
            self.assertEqual(weight["status"], "pinned")

    def test_unknown_goal_role_and_model_are_rejected(self):
        with self.assertRaises(ValueError):
            make_plan("unknown", hardware=host())
        with self.assertRaises(ValueError):
            make_plan("rotation", ["vision.detect=unknown"], hardware=host())


class Artifacts(unittest.TestCase):
    def test_sha_mismatch_refuses_and_removes_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.bin"
            source.write_bytes(b"tampered")
            target = Path(directory) / "models" / "model.bin"
            weight = {"name": "model.bin", "url": source.as_uri(), "sha256": "0" * 64,
                      "size_bytes": 8, "size_source": "estimated"}
            with self.assertRaisesRegex(SetupError, "sha256 mismatch"):
                download_weight(weight, target)
            self.assertFalse(target.exists())
            self.assertFalse(target.with_name("model.bin.partial").exists())

    def test_existing_sha_mismatch_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model.bin"
            target.write_bytes(b"user data")
            with self.assertRaisesRegex(SetupError, "existing file sha256 mismatch"):
                download_weight({"name": "model.bin", "url": target.as_uri(), "sha256": "0" * 64}, target)
            self.assertEqual(target.read_bytes(), b"user data")

    def test_apply_custom_checkpoint_and_idempotent_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "fine-tuned.pth"
            checkpoint.write_bytes(b"real local contract input")
            plan = make_plan("rotation", ["vision.detect=custom:" + str(checkpoint)], hardware=host())
            profile = root / "profile.json"
            prefix = root / "prefix"
            with patch("gamst.crew.setup._install_environment", return_value=(str(root / "python"), "tested")):
                first = apply_plan(plan, prefix=prefix, profile_path=profile, models_dir=root / "models")
                second = apply_plan(plan, prefix=prefix, profile_path=profile, models_dir=root / "models")
            self.assertEqual(first["profile"], second["profile"])
            value = json.loads(profile.read_text())
            self.assertEqual(value["version"], 1)
            self.assertEqual(set(value["roles"]), set(plan["roles"]))
            copied = Path(value["roles"]["vision.detect"]["config"]["path"])
            self.assertEqual(hashlib.sha256(copied.read_bytes()).hexdigest(), hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            self.assertTrue(all(item["cached"] for item in second["artifacts"] if item["name"] == checkpoint.name))

    def test_apply_rejects_unpinned_and_tampered_package_plans(self):
        unpinned = make_plan("depth", hardware=host())
        with self.assertRaisesRegex(SetupError, "no complete pinned download"):
            apply_plan(unpinned, prefix="/unused", profile_path="/unused/profile.json")
        plan = make_plan("rotation", hardware=host())
        plan["install"][0]["packages"].append("unplanned-model==1")
        with self.assertRaisesRegex(SetupError, "planned packages differ|unplanned|install"):
            apply_plan(plan, prefix="/unused", profile_path="/unused/profile.json")


class MCP(unittest.IsolatedAsyncioTestCase):
    """The setup tools live on the one GAMST MCP server and call the same make_plan/apply_plan."""

    async def test_tools_list_and_plan(self):
        from gamst import Gamst
        from gamst.mcp import MCPServer
        async with Gamst(consent=True) as human:
            server = MCPServer(human)
            listed = await server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
            self.assertTrue({"plan_setup", "apply_setup"} <= {tool["name"] for tool in listed["result"]["tools"]})
            with patch("gamst.crew.setup.detect_hardware", return_value=host()):
                plan = await server.call_tool("plan_setup", {"goals": ["rotation"]})
                recipe = await server.call_tool("plan_setup", {"roles": {"vision.rotation": "gamst.rotation"},
                                                               "dependencies": False, "isolation": "current"})
            self.assertIn("vision.rotation", plan["roles"])
            self.assertEqual(plan["roles"]["vision.detect"]["model"], "rfdetr.nano")
            self.assertEqual(set(recipe["roles"]), {"vision.rotation"})
            with self.assertRaisesRegex(ValueError, "exactly one"):
                await server.call_tool("apply_setup", {})
            with patch("gamst.crew.setup.apply_plan", return_value={"profile": "p"}) as apply:
                self.assertEqual(await server.call_tool("apply_setup", {"plan": plan, "models_dir": "/m"}), {"profile": "p"})
            self.assertEqual(apply.call_args.kwargs["models_dir"], "/m")
            await server.close()


class CommandLine(unittest.TestCase):
    def test_gamst_setup_runs_in_process(self):
        from gamst import cli
        output = io.StringIO()
        with patch("gamst.crew.setup.detect_hardware", return_value=host()), patch("sys.stdout", output):
            self.assertEqual(cli._dispatch_setup(["plan", "--goals", "body", "--json"]), 0)
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["roles"]["vision.pose"]["model"], "mediapipe.pose_lite")


class CurrentInterpreter(unittest.TestCase):
    def test_current_isolation_installs_into_this_interpreter_without_base(self):
        import platform
        import sys
        plan = make_plan(roles={"vision.pose": "mediapipe.pose_full"}, dependencies=False, isolation="current",
                         hardware=host(python=platform.python_version()))
        self.assertEqual([entry["environment"] for entry in plan["install"]], ["current"])
        self.assertFalse(any(item.startswith("gamst-sdk") for item in plan["install"][0]["packages"]))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weight = plan["roles"]["vision.pose"]["weights"][0]
            calls = []
            with patch("gamst.crew.setup._run", side_effect=lambda args: calls.append(args)), \
                    patch("gamst.crew.setup.download_weight",
                          side_effect=lambda w, dest: {"name": w["name"], "path": str(dest), "sha256": w["sha256"],
                                                       "bytes": 4, "cached": False}):
                result = apply_plan(plan, prefix=root / "prefix", profile_path=root / "profile.json",
                                    models_dir=root / "models")
            self.assertEqual(calls[0][:3], [sys.executable, "-m", "pip"])
            entry = result["roles"]["vision.pose"]
            self.assertEqual(entry["python"], sys.executable)
            self.assertEqual(entry["config"]["path"], str((root / "models" / "mediapipe.pose_full" / weight["name"]).absolute()))

    def test_plan_for_another_interpreter_is_refused(self):
        import platform
        plan = make_plan(roles={"vision.pose": "mediapipe.pose_full"}, dependencies=False, isolation="current",
                         hardware=host(python="3.99.0"))
        self.assertEqual(plan["install"][0]["python"], platform.python_version())
        plan["install"][0]["python"] = "3.99.0"  # e.g. a plan saved on another machine
        with self.assertRaisesRegex(SetupError, "different interpreter"):
            apply_plan(plan, prefix="/unused", profile_path="/unused/profile.json")


if __name__ == "__main__":
    unittest.main()

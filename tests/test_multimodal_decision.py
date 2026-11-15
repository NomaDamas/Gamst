import unittest

from humansense import HumanSense, SensorMetadata, CapabilitySpec, OutputField
from humansense.mcp import MCPServer
from humansense.schema import timestamp


class MultimodalDecision(unittest.IsolatedAsyncioTestCase):
    async def test_internal_provider_fuses_registered_modalities_into_context(self):
        async with HumanSense(consent=True) as human:
            vision = CapabilitySpec("test.objects", "scene.objects", "video", "fixture.vision", "1",
                (OutputField("objects", "array"),), "frame", freshness_ms=5000)
            audio = CapabilitySpec("test.audio", "audio.events", "audio", "fixture.audio", "1",
                (OutputField("events", "array"),), "frame", freshness_ms=5000)
            human.register_capability(vision)
            human.register_capability(audio)
            now = timestamp() - 100
            await human.push_observation(vision.id, {"objects": [{"label": "cup"}]},
                sensor=SensorMetadata("cam", "rgb", 0), timestamp_ms=now)
            await human.push_observation(audio.id, {"events": [{"label": "speech"}]},
                sensor=SensorMetadata("mic", "array", 0), timestamp_ms=now + 1)
            result = await human.decide_multimodal(timestamp_ms=now + 2, window="5s")
            value = result["observation"]["value"]
            self.assertEqual(value["decision"], "fuse_multimodal_context")
            self.assertEqual(set(value["modalities"]), {"audio", "video"})
            self.assertEqual(len(value["evidence_ids"]), 2)
            context = await human.get_context(window="5s")
            decisions = [v for v in context["observations"].values()
                         if v.get("name") == "decision.multimodal"]
            self.assertEqual(len(decisions), 1)
            self.assertEqual(decisions[0]["value"]["evidence_ids"], value["evidence_ids"])

    async def test_stale_or_single_modality_abstains_and_mcp_can_invoke(self):
        async with HumanSense(consent=True) as human:
            spec = CapabilitySpec("test.objects", "scene.objects", "video", "fixture.vision", "1",
                (OutputField("objects", "array"),), "frame", freshness_ms=10)
            human.register_capability(spec)
            now = timestamp() - 100
            await human.push_observation(spec.id, {"objects": []},
                sensor=SensorMetadata("cam", "rgb", 0), timestamp_ms=now)
            server = MCPServer(human)
            modules = await server.call_tool("list_decision_modules")
            self.assertEqual(modules["modules"][0]["provider_kind"], "internal_perception_module")
            result = await server.call_tool("decide_multimodal", {"timestamp_ms": now + 100, "window": "1s"})
            self.assertTrue(result["observation"]["value"]["abstained"])
            self.assertEqual(result["observation"]["value"]["decision"], "request_additional_modality")
            self.assertFalse("raw" in result["observation"]["value"])

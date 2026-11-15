import json
import time
import unittest
from unittest import mock

from gamst.providers.intern_decision import (
    FIXED_DECISIONS,
    MODEL_PROFILES,
    InternDecisionProvider,
    decision_provider,
)
from gamst.providers.multimodal_decision import MultimodalDecisionProvider


NOW = 1_000_000


def record(record_id, capability, modality, value, timestamp_ms=NOW - 10, freshness_ms=5000, **extra):
    result = {
        "type": "observation.signal",
        "id": record_id,
        "timestamp_ms": timestamp_ms,
        "observation": {
            "modality": modality,
            "capability": {"id": capability, "freshness_ms": freshness_ms},
            "value": value,
        },
    }
    result.update(extra)
    return result


def response(choice, confidence):
    remaining = (1.0 - confidence) / (len(FIXED_DECISIONS) - 1)
    probabilities = {name: remaining for name in FIXED_DECISIONS}
    probabilities[choice] = confidence
    return {
        "model": "fake",
        "answers": {
            "decision": {
                "type": "choice",
                "choice": choice,
                "decision": choice,
                "confidence": confidence,
                "probabilities": probabilities,
                "source": "local",
            },
        },
    }


class FakeModel:
    def __init__(self, result):
        self.result = result
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class InternDecisionTest(unittest.TestCase):
    def provider(self, model, **options):
        provider = InternDecisionProvider(model, **options)
        self.addCleanup(provider.close)
        return provider

    def test_envelope_is_valid_and_evidence_ids_are_host_owned(self):
        model = FakeModel(response("fuse_multimodal_context", 0.9))
        model.result["evidence_ids"] = ["model-injected"]
        model.result["answers"]["decision"]["evidence_ids"] = ["also-injected"]
        provider = self.provider(model)
        records = [
            record("host-video", "test.video", "video", {"objects": [{"label": "cup"}]}),
            record("host-audio", "test.audio", "audio", {"events": ["speech"]}, timestamp_ms=NOW - 9),
        ]
        value = provider.decide(records, now_ms=NOW, input_window_ms=5000)
        self.assertEqual(provider.spec.validate(value), value)
        self.assertEqual(value["decision"], "fuse_multimodal_context")
        self.assertFalse(value["abstained"])
        self.assertEqual(set(value["evidence_ids"]), {"host-video", "host-audio"})
        self.assertNotIn("model-injected", value["evidence_ids"])
        self.assertEqual(value["selected_capabilities"], ["test.audio", "test.video"])
        self.assertEqual(value["evidence_ages_ms"], [9, 10])
        request = model.requests[0]
        self.assertEqual(tuple(request["questions"]["decision"]["criteria"]), FIXED_DECISIONS)
        self.assertNotIn("images", request)
        self.assertNotIn("id", request["state"]["observations"][0])
        self.assertNotIn("evidence_ids", request["state"])

    def test_raw_media_is_never_serialized_to_model(self):
        model = FakeModel(response("escalate_vlm", 0.8))
        provider = self.provider(model)
        records = [
            record("host-video", "test.video", "video", {"objects": []}, raw=b"unbounded-media"),
            record("host-audio", "test.audio", "audio", {"events": []}, samples=b"pcm"),
        ]
        value = provider.decide(records, now_ms=NOW)
        self.assertEqual(value["decision"], "escalate_vlm")
        encoded = json.dumps(model.requests[0])
        self.assertNotIn("unbounded-media", encoded)
        self.assertNotIn("pcm", encoded)
        self.assertNotIn("raw", model.requests[0])
        self.assertNotIn("samples", model.requests[0])
        self.assertNotIn("images", model.requests[0])

    def test_timeout_abstains(self):
        class SlowModel(FakeModel):
            def __call__(self, request):
                time.sleep(0.1)
                return super().__call__(request)

        provider = self.provider(SlowModel(response("escalate_vlm", 0.9)), deadline_ms=5)
        value = provider.decide([
            record("v", "test.video", "video", {}),
            record("a", "test.audio", "audio", {}),
        ], now_ms=NOW)
        self.assertTrue(value["abstained"])
        self.assertEqual(value["decision"], "insufficient_evidence")
        self.assertEqual(value["reason"], "deadline_exceeded")

    def test_low_confidence_invalid_choice_and_schema_failure_abstain(self):
        cases = [
            (response("escalate_vlm", 0.59), "below_min_confidence"),
            (response("escalate_vlm", 0.9), "invalid_model_output"),
            ({"answers": {"decision": {"type": "score"}}}, "invalid_model_output"),
        ]
        cases[1][0]["answers"]["decision"]["choice"] = "model_invented_action"
        for result, reason in cases:
            with self.subTest(reason=reason, result=result):
                provider = self.provider(FakeModel(result))
                value = provider.decide([
                    record("v", "test.video", "video", {}),
                    record("a", "test.audio", "audio", {}),
                ], now_ms=NOW)
                self.assertTrue(value["abstained"])
                self.assertEqual(value["decision"], "insufficient_evidence")
                self.assertEqual(value["reason"], reason)
                provider.spec.validate(value)

    def test_stale_evidence_abstains_without_calling_model(self):
        model = FakeModel(response("fuse_multimodal_context", 0.9))
        provider = self.provider(model)
        value = provider.decide([
            record("old", "test.video", "video", {}, timestamp_ms=NOW - 100, freshness_ms=10),
        ], now_ms=NOW)
        self.assertTrue(value["abstained"])
        self.assertEqual(value["reason"], "stale_evidence")
        self.assertEqual(model.requests, [])

    def test_factory_defaults_to_rules_and_unavailable_runtime_raises(self):
        provider = decision_provider()
        self.assertIsInstance(provider, MultimodalDecisionProvider)
        with mock.patch("gamst.providers.intern_decision._check_runtime",
                        side_effect=RuntimeError("runtime unavailable")):
            with self.assertRaisesRegex(RuntimeError, "runtime unavailable"):
                decision_provider("intern-decision-2b", model_path="/models/intern-decision")

    def test_model_profiles_are_pinned_and_licensed(self):
        self.assertEqual(MODEL_PROFILES["intern-decision-2b"]["model_id"], "internlm/Intern-Decision-2B")
        self.assertEqual(MODEL_PROFILES["intern-decision-2b"]["revision"],
                         "85c5657708a09884ff9fd8d7f57789b410a97b33")
        self.assertEqual(MODEL_PROFILES["intern-decision-4b"]["model_id"], "internlm/Intern-Decision-4B")
        self.assertEqual(MODEL_PROFILES["intern-decision-4b"]["revision"],
                         "38c9dafbbef52dd04bae943c0c6ac918419adfd5")
        self.assertIn("Apache-2.0", InternDecisionProvider.capability().license)


if __name__ == "__main__":
    unittest.main()

import asyncio
import time
import unittest

from humansense import HumanSense, validate_output


def now():
    return time.time_ns() // 1_000_000


class CoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_evidence_references_bounded_while_window_counts_remain_complete(self):
        async with HumanSense(consent=True) as human:
            for i in range(40):
                result = await human.push_text("please clarify this question")
            summary = result["event_summary"]
            self.assertEqual(summary["counts"]["interaction.clarification_requested"], 40)
            self.assertEqual(summary["counts"]["interaction.utterance_repeated"], 39)
            self.assertTrue(summary["evidence_truncated"])
            self.assertEqual(len(result["evidence"]), 16)
            self.assertLessEqual(len(result["recommended_action"]["evidence_ids"]), 8)
            self.assertEqual(result["predictions"]["conversation"]["repeated_utterance_count"], 40)
            events = await human.get_history(type="observation.event")
            repeated = [r for r in events if r["event"] == "interaction.utterance_repeated"]
            self.assertEqual(len(repeated[-1]["evidence"]), 16)
            self.assertTrue(repeated[-1]["observation"]["evidence_truncated"])
            context = await human.get_context()
            self.assertEqual(context["summary"]["event_count"], 79)
            self.assertEqual(context["summary"]["record_count"], 159)
            self.assertEqual(len(context["evidence"]), 16)

    async def test_cleanup_failure_still_closes_other_plugins_and_database(self):
        class Broken:
            async def process(self, frame): return []
            def close(self): raise RuntimeError("broken")
        class Healthy:
            closed = False
            async def process(self, frame): return []
            def close(self): self.closed = True
        human = HumanSense(consent=True)
        healthy = Healthy()
        human.register(Broken()).register(healthy)
        with self.assertRaisesRegex(RuntimeError, "plugin cleanup"):
            await human.close()
        self.assertTrue(healthy.closed)
        import sqlite3
        with self.assertRaises(sqlite3.ProgrammingError):
            human.store.connection.execute("SELECT 1")

    async def test_future_input_rejected_before_mutating_state(self):
        async with HumanSense(consent=True) as human:
            with self.assertRaises(ValueError):
                await human.push_text("future", timestamp_ms=now()+60_000)
            with self.assertRaises(ValueError):
                await human.push_signal("head.yaw", 1, "video", timestamp_ms=now()+60_000)
            self.assertEqual(await human.get_history(), [])

    async def test_high_rate_video_does_not_evict_text_modality(self):
        async with HumanSense(consent=True) as human:
            ts = now()
            await human.push_text("help me", timestamp_ms=ts)
            from humansense.schema import envelope
            for i in range(1100):
                record = envelope("observation.signal", "local-user", ts,
                    observation={"name": "scene.person_count", "value": 1, "modality": "video"})
                human.store.add(record)
            state = await human.get_state()
            self.assertEqual(state["modalities"]["text"], "available")
            self.assertIn("text.utterance", state["signals"])
            result = await human.push_signal("scene.person_count", 1, "video")
            inference = (await human.get_history(type="human.inference"))[-1]
            self.assertEqual(inference["modalities"]["text"], "available")
            self.assertIsNotNone(inference["recommended_action"])

    async def test_metric_change_is_transition_not_per_frame_event(self):
        async with HumanSense(consent=True, window_ms=1000) as human:
            ts = now()
            for i in range(5):
                await human.push_signal("speech.rate", 2, "audio", timestamp_ms=ts-2000+i)
            for i in range(3):
                await human.push_signal("speech.rate", 4, "audio", timestamp_ms=ts+i)
            events = await human.get_history(type="observation.event")
            self.assertEqual(len([r for r in events if r["event"] == "behavior.metric_changed"]), 1)
            direction = dict(human._metric_directions)
            await human.push_signal("speech.rate", 0, "audio", timestamp_ms=ts-100)
            self.assertEqual(human._metric_directions, direction)
    async def test_explicit_clarification_is_grounded_and_uncalibrated(self):
        async with HumanSense(consent=True) as human:
            result = await human.push_text("다시 설명해 주세요")
            self.assertEqual(set(result["predictions"]), {"conversation", "behavior", "consistency"})
            prediction = result["predictions"]["conversation"]["needs_clarification"]
            self.assertEqual(prediction["status"], "observed")
            self.assertIsNone(prediction["probability"])
            self.assertFalse(prediction["calibrated"])
            self.assertTrue(prediction["evidence_ids"])
            self.assertEqual(result["modalities"], {"text": "available", "video": "unavailable", "audio": "unavailable"})
            self.assertEqual(result["recommended_action"]["type"], "ask_clarifying_question")

    async def test_subjects_do_not_share_repetition_claims_or_baselines(self):
        async with HumanSense(consent=True) as human:
            await human.push_text("What?", subject_id="a")
            result = await human.push_text("What?", subject_id="b")
            self.assertEqual(result["predictions"]["conversation"]["repeated_utterance_count"], 0)
            await human.push_claim("birth_year", 1990, subject_id="a")
            result = await human.push_claim("birth_year", 1991, subject_id="b")
            self.assertFalse(result["predictions"]["consistency"]["contradiction_detected"])
            self.assertEqual((await human.get_trends("b"))["metrics"], {})

    async def test_scope_distinguishes_changes_from_conflicting_assertions(self):
        async with HumanSense(consent=True) as human:
            await human.push_claim("location", "Seoul", scope="2026-11-11")
            result = await human.push_claim("location", "Busan", scope="2026-11-12")
            self.assertFalse(result["predictions"]["consistency"]["contradiction_detected"])
            result = await human.push_claim("location", "Seoul", scope="2026-11-12")
            consistency = result["predictions"]["consistency"]
            self.assertTrue(consistency["contradiction_detected"])
            self.assertFalse(consistency["deception_assessed"])
            self.assertEqual(len(consistency["conflicts"][0]["evidence"]), 2)

    async def test_out_of_order_data_does_not_override_current_state(self):
        async with HumanSense(consent=True) as human:
            ts = now()
            await human.push_signal("head.yaw", 20, "video", timestamp_ms=ts)
            old = await human.push_signal("head.yaw", 10, "video", timestamp_ms=ts-1000)
            self.assertIn("out_of_order", old["quality"]["flags"])
            state = await human.get_state()
            self.assertEqual(state["signals"]["head.yaw"]["observation"]["value"], 20)

    async def test_baseline_requires_five_prior_samples(self):
        async with HumanSense(consent=True, window_ms=1000) as human:
            ts = now()
            for i in range(5):
                await human.push_signal("speech.rate", 2, "audio", timestamp_ms=ts-2000+i*10)
            await human.push_signal("speech.rate", 3, "audio", timestamp_ms=ts)
            metric = (await human.get_trends(window=1000, end_ms=ts))["metrics"]["speech.rate"]
            self.assertTrue(metric["baseline_available"])
            self.assertAlmostEqual(metric["relative_change"], .5)
            await human.delete_data()
            await human.push_signal("speech.rate", 3, "audio")
            metric = (await human.get_trends())["metrics"]["speech.rate"]
            self.assertFalse(metric["baseline_available"])
            self.assertIsNone(metric["relative_change"])

    async def test_consent_pause_delete_and_retention(self):
        async with HumanSense() as human:
            with self.assertRaises(PermissionError):
                await human.push_text("hello")
        async with HumanSense(consent=True, retention_hours=1) as human:
            await human.push_text("old", timestamp_ms=now()-7_200_000)
            self.assertEqual(await human.get_history(hours=24), [])
            await human.pause()
            with self.assertRaises(RuntimeError):
                await human.push_text("blocked")
            await human.resume()
            await human.push_text("new")
            await human.delete_data()
            self.assertEqual(await human.get_history(), [])
            self.assertEqual((await human.get_context())["summary"]["record_count"], 0)

    async def test_model_versions_and_low_quality_samples_do_not_mix_baselines(self):
        async with HumanSense(consent=True, window_ms=1000) as human:
            ts = now()
            for i in range(5):
                await human.push_signal("speech.rate", 2, "audio", timestamp_ms=ts-2000+i, model_version="old")
            await human.push_signal("speech.rate", 4, "audio", timestamp_ms=ts, model_version="new")
            metric = (await human.get_trends(window=1000, end_ms=ts))["metrics"]["speech.rate"]
            self.assertFalse(metric["baseline_available"])
            await human.push_signal("head.yaw", 100, "video", quality="insufficient")
            self.assertNotIn("head.yaw", (await human.get_trends())["metrics"])

    async def test_subscribers_are_independent_and_bounded(self):
        async with HumanSense(consent=True, queue_size=1) as human:
            one, two = human.observe(), human.observe()
            first = asyncio.create_task(anext(one))
            second = asyncio.create_task(anext(two))
            await asyncio.sleep(0)
            await human.push_text("help me")
            self.assertEqual((await first)["id"], (await second)["id"])
            await human.push_text("help me")
            await human.push_text("help me")
            self.assertGreater(human.dropped_records, 0)
            await one.aclose()
            await two.aclose()

    async def test_response_delay_uses_explicit_turn_anchor(self):
        async with HumanSense(consent=True) as human:
            ts = now()
            await human.push_text("question", timestamp_ms=ts-1000, role="assistant")
            await human.push_text("answer", timestamp_ms=ts)
            events = await human.get_history(type="observation.event")
            response = [e for e in events if e["event"] == "interaction.response_received"]
            self.assertEqual(response[0]["observation"]["response_delay_ms"], 1000)
            self.assertEqual(response[0]["observation"]["interpretation"], "timing_only")

    async def test_vad_transitions_emit_once_and_energy_is_not_speech(self):
        async with HumanSense(consent=True) as human:
            for value in (False, True, True, False, False):
                await human.push_signal("speech.activity", value, "audio", model_id="test-vad", model_version="1")
            await human.push_signal("audio.energy_active", True, "audio")
            events = await human.get_history(type="observation.event")
            self.assertEqual([r["event"] for r in events], ["speech.started", "speech.stopped"])

    async def test_pause_and_deletion_reset_plugin_buffers(self):
        class Buffered:
            resets = 0
            async def process(self, frame): return []
            def reset(self): self.resets += 1
        async with HumanSense(consent=True) as human:
            plugin = Buffered()
            human.register(plugin)
            await human.pause()
            await human.resume()
            await human.delete_data()
            self.assertEqual(plugin.resets, 2)

    async def test_transcript_routes_to_conversation_and_keeps_model_provenance(self):
        class Transcript:
            name = "asr"
            modalities = ("audio",)
            async def process(self, frame):
                return [{"name": "speech.transcript", "value": "explain again", "model_id": "test-asr", "model_version": "1"}]
        async with HumanSense(consent=True) as human:
            human.register(Transcript())
            result = await human.push_audio([0])
            self.assertIsNotNone(result[-1]["recommended_action"])
            state = await human.get_state()
            self.assertEqual(state["signals"]["speech.transcript"]["models"][0]["id"], "test-asr")

    async def test_deletion_discards_in_flight_model_results(self):
        entered, release = asyncio.Event(), asyncio.Event()
        class Delayed:
            modalities = ("audio",)
            async def process(self, frame):
                entered.set()
                await release.wait()
                return [{"name": "speech.transcript", "value": "private", "model_id": "asr", "model_version": "1"}]
        async with HumanSense(consent=True) as human:
            human.register(Delayed())
            task = asyncio.create_task(human.push_audio([0]))
            await entered.wait()
            await human.delete_data()
            release.set()
            self.assertEqual(await task, [])
            self.assertEqual(await human.get_history(), [])


class ContractTests(unittest.TestCase):
    def test_consent_does_not_accept_truthy_strings(self):
        with self.assertRaises(TypeError):
            HumanSense(consent="false")

    def test_rejects_unvalidated_probability(self):
        for record in ({"probability": .8, "calibrated": False}, {"probability": .8, "calibrated": True}, {"probability": None, "calibrated": True}):
            with self.assertRaises(ValueError):
                validate_output(record)
        validate_output({"probability": None, "calibrated": False})

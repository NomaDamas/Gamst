"""A Crew-launched worker is a Pipeline module (replaces GamstCrew's CoreProviderRuntime test)."""
import asyncio
import sys
import unittest

from gamst import Gamst, SensorMetadata
from gamst.crew.modules import RoleUnavailable, WorkerModule, open_worker, role_module
from gamst.pipeline import Pipeline, PipelineIntake
from gamst.plugins import InputFrame
from gamst.schema import timestamp

CONFIG = {
    "id": "court-cal",
    "reference_id": "view-1",
    "source_id": "cam",
    "sensor_id": "points",
    "input_frame": "camera.image",
    "output_frame": "court.xy",
    "image_size": [640, 480],
    "valid_roi": [0.1, 0.1, 0.9, 0.9],
    "image_to_plane": [[10, 0, 0], [0, 20, 0], [0.5, 0, 1]],
}
POINTS = {"reference_id": "view-1", "image_size": [640, 480],
          "points": [{"id": "contact", "xy": [0.5, 0.5], "on_plane": True}]}


class WorkerAsPipelineModule(unittest.IsolatedAsyncioTestCase):
    async def test_worker_output_is_committed_through_the_pipeline(self):
        async with Gamst(consent=True) as human:
            module = await open_worker(sys.executable, "geometry.ground_plane", CONFIG)
            self.assertIsInstance(module, WorkerModule)
            pipeline = Pipeline(human, "cam")
            pipeline.add_input("points")
            pipeline.add_module("plane", module, consumes=["points"], spec=module.spec, queue=1)
            try:
                sensor = SensorMetadata("cam", "points", 1, "camera.image", calibration_id="court-cal",
                                        stream_epoch="epoch-1")
                await pipeline.feed("points", InputFrame(POINTS, "video", timestamp(), sensor=sensor))
                await pipeline.drain()
                status = pipeline.status()["modules"]["plane"]
                self.assertEqual(status["processed"], 1, status)
                self.assertEqual(status["queue"], 1)
                found = await human.get_observations(capability_id=module.spec.id)
                self.assertEqual(found["returned_count"], 1)
            finally:
                await pipeline.close()
            self.assertTrue(module.worker.closed)

    async def test_profile_roles_without_a_worker_are_reported(self):
        async with Gamst(consent=True) as human:
            with self.assertRaisesRegex(RoleUnavailable, "input role"):
                await role_module(human, "input.video", {"model": "opencv", "config": {}})
            with self.assertRaisesRegex(RoleUnavailable, "no isolated worker"):
                await role_module(human, "audio.asr", {"model": "faster_whisper.small", "python": None,
                                                       "provider_profile": None, "config": {}})
            built = await role_module(human, "vision.rotation", {"model": "gamst.rotation", "python": "/venv/python",
                                                                  "provider_profile": "vision.rotation",
                                                                  "config": {"region_source": "motion"}})
            self.assertTrue(built["in_process"])  # reads committed records, so never a subprocess
            self.assertIsNone(built["spec"])
            self.assertEqual(built["capability"], "vision.rotation")


class Lanes(unittest.IsolatedAsyncioTestCase):
    async def test_a_slow_module_drops_oldest_frames_and_never_blocks_the_input(self):
        release = asyncio.Event()
        seen = []

        class Slow:
            async def process(self, frame):
                await release.wait()
                seen.append(frame.data)
                return []

        class Fast:
            def process(self, frame):
                seen.append(("fast", frame.data))
                return []

        async with Gamst(consent=True) as human:
            pipeline = Pipeline(human, "lanes")
            pipeline.add_input("camera")
            pipeline.add_module("slow", Slow(), consumes=["camera"], queue=2)
            pipeline.add_module("fast", Fast(), consumes=["camera"])
            now = timestamp()
            for index in range(5):
                await pipeline.push("camera", index, timestamp_ms=now + index, modality="video")
            self.assertEqual([item for item in seen if item[0] == "fast"], [("fast", i) for i in range(5)])
            release.set()
            await pipeline.drain()
            status = pipeline.status()["modules"]["slow"]
            # the lane had not run yet: latest two frames win, three were dropped and counted
            self.assertEqual([item for item in seen if not isinstance(item, tuple)], [3, 4])
            self.assertEqual(status["dropped"], 3)
            await pipeline.close()

    async def test_engine_intake_feeds_a_host_fed_input_and_resets_it(self):
        frames = []

        class Recorder:
            resets = 0

            def process(self, frame):
                frames.append(frame.data)
                return []

            def reset(self):
                Recorder.resets += 1

        async with Gamst(consent=True) as human:
            pipeline = Pipeline(human, "profile")
            pipeline.add_input("camera", modality="video")
            pipeline.add_module("recorder", Recorder(), consumes=["camera"])
            human.register(PipelineIntake(pipeline, "camera", "video"))
            await human.push_video("frame-1")
            await pipeline.reset()
            await human.push_video("frame-2")
            self.assertEqual(frames, ["frame-1", "frame-2"])
            self.assertEqual(Recorder.resets, 1)
            with self.assertRaises(ValueError):
                PipelineIntake(pipeline, "missing", "video")
            await pipeline.close()


if __name__ == "__main__":
    unittest.main()

import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from benchmarks.models import run


def args(**overrides):
    values=dict(consent=True,repeats=1,yolo_model=None,onnx_model=None,face_model=None,
                vad_model=None,asr_model=None,image=None,wav=None)
    values.update(overrides)
    return SimpleNamespace(**values)


class ModelBenchmarkTests(unittest.IsolatedAsyncioTestCase):
    async def test_preflight_rejects_missing_input_before_loading_any_model(self):
        with patch('humansense.yolo.UltralyticsYOLOPlugin') as model:
            with self.assertRaisesRegex(ValueError,'require --image'):
                await run(args(yolo_model='model.pt'))
            model.assert_not_called()
        with self.assertRaisesRegex(PermissionError,'consent'):
            await run(args(consent=False))

    async def test_partial_model_initialization_failure_releases_prior_model(self):
        class Plugin:
            name='fixture'
            modalities=('video',)
            closed=False
            async def process(self,frame):return []
            def close(self):self.closed=True
        plugin=Plugin()
        with tempfile.TemporaryDirectory() as td:
            image=Path(td)/'input.jpg'; image.write_bytes(b'fixture')
            with patch('humansense.yolo.UltralyticsYOLOPlugin',return_value=plugin), \
                 patch('humansense.plugins.MediaPipeFacePlugin',side_effect=RuntimeError('model load failed')):
                with self.assertRaisesRegex(RuntimeError,'model load failed'):
                    await run(args(yolo_model='model.pt',face_model='face.task',image=str(image)))
            self.assertTrue(plugin.closed)

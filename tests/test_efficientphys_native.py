import hashlib
import io
import subprocess
import sys
import unittest

from humansense.providers import efficientphys_native as native

try:
    import numpy as np
    import torch
except ImportError:
    np = torch = None


class Configuration(unittest.TestCase):
    def test_import_and_artifact_rejection(self):
        run = subprocess.run([sys.executable, '-c',
            "import humansense.providers.efficientphys_native; import sys; assert not {'torch','numpy'} & sys.modules.keys()"], capture_output=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        with self.assertRaises(ValueError): native.model_class(b'raise RuntimeError("must not execute")')
        for size, depth in ((True, 10), (64, 10), (72, True), (72, 20)):
            with self.assertRaises(ValueError): native._geometry(size, depth)
        with self.assertRaises(ValueError):
            native.EfficientPhysNative.from_local('relative', '/missing', checkpoint_sha256='a'*64)


@unittest.skipIf(torch is None, 'optional NumPy/Torch absent')
class Bridge(unittest.TestCase):
    def test_layout_boundary_duplication_ownership_and_no_normalization(self):
        class Difference(torch.nn.Module):
            def forward(self, data):
                self.seen = data.clone()
                return (data[1:]-data[:-1]).mean((1, 2, 3))[:, None]
        model = Difference().eval()
        runtime = native.EfficientPhysNative(model, image_size=36)
        rgb = np.broadcast_to(np.arange(20, dtype=np.float32)[:, None, None, None], (20,36,36,3)).copy()
        rgb[..., 1] += 100
        value = runtime.predict(rgb)
        np.testing.assert_array_equal(value, [1.]*19+[0.])
        np.testing.assert_array_equal(model.seen[:-1].numpy(), rgb.transpose(0,3,1,2))
        rgb[:] = 0
        self.assertEqual(model.seen[0,1,0,0].item(), 100)
        value[:] = 9
        self.assertEqual(runtime.predict(rgb)[-1], 0)
        runtime.close(); runtime.close()
        with self.assertRaises(RuntimeError): runtime.predict(rgb)

    def test_bounds_and_malformed_outputs(self):
        class Output(torch.nn.Module):
            def forward(self, data): return self.output
        model = Output().eval()
        runtime = native.EfficientPhysNative(model, image_size=36)
        good = np.zeros((10,36,36,3), np.float32)
        for bad in (good.astype(np.float64), good[:9], np.zeros((310,36,36,3),np.float32),
                    np.zeros((10,3,36,36),np.float32), good+np.nan):
            with self.assertRaises(ValueError): runtime.predict(bad)
        for output in (torch.zeros(10),torch.zeros(10,1,dtype=torch.float64),torch.full((10,1),float('nan'))):
            model.output = output
            with self.assertRaises(ValueError): runtime.predict(good)

    def test_strict_weights_and_dataparallel_prefix(self):
        model = torch.nn.Linear(3,1).float()
        def payload(state):
            stream = io.BytesIO(); torch.save(state, stream); return stream.getvalue()
        state = {k:v.detach().clone() for k,v in model.state_dict().items()}
        native._state(model, payload(state), torch)
        native._state(model, payload({'module.'+k:v for k,v in state.items()}), torch)
        for bad in ({'state_dict':state}, {'weight':state['weight']},
                    {'module.weight':state['weight'],'bias':state['bias']},
                    {**state,'weight':torch.zeros(1,3,dtype=torch.float64)},
                    {**state,'bias':torch.full((1,),float('inf'))}):
            with self.assertRaises(ValueError): native._state(model,payload(bad),torch)

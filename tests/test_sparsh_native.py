import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from humansense.providers.sparsh_native import (
    SparshSlipNative, _header, construct, source_graph, strict_state,
)
from humansense.providers.sparsh_sources import STATE_SHAPES

TORCH = all(importlib.util.find_spec(x) for x in ('torch', 'safetensors', 'numpy'))
SOURCE = os.environ.get('GAMST_TEST_SPARSH_SOURCE')


def packed(header, data=b''):
    raw = json.dumps(header).encode()
    return len(raw).to_bytes(8, 'little') + raw + data


class HeaderTests(unittest.TestCase):
    def test_strict_bounded_metadata(self):
        valid = {'weight': {'dtype':'F32', 'shape':[2], 'data_offsets':[0,8]}}
        self.assertEqual(_header(packed(valid, b'\0'*8)), valid)
        for obj in ({}, {'x':[]}, {'x':{'dtype':'F32','shape':[0],'data_offsets':[0,0]}},
                    {'x':{'dtype':'F16','shape':[2],'data_offsets':[0,4]}},
                    {'x':{'dtype':'F32','shape':[2],'data_offsets':[]}},
                    {'x':{'dtype':'F32','shape':[2],'data_offsets':[1,9]}},
                    {'x':{'dtype':'F32','shape':[True],'data_offsets':[0,4]}},
                    {'__metadata__':{'note':'not canonical'}}):
            with self.subTest(obj=obj), self.assertRaises(ValueError):
                _header(packed(obj, b'\0'*8))
        for data in (b'', b'\0'*8, (131073).to_bytes(8,'little')+b' '*131073,
                     packed(valid, b'\0'*9)):
            with self.assertRaises(ValueError):
                _header(data)
        raw=b'{"x":{},"x":{}}'
        with self.assertRaises(ValueError):
            _header(len(raw).to_bytes(8,'little')+raw)

    def test_sources_missing_or_modified_fail_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                source_graph(tmp)
            target=Path(tmp)/'LICENSE.md'
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(b'raise AssertionError("must not execute")')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                source_graph(tmp)
        with self.assertRaises(ValueError):
            source_graph('.')

    def test_head_missing_no_graph_allocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'encoder';p.write_bytes(b'x')
            import hashlib
            with patch('humansense.providers.sparsh_native.construct') as builder:
                with self.assertRaises(FileNotFoundError):
                    SparshSlipNative.from_local(tmp,str(p),str(Path(tmp)/'missing'),
                        encoder_sha256=hashlib.sha256(b'x').hexdigest(),head_sha256='0'*64)
                builder.assert_not_called()


@unittest.skipUnless(TORCH, 'optional existing torch/safetensors/numpy runtime')
class RuntimeTests(unittest.TestCase):
    def test_constructor_contract_and_strict_tiny_state(self):
        import torch
        from safetensors.torch import save
        encoder=Mock();head=Mock()
        encoder.vit_base.side_effect=lambda **kw:torch.nn.Linear(2,3)
        head.SlipForceProbe.side_effect=lambda **kw:torch.nn.Linear(3,2)
        a,b=construct(encoder,head)
        encoder.vit_base.assert_called_once_with(img_size=(320,240),patch_size=16,
            in_chans=6,pos_embed_fn='sinusoidal',num_register_tokens=1)
        head.SlipForceProbe.assert_called_once_with(embed_dim='base',num_heads=12,
            depth=1,attn_pool=True,num_classes=2,dropout=0.0)
        self.assertFalse(a.training);self.assertFalse(b.training)
        original={k:v.clone() for k,v in a.state_dict().items()}
        strict_state(a,save(original))
        for k,v in original.items():self.assertTrue(torch.equal(a.state_dict()[k],v))
        for state in ({'weight':original['weight']},
                      dict(original,weight=torch.zeros(2,3)),
                      dict(original,weight=original['weight'].double()),
                      dict(original,weight=torch.full_like(original['weight'],float('nan')))):
            with self.assertRaises(ValueError):strict_state(a,save(state))

    def test_preallocation_rejects_wrong_graph(self):
        import hashlib
        import torch
        from safetensors.torch import save
        payload=save({'wrong':torch.zeros(1)})
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'state';path.write_bytes(payload)
            with patch('humansense.providers.sparsh_native.construct') as builder:
                with self.assertRaisesRegex(ValueError,'keys/shapes'):
                    SparshSlipNative.from_local(tmp,str(path),str(path),
                        encoder_sha256=hashlib.sha256(payload).hexdigest(),
                        head_sha256=hashlib.sha256(payload).hexdigest())
                builder.assert_not_called()

    def test_local_loader_roundtrip_and_nonfinite_before_allocation(self):
        import hashlib
        import torch
        from safetensors.torch import save
        encoder=torch.nn.Linear(2,3).eval();head=torch.nn.Linear(3,2).eval()
        shapes=[{k:list(v.shape) for k,v in m.state_dict().items()} for m in (encoder,head)]
        with tempfile.TemporaryDirectory() as tmp:
            paths=[Path(tmp)/'encoder',Path(tmp)/'head']
            for path,model in zip(paths,(encoder,head)):path.write_bytes(save(model.state_dict()))
            def load():
                return SparshSlipNative.from_local(tmp,*map(str,paths),
                    encoder_sha256=hashlib.sha256(paths[0].read_bytes()).hexdigest(),
                    head_sha256=hashlib.sha256(paths[1].read_bytes()).hexdigest())
            with patch('humansense.providers.sparsh_native.STATE_SHAPES',shapes), \
                 patch('humansense.providers.sparsh_native.source_graph',return_value=(None,None)), \
                 patch('humansense.providers.sparsh_native.construct',return_value=(encoder,head)) as builder:
                runtime=load()
                self.assertEqual(runtime.identity['trained_quality_verified'],False)
                self.assertIs(runtime._encoder,encoder);runtime.close()
                builder.reset_mock()
                state=head.state_dict();state['weight']=torch.full_like(state['weight'],float('inf'))
                paths[1].write_bytes(save(state))
                with self.assertRaisesRegex(ValueError,'nonfinite'):load()
                builder.assert_not_called()

    def runtime(self):
        import torch
        runtime=object.__new__(SparshSlipNative)
        runtime._lock=threading.RLock()
        runtime._encoder=lambda x:x
        runtime._head=lambda x:{'slip':torch.tensor([[1.,2.]]),'force':torch.tensor([[0.,-.5,1.]])}
        return runtime

    def test_input_output_and_closed_bounds(self):
        import numpy as np
        import torch
        runtime=self.runtime();data=np.zeros((1,6,320,240),dtype=np.float32)
        result=runtime.predict(data)
        np.testing.assert_array_equal(result['slip_logits'],[1,2])
        for bad in (data.astype(np.float64),data[:,:,:2],data+2,data+float('nan')):
            with self.assertRaises(ValueError):runtime.predict(bad)
        for out in ({}, {'slip':torch.zeros(1,2),'force':torch.ones(1,3)*2},
                    {'slip':torch.zeros(2),'force':torch.zeros(1,3)},
                    {'slip':torch.full((1,2),float('nan')),'force':torch.zeros(1,3)}):
            runtime._head=lambda x,out=out:out
            with self.assertRaises(ValueError):runtime.predict(data)
        runtime.close();runtime.close()
        with self.assertRaises(RuntimeError):runtime.predict(data)

    def test_close_waits_for_inflight_owned_inference(self):
        import numpy as np
        runtime=self.runtime();entered=threading.Event();release=threading.Event()
        closed=threading.Event();errors=[]
        def encoder(data):
            entered.set()
            if not release.wait(5):raise AssertionError('fixture watchdog')
            return data
        runtime._encoder=encoder
        def predict():
            try:runtime.predict(np.zeros((1,6,320,240),dtype=np.float32))
            except BaseException as exc:errors.append(exc)
        worker=threading.Thread(target=predict);worker.start()
        self.assertTrue(entered.wait(5))
        closer=threading.Thread(target=lambda:(runtime.close(),closed.set()));closer.start()
        self.assertFalse(closed.is_set())
        release.set();worker.join(5);closer.join(5)
        self.assertFalse(worker.is_alive());self.assertFalse(closer.is_alive())
        self.assertEqual(errors,[]);self.assertTrue(closed.is_set())


@unittest.skipUnless(TORCH and SOURCE, 'requires explicit local pinned source; never downloads')
class OfficialGraphTests(unittest.TestCase):
    def test_official_base_architecture_and_actual_forward(self):
        import torch
        before={k:v for k,v in sys.modules.items() if k.startswith('tactile_ssl')}
        encoder_module,head_module=source_graph(SOURCE)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(14)
            encoder,head=construct(encoder_module,head_module)
        self.assertEqual([{k:list(v.shape) for k,v in model.state_dict().items()}
                          for model in (encoder,head)],STATE_SHAPES)
        self.assertEqual(len(encoder.blocks),12)
        self.assertEqual(encoder.patch_embed.proj.in_channels,6)
        self.assertEqual(encoder.patch_embed.num_patches,300)
        self.assertEqual(encoder.norm.eps,1e-6)
        self.assertEqual(head.fc_norm.eps,1e-5)
        self.assertEqual(tuple(head.pooler.query_tokens.shape),(1,1,768))
        with torch.inference_mode():
            latent=encoder(torch.zeros(1,6,320,240))
            output=head({'latent':latent})
        self.assertEqual(tuple(latent.shape),(1,300,768))
        self.assertEqual(tuple(output['slip'].shape),(1,2))
        self.assertEqual(tuple(output['force'].shape),(1,3))
        self.assertTrue(torch.isfinite(output['slip']).all().item())
        self.assertLessEqual(output['force'].abs().max().item(),1)
        self.assertEqual({k:v for k,v in sys.modules.items() if k.startswith('tactile_ssl')},before)
        from safetensors.torch import save
        payload=save(head.state_dict())
        strict_state(head,payload)
        with torch.inference_mode():restored=head({'latent':latent})
        self.assertTrue(torch.equal(restored['slip'],output['slip']))
        self.assertTrue(torch.equal(restored['force'],output['force']))


if __name__=='__main__':unittest.main()

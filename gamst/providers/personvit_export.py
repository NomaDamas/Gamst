"""One-time conversion of the official PersonViT ViT-B/16 MSMT17 re-ID checkpoint to ONNX (the ``vision.body_identity``
default since the owner's decision of 2026-11-27: "무조건 SOTA").

PersonViT (hustvl/PersonViT, code Apache-2.0; weights on the authors' Hugging Face repo ``lakeAGI/PersonViTReID``,
pretrained on LUPerson, which is non-commercial; fine-tuned on MSMT17, research terms) is a TransReID ViT. The network is
built from the official ``transreid_pytorch/model/backbones/vit_pytorch.py`` at a pinned revision (its sha256 is checked;
the file is read, never installed). The checkpoint is read with ``torch.load(weights_only=True)`` (tensors only, nothing
is unpickled), its backbone keys (``base.*``) are loaded strictly, and the test feature of the official config
(``configs/msmt17/vit_base.yml``: ``TEST.NECK_FEAT 'before'``, no camera or view embedding, no JPM) is exported: the
backbone's normed class token, 768-d, for a 256x128 crop normalised with mean 0.5 and std 0.5::

    python -m gamst.providers.personvit_export vit_pytorch.py transformer_120.pth OUT.onnx

It prints the ONNX file's sha256 and the largest difference between the PyTorch and ONNX features on random crops.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json

SOURCE_REVISION = '7dc5b37d09eb99fcf580e424e40509e7bc7d59a6'      # hustvl/PersonViT
HEIGHT, WIDTH = 256, 128


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def build(source, weights, *, source_sha256=None, weights_sha256=None):
    import torch
    for path, expected in ((source, source_sha256), (weights, weights_sha256)):
        if expected is not None and sha256(path) != expected:
            raise ValueError(f'{path} does not match its pinned sha256 {expected}')
    spec = importlib.util.spec_from_file_location('personvit_vit_official', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = torch.load(weights, map_location='cpu', weights_only=True)
    state = state.get('state_dict', state) if isinstance(state, dict) else state
    state = {key[7:] if key.startswith('module.') else key: value for key, value in state.items()}
    base = {key[5:]: value for key, value in state.items() if key.startswith('base.')}
    if any(key.startswith(('fcneck', 'b1.', 'b2.')) for key in state):
        raise ValueError('this checkpoint has a reduced feature or JPM branches; the export covers the plain test feature')
    model = module.vit_base_patch16_224_TransReID(img_size=(HEIGHT, WIDTH), stride_size=[16, 16], drop_path_rate=0.,
                                                  camera=0, view=0, sie_xishu=3.0, pool_method='None', stem_conv=False)
    model.load_state_dict(base, strict=True)
    return model.eval(), sorted({key.split('.', 1)[0] for key in state})


def main(argv=None):
    import numpy as np
    import torch
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('source')
    parser.add_argument('weights')
    parser.add_argument('out')
    parser.add_argument('--source-sha256')
    parser.add_argument('--weights-sha256')
    args = parser.parse_args(argv)
    model, parts = build(args.source, args.weights, source_sha256=args.source_sha256, weights_sha256=args.weights_sha256)
    example = torch.zeros(1, 3, HEIGHT, WIDTH)
    torch.onnx.export(model, example, args.out, input_names=['image'], output_names=['feature'], opset_version=17,
                      dynamic_axes={'image': {0: 'batch'}, 'feature': {0: 'batch'}}, dynamo=False)
    crops = torch.from_numpy(np.random.default_rng(0).normal(size=(4, 3, HEIGHT, WIDTH)).astype(np.float32))
    with torch.inference_mode():
        reference = model(crops).numpy()
    result = dict(out=args.out, sha256=sha256(args.out), checkpoint_parts=parts, feature=int(reference.shape[1]),
                  source_revision=SOURCE_REVISION, weights_sha256=sha256(args.weights), source_sha256=sha256(args.source))
    try:
        import onnxruntime
        session = onnxruntime.InferenceSession(args.out, providers=['CPUExecutionProvider'])
        exported = session.run(None, {'image': crops.numpy()})[0]
        result['max_abs_diff'] = float(np.abs(exported - reference).max())
    except ImportError:
        result['max_abs_diff'] = None
    print(json.dumps(result))


if __name__ == '__main__':
    main()

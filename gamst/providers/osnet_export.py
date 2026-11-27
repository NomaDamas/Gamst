"""One-time conversion of the official OSNet-AIN re-ID checkpoint to ONNX (the ``vision.body_identity`` default).

The weights (``kaiyangzhou/osnet`` on Hugging Face, MIT, sha256 pinned in the registry) are a PyTorch state dict of
torchreid's ``osnet_ain_x1_0``. The network is built from the official source file ``torchreid/models/osnet_ain.py``
at a pinned revision (its sha256 is checked; the file is read, never installed), the state dict is loaded strictly
(only the identity classifier of the training set is left out), and the eval-mode network (the 512-d feature before
the classifier, as torchreid's ``FeatureExtractor`` returns) is exported with a dynamic batch. GAMST then runs the ONNX
file with ONNX Runtime (CPU or CoreML) through :class:`gamst.providers.person_match.ReIDEmbedder`; torch is needed only
here::

    python -m gamst.providers.osnet_export osnet_ain.py WEIGHTS.pth OUT.onnx

It prints the ONNX file's sha256 and the largest difference between the PyTorch and ONNX features on random crops.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json

SOURCE_REVISION = 'f8cd150fdf77e8d9e1ed143b7f308c2c609ded50'      # KaiyangZhou/deep-person-reid
SOURCE_SHA256 = 'a43b495866a4b1645cd134a30c3011592e8c6826ba4781955a135588cf46cd25'
WEIGHTS_SHA256 = '8a07e8da38946f7cee37f4561617bf8b6d2fe8f3a4027852893ea092e46d919f'
HEIGHT, WIDTH = 256, 128


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def build(source, weights):
    import torch
    for path, expected in ((source, SOURCE_SHA256), (weights, WEIGHTS_SHA256)):
        if sha256(path) != expected:
            raise ValueError(f'{path} does not match its pinned sha256 {expected}')
    spec = importlib.util.spec_from_file_location('osnet_ain_official', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = torch.load(weights, map_location='cpu', weights_only=True)
    state = state.get('state_dict', state)
    state = {key[7:] if key.startswith('module.') else key: value for key, value in state.items()}
    classes = state['classifier.weight'].shape[0]
    model = module.osnet_ain_x1_0(num_classes=classes, pretrained=False, loss='softmax')
    model.load_state_dict(state, strict=True)
    return model.eval(), classes


def main(argv=None):
    import numpy as np
    import torch
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('source')
    parser.add_argument('weights')
    parser.add_argument('out')
    args = parser.parse_args(argv)
    model, classes = build(args.source, args.weights)
    example = torch.zeros(1, 3, HEIGHT, WIDTH)
    torch.onnx.export(model, example, args.out, input_names=['image'], output_names=['feature'], opset_version=17,
                      dynamic_axes={'image': {0: 'batch'}, 'feature': {0: 'batch'}}, dynamo=False)
    crops = torch.from_numpy(np.random.default_rng(0).normal(size=(4, 3, HEIGHT, WIDTH)).astype(np.float32))
    with torch.inference_mode():
        reference = model(crops).numpy()
    result = dict(out=args.out, sha256=sha256(args.out), classes=classes, feature=int(reference.shape[1]),
                  source_revision=SOURCE_REVISION)
    try:
        import onnxruntime
        session = onnxruntime.InferenceSession(args.out, providers=['CPUExecutionProvider'])
        exported = session.run(None, {'image': crops.numpy()})[0]
        result['max_abs_diff'] = float(np.abs(exported - reference).max())
    except ImportError:
        result['max_abs_diff'] = None      # checked where ONNX Runtime is installed
    print(json.dumps(result))


if __name__ == '__main__':
    main()

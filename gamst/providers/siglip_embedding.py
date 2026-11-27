"""SigLIP image embeddings of player crops (the Roboflow Sports team split's model), in the worker's environment.

Roboflow Sports (MIT, ``sports/common/team.py``) embeds player crops with ``google/siglip-base-patch16-224``
(Apache-2.0), reduces them with UMAP and clusters them with k-means. Here the worker returns the pooled vision
embedding; :class:`gamst.providers.sports_team.TeamSplit` reduces (PCA) and clusters. Only the pinned
``model.safetensors`` is read: the vision tower's configuration is SigLIP base/16 at 224 px (the defaults of
``transformers.SiglipVisionConfig``), and the pre-processing is that checkpoint's (``preprocessor_config.json``:
resize to 224 x 224, scale to [0, 1], normalise with mean 0.5 and std 0.5), so no unpinned file is needed.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


class SigLIPEmbedder:
    def __init__(self, model, device='cpu'):
        self.model, self.device = model, device

    @classmethod
    def from_local(cls, path, sha256, device='cpu', **_):
        import torch
        from safetensors.torch import load_file
        from transformers import SiglipVisionConfig, SiglipVisionModel
        if not isinstance(sha256, str) or not re.fullmatch('[0-9a-f]{64}', sha256):
            raise ValueError('a lowercase SHA256 of model.safetensors is required')
        path = Path(path).expanduser().resolve()
        path = path / 'model.safetensors' if path.is_dir() else path
        if not path.is_file() or _sha256(path) != sha256:
            raise ValueError('model.safetensors with the pinned SHA256 required')
        model = SiglipVisionModel(SiglipVisionConfig())
        state = {key[len('vision_model.'):]: value for key, value in load_file(str(path)).items()
                 if key.startswith('vision_model.')}
        missing, unexpected = model.vision_model.load_state_dict(state, strict=False)
        if missing:
            raise ValueError(f'the checkpoint lacks SigLIP vision weights: {missing[:3]}')
        model.eval().to(device)
        torch.set_grad_enabled(False)
        return cls(model, device)

    def embed(self, image_bgr):
        import numpy as np
        import torch
        from PIL import Image
        image = np.asarray(Image.fromarray(np.ascontiguousarray(image_bgr[:, :, ::-1])).resize((224, 224), Image.BICUBIC),
                           dtype=np.float32) / 255.
        tensor = torch.from_numpy(((image - .5) / .5).transpose(2, 0, 1)[None]).to(self.device)
        output = self.model(pixel_values=tensor).pooler_output[0]
        return torch.nn.functional.normalize(output, dim=0).cpu().numpy().astype('<f4').tolist()

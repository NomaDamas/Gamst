"""Local UNTRAINED graph wiring check; never a physiological-quality benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time

from humansense.providers.efficientphys_native import EfficientPhysNative, model_class, SOURCE_SHA256
from humansense.providers.vocal_expression import _read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--runtime-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = Path(args.source).resolve(strict=True)
    root = Path(args.runtime_root).resolve(strict=True)
    if not root.is_dir(): raise ValueError('existing runtime directory required')
    blob = _read(source, 16384, SOURCE_SHA256)
    import numpy as np
    import torch
    with torch.device('cpu'), torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        model = model_class(blob)(img_size=72, frame_depth=10).float().cpu().eval()
    # Generated random parameters are test artifacts, not downloaded/trained weights.
    with tempfile.TemporaryDirectory(prefix='gamst-efficientphys-fixture-', dir=root) as directory:
        checkpoint = Path(directory)/'untrained.pth'
        torch.save({'module.'+k:v for k,v in model.state_dict().items()}, checkpoint)
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        runtime = EfficientPhysNative.from_local(str(source), str(checkpoint), checkpoint_sha256=digest)
        try:
            rgb = np.random.default_rng(17).normal(size=(180,72,72,3)).astype(np.float32)
            started = time.monotonic()
            actual = runtime.predict(rgb)
            elapsed = time.monotonic()-started
            # Independent direct graph invocation with official trainer's terminal duplication.
            raw = torch.from_numpy(rgb.transpose(0,3,1,2).copy())
            with torch.inference_mode():
                expected = model(torch.cat((raw,raw[-1:]),0))[:,0].numpy()
            np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
            report = dict(trained_weights=False, quality_acceptance=False, live_capture_latency=False,
                source_sha256=SOURCE_SHA256, fixture_checkpoint_sha256=digest,
                torch_version=torch.__version__, input_frames=180, output_samples=len(actual),
                parameters=sum(x.numel() for x in model.parameters()), exact_direct_graph_parity=True,
                fixture_inference_seconds=elapsed)
        finally:
            runtime.close()
    with Path(args.output).open('x') as out: json.dump(report,out,indent=2)
    print(json.dumps(report))


if __name__ == '__main__': main()

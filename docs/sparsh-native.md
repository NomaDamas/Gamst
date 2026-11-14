# Supplied local Sparsh graph

This bridge runs the exact official ViT-base plus SlipForceProbe on an already
preprocessed `(1,6,320,240)` float32 CPU array in `[0,1]`. It returns two slip
logits and a three-component **unitless normalized delta-force** diagnostic.
`no_slip` does not mean `no_contact`. There is no physical force calibration,
contact classifier, pose estimator, image preprocessing, streaming memory,
LLM call, trained-quality or realtime claim in this slice. #236 owns the paired
input and Context adapter; #226 retains trained-head and real-device acceptance.

## Sources, assets and rights

Official source pin:
[facebookresearch/sparsh fee6a05a](https://github.com/facebookresearch/sparsh/tree/fee6a05a97330eca014b59c71ac41596fe3974a7).
The caller supplies an absolute local source checkout. `sparsh_sources.py` pins
every executed source file, root license hash and exact state shapes. No source
code is vendored. The loader executes the original source in private modules,
resolving only the explicit graph exports; it never imports the upstream eager
training package or registers/replaces `tactile_ssl` in `sys.modules`. All source
hashes are checked before any source executes. This is import isolation, **not a
security sandbox**. Torch/NumPy/einops are trusted installed runtime dependencies.
The original xformers ImportError fallback is selected per private import, without
changing global environment. CPU torch attention is the only supported backend.

The root license is CC-BY-NC 4.0; some DINO-derived files have Apache headers,
and downstream files have their own headers. These facts do not make the entire
upstream package commercially permissive. Review source and separately supplied
encoder/head licenses before use. SDK adapter code retains its repository license.

Provide **two canonical safetensors files**: encoder and trained SlipForceProbe.
Each contains precisely its module's `state_dict()` keys, without `module.`,
`model_encoder.`, `model_task.` prefixes or safetensors metadata. Only contiguous
F32 tensors with exact pinned shapes and finite values are accepted. Header limit
128 KiB, maximum512 tensors, encoder file400 MiB and head32 MiB. Files are read
into bounded immutable bytes and hashes verified; metadata and finite values are
checked before graph construction; strict loading checks the constructed graph
again. These are artifact bounds, not an OS memory limit. Peak memory includes
original bytes, decoded tensors and graph, and can substantially exceed file size.

The official release's internal checkpoint keys were **not** verified. There is
no claim that its `.ckpt` or `.safetensors` file can be passed directly. There is no
legacy pickle loader, automatic conversion, hub request or random-head fallback.
A separately audited conversion must establish provenance and exact canonical
keys; a fabricated head with valid shapes cannot establish trained quality.

```python
from humansense.providers.sparsh_native import SparshSlipNative

native = SparshSlipNative.from_local(
    "/absolute/supplied/sparsh",
    "/absolute/supplied/encoder.safetensors",
    "/absolute/supplied/slipforce.safetensors",
    encoder_sha256="<verified lowercase SHA256>",
    head_sha256="<verified lowercase SHA256>",
)
try:
    result = native.predict(preprocessed_pair)
finally:
    native.close()
```

`predict` and `close` serialize under one lock; close waits for any inference
holding that lock and releases module references. This synchronous primitive
cannot kill an in-flight native call. An async adapter must own and drain its
thread operation through cancellation; do not discard a `to_thread` future and
claim resources released. Source/profile epochs and stale-result rejection are
Context-layer responsibilities. `identity` records hashes, graph, source revision,
selected backend and unverified trained quality. It is descriptive metadata, not
cryptographic runtime attestation.

## Software validation

Existing local runtime tested: Python3.10.11, torch2.8.0, safetensors0.7.0,
einops0.8.2, NumPy2.2.6. No dependency installation or trained-weight download was
performed. Torch2.8.0 is checked on load; the optional `sparsh` dependency group
records the selected versions. A future version requires separate parity evidence.

`python -m unittest discover -s tests -p test_sparsh_native.py -v` checks bounded
metadata, missing/invalid assets, constructor contract, strict tiny fixture states,
input/output validation and owned close. These fixtures do not establish trained
model behavior. Core installations skip optional Torch tests.

With an explicitly supplied source checkout and existing optional dependencies:

```sh
GAMST_TEST_SPARSH_SOURCE=/absolute/supplied/sparsh \
  python -m unittest discover -s tests -p test_sparsh_native.py -v
```

This optional test executes the **actual official base graph** with deterministic
synthetic initialization, checks the pinned shapes, 300 patch tokens, normalization,
query pooling, finite output and native head serialization parity. It never
fetches source or weights. Passing this test is graph/software evidence, not the
trained/contact/physical acceptance required by #226. No cheaper/faster claim is
supported by these tests.

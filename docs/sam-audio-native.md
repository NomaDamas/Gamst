# Local SAM-Audio text/manual temporal adapter

The opt-in `sam_audio.text_temporal.local` profile implements the software path in [#240](https://github.com/NomaDamas/Gamst/issues/240): verified local artifacts → pinned official processor/model → capture-bound metadata and privately owned target/residual bytes. It starts no default worker, downloads no assets and preserves ambient audio. Actual trained-checkpoint compatibility, quality, cold/warm latency and peak RAM/GPU measurements remain [#203](https://github.com/NomaDamas/Gamst/issues/203). Software fixtures do not complete #201/#187/#29.

An application explicitly selects an existing Python **3.11 or later** environment and supplies an [artifact manifest](sam-audio-artifacts.md) with its frozen canonical SHA256. No package/model installation or device access is performed by this adapter. Discovery validates configuration and hashes its resolved identity without importing NumPy, Torch, Transformers or upstream SAM.

```python
from humansense import ProcessProvider

config = {
    "manifest_path": "/absolute/local/sam-manifest.json",
    "manifest_sha256": frozen_manifest_digest,
    "prompt": "the ringing telephone",
    "anchors": [["+", 0.1, 0.3]],  # seconds relative to each complete window
    "device": "cpu",             # cpu | cuda, no automatic device choice
    "ode_steps": 32,              # integer 1..64, midpoint
    "random_seed": 0,             # integer 0..2**63-1
}
owner = human.enable_separation_media(
    subject_id="alice", source_id="mic", sensor_id="device")
provider = await ProcessProvider.open(
    "/absolute/existing/python3.11", "sam_audio.text_temporal.local", config,
    separation_owner=owner)
```

The absolute manifest path, frozen digest, prompt, anchors, resolved defaults, native adapter version and fixed window/output policy bind the capability version. Unknown options fail. Changing the prompt/config requires a newly opened capability and fresh owner. No fixture backend/config option is accepted in production. Follow [private media ownership](sam-audio-media.md) for explicit retrieval, revocation and TTL; Context/API/MCP expose metadata only and install no remote waveform retrieval method.

## Source and runtime identity

The loader requires the complete fixed Python inventory in `sam_audio_source_hashes.py`: 20 SAM package files at `bb4c6999d2677c7402360e426afc01ddfad6dce0`, 43 PE `core` files at `e72b6810b1133e1c879f2cc965d276eb73803f1f`, and 11 DACVAE files at `414c20785fc3a28373073ea8ef7a1316eeeaca6e`. Each fixed SHA256 was derived from the official pinned Git tree after checking the downloaded file's length and Git blob SHA1. A caller's self-consistent inventory or revision string cannot substitute its own code. Legitimate zero-byte Python package files are supported; configs/weights/tokenizers/manifests remain nonempty.

Before any owned upstream import, already-loaded `sam_audio`, `core` or `dacvae` modules are rejected. A worker-local import finder executes verified immutable source snapshots with their explicit origins, intercepts every owned module name and rejects undeclared modules. It never executes adjacent/stale `.pyc` files or reads changed source a second time. It is removed on failure/close; imported upstream modules intentionally require a fresh worker for another load. This does not authenticate third-party native binaries or code already controlling Python.

The manifest must freeze exact installed versions of Torch, torchaudio, torchvision, Transformers, safetensors, huggingface-hub, NumPy, torchdiffeq, torchcodec, timm, einops and descript-audiotools, plus the loaded transitive dependency closure. Exact Python patch and distribution metadata are checked; undeclared loaded third-party roots fail. These checks establish declared identity, not compatibility. The official PE graph includes **both text and vision** even for text-only inference. Its `CLIP.from_config` default does not fetch pretrained PE weights.

The worker sets Hub/Transformers offline flags and rejects Python socket connections, Hub download helpers, Torch download helpers and unsupported imagebind/CLAP/aesthetics imports while constructing or running the model. Span prediction and both rankers are explicitly `None` in effective config, including when omitted in original JSON. This is an application-level guard in an owned worker, **not an OS network sandbox**. Existing required packages/assets must be trusted; no claim covers arbitrary native code or a compromised interpreter.

## Exact local loading and defensive bounds

Original SAM config bytes/hash and effective config/hash remain separate. Effective config sets the verified local T5 root, disabled auxiliaries, explicit 48kHz codec rate/derived hop, and bounded text maximum. Remaining graph settings derive from supplied verified config; no named small architecture is invented.

A worker-local subclass adapts only the official `T5TextEncoder` constructor. It constructs the builtin `T5EncoderModel` from verified local config and reuses the official forward method. The builtin fast tokenizer receives a private worker-owned temporary directory of verified immutable tokenizer/config snapshots with `local_files_only=True`, `trust_remote_code=False`; it never reads mutable original assets. The parent creates its mode0700 directory inside the current working directory, stages at most about128MiB (excluding weights), and removes only its own files after successful startup or after child reap on failure/cancellation, including a killed worker. This requires a writable application working directory. Artifact files are also reverified around construction. Tokenizer settings and auxiliary special-token maps use closed bounded T5 schemas: asset-reference/unknown fields and slow-conversion requests fail, including legacy map keys that Transformers would otherwise merge into constructor arguments. Added-token maps permit only bounded token strings and integer IDs, and every auxiliary file is validated before staging or construction. Explicit builtin type, staged `tokenizer_file`, staged optional `vocab_file`, and `from_slow=False` prevent tokenizer config from redirecting reads outside the verified inventory. Ordinary special-token settings and the exact known Hugging Face unbounded `model_max_length` sentinel are supported; actual forward truncation still uses the explicit effective1–512 maximum. No `T5EncoderModel.from_pretrained`, repo-ID fallback or remote-code entry point is used.

The first supported weight policy is **float32, unsharded, encoder-only T5 safetensors**. Exact encoder keys/shapes/dtypes/finite values are required. `shared.weight` and `encoder.embed_tokens.weight` may use one stored tied alias; when both occur their values must agree. Full T5 decoder/head tensors, shards, unknown omissions and implicit dtype conversions fail. A standard full T5 checkpoint is therefore not claimed supported. This narrow policy still needs actual authorized asset compatibility evidence.

SAM core tensors must exactly cover the constructed graph outside `text_encoder.*`; independently reconciled T5 tensors then complete the exact full graph. Disabled auxiliary tensors and extra/missing/shape/dtype/nonfinite entries fail. Loading explicitly calls `torch.nn.Module.load_state_dict(..., strict=True)` to bypass SAM's permissive override (including its no-load `strict=False` branch).

Graph planning first constructs the actual configuration on Torch's meta device, including mandatory PE tensors. Tensor count is capped at100,000 and prospective parameter/buffer storage at16GiB. Recursive config size/depth/scalar/layer/head bounds prevent unbounded graph loops. Builtin T5 constructor results are checked again after alias/default resolution and before either graph; label-map counts are capped at128 before the config constructor can expand them. Actual codec config values (including pinned constructor defaults) must have1–16 positive bounded encoder/decoder rates with identical product equal to the verified hop; this prevents an unrelated decoder expansion from creating huge waveforms before output validation. CPU graph reconstruction preserves initialized nonpersistent buffers. Unsupported meta construction fails; it has not been demonstrated with an actual small checkpoint in this phase.

SAM weights load from one immutable SHA256-verified ZIP snapshot. Only uncompressed/unencrypted unique Torch ZIP entries are accepted, with bounded entry count, aggregate storage16GiB and pickle metadata32MiB. `weights_only=True` planning runs inside Torch FakeTensorMode, not a plain meta map that can first allocate CPU storage. A scoped weights-only persistent-loader guard reconciles every declared storage count/type with the exact ZIP data record before storage creation. Record offsets identify shared storages; aggregate bytes and each tensor offset/stride extent are checked before CPU materialization. This does not trust FakeTensorMode storage byte estimates (Torch2.8 overstates typed sizes). Only primitive module-version metadata and plain tensor state are accepted; hidden tensor attributes/hooks fail. The same immutable bytes then undergo guarded CPU loading and exact finite validation. This narrow Torch reader requires the tested internal weights-only and checkpoint-offset seams; unsupported runtimes fail without a legacy fallback. T5 uses one immutable verified safetensors snapshot (8GiB maximum), bounded header32MiB and exact float32 geometry before safe CPU materialization. There is no unsafe pickle fallback.

Declared serialized assets also have an aggregate24.5GiB ceiling checked before source/weight snapshots and graph construction. These are defensive ceilings, **not measured memory requirements**. Snapshot bytes, meta bookkeeping, the CPU graph, checkpoint tensors, tokenizer assets and selected-device copy add concurrent costs; a16GiB tensor ceiling is not a16GiB peak RAM guarantee. Runtime feasibility and actual checkpoint semantics remain acceptance work.

## Complete windows and observations

The Context accepts copied batch-one mono little-endian float32 48kHz windows of1–4 seconds, normalized finite input and at most32 bounded nonoverlapping manual anchors. It rejects partial/oversized/multichannel/wrong-rate input without resampling/mixing. Native inference additionally caps codec feature frames at512. Anchors use the [window helper's](sam-audio-windows.md) exact ceil-hop quantization; the actual official processor's sizes/masks/anchor IDs/alignment must match. Construction directly calls `SAMAudioProcessor(hop, 48000)`; no pretrained processor helper is used.

The official `model.separate` receives `predict_spans=False`, `reranking_candidates=1`, midpoint ODE step size `1/ode_steps` and a controlled seed inside a restored RNG scope. Its one-item target/residual lists must contain float32 tensors of the exact padded codec length. Detached CPU arrays pass through shared output validation, reject nonfinite/amplitude beyond16 values without clipping/normalizing, and trim padding to original samples. Residual is the model's own output; it is never derived by subtraction.

The actual immutable request binds subject/source/sensor, capture versus publication sequence, stream epoch, capture interval, calibration/coordinate/channels, input hash, original samples and padding. Lane changes, sequence regression and nonadvancing capture end require reset. Capture gaps are marked discontinuities; SensorStreams' existing reset-on-gap policy revokes retained pairs and starts a new provider epoch. Context reset clears its ordering state. Windows carry no hidden native continuation or buffered partial tail; finish emits nothing.

Metadata records whole-window collection delay (`samples / 48` milliseconds), measured inference time, zero additional lookahead policy, original/effective artifact identities, settings and discontinuity reason. Truncation counts come from bounded untruncated tokenization compared with the exact tokenizer invocation used by official T5 forward. Only counts/hashes/status are published, never prompt text, token IDs or waveform arrays. Model estimates remain quality-unvalidated, with no additive reconstruction, clock verification, causal streaming or seamless cross-window guarantee. Capture/queue/transport timing and peak memory are separate from measured inference time.

## Software verification and limits

Focused tests cover fixed-source and wrong/preloaded-origin rejection, immutable source execution, legitimate empty packages, exact dependency declarations, strict tensor/alias/layout rejection, meta/archive bounds, socket/Hub denial, constructor-only local flags, permissive-loader bypass, actual separate settings, tokenizer truncation and malformed outputs. A tiny synthetic graph is unmistakably a fixture, never a trained model.

The owned subprocess test follows the actual reserved profile, worker constructor, Context, SensorStreams, registered observation and MCP path, replacing only native model loading with a labeled fixture backend. It checks ambient preservation, explicit local retrieval, gap reset/epoch revocation and process cleanup. Existing media tests cover TTL, reset races, cancellation/timeout/reap, independent sensor cadence and metadata-only network subscription; legacy JSON/OpenEB checks remain required.

Optional local parity executes the authenticated official processor source using installed Torch/torchaudio and inert optional import fixtures, without constructing a model:

```sh
GAMST_SAM_PROCESSOR_SOURCE=/absolute/already-present/sam_audio/processor.py \
  python -m unittest discover -s tests -p test_sam_audio_native.py -v
```

The default test skips this parity case when no explicit local source is supplied. A separate test constructs a builtin tokenizer from a synthetic vocabulary, removes its parent-owned snapshot directory, then tokenizes repeatedly; it skips with an explicit reason when installed Transformers is absent/incompatible. CI runs native tensor/constructor fixtures after its existing optional Torch/safetensors installs and Context fixtures in the NumPy job. The development machine has Python3.10, incompatible installed Transformers/Hub versions and missing native dependencies; no packages were installed or repaired. Actual model loading/inference and trained quality/performance remain unclaimed.

Official sources: [SAM model/loader/separate](https://github.com/facebookresearch/sam-audio/blob/bb4c6999d2677c7402360e426afc01ddfad6dce0/sam_audio/model/model.py), [T5 constructor/forward](https://github.com/facebookresearch/sam-audio/blob/bb4c6999d2677c7402360e426afc01ddfad6dce0/sam_audio/model/text_encoder.py), [processor](https://github.com/facebookresearch/sam-audio/blob/bb4c6999d2677c7402360e426afc01ddfad6dce0/sam_audio/processor.py), [PE config construction](https://github.com/facebookresearch/perception_models/blob/e72b6810b1133e1c879f2cc965d276eb73803f1f/core/vision_encoder/pe.py).

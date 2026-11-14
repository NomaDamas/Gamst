# SAM-Audio offline artifact preflight

This is the file/declaration slice of [#234](https://github.com/NomaDamas/Gamst/issues/234), required by native integration [#201](https://github.com/NomaDamas/Gamst/issues/201). It does not load a model, attest upstream source authenticity or establish checkpoint compatibility. Native import-origin/source verification and actual tensor coverage remain required before inference; this slice does not close those requirements.

```python
from humansense.sam_audio_artifact import read_manifest, verify_manifest, verify_runtime

manifest = read_manifest("/absolute/local/manifest.json", expected_sha256=frozen_digest)
checked = verify_manifest(manifest, expected_sha256=frozen_digest)
# Run separately inside the explicitly selected native worker:
runtime = verify_runtime(manifest, expected_sha256=frozen_digest)
```

The expected digest is the canonical JSON SHA256 frozen by the application outside this check. `read_manifest` rejects duplicate keys, nonfinite values and oversized JSON; dictionaries already decoded elsewhere cannot preserve duplicate-key evidence. No command downloads packages or assets. The module has no ML imports.

The exact version1 declaration has `schema_version`, `sources`, `runtime`, `sam`, and `t5`:

- `sources` has `sam_audio`, `perception_models`, `dacvae`. Each has an absolute `root`, the required `revision`, and `files` mapping relative Python paths to SHA256. Pins are respectively `bb4c6999d2677c7402360e426afc01ddfad6dce0`, `e72b6810b1133e1c879f2cc965d276eb73803f1f`, `414c20785fc3a28373073ea8ef7a1316eeeaca6e`. Each group declares1..1024 files, at most4MiB/file and64MiB/group. Only declared files are checked; undeclared import closure and actual executed code are not verified. A matching revision string and caller-supplied hashes do not prove official provenance.
- `runtime` has exact Python patch version `python` (3.11 or later) and1..128 normalized distribution names/exact version strings in `packages`. The worker comparison checks these declared metadata versions, including local suffixes. It does not prove transitive coverage, binary identity, runtime compatibility or memory feasibility.
- `sam` has `config_path`, `config_sha256`, `checkpoint_path`, `checkpoint_sha256`. Config is bounded to128KiB, checkpoint to16GiB. Config bytes are retained as a detached snapshot; weight hashing streams bounded chunks.
- `t5` has an absolute `root` and exact `files` inventory. Required: `config.json`, `tokenizer_config.json`, `tokenizer.json`, `model.safetensors`. Optional: `special_tokens_map.json`, `added_tokens.json`, `spiece.model`. No other files/directories, shards or symlinks are accepted in this first layout. Model weights are bounded to8GiB; the two config files to128KiB; other tokenizer assets to32MiB. Config must declare builtin `t5`; tokenizer class must be builtin `T5Tokenizer` or `T5TokenizerFast`. Remote-code mappings are rejected. Parsing or successful loading of tensor/tokenizer contents is a native-loader responsibility.

Paths must be normalized absolute local paths, with no symlinked parents/files or traversal. Files must be nonempty regular files; FIFOs/devices are rejected before reading. Open-file identity, size and modification/change times are checked around hashing. This detects ordinary replacement/modification during inspection, not adversarial filesystem control. Large weights are **not** immutable snapshots: the native loader must reverify the actual assets it consumes and must not treat this historic preflight as a load-time identity guarantee.

Original config is preserved. Effective config explicitly disables span prediction and both rankers and points the text encoder at the verified local T5 root, including when disable keys were absent. Codec sample rate must be explicit48kHz, encoder-rate product1..48,000, text maximum1..512. Other graph settings are not invented or validated as a named small-model architecture. Original/effective/runtime/T5 digests are returned separately.

`declared_file_bytes_verified` only describes declared content matching at inspection. `imported_source_verified`, `tensor_coverage_verified`, and `runtime_compatibility_verified` stay false. Tests use unmistakable fake file contents to ensure hashes never imply trained-model evidence. Native graph construction, strict SAM/T5 tensor policy, source execution verification and real quality/latency/memory remain open under #201/#203/#234.

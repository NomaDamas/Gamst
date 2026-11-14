# PG-InstructBLIP artifact preflight prerequisite

`python -m humansense.pg_artifact` inspects a frozen local inventory. It does **not**
load a checkpoint, construct a graph, register an inference provider, produce
physical-concept Context, or establish model readiness. #249 is the inspection
software slice. Actual native runtime, captured-object Context and trained-model
acceptance remain required by #242, #105 and #38.

The publisher lists `pgvlm_weights.bin` at24,999,831,084 bytes (about23.28GiB), SHA256
`6b3923683eb83405ca12d81e10a564e7523efe72a1bdf6efa9b493545984bd1e`.
This is publisher metadata; no checkpoint tensor bytes or key schema were inspected
during implementation. No package installation, model download or hardware use
was performed. Stock startup is not a lightweight or proven realtime route.

## Immutable identities and graph boundary

- Official [PG model source](https://huggingface.co/bidiptas/PG-InstructBLIP/tree/7caa1683fbc3f6889868accf25c2074aee83324c)
  is pinned to `7caa1683fbc3f6889868accf25c2074aee83324c`.
- [Historical LAVIS candidate](https://github.com/salesforce/LAVIS/tree/ac8fc98c93c02e2dfb727e24a361c4c309c8dbbc)
  is `ac8fc98c93c02e2dfb727e24a361c4c309c8dbbc`. The authors did not pin it; its date
  does not establish the author runtime or compatibility.
- `humansense.pg_sources.SOURCE_HASHES` lists the exact known PG source and selected
  candidate graph/config/license files. This is **not** the complete executed
  import closure. Matching these hashes alone cannot authorize native execution.
- The official test constructs with Q-Former text enabled, loads its state, then
  sets `qformer_text_input=False`. Constructing withFalse deletes text modules and
  changes the checkpoint key inventory. The manifest fixes constructionTrue and
  inferenceFalse separately; strict full retained-parameter coverage is future work.

The original initialization chain requests BERT tokenizer/config/weights,
Flan-T5-XXL tokenizer/config/weights, EVA weights and configured InstructBLIP weights
before applying the local PG checkpoint. This utility requires separate inventory
roles for every one of those dependencies. Base-asset revisions/digests/licenses
are caller declarations, not authenticated by the PG revision. A complete PG state
might permit an architecture-only path, but that cannot be inferred without actual
safe state inspection and strict graph coverage. No base requirement is silently
skipped on that assumption.

## Frozen schema v1

Top-level keys are exactly `schema_version`, `bundle_root`, `sources`, `artifacts`,
`runtime`, `execution`. `bundle_root` is a normalized absolute local directory.

`sources` has exactly `pg` and `lavis_candidate`. Each contains `revision` and
`files`. File names/known hashes must exactly match `SOURCE_HASHES`; each entry is
`{"path": "relative/path", "size": 123, "sha256": "..."}`. Paths are relative to
bundle_root, with no absolute spelling, traversal, backslashes or repeated/dot
components. Each declared path is unique across the entire manifest.

`artifacts` has exactly these roles:

- `pg_weights`
- `bert_config`, `bert_tokenizer`, `bert_weights`
- `t5_config`, `t5_tokenizer`, `t5_weights`
- `eva_weights`, `instructblip_weights`

Each role has `origin`, `revision`, `license`, `files`. Origins exactly match the
reference chain in `ORIGINS`; revision is a40-character lowercase commit or the
literal `unresolved`. License is a bounded declaration, not a permission verdict.
`files` contains1..64 exact path/size/SHA entries. PG weights must be one file named
`pgvlm_weights.bin` with the exact publisher revision/size/hash above. Synthetic
files cannot be labeled as that trained artifact. Config/tokenizer/base-weight
content semantics and complete tensor/shard/alias coverage remain unverified.

`runtime` contains an exact Python patch version and exact versions for `torch`,
`torchvision`, `transformers`, `timm`, `fairscale`, `omegaconf`, `sentencepiece`,
`accelerate`, `safetensors`, `numpy`, `pillow`. The historical candidate constrains
transformers4.33.2, timm0.4.12 and fairscale0.4.4. Other exact versions are caller
selections requiring future parity evidence. The report compares installed package
metadata without importing those packages. Even a complete version match is not
runtime compatibility. Existing inspected environment mismatches included
transformers4.57.6 and missing timm/fairscale; no downgrade or installation occurred.

`execution` contains exactly:

```json
{
  "construction_qformer_text_input": true,
  "inference_qformer_text_input": false,
  "device": "cpu",
  "precision": "float32",
  "batch_size": 1,
  "image_size": 224,
  "prompt_tokens": 128,
  "max_new_tokens": 32,
  "num_beams": 5,
  "num_captions": 3,
  "use_nucleus_sampling": false,
  "length_penalty": 0,
  "repetition_penalty": 1
}
```

The only other device/precision pair is `cuda`/`reference_mixed`; this names an
unvalidated planned reference route, not GPU support. Prompt tokens1..128,
new tokens1..32, beams2..5, captions1..min(3,beams), integer length penalty-5..5,
integer repetition penalty1..5. Sampling is disabled. These limits are declarations
for future native validation, not proof that tokenization or generation occurred.

## Inspection and safety bounds

Canonicalize and freeze the manifest using `manifest_digest(manifest)` before
calling `inspect_manifest(manifest, expected_sha256=...)` or the actual command:

```sh
python -m humansense.pg_artifact \
  --manifest /absolute/local/manifest.json --sha256 <canonical-manifest-sha256>
```

Default mode reads only the bounded manifest plus installed distribution metadata.
Add `--inspect-files` only when intending to read **all** declared local assets:

```sh
python -m humansense.pg_artifact \
  --manifest /absolute/local/manifest.json --sha256 <canonical-manifest-sha256> \
  --inspect-files
```

Missing files are listed explicitly; a wrong hash, size, type or changed file fails.
No filesystem discovery, cache fallback, network access, pickle/tensor decoding,
`weights_only` load or graph allocation takes place. Strict duplicate/extra fields,
nonfinite JSON, booleans in integer bounds and unknown roles are rejected.

Manifest bound1MiB, total files128, source files1MiB each, config/tokenizer files32MiB,
weight files64GiB, declared aggregate128GiB. The existing audited local artifact
reader streams at most1MiB per read, rejects symlinked ancestors/files and nonregular
files, checks the opened descriptor and before/open/after identities, and detects
normal append/truncate/replacement during inspection. The25GB hash is substantial
IO despite bounded memory. Peak native CPU/VRAM and load cost are **unknown**.

Verification describes the **inspected snapshot of a trusted immutable bundle**.
It is not a lock or protection against later replacement or a hostile writer who
restores metadata. The future native loader must reverify/use verified immutable
owned snapshots; do not reuse this report as execution authorization.

Every report retains `native_ready=false`, `graph_coverage_verified=false`,
`legacy_checkpoint_schema_verified=false`, `runtime_compatibility_verified=false`,
`resource_admission="unavailable"`, `actual_acceptance="incomplete"`. License and
complete import-closure verification also remain false. Source/publisher identity
and caller-declared base assets are explicitly distinguished.

Actual native follow-up requires an isolated, verified exact-source constructor
route with all pretrained downloads replaced, safe supplied checkpoint inspection
and reviewed canonical conversion, complete keys/shapes/dtypes/aliases/buffers,
no random/meta leftovers, exact generation/tokenizer contracts and real reference
parity, resource/lifecycle/held-out/license acceptance. This utility does not satisfy
those requirements and never substitutes a generic caller-injected graph.

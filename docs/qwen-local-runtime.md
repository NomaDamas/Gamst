# Local Qwen ASR worker profile

The `qwen3.asr` ProcessProvider profile connects verified local bundle files to the Qwen streaming adapter. This is an implemented loader/profile with fixture validation, **not a tested Qwen inference deployment**. Actual checkpoint execution and supported-device measurements remain required in #68. No installation or download happens during discovery.

## Bundle contract

Supply an absolute local directory and a `files` mapping of filename to independently trusted lowercase SHA256. Required files are `config.json`, `generation_config.json`, `preprocessor_config.json`, `tokenizer_config.json`, `chat_template.json`, `merges.txt`, `vocab.json` and `model.safetensors`. Optional README.md, LICENSE and .gitattributes must also be declared and hashed if present. This layout follows the [official Qwen3-ASR-0.6B bundle](https://huggingface.co/Qwen/Qwen3-ASR-0.6B/tree/main); sharded, converted, quantized and other layouts need separate profiles and validation.

The loader accepts only a clean flat bundle: no extra files/directories, symlinks, model IDs or URLs. Export the selected files into a dedicated directory rather than pointing at a symlinked shared cache. Do not remove or modify shared caches to satisfy this format. Files must be nonempty; weight size is limited to 8 GiB and other files to 32 MiB each, with model config additionally limited to 1 MiB. All file hashes are checked before optional inference imports. The model configuration must identify Qwen3-ASR's expected architecture. A canonical hash of the filename/hash mapping enters capability version provenance.

Hashes identify the bytes supplied; they do not authenticate an untrusted manifest or validate tensor contents/recognition quality. Keep the directory immutable and trusted throughout loading/execution. Checks are not an OS sandbox against concurrent hostile file replacement. Safetensors parsing and model compatibility still belong to the actual loader. Do not generate an acceptance hash from an unknown file and call that external authenticity evidence.

## Explicit runtime profile

The source-audited candidate profile pins qwen-asr 0.0.6, vLLM 0.14.0 and transformers 4.57.6, checked before import. These versions come from the [audited Qwen source](https://github.com/QwenLM/Qwen3-ASR/blob/7c6daf77a2421100f5fb066495372c00129d39ff/pyproject.toml); their combined device/runtime behavior is not yet validated here.

The owned child receives HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1, disabled HF/vLLM usage telemetry and VLLM_ENABLE_V1_MULTIPROCESSING=0 before startup. The loader selects the same verified directory for model and tokenizer, safetensors-only loading, trust_remote_code=false, eager execution, one sequence, one-device `uni` execution, 128 generated tokens per call and a 0.5 GPU memory utilization target. The latter is an engine setting, not a measured memory guarantee. No forced aligner is loaded. Adapter defaults are 20 ms hops, 2-second decode chunks and a 30-second segment cap.

The [vLLM 0.14 environment source](https://github.com/vllm-project/vllm/blob/v0.14.0/vllm/envs.py) defines the multiprocessing switch. Process-tree cleanup and device compatibility still need real-runtime validation. Offline flags constrain supported library resolution; they are not a general network sandbox for arbitrary installed code. Use a trusted selected Python environment. Direct `from_local` callers must configure the required environment before importing inference libraries and own engine lifetime themselves; the ProcessProvider path supplies those settings and owns child teardown.

```python
from humansense import ProcessProvider

config = {'path': '/absolute/path/to/verified-qwen-bundle',
          'files': trusted_file_hash_mapping}
spec = ProcessProvider.capability('qwen3.asr', config)  # metadata only
# Explicit runtime activation; does not install missing files/dependencies:
# provider = await ProcessProvider.open('/absolute/path/to/runtime/bin/python',
#     'qwen3.asr', config, startup_timeout=120, inference_timeout=30)
```

A deployment may need a larger explicit startup timeout for hashing/loading. Missing packages, mismatched versions, incomplete bundles or unsupported hardware fail; nothing is silently downloaded or restarted. Model/runtime licenses must be reviewed separately. The official model page labels the selected checkpoint Apache-2.0, but the actual manifest/license and dependency distribution remain deployment evidence.

## Validation status

`python3 -m unittest discover -s tests -p test_qwen_artifact.py -v` uses generated non-model files and a labeled loader double: hash identity, discovery, changed/extra/missing-path rejection, symlinks, offline/version guards and exact local loader arguments. It does not deserialize real tensors or start vLLM. Existing process tests cover owned IPC and timeout mechanics, not GPU runtime internals. No weights or new environments were installed.

#67 retains remote MCP/live-API controls and remaining integration. #68 requires real Korean/English streaming, segment-boundary accuracy, process-tree cleanup, sustained latency/resources and native behavior evidence. Neither issue nor the overall Goal is complete from these fixture tests.

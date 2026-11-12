# Optional local SAM3 segmentation

The official `facebook/sam3` model runs in a separate Python worker. It segments objects matching a text prompt (default `person`) and emits normalized bounding boxes, confidence scores and outline polygons. These observations can be drawn over the camera and routed into structured context. They do not supply emotion classifications or identity recognition.

This integration samples independent images, by default once every two seconds. It is not the native SAM3 video tracker. Inference runs off the capture path; the queued provider keeps only the newest pending frame. CPU inference can take longer than the sampling interval; SAM3 throughput is not a camera FPS guarantee.

## Installation

Accept the model's terms and request access at [the official model repository](https://huggingface.co/facebook/sam3), then authenticate locally with `hf auth login`. Never put an access token in chat or source code.

```sh
.venv/bin/python scripts/setup_sam3.py install
.venv/bin/python scripts/setup_sam3.py status
```

The installer checks access before downloading dependencies. It pins the model to the current repository commit and places weights and a separate Transformers 5 runtime under `.local-models/sam3/`. This preserves the main voice runtime's Transformers 4 dependencies. The optional `--device mps` or `--device cuda` selects hardware explicitly; CPU is the default. Only official safetensors/config/tokenizer files are downloaded; inference uses local files with no remote custom code.

`ready.json` means files downloaded and runtime imports verified, not successful inference on this machine. The first enabled request loads the model. A missing gated-model approval remains an installation blocker; the application must not describe SAM3 as installed when that check fails.

## Stream contract

`LocalSAM3Plugin` emits `scene.segments` and `scene.segment_count`, preserving frame timestamp, subject scope, provider name and pinned revision. Each segment contains normalized `bbox`, `score`, `polygons`, and the text `prompt`. Limits are 16 objects, 4 polygons per object and 128 vertices per polygon. These are per-image segments without persistent person identity.

Frames are resized to at most 1024 pixels on either edge and transmitted as JPEG through process pipes; neither original frames nor JPEGs are written to disk. Runtime diagnostics are drained without retaining raw requests. Reset invalidates pending responses and stops only the worker created by this provider. Timeout or malformed output also stops that worker, allowing the next request to restart it.

## Removal

Stop the local dashboard first, then run:

```sh
.venv/bin/python scripts/setup_sam3.py remove --yes
```

Removal verifies checkout ownership and open file handles before deleting only `.local-models/sam3`. Other models and shared Hugging Face caches are untouched.

Implementation follows the official [Transformers SAM3 model API](https://huggingface.co/docs/transformers/model_doc/sam3), using `Sam3Model`, `Sam3Processor` and `post_process_instance_segmentation`.


After installation, restart `scripts/run_local.py`; it discovers `ready.json`
automatically. The Settings model switch controls real SAM3 work; the camera's
SAM3 regions button separately controls drawing. Compact MCP context includes
boxes, scores and prompt; display geometry includes polygons. The dashboard's
SAM3 status reports configured versus successful inference separately.

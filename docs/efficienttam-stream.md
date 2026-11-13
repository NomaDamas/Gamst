# EfficientTAM bounded streaming bridge

gamst's required [mask-tracking family #19](https://github.com/NomaDamas/Gamst/issues/19) needs a video path in addition to sampled SAM image segmentation. `EfficientTAMStream` is the internal state-management component for that path. **It is not yet a registered provider, a verified checkpoint loader, or evidence of real model execution.** Remaining integration and actual acceptance are required in [#127](https://github.com/NomaDamas/Gamst/issues/127).

## Responsibility and boundaries

```mermaid
flowchart LR
    C[Capture ownership and gap checks: integration pending] --> B[EfficientTAMStream: bounded forward state]
    P[Exact-capture seed boxes: integration pending] --> B
    B --> M[Pinned tiny512 predictor: native loader pending]
    M --> O[Boolean current-frame masks]
    O --> S[Context schema and worker integration: pending]
```

The bridge accepts one owned BGR uint8 image per `step`, with axes 32..2048 and at most 1,048,576 pixels. The initial call requires one to four pixel `xyxy` boxes on that image. Subsequent calls cannot add objects or correct old frames. Returns are owned boolean `N×H×W` masks in initial prompt order; IDs 1..N are session-local indices, not recognized identities. Thresholding uses native mask logits `>0`, not calibrated confidence. There is no automatic object detection, presence certainty, reidentification or box reuse after reset.

The outer integration must bind prompts to the exact captured frame and reset on timestamp/sequence gaps, source/epoch changes and dropped frames. This internal component receives no timestamps and therefore **cannot detect these discontinuities itself**. Geometry changes, policy drift, native errors, malformed output and a configurable session limit (default/maximum 1,000,000 frames) invalidate all state. Reset requires fresh initial boxes; close releases the predictor reference and is terminal. The caller must not mutate or concurrently use the supplied predictor.

## Pinned source and state proof

Source target: [EfficientTAM revision abcd061](https://github.com/yformer/EfficientTAM/tree/abcd061ebd3cc6e7527d152d75b890126aaa53f6), Apache-2.0 upstream code; checkpoint provenance/license remains to be verified by the native loader. The integration targets `efficienttam_ti_512x512.yaml`, evaluation mode, mask-memory count 7, temporal stride 1, maximum object pointers 16, past-only pointers and no pointer temporal encoding. Compilation, optimized VOS, hole filling and surrounding-memory clearing must be disabled by the loader. The bridge checks the attention/propagation policy, but does not attest package/source identity or disable an already compiled model.

The official `init_state(video_path)` loads a whole video. This bridge constructs the same state fields for the current image and runs only `propagate_in_video(start_frame_idx=current, max_frame_num_to_track=0, reverse=False)`. In this pinned implementation the endpoint is inclusive: zero means precisely the current frame. The generator is fully consumed and checked for exactly one result before state pruning.

At the end of frame *t*, each object retains the initial conditioning output and non-conditioning outputs *max(1,t−14)..t*. At *t+1*, mask attention requests at most six prior frames and object-pointer attention at most fifteen: all are retained. Absolute indices and `num_frames=t+1` are preserved; they must not be replaced by retained-image count. Prompt records remain at frame zero, temporary outputs must be empty, tracked-frame metadata keeps only the current frame, and image/backbone caches are cleared after propagation. Shared positional constants remain in state. No future image or lookahead is supplied.

This is a bound on retained state references, not a measured CPU/GPU memory ceiling. Native tensors, allocator caches, operator workspaces and weights still consume memory. Actual full-clip parity, long-run memory plateau and throughput require #127. The bridge does not claim realtime FPS or quality equivalence from substituted arithmetic tests.

Preprocessing follows the pinned JPEG loader's arithmetic on predecoded pixels: BGR→RGB, Pillow bicubic 512×512 resize, uint8 division in float64, conversion to float32, then ImageNet mean/std normalization. No JPEG re-encoding is introduced. Compare against the same predecoded pixels for native parity; different video/JPEG decoding can itself change masks.

## Validation and remaining integration

Run `python3 -m unittest discover -s tests -p test_efficienttam_stream.py -v` with NumPy, Pillow and Torch available. Tests exercise 1,000 frames/two objects, all prior-memory addresses needed by the fixed attention policy, cache/metadata bounds, malformed output and late generator errors, fresh-prompt requirements, input limits, policy drift, terminal close and exact preprocessing arithmetic. Model math is substituted. No EfficientTAM package or weights were installed for these tests.

The next implementation must add verified local source/config/checkpoint loading in an isolated runtime, a concrete `ProcessProvider` profile, capture-bound `process_regions` seeds, `needs_prompt` after gaps, compact mask context, SensorStreams/MCP integration and actual model acceptance. Merely accepting an arbitrary predictor here does not satisfy the family deliverable. Existing `sam3.py` remains the sampled image path; neither this internal bridge nor SAM image inference is advertised as completed video tracking.

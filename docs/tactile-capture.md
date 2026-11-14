# Tactile input and capture

This is acquisition and provenance support for required family17, [#224](https://github.com/NomaDamas/Gamst/issues/224). It provides local DIGIT/GelSight Mini frames, a bounded recording reader and registered `tactile.input` observations. It does **not** run Sparsh or infer contact, slip, surface geometry or force. The supplied task-head runtime [#225](https://github.com/NomaDamas/Gamst/issues/225) and actual trained-head/device evidence [#226](https://github.com/NomaDamas/Gamst/issues/226) remain required; [#188](https://github.com/NomaDamas/Gamst/issues/188) and family17 are incomplete.

## Responsibility and bounds

`TactileSource` binds an explicit sensor model/serial/source epoch to native width/height, BGR or RGB order, reference bytes, calibration/illumination identifiers, coordinate frame, clock and subject. These are declarations. Hashing a reference does not prove that it depicts no contact or that a sensor is physically calibrated. `subject_id` is an application owner, not a verified person.

`tactile_frame()` packs raw uint8 pixels with this binding, sequence, timestamp, acquisition read interval, backend identity and SHA256. The strict packet is `GTACT01\n`, a four-byte big-endian JSON header length, an at-most8192-byte header, and exactly H×W×3 bytes. Width≤640 and height≤480; maximum pixels921600bytes. Duplicate/unknown JSON fields, corrupt hashes, noninteger/out-of-bound times, incorrect byte lengths and mismatched InputFrame metadata are rejected. No JPEG, pickle, unsafe deserialization, image codec or model is needed to read a packet.

The source retains the caller's bounded reference bytes. Packets carry its hash, not the reference itself; retain the original local reference separately for a later task model. The bridge never performs color conversion, reference subtraction, normalization, cropping, rotation or resizing. The trained model's preprocessing must be applied separately and verified against these identities.

`TactileWindow` keeps at most six packets. After six consecutive increasing captures it yields `(current, t-5)`; it never duplicates the first frame to synthesize history. A capture sequence gap, a time gap greater than configurable `max_gap_ms` (1..1000, default100), or changes in source/reference/lighting/calibration/clock/geometry reset the window. Reordered/nonincreasing timestamps or sequences clear history and raise. Invalid packets also clear history. Six frames by count do not guarantee the training cadence: task-specific cadence and model validation remain #225. There is no future lookahead or inference scheduler. The caller chooses when to request/submit another frame.

## Explicit native capture

Imports and construction open no devices. `TactileCapture(source, consent=True)` is required, followed by an explicit `open_digit()` or `open_gelsight(device=...)`. The bridge does not probe arbitrary cameras or auto-select a device. The official DIGIT constructor enumerates DIGIT device metadata to resolve the explicitly supplied serial. No automatic dependency/weight download, GUI or recording is performed.

The DIGIT bridge verifies installed `digit_interface.digit` and `digit_handler` Python wrapper hashes against [official revision87a28bbf](https://github.com/facebookresearch/digit-interface/blob/87a28bbf2beee8008a5308e9a12d72e1bc4fefb9/digit_interface/digit.py). This checks those wrappers, not all native binaries or the physical device. It calls `Digit(serial, name)`, `connect`, `set_resolution`, `set_fps`, `set_intensity`, `get_frame(transpose=True)` and `disconnect`. `transpose=True` is intentional: the default method transposes/flips pixels. Native reference geometry must be320×240 or640×480; QVGA permits30/60fps and VGA15/30fps. Requested fps/intensity must match the source's `configured_fps`/`configured_led_intensity`. Driver acceptance is not an achieved-rate guarantee.

GelSight uses the documented [official OpenCV camera API](https://github.com/gelsightinc/gsrobotics/blob/321d6a22da64529138ff10237335038fd8c5189f/utilities/gelsightmini.py): explicit `VideoCapture(device)`, `isOpened`, set width/height, `read`, `release`. It does not import or copy the GUI SDK or its crop/rescale pipeline. Select an explicit numeric local camera index or `/dev/` device path; files/URLs are rejected. This API cannot independently authenticate that the selected webcam is a GelSight device. The caller binds the serial/model; actual device validation remains required. GelSight fps and LEDs are not configured, so their source settings must be `None`. Each returned image must match declared uint8 BGR geometry exactly; no driver resize fallback is applied. Installed OpenCV version and reference API revision accompany packets.

```python
from humansense.providers.tactile import TactileSource
from humansense.providers.tactile_capture import TactileCapture

# local_reference_bgr is caller-supplied raw uint8 BGR bytes, exactly320*240*3.
source = TactileSource(
    source_id='touch-left', sensor_id='finger-left', sensor_model='digit',
    serial='D00000', source_epoch='connection-1', calibration_id='reference-1',
    illumination_id='led15-reference-1', coordinate_frame='digit-native',
    clock_id='host-unix', width=320, height=240, reference=local_reference_bgr,
    configured_fps=30, configured_led_intensity=15,
)
async with TactileCapture(source, consent=True) as capture:
    await capture.open_digit(fps=30, intensity=15)
    tactile = await capture.next_frame()
    # Explicit caller-owned routing: await streams.submit(tactile)
```

Native timestamps are **host Unix-millisecond read completion**, bounded by the host interval around the read. They are not hardware exposure timestamps. A backwards host clock or interval longer than60seconds rejects the packet. A caller-supplied `clock_id` identifies that host clock session; reconnect/reference changes require a new source/session epoch. Read completion and actual inference latency are different measurements.

Pull-based reads serialize under a lock; the bridge holds one native handle and does not spawn a producer queue. Constructor/read/release execute in owned worker tasks. Cancellation drains the native operation and releases a late handle before propagating `CancelledError`; concurrent close prevents late frame publication. Repeated close shares cleanup, and native failures remain in bounded `status()['errors']`. Failed partial startup is cleaned; this session cannot restart. A caller cancelled while merely waiting to acquire the operation lock has not changed ownership. Native blocked I/O cannot be interrupted by asyncio, so shutdown has no hard deadline. Applications requiring one need a supervised process. No new session should reuse the same physical device until the old operation has completed.

## Recording replay and Context

A local absolute JSON manifest contains exact fields:

```json
{"schema_version":1,"packets":[{"file":"touch000.gtact","sha256":"<64 lowercase hex characters>"}]}
```

The manifest is bounded to256KiB/1024 entries. Packet names are unique local basenames ending`.gtact`; resolved paths must stay in its directory. Each file is read with the packet size bound and its full SHA256 checked. Reading is incremental; an error in a later packet makes the replay incomplete, even if earlier observations were emitted. Preserve each packet's original source/clock/reference identities and timestamps; replay never substitutes current time or manufactures acquisition metadata. `recording_frames(absolute_manifest)` yields original tactile InputFrames. It opens no sensor.

Run the implemented command:

```sh
python3 -m humansense.tactile_replay --manifest /absolute/path/recording.json --max-gap-ms 100
```

It prints bounded acquisition observations as JSON lines, not raw pixels, task inference or an acceptance verdict. To reach the SDK's normal Context/MCP surface, register `TactileInputContext.capability()` with `SensorStreams`, open its explicit source/sensor lane, submit tactile InputFrames, then drain. SensorStreams owns its runtime epoch separately from the recorded `source_epoch`. The provider publishes `tactile.input` with input hashes, declared sensor/reference identities, acquisition intervals, raw geometry, backend provenance and `window_ready`; `task_inference_performed` and physical calibration verification remain false. It never calls an LLM.

## Dependencies, licenses and validation

Packet/replay support needs only the standard library. Native capture requires caller-installed NumPy/OpenCV; DIGIT additionally requires the pinned external interface. No dependency or asset was installed for this change. The archived DIGIT interface uses [CC-BY-NC4.0](https://github.com/facebookresearch/digit-interface/blob/87a28bbf2beee8008a5308e9a12d72e1bc4fefb9/LICENSE). The referenced GelSight SDK uses [GPL3](https://github.com/gelsightinc/gsrobotics/blob/321d6a22da64529138ff10237335038fd8c5189f/LICENSE); this bridge invokes OpenCV directly and does not redistribute that SDK. Sparsh's separate source/weight licensing must be checked for #225. These dependency boundaries do not grant commercial permission for upstream software or checkpoints.

`python3 -m unittest discover -s tests -p 'test_tactile.py' -v` exercises actual byte parsing/hash validation, temporal ordering/reset/bounds, the executable replay command, SensorStreams-to-Context, both concrete native API call shapes, constructor/read/close cancellation, delayed release and partial-start errors. Native devices are fake fixtures. There is no trained model, real tactile recording, hardware throughput, contact/slip accuracy or force calibration evidence. Those are mandatory distinct gates in #226.

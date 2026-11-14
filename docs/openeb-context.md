# OpenEB EventCD context

`openeb.eventcd` converts explicit EventCD packets into activity measurements.
It does not identify objects, people, emotion, motion in meters, depth or presence.
Event cameras report pixel brightness changes; noise, flicker and illumination
changes also generate activity. Native capture is a separate explicit source;
the context worker never opens hardware or installs software.

## Runtime responsibilities

`OpenEBCapture` owns a synchronous installed OpenEB iterator selected by exact
camera serial or existing absolute RAW recording path. It emits bounded gamst
packets with native times and caller-declared capture-clock mapping.
`OpenEBContext`/`ProcessProvider('openeb.eventcd')` validate these packets and produce
numeric observations. SensorStreams/MCP retain and deliver observations using the
existing user-selected cadence. Raw events never enter LLM Context.

This is a native event stream, not a stateful neural model or ordinary video FPS.
A slice is returned after8192events or10000microseconds by the mixed SDK iterator.
The adapter emits at most8192events per packet; this does not bound hidden native
SDK/driver buffers or native read time. The CPU context uses NumPy only; capability
discovery imports no NumPy or OpenEB. No model/checkpoint is used.

## Pin and evidence

Inspected [OpenEB5.2.0](https://github.com/prophesee-ai/openeb/tree/9003b5416676e78ba994d912087486cfa94fae73)
commit9003b5416676e78ba994d912087486cfa94fae73. The installed capture adapter checks
these Python source SHA256 values before constructing the native iterator:

- `metavision_core/event_io/events_iterator.py`: `f4ada10870fc14488e4fd14df9a8d394f4fd37607b6830cf103c5c7b918991b8`.
- `metavision_core/event_io/raw_reader.py`: `d9976c236951968546780ed1563cd46e0e7d06d718104c979f4609712ca8480d`.

These checks identify installed wrapper sources, not executed bytecode, native
binary, transitive imports or device-plugin attestation. The normal installed
Python/native runtime is trusted. Different wrapper revisions fail explicitly;
other OpenEB releases are not declared compatible. OpenEB source is Apache2.0;
recordings, plugins and devices need their own provenance/licenses.

Actual OpenEB/native SDK and camera are unavailable in the current environment
and were not installed. Generated EventCD arrays test the adapter's real numeric
execution. Native API-contract tests substitute the reader and do not count as
actual SDK decoding or device evidence. Required acceptance is [#140](https://github.com/NomaDamas/Gamst/issues/140),
under family19 [#35](https://github.com/NomaDamas/Gamst/issues/35).

## Context input and clock

```python
from humansense import ProcessProvider

config = dict(width=640, height=480,
              sensor_origin_us="0", capture_origin_ms=known_capture_origin_ms,
              calibration_id="camera-clock-calibration")
spec = ProcessProvider.capability("openeb.eventcd", config)
worker = await ProcessProvider.open("/absolute/python", "openeb.eventcd", config)
# await worker.process(InputFrame(packet, "events", mapped_end_ms, sensor=...))
```

For an existing native EventCD producer, use
`event_packet(events, width=..., height=..., start_us=..., end_us=...)` from
`humansense.providers.openeb`. Input is a plain one-dimensional NumPy structured
array with integer fields in x,y,p,t order. Masked/subclass arrays, noninteger
fields, out-of-image coordinates, polarities other than0/1, unordered/negative
timestamps and more than8192events are rejected. Input record itemsize is limited to32bytes (at most262144input bytes), before
copying; the native16byte aligned EventCD layout is supported. Native padding is
never serialized.

The gamst-specific packet is **not** an OpenEB RAW/DAT encoding: little-endian
header `<8sHHQQI` contains magic `GECD001\n`, width,height,start_us,end_us,count;
each row `<HHBQ` contains x,y,p,t; a little-endian CRC32 covers header and rows.
Maximum106532bytes. Exact length, CRC and all semantic fields are checked before
use. CRC detects corruption, not authenticity. Geometry axes1..4096 and at most
4096×2160pixels; packet duration0..1000000us. Empty arrays are valid.

Original native microseconds are nonnegative int64 and appear as exact decimal
strings in Context to avoid JSON number precision loss. Clock origin is also an
explicit decimal string. Caller chooses mapping:

`capture_end_ms = capture_origin_ms + (native_end_us - sensor_origin_us) // 1000`.

Outer InputFrame timestamp must exactly match this mapping and its sensor must
have the matching calibration ID and declared pixel coordinate frame. Mapping
configuration is not calibration proof; `clock_calibration_verified=False`.
Use actual camera clock calibration, not current processing time. A packet cannot
predate the declared native origin. Capture milliseconds must fit0..2^53-1.

Native event times lie within the closed interval[start_us,end_us]. At a
count-limited boundary multiple distinct events may have the same microsecond;
a following packet may start at the prior end. Such events are not deduplicated.
Zero-duration intervals have `event_rate_hz=None`; a positive-duration empty slice
has rate0 and no centroid. Event rate counts reported events/second, not frames,
objects or frequency of an independently measured physical phenomenon.

Within an owner, original capture sequence (or sequence fallback) must increase
and intervals cannot overlap backward. Adjacent sequence and start==prior end
retain the state segment; skipped sequence/time gaps or subject/source/sensor/
epoch/coordinates/channels/sequence-mode change start a new segment. A different
clock calibration requires a matching newly configured provider; mismatches reject.
Malformed input clears continuity before raising. No missing events are invented.
Reset clears state; close is idempotent and permanent.

## Explicit native source

```python
from humansense.providers.openeb_capture import OpenEBCapture

options = dict(source_id="event-camera", sensor_id="events",
               sensor_origin_us="0", capture_origin_ms=known_capture_origin_ms,
               calibration_id="camera-clock-calibration", coordinate_frame="sensor-pixels")
with OpenEBCapture.recording("/absolute/recording.raw", **options) as source:
    # source.config includes dimensions read from the SDK; use it to build spec.
    frame = source.next_frame()
# Or explicitly: OpenEBCapture.camera("EXACT_SERIAL", **options)
```

No empty-string/default camera or relative path is accepted. Native start_ts must
be a multiple of10000us, matching EventsIterator's mixed-mode seek constraint.
The reader's `get_current_time()` supplies slice boundaries, not the first/last
event alone; this preserves empty intervals and zero-duration count-limited slices.
Native NumPy integer timestamps are converted exactly to Python integers.
Full datatype validation and owned field copies precede packet publication.

Exhaustion/error closes the source. Explicit close also stops constructor-started
RAW acquisition before the first read. Pinned RawReaderBase context-manager
cleanup is used; tests validate calls but real driver cleanup is unverified.
Reads/close are serialized. **Close cannot interrupt a blocked native read.**
Use [OpenEBProcess](openeb-process.md) for owned source-process supervision and
asynchronous deadlines/full-duplex isolation. Do not advertise the synchronous source as hard-cancellable or demonstrated
nonblocking live audio/video. The isolated context worker itself uses the existing
bounded ProcessProvider lifecycle and does not call hardware.

Only RAW/camera capture is selected here; DAT/HDF5 formats advertised by upstream
are not implicitly supported by this capture adapter. Direct EventCD input from
other explicit producers remains possible. Replay does not automatically pace
wall time or turn recording-relative timestamps into actual live capture time.

## Output and tests

Activity includes count, positive/negative counts, interval duration, event rate,
polarity fraction, number/fraction of unique active pixels, event-weighted pixel
centroid and two8×8row-major count grids. Grid cells use floor(x*8/width),
floor(y*8/height); all events are accounted for, including repeated same-pixel
activity. The centroid is an activity statistic, not a tracked object's location.
Summary includes native interval, mapped capture end, dimensions, original
sequence, state segment, continuity and packet hash. Completeness of native event
capture is `not_verified`; detection/physical-motion flags are false.

```sh
python3 -m unittest discover -s tests -p test_openeb.py -v
```

Nine tests cover exact wire layout and histogram/rate oracles, empty/equal-time
slices, int64 time precision, gaps/overlap/clock/lifecycle, malformed CRC/fields,
masked/oversized/padded inputs, maximum payload, installed-API call shape/early closure
and a real owned context worker→SensorStreams→MCP path with generated events.
They do not prove actual RAW decoding, native memory bounds, camera accuracy or
latency. Real SDK/recording/device evidence, native-supervision acceptance, overflow/disconnect,
clock calibration, synchronized stimuli and measured p50/p95/resources remain
mandatory in #140. No hardware/model acceptance is closed by fixtures.

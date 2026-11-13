# TI radar context

```mermaid
flowchart LR
  A[Host UART capture and clock mapping] --> B[Bounded incremental packet framer]
  B --> C[Explicit xWR6843 decoder]
  C --> D[Metric tracks and point summaries]
  D --> E[gamst SDK observations and MCP]
```

The host owns the physical radar, firmware, CLI configuration, serial reads and
capture-clock mapping. `TIRadarPackets` frames byte chunks; `TIRadar` decodes a
complete packet and publishes sensor estimates. Neither starts a serial port or
radio. The radar firmware performs tracking; this adapter does not run a neural
network, infer identity or establish wall penetration. Hardware acceptance is
required in [#123](https://github.com/NomaDamas/Gamst/issues/123), under#31.

## Explicit wire format

The primary format reference is TI's
[Radar Toolbox3.30.00.06 UART specification](https://dev.ti.com/tirex/explore/content/radar_toolbox_3_30_00_06/software_docs/Understanding_UART_Data_Output_Format.html).
This adapter selects little-endian40-byte headers, platform0xA6843, subframe0,
32-byte packet alignment and TLV lengths excluding the8-byte TLV header.
The caller must supply the exact `sdk_version` uint32 encoded by their firmware;
the value is checked on every packet. Other platform/header/length profiles are
not auto-detected. Header compatibility does not authenticate a firmware build.
Record the binary hash, SDK/toolbox versions and CLI configuration separately.

Supported payloads:

- Type1: Cartesian x/y/z in meters and radial Doppler in m/s, four float32 values.
- Type7: Cartesian-point SNR and noise, uint16 in0.1dB units, following this
  selected format specification. Firmware with signed side-info requires a
  separately validated profile; no signedness guessing is performed.
- Type1020: five float32 scales then elevation/azimuth int8, Doppler int16,
  range/SNR uint16. Outputs radians, meters, m/s and scaled SNR in dB.
- Type1010: uint32 track ID and27float32 values,112bytes per track. Retains
  position, velocity, acceleration, covariance, gating gain and firmware confidence.
- Type1012: uint32 track ID and maximum/minimum z float32 values,12bytes per entry.
- Type1011: counts of associated/unassociated point labels. No point-track join.
  Some people-tracking firmware emits association indices for the previous frame;
  this adapter deliberately preserves that boundary rather than assigning them
  to current points. Hardware acceptance must establish the selected firmware's
  timing before any future association implementation.
- Type1021: firmware presence flag0/1. This is not verified human presence.

Other TLVs are skipped after length validation; their numeric types are reported.
Duplicate TLVs, ambiguous Cartesian+compressed clouds, nonfinite values, reserved
track IDs, invalid bounds, unsupported versions and malformed padding lengths fail
closed. Padding byte values are unspecified and not assumed zero. Missing track,
height or presence payloads are `null`, not a verified empty scene. An explicitly
empty target list is `[]`. Point counts must match a supplied cloud payload;
without a cloud the header count is reported as undecoded.

## Bounds and context meaning

A packet is64..65536bytes, with at most32TLVs,512points and32tracks. The framer
accepts at most4096bytes per feed, retains fewer than65536bytes between calls and
can return at most64small packets per call (or a previously buffered large packet
plus the remaining input). It searches for the eight-byte magic after noise or
an impossible length. No wire checksum exists: a plausible false header can wait
for its declared packet length. The host must reset framing on timeout/reconnect
and handle parse errors; framing alone does not establish packet integrity.

Point context includes total decoded count, minimum/maximum range, mean radial
velocity and the nearest16points with explicit retained count/selection. This
summary preserves extrema and bounded work without pretending to expose every
point. The full bounded target list is retained. Track `range_to_frame_origin_m`
and speed are vector norms. Firmware confidence is not a calibrated probability;
covariance is passed through, not asserted to be a validated uncertainty model.
No smoothing, force/contact inference, vital signs or person classification is added.

`coordinate_frame` and `calibration_id` must match input SensorMetadata. Firmware
may already transform tracks using sensor-position configuration; the host must
document whether point and track frames differ. This adapter performs no transform.
Do not assume track-origin distance equals physical radar range after translation.
`hardware_verified`, `calibration_verified`, `clock_mapping_verified`,
`identity_verified`, `through_wall_verified` and `confidence_calibrated` remain false.

The native streaming mode reflects the device's continuously maintained tracker.
The adapter has no lookahead and retains only continuity metadata. Capture time
is a host-mapped integer milliseconds value in0..2**53-1, not UART arrival time.
The header CPU cycle counter is reported without converting it to a wall clock.
Track IDs are scoped to the returned session, not identities. Host source,
sensor, subject, epoch or sequence-mode changes, missing frames and excessive
gaps start a new session. Firmware frame numbers support uint32 wrap. Duplicate
firmware frames and reversed/repeated host capture time or sequence fail and
clear continuity. A firmware restart cannot always be detected from counters
alone: the host must supply a fresh stream epoch when reconnecting/resetting.
Default freshness/gap limit is1000ms and configurable1..10000ms. Close is permanent.

## Use

Python standard library only; no model weights or optional package installation.
Adapter license is Apache-2.0; TI firmware/software licensing is separate and
must be checked for the actual binary. Read serial data only through your
application's explicitly configured capture driver.

```python
from humansense import ProcessProvider, SensorMetadata, SensorStreams
from humansense.plugins import InputFrame
from humansense.providers.radar import TIRadarPackets

config = dict(sdk_version=firmware_sdk_version_word,
              coordinate_frame="radar_local", calibration_id="radar_cal1")
spec = ProcessProvider.capability("ti.radar", config)
streams = SensorStreams(human)  # Existing consented HumanSense instance.
async def create():
    return await ProcessProvider.open(worker_python, "ti.radar", config)
streams.register(spec, create)
await streams.open("radar1", "radar", [spec.id])
framer = TIRadarPackets()
# In the host-owned capture loop; each chunk is at most4096bytes:
for packet in framer.feed(chunk):
    timestamp_ms, capture_sequence = map_packet_capture_time(packet)
    await streams.submit(InputFrame(packet, "radar", timestamp_ms,
        sensor=SensorMetadata("radar1", "radar", capture_sequence,
            coordinate_frame="radar_local", calibration_id="radar_cal1")))
await streams.drain()
# At end/reconnect, finish/close this stream and reset the host framer.
await streams.finish("radar1", "radar")
```

`map_packet_capture_time` is an application-supplied calibrated mapping, not an
SDK function. The host must bound its serial queue and observe dropped packets.
ProcessProvider uses the existing owned-worker lifecycle and byte transport.
Observations are available through the normal SDK/MCP `get_observations` path;
raw UART packets are not copied into context.

## Evidence and required follow-up

`python3 -m unittest discover -s tests -p test_radar.py -v` runs generated wire
fixtures: known geometry, signed/unsigned compression, malformed data, every
split point, resynchronization, bounds, continuity/wrap, missing versus empty
snapshots, dependency-light discovery and actual unpatched decoding in an owned
worker through SensorStreams/MCP. No fixture is claimed as captured hardware.
Real firmware parity, measured clock mapping, clutter/trajectory accuracy,
capture-to-context p50/p95 and sustained live duplex/resource tests remain
mandatory in#123. No live performance, detection quality or through-wall claim
can be derived from these software tests.

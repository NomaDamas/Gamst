# Radiometric thermal context

`humansense.providers.thermal.TLinearThermal` converts assembled, row-major uint16 TLinear image payloads to bounded ROI Celsius summaries. It uses Python's standard library, requires no weights or optional packages, and never opens a camera. The SDK implementation is Apache-2.0; camera firmware/drivers retain their own licenses. This is an executable decoder/statistics provider, not a VoSPI packet assembler, USB driver, RGB temperature estimator or radiometric JPEG decoder.

FLIR documents [TLinear temperature conversion](https://oem.flir.com/learn/thermal-integration-made-easy/radiometry-with-lepton/) and [radiometric spot-meter operation](https://www.oem.flir.com/en-150/developer/lepton-family/project-lepton-spot-meter-code/). Declare the sensor's actual resolution explicitly: 0.01 or 0.1 Kelvin/count. Celsius = count × resolution − 273.15. The SDK does not infer the resolution, byte order or radiometric mode from pixel values.

## Input and runtime responsibilities

The capture adapter must verify radiometry and TLinear mode, disable AGC on this output, remove transport headers/telemetry, reject corrupted/incomplete packets and suppress frames during FFC or invalid sensor states. A colour preview is not acceptable input. Supply the actual image dimensions and little/big endian order; exact payload length is checked. There is no default calibration profile or emissivity. `calibration_id` identifies an application-owned record of sensor model/serial, mode, scale, emissivity, gain, window transmission, reflected/environment temperature, reference checks and acquisition settings. The same ID must accompany each `SensorMetadata`; configuration changes require a new provider/profile. Matching IDs verify consistency, not physical calibration accuracy.

`emissivity` reports the sensor setting; this provider does not apply emissivity correction a second time. `physical_accuracy_verified=false` is always emitted because the decoder cannot certify absolute measurement accuracy. External validation must not silently turn this into a ground-truth temperature claim. RGB alignment is also explicitly unavailable.

```python
from humansense import HumanSense, SensorStreams, SensorMetadata
from humansense.plugins import InputFrame
from humansense.providers.thermal import TLinearThermal

# Inside an async application; capture supplies payload/captured_at_ms/sequence.
async def observe(payload, captured_at_ms, sequence):
    async with HumanSense(consent=True) as human:
        streams = SensorStreams(human)
        spec = TLinearThermal.capability()
        streams.register(spec, lambda: TLinearThermal(
            width=160, height=120, kelvin_per_count=.01, byteorder='little',
            calibration_id='lepton-profile-1', emissivity=.95,
            rois={'scene': (0, 0, 160, 120)}))
        await streams.open('device', 'lepton', [spec.id])
        await streams.submit(InputFrame(payload, 'thermal', captured_at_ms,
            sensor=SensorMetadata('device', 'lepton', sequence,
                                  calibration_id='lepton-profile-1')))
        await streams.drain()
        return await human.get_context()
```

For continuous capture, keep the runtime/manager open and repeatedly submit frames with advancing sequence/capture timestamps. Capture, inference and LLM subscription clocks remain independent. Processing is per frame, zero lookahead, with no retained pixel history; stream queues, gaps, epochs, lifecycle and context freshness use `SensorStreams`. No raw pixels enter observations or MCP context.

## Bounds and unavailable measurements

Images are at most 640×512 (655360 input bytes), with 1–16 ROIs. Bounds are `(left, top, right, bottom)` in thermal image pixels, right/bottom exclusive. At most one decoded image and 16 bounded output summaries exist per invocation. The provider conservatively excludes codes 0 and 65535; this is not a complete sensor-specific bad-pixel detector. Capture must supply already corrected/validated images. Each ROI reports valid/invalid pixel counts and fraction. All-invalid regions have `available=false` and null statistics; no zero-degree temperature is fabricated. These numerical ranges are encoding ranges, not the camera's guaranteed temperature/accuracy range.

## Validation status

`python3 -m unittest discover -s tests -p test_thermal_provider.py -v` checks known numeric conversion, both resolutions and endianness, ROI ownership, missing pixels, malformed inputs, reset/close and actual SensorStreams-to-context integration. Inputs are generated numeric fixtures, not recorded camera packets or hardware measurements. Python 3.10.11 was used locally. No physical device, calibration target, acquisition-driver compatibility or live accuracy/latency evidence is claimed. Required live sensor validation, capture transport integration and RGB registration remain open under #30; this provider does not complete that capability family.

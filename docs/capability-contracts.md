# Sensory capability contracts

The capability registry describes a provider output without importing the provider, downloading weights or opening sensors. It complements the existing `PerceptionPlugin` API and does not change legacy `push_audio`, `push_video` or `push_signal` calls.

```python
from humansense import HumanSense, CapabilitySpec, OutputField, SensorMetadata

spec = CapabilitySpec(
    id="thermal.roi", output_name="thermal.surface", modality="thermal",
    provider="my.radiometric.sensor", version="1.0",
    fields=(OutputField("mean", "number", "degC"),
            OutputField("region", "object")),
    streaming_mode="frame", freshness_ms=1000,
    interpretation="measurement", sensor_requirements=("radiometric",),
)

async def publish(measured_temperature, region, capture_time_ms):
    async with HumanSense(consent=True) as human:
        human.register_capability(spec)
        # These values must come from the caller's actual calibrated sensor.
        record = await human.push_observation(
            "thermal.roi", {"mean": measured_temperature, "region": region},
            sensor=SensorMetadata("camera1", "lepton1", sequence=1,
                                  calibration_id="sensor-calibration-v1"),
            timestamp_ms=capture_time_ms,
        )
        return record
```

`OutputField` validates declared field names and types, not a full JSON Schema. The provider must validate semantic constraints such as bounding box ordering, calibration validity and physical ranges. Field types are string, number, integer, boolean, object and array. Extra top-level fields, nonfinite values, raw media fields, non-JSON objects and oversized/deep structures are rejected. Outputs are copied before storage so caller mutation cannot change evidence. Probability calibration checks remain in force.

Supported modality identifiers are video, audio, text, thermal, radar, csi, imu, tactile, environment and events. New nonlegacy observations require a registered contract and `SensorMetadata`. Metadata identifies source, physical sensor, sequence, channel names, coordinate frame and calibration reference. Sequence ordering and provider-specific gap resets are runtime responsibilities, not enforced by this metadata class. `InputFrame.sensor` carries this metadata without downmixing or modifying raw input.

Streaming declarations distinguish native, window, frame and offline behavior. Window mode requires a window duration; lookahead is explicit. Registering an offline provider does not make it causal. Declared freshness controls observation freshness independently of delivery cadence.

In `observation_packet`, legacy signals keep their original keys. Registered outputs use `@capability:` followed by a JSON-encoded tuple of output name, capability ID, source ID and sensor ID. Consumers should iterate values and use their `name`, `capability` and `sensor` metadata instead of constructing keys. Different sensors and providers cannot overwrite each other's latest registered observations. Context remains limited to the newest 32 entries; registered payloads individually remain within the 64 KiB contract budget. Aggregate transport/token budgets belong to the delivery layer.

The registry and observation contract do not implement a physical sensor driver or prove a model's accuracy. See the [acceptance map](perception-acceptance.md) for actual provider and device validation requirements.

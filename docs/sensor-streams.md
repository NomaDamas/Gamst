# Independent sensor streams

`SensorStreams` adds an explicit typed-provider input path alongside the legacy media router. A source/sensor pair owns one modality and subject. Each selected capability receives its own provider instance, bounded queue and copied input. Synchronous inference runs on a worker thread; asynchronous providers must avoid blocking the event loop themselves.

```python
from humansense import SensorStreams

# spec is a CapabilitySpec; Provider implements process(frame), reset and close.
streams = SensorStreams(human, capacity=2)
streams.register(spec, Provider)  # metadata only; no construction
await streams.open("camera1", "sensor1", [spec.id])  # explicit construction
await streams.submit(frame)  # InputFrame carrying matching SensorMetadata
await streams.drain()
status = streams.status()
await streams.clear("camera1", "sensor1")
```

The provider's `process` returns a dictionary matching its capability fields, or `None` if no output is ready. It may be synchronous or asynchronous. Stateful providers must implement `reset`; a sequence gap, dropped queued input or processing error resets affected state before the next input. Different sources do not share provider instances. Frames preserve channel dimensions and sample rate; no implicit downmix occurs. Audio cleanup, playback-reference AEC and ASR branching belong to audio providers. Independent input lanes can continue while the consumer plays speech, but actual acoustic echo cancellation is a separate capability.

Inputs must carry nondecreasing capture timestamps and strictly increasing sequence numbers. Reopening after clear starts a new provider instance and sequence epoch. Late-generation outputs after pause, data deletion or close are discarded. Lifecycle operations await synchronous inference before closing its backend; a backend that never returns can delay cleanup and must provide its own bounded inference timeout. The SDK does not terminate arbitrary native inference threads.

Queues drop the oldest pending input at capacity. Packet and total in-flight payload budgets reject new submissions before consuming sequence numbers. Numeric lists, bytes, bounded JSON and array-like numeric buffers are copied; raw input is never persisted in the observation store. Budget accounting bounds admitted payloads, not all Python/backend allocation or model weight memory. Providers must keep their own model/cache allocations bounded.

Status exposes accepted/processed/dropped/error/reset counts, queue depth and processing latency. The latency counter covers provider processing and publishing, not camera capture latency. User context delivery cadence remains independent and is handled by the context/transport layer. Discovery does not construct providers. Pause/reset clears opened streams, so callers must explicitly reopen them after resuming.

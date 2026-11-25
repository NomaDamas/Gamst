# Resource allocation foundation

`ResourceAllocator` is a bounded accounting ledger for provider reservations.
Deployment discovery supplies a `ResourceCapacity` for CPU cores, RAM bytes, GPU
count and VRAM bytes; the allocator does not probe hardware or claim that a
reservation changes host scheduling. A `ResourceRequest` is admitted atomically
only when every dimension fits the remaining capacity. Overcommit is rejected
without partially reserving any dimension.

```python
from gamst.crew import ResourceAllocator, ResourceCapacity, ResourceRequest

allocator = ResourceAllocator(ResourceCapacity(
    cpu_cores=8, ram_bytes=32 * 1024**3, gpu_count=1,
    vram_bytes=24 * 1024**3,
))
reservation = allocator.reserve(
    ResourceRequest(cpu_cores=2, ram_bytes=4 * 1024**3,
                    gpu_count=1, vram_bytes=8 * 1024**3),
    owner="vision",
)
try:
    # Start the provider only after this reservation succeeds.
    ...
finally:
    allocator.release(reservation)
```

Release requires the exact opaque reservation token and rejects double release
or forged copies. `max_reservations` bounds the ledger itself. `telemetry()`
reports capacity, reserved and available vectors, active owners and counters for
attempts, accepts, overcommit/capacity rejections and releases. The counters are
accounting evidence, not OS-level usage measurements.

`RuntimeManager(resource_allocator=...)` accepts a request on registration. The
reservation is made before adapter startup, released if startup fails, and
released after adapter shutdown. `resource_telemetry()` returns the allocator
view or `{ "enabled": false }` when no allocator is configured. Automatic
resource resizing, GPU process isolation and host usage sampling remain outside
this foundation slice. `HostLimitController` is an explicit opt-in seam for
the host boundary. `mode="accounting"` (the default) performs no host
mutation. On Linux with a caller-supplied writable cgroup-v2 root,
`mode="strict"` can attach CPU and RAM limits before the provider is accepted;
GPU/VRAM and missing controllers fail closed. `mode="best_effort"` reports
`unsupported` or `unavailable` without pretending enforcement occurred.
macOS and Windows currently report `unsupported` until native controllers are
implemented. A fake cgroup fixture tests the file contract, but is not host
acceptance evidence.

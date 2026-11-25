"""One model process of a ParallelCrew run.

Runs inside the model's own Python environment. It constructs the provider
through GAMST's own ``provider_worker._construct`` (the same code path as
``ProcessProvider``), enforces its GPU memory budget before loading weights,
reads frames/audio from the shared media rings, and writes one JSON line per
result or metrics tick to its private stdout descriptor.

Usage: python -m gamst.crew.parallel_worker '<spec json>'
"""
from __future__ import annotations

import json
import os
import sys
import time


def _budget(spec: dict) -> dict:
    budget = spec.get("gpu_budget_mib")
    info = {"budget_mib": budget, "enforced": False}
    if not budget:
        return info
    try:
        import torch
    except ImportError:
        return info
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 2**20
        torch.cuda.set_per_process_memory_fraction(min(1.0, budget / total), 0)
        info.update(enforced=True, total_mib=round(total))
    return info


def _gpu_used_mib() -> float | None:
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.memory_reserved(0) / 2**20, 1)
    except ImportError:
        pass
    return None


def main() -> None:
    spec = json.loads(sys.argv[1])
    os.environ["GAMST_PROVIDER_WORKER"] = "1"
    out = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    def emit(kind: str, **fields) -> None:
        out.write(json.dumps({"type": kind, "worker": spec["name"], **fields}, allow_nan=False) + "\n")

    budget = _budget(spec)
    from gamst.capabilities import SensorMetadata
    from gamst.plugins import InputFrame
    from gamst.provider_worker import _construct
    from gamst.crew.media_bus import AudioRing, FrameRing

    started = time.time()
    provider, capability = _construct(spec["profile"], spec["config"])
    emit("ready", load_s=round(time.time() - started, 2), capability_id=capability.id,
         capability_version=capability.version, budget=budget, pid=os.getpid())

    source = spec["input"]
    kind = source["kind"]
    ring = FrameRing.attach(spec["frame_ring"]) if kind == "video" else AudioRing.attach(spec["audio_ring"])
    epoch_ms = spec["media_epoch_ms"]
    sensor_id = spec.get("sensor_id", "camera0" if kind == "video" else "mic0")
    channels = () if kind == "video" else ("mono",)
    stream_epoch = spec.get("stream_epoch", "run0")
    stride = int(source.get("stride", 1))
    mode = source.get("mode", "every")
    chunk = int(source.get("chunk_ms", 0)) * (ring.sample_rate // 1000) if kind == "audio" else 0
    cursor, sequence = (0, 0)
    processed = dropped = errors = 0
    busy_s = 0.0
    last_metrics = time.time()
    stop_file = spec.get("stop_file")

    def metrics(final: bool = False) -> None:
        emit("metrics", processed=processed, dropped=dropped, errors=errors, busy_s=round(busy_s, 3),
             gpu_reserved_mib=_gpu_used_mib(), final=final,
             cursor=cursor, head=(ring.write_seq if kind == "video" else ring.written))

    try:
        while True:
            if stop_file and os.path.exists(stop_file):
                break
            if kind == "video":
                head = ring.write_seq
                if head < 0 or cursor > head:
                    if spec.get("end_flag") and os.path.exists(spec["end_flag"]) and cursor > ring.write_seq:
                        break
                    time.sleep(0.002)
                    continue
                if mode == "latest":
                    target = head - head % stride
                    if target < cursor:
                        time.sleep(0.002)
                        continue
                    dropped += max(0, (target - cursor) // stride)
                    cursor = target
                elif cursor < ring.oldest_available():
                    skip_to = ring.oldest_available()
                    skip_to += (-skip_to) % stride
                    dropped += (skip_to - cursor) // stride
                    cursor = skip_to
                item = ring.read(cursor)
                if item is None:
                    dropped += 1
                    cursor += stride
                    continue
                pts_ms, wall_ns, data = item
                frame_seq = cursor
                cursor += stride
                sample_rate = None
            else:
                if ring.written < cursor + chunk:
                    if spec.get("end_flag") and os.path.exists(spec["end_flag"]):
                        break
                    time.sleep(0.002)
                    continue
                data = ring.read(cursor, chunk)
                if data is None:
                    dropped += 1
                    cursor = ring.written - chunk
                    cursor -= cursor % chunk
                    continue
                frame_seq = cursor // chunk
                pts_ms = (cursor + chunk) * 1000 // ring.sample_rate
                wall_ns = time.time_ns()
                cursor += chunk
                sample_rate = ring.sample_rate
                if source.get("dtype") == "float32":  # providers that take normalized PCM
                    data = data.astype("float32") / 32768.0
            sensor = SensorMetadata(source_id=spec.get("source_id", "media0"), sensor_id=sensor_id,
                                    sequence=sequence, channels=channels, stream_epoch=stream_epoch,
                                    coordinate_frame=source.get("coordinate_frame"))
            sequence += 1
            frame = InputFrame(data=data, modality=kind, timestamp_ms=epoch_ms + pts_ms,
                               subject_id=spec.get("subject_id", "local-user"),
                               sample_rate=sample_rate, sensor=sensor)
            begin = time.time()
            try:
                result = provider.process(frame)
                if result is not None:
                    result = capability.validate(result)
            except Exception as exc:  # isolate one failing frame; count it
                errors += 1
                emit("error", seq=frame_seq, pts_ms=pts_ms, error=type(exc).__name__, detail=str(exc)[:300])
                result = None
                if errors >= int(spec.get("max_errors", 20)):
                    raise
            end = time.time()
            busy_s += end - begin
            processed += 1
            if result is not None:
                emit("result", seq=frame_seq, pts_ms=pts_ms, timestamp_ms=epoch_ms + pts_ms,
                     available_ns=wall_ns, done_ns=time.time_ns(),
                     latency_ms=round((time.time_ns() - wall_ns) / 1e6, 2),
                     compute_ms=round((end - begin) * 1000, 2), value=result)
            if end - last_metrics >= 1.0:
                last_metrics = end
                metrics()
        finish = getattr(provider, "finish", None)
        if callable(finish):
            result = finish()
            if result is not None:
                emit("result", seq=-1, pts_ms=None, timestamp_ms=None, final=True,
                     value=capability.validate(result))
    finally:
        metrics(final=True)
        close = getattr(provider, "close", None)
        if callable(close):
            close()
        out.flush()


if __name__ == "__main__":
    main()

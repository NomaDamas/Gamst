"""ParallelCrew: every model at once on one GPU, never one model at a time.

Not exercised by the local test suite (it needs ``nvidia-smi`` and one CUDA GPU); the code is
from the former GamstCrew package, unchanged apart from the worker module path.

- One OS process per model (its own Python environment and CUDA context).
- GPU memory budgets are reserved up front against measured device capacity
  (all-or-nothing, no overcommit); torch workers additionally cap their
  allocator, vLLM servers get ``--gpu-memory-utilization`` from the budget,
  and the monitor flags any process whose measured usage exceeds its budget.
- One producer decodes the media once into shared-memory rings; every worker
  reads the same frames and the same media clock.
- A monitor samples per-process GPU memory and worker throughput every second
  and records realtime factor, drops and latency.

The run writes ``results.jsonl`` (every model output with media timestamps),
``monitor.jsonl`` (1 Hz samples) and ``summary.json``.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .resources import ResourceAllocator, ResourceCapacity, ResourceRequest, ResourceUnavailable

MIB = 2**20


@dataclass
class WorkerSpec:
    name: str
    python: str
    profile: str
    config: dict
    input: dict
    gpu_budget_mib: int = 0
    env: dict = field(default_factory=dict)
    max_errors: int = 20


@dataclass
class ServerSpec:
    """An external model server (e.g. vLLM) supervised as one more process."""
    name: str
    argv: list
    health_url: str
    gpu_budget_mib: int
    env: dict = field(default_factory=dict)
    startup_timeout_s: int = 900


def gpu_capacity_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout.split()
    if len(out) != 1:
        raise RuntimeError("ParallelCrew expects exactly one visible GPU")
    return int(out[0])


def gpu_processes() -> dict[int, int]:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    usage = {}
    for line in out.strip().splitlines():
        pid, used = (part.strip() for part in line.split(","))
        usage[int(pid)] = int(used)
    return usage


def _descendants(pid: int) -> set[int]:
    found, frontier = {pid}, [pid]
    while frontier:
        parent = frontier.pop()
        try:
            children = Path(f"/proc/{parent}/task/{parent}/children").read_text().split()
        except OSError:
            continue
        for child in map(int, children):
            if child not in found:
                found.add(child)
                frontier.append(child)
    return found


def _percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))], 2)


class ParallelCrew:
    def __init__(self, out_dir: str, *, workers: list[WorkerSpec], servers: list[ServerSpec] = (),
                 gamst_root: str | None = None, reserve_mib: int = 1024, strict_budget: bool = True):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.workers, self.servers = list(workers), list(servers)
        # Workers import gamst (and gamst.crew) from this checkout unless their env has it.
        self.gamst_root = gamst_root or str(Path(__file__).resolve().parents[2])
        self.strict_budget = strict_budget
        self.capacity_mib = gpu_capacity_mib()
        self.allocator = ResourceAllocator(ResourceCapacity(gpu_count=1, vram_bytes=(self.capacity_mib - reserve_mib) * MIB))
        self.reservations = {}
        self.procs: dict[str, subprocess.Popen] = {}
        self.state: dict[str, dict] = {}
        self.latency: dict[str, list] = {}
        self.violations: list[dict] = []
        self.lock = threading.Lock()
        self.results = open(self.out / "results.jsonl", "a", buffering=1)
        self.monitor_log = open(self.out / "monitor.jsonl", "a", buffering=1)
        self.stop_file = str(self.out / "STOP")
        self.end_flag = str(self.out / "MEDIA_END")
        for path in (self.stop_file, self.end_flag):
            if os.path.exists(path):
                os.remove(path)
        self._stop = threading.Event()

    # -- budgets -------------------------------------------------------------
    def reserve_all(self) -> dict:
        for spec in self.servers + self.workers:
            try:
                self.reservations[spec.name] = self.allocator.reserve(
                    ResourceRequest(vram_bytes=spec.gpu_budget_mib * MIB), owner=spec.name)
            except ResourceUnavailable as exc:
                raise RuntimeError(f"GPU budget overcommit at {spec.name}: "
                                   f"asked {spec.gpu_budget_mib} MiB, free {exc.available.vram_bytes // MIB} MiB") from exc
        return self.allocator.telemetry()

    # -- servers ---------------------------------------------------------------
    def start_servers(self) -> None:
        for spec in self.servers:
            log = open(self.out / f"{spec.name}.log", "a")
            env = {**os.environ, **spec.env}
            proc = subprocess.Popen(spec.argv, stdout=log, stderr=subprocess.STDOUT, env=env,
                                    start_new_session=True)
            self.procs[spec.name] = proc
            self.state[spec.name] = {"kind": "server", "budget_mib": spec.gpu_budget_mib, "pid": proc.pid}
            deadline = time.time() + spec.startup_timeout_s
            while time.time() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f"server {spec.name} exited with {proc.returncode}")
                try:
                    urllib.request.urlopen(spec.health_url, timeout=2)
                    break
                except Exception:
                    time.sleep(3)
            else:
                raise RuntimeError(f"server {spec.name} not healthy within {spec.startup_timeout_s}s")
            self.state[spec.name]["ready_s"] = round(spec.startup_timeout_s - (deadline - time.time()), 1)

    # -- workers ---------------------------------------------------------------
    def start_workers(self, frame_ring: str | None, audio_ring: str | None, media_epoch_ms: int,
                      stream_epoch: str) -> None:
        for spec in self.workers:
            payload = {"name": spec.name, "profile": spec.profile, "config": spec.config, "input": spec.input,
                       "gpu_budget_mib": spec.gpu_budget_mib, "frame_ring": frame_ring, "audio_ring": audio_ring,
                       "media_epoch_ms": media_epoch_ms, "stream_epoch": stream_epoch,
                       "stop_file": self.stop_file, "end_flag": self.end_flag, "max_errors": spec.max_errors}
            env = {**os.environ, **spec.env,
                   "PYTHONPATH": os.pathsep.join([self.gamst_root, spec.env.get("PYTHONPATH", "")])}
            proc = subprocess.Popen([spec.python, "-m", "gamst.crew.parallel_worker", json.dumps(payload)],
                                    stdout=subprocess.PIPE, stderr=open(self.out / f"{spec.name}.log", "a"),
                                    env=env, text=True, start_new_session=True)
            self.procs[spec.name] = proc
            self.state[spec.name] = {"kind": "worker", "budget_mib": spec.gpu_budget_mib, "pid": proc.pid,
                                     "ready": False}
            self.latency[spec.name] = []
            threading.Thread(target=self._pump, args=(spec.name, proc), daemon=True).start()

    def _pump(self, name: str, proc: subprocess.Popen) -> None:
        for line in proc.stdout:
            try:
                message = json.loads(line)
            except ValueError:
                continue
            kind = message.get("type")
            with self.lock:
                state = self.state[name]
                if kind == "ready":
                    state.update(ready=True, load_s=message["load_s"], capability_id=message["capability_id"],
                                 capability_version=message["capability_version"], budget=message["budget"])
                elif kind == "metrics":
                    state.update({k: message[k] for k in ("processed", "dropped", "errors", "busy_s",
                                                         "gpu_reserved_mib", "cursor", "head")})
                elif kind == "result":
                    if message.get("latency_ms") is not None:
                        self.latency[name].append(message["latency_ms"])
                    self.results.write(line if line.endswith("\n") else line + "\n")
                elif kind == "error":
                    state["last_error"] = message.get("error")
                    self.results.write(line if line.endswith("\n") else line + "\n")
        with self.lock:
            self.state[name]["exit_code"] = proc.wait()

    def wait_ready(self, timeout_s: int = 900) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            with self.lock:
                pending = [n for n, s in self.state.items() if s["kind"] == "worker" and not s.get("ready")]
                dead = [n for n in pending if "exit_code" in self.state[n]]
            if dead:
                raise RuntimeError(f"workers exited before ready: {dead}")
            if not pending:
                return
            time.sleep(1)
        raise RuntimeError(f"workers not ready: {pending}")

    # -- monitor ---------------------------------------------------------------
    def _monitor(self, media_clock) -> None:
        while not self._stop.is_set():
            usage = gpu_processes()
            sample = {"t_wall": time.time(), "media_s": media_clock(), "gpu_total_used_mib": sum(usage.values()),
                      "models": {}}
            with self.lock:
                for name, state in self.state.items():
                    pids = _descendants(state["pid"])
                    used = sum(usage.get(pid, 0) for pid in pids)
                    state["gpu_used_mib"] = used
                    state["gpu_peak_mib"] = max(state.get("gpu_peak_mib", 0), used)
                    sample["models"][name] = {k: state.get(k) for k in (
                        "gpu_used_mib", "processed", "dropped", "errors", "busy_s")}
                    if state["budget_mib"] and used > state["budget_mib"] * 1.05:
                        violation = {"t_wall": sample["t_wall"], "model": name, "used_mib": used,
                                     "budget_mib": state["budget_mib"]}
                        self.violations.append(violation)
                        if self.strict_budget and state["kind"] == "worker":
                            self.procs[name].send_signal(signal.SIGTERM)
            self.monitor_log.write(json.dumps(sample) + "\n")
            self._stop.wait(1.0)

    def start_monitor(self, media_clock) -> None:
        threading.Thread(target=self._monitor, args=(media_clock,), daemon=True).start()

    # -- shutdown & summary ----------------------------------------------------------
    def finish_media(self) -> None:
        Path(self.end_flag).touch()

    def drain(self, timeout_s: int = 600) -> None:
        deadline = time.time() + timeout_s
        for name, spec in ((s.name, s) for s in self.workers):
            proc = self.procs[name]
            try:
                proc.wait(timeout=max(1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                Path(self.stop_file).touch()
                proc.wait(timeout=30)

    def stop(self) -> None:
        Path(self.stop_file).touch()
        self._stop.set()
        for name, proc in self.procs.items():
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for proc in self.procs.values():
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
        for reservation in self.reservations.values():
            self.allocator.release(reservation)
        self.reservations.clear()

    def summary(self, media_duration_s: float, wall_s: float) -> dict:
        models = {}
        with self.lock:
            for name, state in self.state.items():
                entry = {k: state.get(k) for k in ("kind", "budget_mib", "gpu_peak_mib", "load_s", "ready_s",
                                                   "processed", "dropped", "errors", "busy_s", "capability_id",
                                                   "capability_version", "exit_code", "last_error")}
                if state["kind"] == "worker" and state.get("processed"):
                    seen = state["processed"] + (state.get("dropped") or 0)
                    entry["drop_rate"] = round((state.get("dropped") or 0) / seen, 4) if seen else None
                    entry["busy_rtf"] = round(state["busy_s"] / media_duration_s, 4) if media_duration_s else None
                    entry["compute_ms_mean"] = round(1000 * state["busy_s"] / state["processed"], 2)
                    entry["latency_ms_p50"] = _percentile(self.latency[name], .5)
                    entry["latency_ms_p95"] = _percentile(self.latency[name], .95)
                models[name] = entry
        result = {"gpu_capacity_mib": self.capacity_mib, "media_duration_s": media_duration_s, "wall_s": wall_s,
                  "realtime_factor_wall": round(wall_s / media_duration_s, 4) if media_duration_s else None,
                  "budget_violations": self.violations[:50], "models": models,
                  "allocator": self.allocator.telemetry()}
        (self.out / "summary.json").write_text(json.dumps(result, indent=1))
        return result

from __future__ import annotations
import asyncio
import json
import platform
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping
from .packet import (
    EventPacket,
    MAX_PACKET_BYTES,
    PacketIntegrityCounter,
)
from .host_limits import HostLimitController, HostLimitError, HostLimitReport
from .resources import ResourceRequest


@dataclass
class ProcessRuntime:
    argv: tuple[str, ...]
    descriptor: object
    max_line: int = MAX_PACKET_BYTES
    process: asyncio.subprocess.Process | None = None
    integrity_counter: PacketIntegrityCounter = field(default_factory=PacketIntegrityCounter)
    host_controller: HostLimitController | None = None
    resource_request: ResourceRequest | None = None
    host_limit_report: HostLimitReport | None = None
    stderr_limit: int = 64
    _stderr_task: asyncio.Task | None = field(default=None, init=False, repr=False)
    _stderr_tail: deque[str] = field(default_factory=lambda: deque(maxlen=64), init=False, repr=False)
    _last_error: str | None = field(default=None, init=False, repr=False)

    def configure_host_limits(self, controller: HostLimitController | None,
                              request: ResourceRequest | None) -> None:
        if controller is not None and not isinstance(controller, HostLimitController):
            raise TypeError("host_controller must be a HostLimitController")
        if request is not None and not isinstance(request, ResourceRequest):
            raise TypeError("resource_request must be a ResourceRequest")
        if self.process is not None:
            raise RuntimeError("host limits must be configured before start")
        self.host_controller = controller
        self.resource_request = request

    async def _read_line(self):
        try:
            return await self.process.stdout.readuntil(b"\n")
        except asyncio.LimitOverrunError as exc:
            raise ValueError("runtime line exceeds line budget") from exc
        except asyncio.IncompleteReadError as exc:
            if not exc.partial:
                return b""
            raise ValueError("runtime returned a truncated line") from exc

    async def start(self):
        if not self.argv:
            raise ValueError("runtime argv cannot be empty")
        self.process = await asyncio.create_subprocess_exec(
            *self.argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=self.max_line,
        )
        self._stderr_task = asyncio.create_task(self._capture_stderr(), name="gamst-crew:stderr")
        try:
            if self.host_controller is not None:
                request = self.resource_request or ResourceRequest()
                self.host_limit_report = self.host_controller.attach(self.process.pid, request)
            ready = await asyncio.wait_for(self._read_line(), 5)
            if not ready:
                raise RuntimeError("runtime exited before readiness")
            message = json.loads(ready)
            if message.get("type") != "ready":
                raise RuntimeError("runtime readiness message required")
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            self._last_error = str(exc)
            await self.stop()
            raise RuntimeError("runtime readiness must be JSON") from exc
        except BaseException:
            self._last_error = "runtime failed during startup"
            await self.stop()
            raise

    async def _capture_stderr(self):
        """Drain stderr so a noisy provider cannot deadlock on its pipe."""
        if self.process is None or self.process.stderr is None:
            return
        try:
            while True:
                line = await self.process.stderr.readline()
                if not line:
                    return
                text = line.decode("utf-8", errors="replace").rstrip("\r\n")
                self._stderr_tail.append(text[-2048:])
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # diagnostics must never break lifecycle
            self._stderr_tail.append(f"stderr capture failed: {exc}")

    async def submit(self, packet: EventPacket | Mapping[str, Any] | bytes):
        if self.process is None or self.process.returncode is not None:
            self._last_error = "runtime is not running"
            raise RuntimeError("runtime is not running")
        if isinstance(packet, EventPacket):
            parsed = EventPacket.from_bytes(
                packet.to_bytes(), counter=self.integrity_counter
            )
            payload = parsed.to_dict()
        elif isinstance(packet, (bytes, bytearray, memoryview)):
            parsed = EventPacket.from_bytes(packet, counter=self.integrity_counter)
            payload = parsed.to_dict()
        elif isinstance(packet, Mapping):
            parsed = EventPacket.from_dict(packet, counter=self.integrity_counter)
            payload = parsed.to_dict()
        else:
            raise TypeError("runtime packet must be an EventPacket or packet mapping")
        raw = (
            json.dumps(
                {"type": "packet", "packet": payload},
                ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        if len(raw) > self.max_line:
            raise ValueError("runtime packet exceeds line budget")
        self.process.stdin.write(raw)
        await self.process.stdin.drain()
        try:
            response = await asyncio.wait_for(self._read_line(), 5)
        except Exception as exc:
            self._last_error = str(exc)
            raise
        if not response:
            raise RuntimeError("runtime closed stdout")
        if len(response) > self.max_line:
            raise ValueError("runtime response exceeds line budget")
        return json.loads(response)

    async def stop(self):
        if self.process is None:
            return
        process, self.process = self.process, None
        stderr_task, self._stderr_task = self._stderr_task, None
        try:
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), 2)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
        finally:
            if stderr_task is not None:
                stderr_task.cancel()
                await asyncio.gather(stderr_task, return_exceptions=True)
            if self.host_controller is not None and self.process is None:
                try:
                    self.host_limit_report = self.host_controller.detach(process.pid)
                except HostLimitError as exc:
                    # Shutdown must not mask the original startup/stop failure.
                    self.host_limit_report = HostLimitReport(
                        self.host_controller.mode,
                        "unavailable",
                        (self.host_controller.system or platform.system()).lower(),
                        str(exc),
                    )
            process._transport.close()

    def health(self):
        return {
            "running": self.process is not None and self.process.returncode is None,
            "returncode": None if self.process is None else self.process.returncode,
            "integrity_rejected": self.integrity_counter.integrity_rejected,
            "stderr_tail": list(self._stderr_tail),
            "last_error": self._last_error,
            "host_limits": None if self.host_limit_report is None else self.host_limit_report.as_dict(),
        }

"""Opt-in native tactile I/O. No device/dependency imports until explicit open.

Native calls cannot be interrupted. Cancellation drains and releases owned
handles; use a supervised process if an application needs a hard stop deadline.
"""
import asyncio
import hashlib
import importlib
from pathlib import Path
import time

from .tactile import TactileSource, tactile_frame

DIGIT_REVISION = '87a28bbf2beee8008a5308e9a12d72e1bc4fefb9'
GELSIGHT_API_REVISION = '321d6a22da64529138ff10237335038fd8c5189f'
DIGIT_SOURCES = {
    'digit': '44ec168b168732bcdd92482252c8986b85b279830c7ecabe671eb3b7b6f29604',
    'digit_handler': '08c70f4134ccf84fb3622a5cc05dde23d73bce51db12b459ceb5b3db5708f59c',
}


def _digit_class():
    try:
        modules = {}
        for name, expected in DIGIT_SOURCES.items():
            module = importlib.import_module('digit_interface.' + name)
            with Path(module.__file__).open('rb') as handle: payload = handle.read(131073)
            if len(payload) > 131072 or hashlib.sha256(payload).hexdigest() != expected:
                raise RuntimeError('installed DIGIT wrapper does not match pinned source')
            modules[name] = module
        return modules['digit'].Digit
    except ImportError as exc:
        raise RuntimeError('explicit DIGIT capture needs caller-installed pinned digit-interface') from exc


def _opencv():
    try:
        return importlib.import_module('cv2')
    except ImportError as exc:
        raise RuntimeError('explicit tactile capture needs caller-installed OpenCV and NumPy') from exc


async def _owned(function):
    """Drain one native call; outcome values avoid abandoned-shield exceptions."""
    def invoke():
        try: return function(), None
        except BaseException as exc: return None, exc
    task = asyncio.create_task(asyncio.to_thread(invoke))
    cancelled = False
    while not task.done():
        try: await asyncio.shield(task)
        except asyncio.CancelledError: cancelled = True
    value, error = task.result()
    return value, error, cancelled


class TactileCapture:
    """Single native handle with pull-based backpressure and serialized release.

    Source reference must match raw native BGR geometry. There are no automatic
    rotations, reference capture, camera probing, GUI, recording or restarts.
    """
    def __init__(self, source, *, consent=False):
        if not isinstance(source, TactileSource) or source.color_order != 'BGR':
            raise ValueError('native capture requires declared BGR TactileSource')
        if consent is not True: raise PermissionError('explicit tactile capture consent=True required')
        self.source = source
        self._lock = asyncio.Lock()
        self._handle = None
        self._release = None
        self._read = None
        self._opened = False
        self._stopping = False
        self._close_task = None
        self._errors = []
        self.sequence = 0
        self.backend = self.version = self.revision = None

    def _record(self, operation, error):
        if error is not None:
            self._errors.append(f'{operation}: {error}'[:1024])
            self._errors[:] = self._errors[-20:]

    async def _release_owned(self):
        handle, release = self._handle, self._release
        self._handle = self._release = self._read = None
        self._opened = False
        if handle is None: return None, False
        _, error, cancelled = await _owned(lambda: release(handle))
        self._record('release', error)
        return error, cancelled

    async def _open(self, constructor):
        # A cancelled caller waiting for this lock has not acquired ownership.
        async with self._lock:
            if self._stopping or self._opened or self._handle is not None:
                raise RuntimeError('tactile capture already opened or closed; construct a new epoch/session')
            _, error, cancelled = await _owned(constructor)
            self._record('open', error)
            if error is not None or cancelled or self._stopping:
                self._stopping = True
                release_error, release_cancelled = await self._release_owned()
                if cancelled or release_cancelled: raise asyncio.CancelledError
                if error is not None: raise error
                if release_error is not None: raise release_error
                raise RuntimeError('tactile startup interrupted by close')
            self._opened = True
            return self

    async def open_digit(self, *, fps=30, intensity=15):
        if self.source.sensor_model != 'digit': raise ValueError('DIGIT source required')
        if (self.source.width, self.source.height) not in ((320, 240), (640, 480)):
            raise ValueError('DIGIT native QVGA/VGA geometry required')
        if type(fps) is not int or fps not in ((30, 60) if self.source.width == 320 else (15, 30)):
            raise ValueError('unsupported DIGIT stream fps')
        if type(intensity) is not int or not 0 <= intensity <= 15: raise ValueError('intensity must be0..15')
        if (fps, intensity) != (self.source.configured_fps, self.source.configured_led_intensity):
            raise ValueError('DIGIT settings must match the declared reference profile')
        def construct():
            Digit = _digit_class()
            cv = _opencv()
            self.backend, self.version, self.revision = 'digit-interface', str(cv.__version__), DIGIT_REVISION
            self._release = lambda handle: handle.disconnect()
            self._handle = Digit(self.source.serial, 'gamst')
            self._handle.connect()
            self._handle.set_resolution(Digit.STREAMS['QVGA' if self.source.width == 320 else 'VGA'])
            self._handle.set_fps(fps)
            self._handle.set_intensity(intensity)
            self._read = lambda: self._handle.get_frame(transpose=True)
        return await self._open(construct)

    async def open_gelsight(self, *, device):
        if self.source.sensor_model != 'gelsight_mini': raise ValueError('GelSight Mini source required')
        if not ((type(device) is int and 0 <= device <= 255) or
                (type(device) is str and device.startswith('/dev/') and len(device) <= 256 and '..' not in device)):
            raise ValueError('explicit local camera index or /dev path required; URLs/files unsupported')
        if self.source.configured_fps is not None or self.source.configured_led_intensity is not None:
            raise ValueError('GelSight bridge does not configure fps or LEDs; declare both unknown')
        def construct():
            cv = _opencv()
            self.backend, self.version, self.revision = 'opencv-gelsight', str(cv.__version__), GELSIGHT_API_REVISION
            self._release = lambda handle: handle.release()
            self._handle = cv.VideoCapture(device)
            if not self._handle.isOpened(): raise RuntimeError('selected GelSight camera did not open')
            self._handle.set(cv.CAP_PROP_FRAME_WIDTH, self.source.width)
            self._handle.set(cv.CAP_PROP_FRAME_HEIGHT, self.source.height)
            def read():
                ok, pixels = self._handle.read()
                if not ok: raise RuntimeError('GelSight frame read failed')
                return pixels
            self._read = read
        return await self._open(construct)

    async def next_frame(self):
        async with self._lock:
            if self._stopping or not self._opened: raise RuntimeError('tactile capture not open')
            start = time.time_ns() // 1_000_000
            pixels, error, cancelled = await _owned(self._read)
            end = time.time_ns() // 1_000_000
            self._record('read', error)
            if error is None and not cancelled and not self._stopping:
                try:
                    import numpy as np
                    if type(pixels) is not np.ndarray or pixels.dtype != np.uint8 or pixels.shape != (self.source.height, self.source.width, 3):
                        raise ValueError('native tactile shape/dtype differs from declared raw reference')
                    # Validated bounds before copying a possibly noncontiguous array.
                    result = tactile_frame(pixels.tobytes(order='C'), self.source, sequence=self.sequence,
                        timestamp_ms=end, read_start_ms=start, read_end_ms=end, acquisition='host_read_end',
                        backend=self.backend, backend_version=self.version, backend_revision=self.revision)
                    self.sequence += 1
                    return result
                except BaseException as exc:
                    error = exc; self._record('packet', error)
            self._stopping = True
            release_error, release_cancelled = await self._release_owned()
            if cancelled or release_cancelled: raise asyncio.CancelledError
            if error is not None: raise error
            if release_error is not None: raise release_error
            raise RuntimeError('tactile read interrupted by close')

    async def _close(self):
        async with self._lock:
            error, _ = await self._release_owned()
            return error

    async def close(self):
        self._stopping = True
        if self._close_task is None: self._close_task = asyncio.create_task(self._close())
        cancelled = False
        while not self._close_task.done():
            try: await asyncio.shield(self._close_task)
            except asyncio.CancelledError: cancelled = True
        error = self._close_task.result()
        if cancelled: raise asyncio.CancelledError
        if error is not None: raise error

    def status(self):
        return dict(opened=self._opened, stopping=self._stopping, captures=self.sequence,
                    errors=list(self._errors), hardware_verified=False)

    async def __aenter__(self): return self
    async def __aexit__(self, *args): await self.close()

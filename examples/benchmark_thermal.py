"""Synthetic thermal compute benchmark; never sensor or accuracy evidence.

Run on each revision with: PYTHONPATH=. python3 examples/benchmark_thermal.py
Compare fixture/output hashes before comparing timing. No downloads or devices.
"""
import hashlib
import json
import platform
import random
import statistics
import struct
import time
import tracemalloc

from humansense import SensorMetadata
from humansense.plugins import InputFrame
from humansense.providers.thermal import TLinearThermal


def run():
    cases = []
    for width, height, count, repeats in ((160, 120, 1, 30), (640, 512, 16, 5)):
        rng = random.Random(20261114)
        values = [rng.choice((0, 65535, rng.randrange(25000, 35000)))
                  for _ in range(width*height)]
        payload = struct.pack('<'+str(len(values))+'H', *values)
        provider = TLinearThermal(width=width, height=height, kelvin_per_count=.01,
            byteorder='little', calibration_id='synthetic', emissivity=.95,
            rois={str(i): (i, i, width-i, height-i) for i in range(count)})
        frame = InputFrame(payload, 'thermal', 0,
            sensor=SensorMetadata('fixture', 'thermal', 0, calibration_id='synthetic'))
        expected = provider.process(frame)
        durations = []
        for _ in range(repeats):
            start = time.perf_counter()
            result = provider.process(frame)
            durations.append((time.perf_counter()-start)*1000)
            assert result == expected
        # Separate allocation pass so tracing does not distort the timing sample.
        tracemalloc.start()
        provider.process(frame)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        cases.append(dict(width=width, height=height, rois=count, repeats=repeats,
            fixture_sha256=hashlib.sha256(payload).hexdigest(),
            output_sha256=hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest(),
            p50_ms=statistics.median(durations),
            p95_ms=sorted(durations)[min(repeats-1, int(repeats*.95))],
            peak_python_allocated_bytes=peak))
        provider.close()
    return dict(evidence='synthetic compute only; no hardware/capture/accuracy validation',
                python=platform.python_version(), platform=platform.platform(), cases=cases)


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))

"""Paced local speech plus silence through the actual Silero worker and context."""
import argparse
import asyncio
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import resource
import sys
import time

from humansense import HumanSense, ProcessProvider, SensorMetadata, SensorStreams
from humansense.plugins import InputFrame
from humansense.schema import timestamp


async def run(args):
    import numpy as np
    import soundfile as sf
    samples, rate = sf.read(args.input, dtype='float32')
    if rate != 16000 or samples.ndim != 1 or not len(samples):
        raise ValueError('supply a nonempty mono 16 kHz recording; resample explicitly upstream')
    samples = np.concatenate((np.zeros(16000, dtype=np.float32), samples,
                              np.zeros(16000, dtype=np.float32)))
    started = time.perf_counter()
    worker = await ProcessProvider.open(args.worker_python, 'silero.activity',
        dict(path=args.model, sha256=args.sha256))
    report = dict(model_sha256=args.sha256, input_sha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        platform=platform.platform(), device='cpu', worker_python=args.worker_python,
        load_ms=(time.perf_counter()-started)*1000,
        parent_dependencies={name: version(name) for name in ('numpy', 'soundfile')},
        replay='paced 20ms chunks, 1s generated silence before and after recording; not live capture',
        duration_ms=len(samples)/16, decisions=[])
    latencies, ages = [], []
    async with HumanSense(consent=True) as human:
        streams = SensorStreams(human)
        streams.register(worker.spec, lambda: worker)
        await streams.open('recording', 'vad', [worker.spec.id])
        origin, start = timestamp(), time.monotonic()
        for sequence, offset in enumerate(range(0, len(samples), 320)):
            block = samples[offset:offset+320]
            await asyncio.sleep(max(0., start+(offset+len(block))/16000-time.monotonic()))
            started = time.perf_counter()
            await streams.submit(InputFrame(block, 'audio', round(origin+(offset+len(block))/16),
                sample_rate=16000, sensor=SensorMetadata('recording', 'vad', sequence)))
            await streams.drain()
            elapsed = (time.perf_counter()-started)*1000
            records = human.store.latest_signals('local-user', 0, timestamp())
            if records and records[0]['observation']['sensor']['sequence'] == sequence:
                latencies.append(elapsed)
                for decision in records[0]['observation']['value']['decisions']:
                    ages.append(timestamp()-decision['end_ms'])
                    report['decisions'].append(dict(decision, start_ms=decision['start_ms']-origin,
                                                   end_ms=decision['end_ms']-origin))
        report['final_freshness'] = [v['fresh'] for v in (await human.get_context())['observations'].values()]
    report['decision_count'] = len(report['decisions'])
    assert report['decision_count'] == len(samples)//512
    assert all(b['start_ms'] == a['end_ms'] for a, b in zip(report['decisions'], report['decisions'][1:]))
    for key, data in (('submit_to_observation_ms', latencies), ('decision_end_to_observation_ms', ages)):
        report[key] = dict(p50=float(np.percentile(data, 50)), p95=float(np.percentile(data, 95)), maximum=max(data))
    report['transitions'] = [d for d in report['decisions'] if d['transition'] != 'none']
    divisor = 1024*1024 if platform.system() == 'Darwin' else 1024
    report['parent_peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/divisor
    child = resource.getrusage(resource.RUSAGE_CHILDREN)
    report['child_peak_rss_mib'] = child.ru_maxrss/divisor
    report['child_cpu_seconds'] = child.ru_utime+child.ru_stime
    report['torch_imported_in_parent'] = 'torch' in sys.modules
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'decisions'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--worker-python', default=sys.executable)
    parser.add_argument('--output', required=True)
    asyncio.run(run(parser.parse_args()))

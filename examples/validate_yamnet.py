"""Replay labeled local PCM16 WAVs through actual YAMNet and context storage."""
import argparse
import asyncio
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import resource
import time
import wave

from humansense import HumanSense, SensorMetadata, SensorStreams, ProcessProvider
from humansense.plugins import InputFrame
from humansense.providers.environment_audio import YAMNetEvents, YAMNET_SHA256
from humansense.schema import timestamp


async def run(args):
    import numpy as np
    from scipy.signal import resample_poly
    labels = {}
    if args.labels_csv:
        with open(args.labels_csv, newline='') as source:
            labels = {row['filename']: row['category'] for row in csv.DictReader(source)}
    started = time.perf_counter()
    if args.worker_python:
        provider = await ProcessProvider.open(args.worker_python, 'yamnet.events', {'path': args.model})
    else:
        provider = YAMNetEvents.from_local(args.model)
    load_ms = (time.perf_counter()-started)*1000
    spec = provider.spec
    report = dict(model_sha256=YAMNET_SHA256, model_version=spec.version, platform=platform.platform(),
        device='cpu', load_ms=load_ms, worker_python=args.worker_python,
        parent_dependencies={name: version(name) for name in ('numpy', 'scipy')},
        replay=('paced' if args.pace else 'accelerated historical')+' prerecorded 20ms blocks; not live microphone capture', fixtures=[])
    timings, capture_ages = [], []
    async with HumanSense(consent=True) as human:
        streams = SensorStreams(human)
        streams.register(spec, lambda: provider)
        await streams.open('ambient', 'replay', [spec.id])
        sequence = 0
        # Monotonic capture timeline ending before now, so accelerated replay
        # never submits future timestamps. Freshness is reported, not assumed.
        duration_ms = 0
        decoded = []
        for filename in args.input:
            with wave.open(filename, 'rb') as wav:
                if wav.getsampwidth() != 2 or wav.getnchannels() != 1:
                    raise ValueError('validation fixtures must be mono PCM16 WAV')
                rate = wav.getframerate()
                samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(np.float32)/32768
            if rate != 16000:
                from math import gcd
                divisor = gcd(rate, 16000)
                samples = resample_poly(samples, 16000//divisor, rate//divisor).astype(np.float32)
            # Resampling can create small overshoots; explicit clipping is
            # recorded rather than silently changing the provider input contract.
            clipped = int(np.count_nonzero(np.abs(samples) > 1))
            samples = np.clip(samples, -1, 1)
            decoded.append((filename, rate, samples, clipped))
            duration_ms += len(samples)/16+1000
        capture_end = timestamp()-int(duration_ms)-1000
        for filename, rate, samples, clipped in decoded:
            if sequence:
                sequence += 1  # Explicit discontinuity: runtime resets and rotates the lane epoch.
            if args.pace:
                capture_end = timestamp()
                clock_start = time.monotonic()
            fixture = dict(file=Path(filename).name, reference_label=labels.get(Path(filename).name),
                sha256=hashlib.sha256(Path(filename).read_bytes()).hexdigest(), original_rate=rate,
                resampler='scipy.signal.resample_poly', clipped_resampled_samples=clipped,
                windows=[])
            for offset in range(0, len(samples), 320):
                block = samples[offset:offset+320]
                if args.pace:
                    # Absolute deadlines prevent inference time and scheduler
                    # delay from accumulating into an artificial capture lag.
                    deadline = clock_start+(offset+len(block))/16000
                    await asyncio.sleep(max(0., deadline-time.monotonic()))
                capture_end += len(block)/16
                started = time.perf_counter()
                await streams.submit(InputFrame(block, 'audio', round(capture_end), sample_rate=16000,
                    sensor=SensorMetadata('ambient', 'replay', sequence)))
                await streams.drain()
                elapsed_ms = (time.perf_counter()-started)*1000
                records = human.store.latest_signals('local-user', 0, timestamp())
                record = records[0] if records else None
                if record and record['observation']['sensor']['sequence'] == sequence:
                    value = record['observation']['value']
                    assert value['window']['sample_count'] == 15600
                    fixture['windows'].append(dict(offset_end_ms=(offset+len(block))/16,
                        submit_to_observation_ms=elapsed_ms,
                        capture_end_to_observation_ms=timestamp()-record['timestamp_ms'], value=value))
                    timings.append(elapsed_ms)
                    capture_ages.append(timestamp()-record['timestamp_ms'])
                sequence += 1
            assert fixture['windows'], 'fixture did not produce any model output'
            report['fixtures'].append(fixture)
            capture_end += 1000
        report['final_context_freshness'] = [v['fresh'] for v in (await human.get_context())['observations'].values()]
    report['processing_latency_ms'] = dict(count=len(timings), p50=float(np.percentile(timings, 50)),
        p95=float(np.percentile(timings, 95)), maximum=max(timings))
    report['capture_end_to_observation_ms'] = dict(p50=float(np.percentile(capture_ages, 50)),
        p95=float(np.percentile(capture_ages, 95)), maximum=max(capture_ages))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['parent_process_peak_rss_mib'] = rss/(1024*1024 if platform.system() == 'Darwin' else 1024)
    child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    report['child_process_peak_rss_mib'] = child_usage.ru_maxrss/(1024*1024 if platform.system() == 'Darwin' else 1024)
    report['child_cpu_seconds'] = child_usage.ru_utime+child_usage.ru_stime
    report['rss_scope'] = 'parent peak including imports; excludes child memory in worker mode'
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ('load_ms', 'processing_latency_ms', 'parent_process_peak_rss_mib')}))
    for fixture in report['fixtures']:
        scores = {}
        for window in fixture['windows']:
            for event in window['value']['events']:
                scores[event['label']] = max(scores.get(event['label'], 0), event['score'])
        print(fixture['reference_label'], sorted(scores.items(), key=lambda item: -item[1])[:6])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--input', nargs='+', required=True)
    parser.add_argument('--labels-csv')
    parser.add_argument('--worker-python')
    parser.add_argument('--output', required=True)
    parser.add_argument('--pace', action='store_true', help='Pace prerecorded chunks at their sample rate')
    asyncio.run(run(parser.parse_args()))

"""Actual local-model replay evidence. No download, camera or microphone access."""
import argparse
import asyncio
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import time

from humansense import HumanSense, SensorMetadata, SensorStreams, ProcessProvider
from humansense.plugins import InputFrame
from humansense.providers.detection import RFDETRObjects
from humansense.schema import timestamp


async def run(args):
    import cv2
    import numpy as np
    data = cv2.imread(args.image)
    if data is None:
        raise ValueError('Cannot decode input image')
    start = time.perf_counter()
    if args.worker_python:
        provider = await ProcessProvider.open(args.worker_python, 'rfdetr.nano',
            dict(path=args.checkpoint, sha256=args.sha256, device=args.device))
        spec = provider.spec
    else:
        provider = RFDETRObjects.from_local(args.checkpoint, sha256=args.sha256, device=args.device)
        spec = provider.capability(provider.version)
    load_ms = (time.perf_counter()-start)*1000
    frames = [data, data, cv2.warpAffine(data, np.float32([[1, 0, 5], [0, 1, 0]]),
              (data.shape[1], data.shape[0])), np.zeros_like(data)]
    packages = ('numpy',) if args.worker_python else ('rfdetr', 'torch', 'torchvision', 'transformers', 'supervision', 'numpy')
    report = dict(model_version=spec.version, device=args.device, platform=platform.platform(),
        worker_python=args.worker_python, parent_python=sys.executable,
        parent_dependencies={name: version(name) for name in packages},
        input_sha256=hashlib.sha256(Path(args.image).read_bytes()).hexdigest(),
        scenario='real image, repeated image, synthetic 5px translation, blank frame; not a natural video',
        load_ms=load_ms, frames=[])
    async with HumanSense(consent=True) as human:
        streams = SensorStreams(human)
        streams.register(spec, lambda: provider)
        await streams.open('replay', 'image', [spec.id])
        capture_ms = 0
        for index, data in enumerate(frames):
            capture_ms = max(capture_ms+1, timestamp())
            start = time.perf_counter()
            await streams.submit(InputFrame(data, 'video', capture_ms,
                sensor=SensorMetadata('replay', 'image', index, coordinate_frame='camera.normalized')))
            await streams.drain()
            elapsed = (time.perf_counter()-start)*1000
            record = human.store.latest_signals('local-user', 0, timestamp())[0]
            if record['observation']['sensor']['sequence'] != index:
                raise AssertionError('Detector lane failed to emit the expected frame')
            value = record['observation']['value']
            await human.remember_objects(record['id'], complete_snapshot=value['summary']['complete_snapshot'])
            context = await human.get_context()
            fresh = sum(entry['fresh'] for entry in context['observations'].values())
            report['frames'].append(dict(sequence=index, submit_to_observation_ms=elapsed,
                fresh_context_observations=fresh,
                stale_context_observations=len(context['observations'])-fresh, value=value))
        report['memory'] = await human.find_objects()
        assert report['frames'][0]['value']['objects'], 'Fixture must produce detections'
        assert not report['frames'][-1]['value']['objects'], 'Blank frame unexpectedly produced detections'
        assert report['memory'] and all(not item['currently_visible'] for item in report['memory'])
    report['parent_loaded_torch'] = 'torch' in sys.modules
    report['parent_loaded_rfdetr'] = 'rfdetr' in sys.modules
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({key: report[key] for key in ('model_version', 'device', 'load_ms')}))
    print('latencies_ms:', [round(row['submit_to_observation_ms'], 2) for row in report['frames']])
    print('labels:', [row['label'] for row in report['frames'][0]['value']['objects']])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default='cpu')
    parser.add_argument('--output', required=True)
    parser.add_argument('--worker-python', help='Run the model in this separately installed Python environment')
    asyncio.run(run(parser.parse_args()))

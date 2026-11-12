"""Optional SAM3 inference in an isolated, project-owned Python worker.

Frames exist only in pipe buffers. This is sampled image segmentation, not the
native SAM3 video tracker, identity recognition or a calibrated emotion model.
"""
from __future__ import annotations

import asyncio
import base64
import json
import math
from pathlib import Path
import sys

MAX_REPLY_BYTES = 1_000_000


def validate_segments(value):
    """Enforce a small normalized geometry contract before crossing into context."""
    if not isinstance(value, list) or len(value) > 16:
        raise ValueError('Invalid SAM3 segment count')
    output = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError('Invalid SAM3 segment')
        box = item.get('bbox', {})
        values = [box.get(k) for k in ('x', 'y', 'width', 'height')]
        score = item.get('score')
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in [*values, score]):
            raise ValueError('SAM3 coordinates and scores must be finite')
        x, y, width, height = values
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1
                and x+width <= 1.000001 and y+height <= 1.000001 and 0 <= score <= 1):
            raise ValueError('SAM3 coordinates must be normalized')
        polygons = item.get('polygons', [])
        if not isinstance(polygons, list) or len(polygons) > 4:
            raise ValueError('Too many SAM3 contours')
        for polygon in polygons:
            if not isinstance(polygon, list) or not 3 <= len(polygon) <= 128:
                raise ValueError('Invalid SAM3 contour')
            for point in polygon:
                if not isinstance(point, list) or len(point) != 2 or any(
                    type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in point):
                    raise ValueError('Invalid SAM3 polygon point')
        output.append({'bbox': dict(zip(('x','y','width','height'), values)),
                       'score': score, 'polygons': polygons})
    return output


class LocalSAM3Plugin:
    name = 'sam3_segmentation'
    modalities = ('video',)

    def __init__(self, model_path, python_executable, revision, prompt='person',
                 interval=2.0, threshold=.5, device='cpu', timeout=180):
        if not Path(model_path).is_dir() or not Path(python_executable).is_file():
            raise ValueError('Install the project-owned SAM3 model and runtime first')
        if not isinstance(revision, str) or not revision:
            raise ValueError('A pinned SAM3 revision is required')
        if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 200:
            raise ValueError('SAM3 prompt must contain 1..200 characters')
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in (interval,threshold,timeout)):
            raise ValueError('SAM3 numeric settings must be finite')
        if not .25 <= interval <= 60 or not 0 <= threshold <= 1 or not 1 <= timeout <= 600:
            raise ValueError('SAM3 settings out of bounds')
        if device not in ('cpu','mps','cuda'):
            raise ValueError('Choose cpu, mps or cuda')
        self.model_path, self.python_executable = str(Path(model_path).resolve()), str(Path(python_executable).absolute())
        self.version, self.prompt = revision, prompt.strip()
        self.interval, self.threshold, self.device, self.timeout = interval, threshold, device, timeout
        self._lock = asyncio.Lock()
        self._process = None
        self._stderr_task = None
        self._generation = self._request_id = 0
        self._closed = False
        self._last_sample = -float('inf')
        self._subject = None
        self.last_error = None
        self.loaded = False

    async def prepare(self, frame):
        from ..plugins import InputFrame
        if self._closed:
            return None
        if self._subject != frame.subject_id:
            self._subject = frame.subject_id
            self._last_sample = -float('inf')
        if frame.timestamp_ms-self._last_sample < self.interval*1000:
            return None
        import cv2
        h, w = frame.data.shape[:2]
        scale = min(1., 1024/max(h,w))
        image = cv2.resize(frame.data, (max(1,round(w*scale)),max(1,round(h*scale))))
        ok, jpeg = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise ValueError('Cannot encode SAM3 frame')
        encoded = base64.b64encode(jpeg).decode()
        if len(encoded) > MAX_REPLY_BYTES - 4096:
            raise ValueError('SAM3 frame too large')
        self._last_sample = frame.timestamp_ms
        return InputFrame({'sam3_jpeg': encoded}, 'video', frame.timestamp_ms, frame.subject_id)

    async def _start_worker(self):
        if self._process and self._process.returncode is None:
            return
        self._process = await asyncio.create_subprocess_exec(
            self.python_executable, str(Path(__file__).resolve()), '--worker', '--model-path', self.model_path,
            '--device', self.device, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, limit=MAX_REPLY_BYTES+1)
        async def drain_stderr(process):
            # Drain diagnostics without retaining raw requests, tokens or unbounded logs.
            while await process.stderr.read(4096):
                pass
        self._stderr_task = asyncio.create_task(drain_stderr(self._process))

    async def _stop_worker(self):
        process, self._process = self._process, None
        self.loaded = False
        stderr_task, self._stderr_task = self._stderr_task, None
        if process and process.returncode is None:
            try: process.terminate()
            except ProcessLookupError: pass
            try: await asyncio.wait_for(process.wait(), 3)
            except asyncio.TimeoutError:
                try: process.kill()
                except ProcessLookupError: pass
                await process.wait()
        if stderr_task:
            await asyncio.gather(stderr_task, return_exceptions=True)

    async def process(self, frame):
        from ..plugins import _signal
        if not isinstance(frame.data, dict) or 'sam3_jpeg' not in frame.data:
            frame = await self.prepare(frame)
            if frame is None:
                return []
        generation = self._generation
        async with self._lock:
            if generation != self._generation or self._closed:
                return []
            self._request_id += 1
            request_id = self._request_id
            try:
                await self._start_worker()
                if generation != self._generation or self._closed:
                    await self._stop_worker()
                    return []
                process = self._process
                request = {'id': request_id, 'jpeg': frame.data['sam3_jpeg'],
                           'prompt': self.prompt, 'threshold': self.threshold}
                process.stdin.write((json.dumps(request)+'\n').encode())
                await process.stdin.drain()
                raw = await asyncio.wait_for(process.stdout.readline(), self.timeout)
                if not raw or len(raw) > MAX_REPLY_BYTES:
                    raise RuntimeError('SAM3 worker exited or returned an oversized result')
                reply = json.loads(raw)
                if reply.get('id') != request_id:
                    raise RuntimeError('SAM3 worker response sequence mismatch')
                if reply.get('error'):
                    raise RuntimeError(str(reply['error'])[:300])
                segments = validate_segments(reply.get('segments'))
            except asyncio.CancelledError:
                await self._stop_worker()
                raise
            except Exception as exc:
                self.last_error = type(exc).__name__ + ': ' + str(exc)[:300]
                await self._stop_worker()
                raise
            if generation != self._generation or self._closed:
                await self._stop_worker()
                return []
            self.last_error = None
            self.loaded = True
            for segment in segments:
                segment['prompt'] = self.prompt
            return [_signal(frame, 'scene.segments', segments, self.name, self.version),
                    _signal(frame, 'scene.segment_count', len(segments), self.name, self.version)]

    async def reset(self):
        self._generation += 1
        self._last_sample = -float('inf')
        self._subject = None
        await self._stop_worker()

    async def close(self):
        self._closed = True
        await self.reset()


def _worker_main(model_path, device):
    # All heavy imports are isolated from the main voice runtime.
    import contextlib
    import io
    with contextlib.redirect_stdout(sys.stderr):
        import cv2
        import numpy as np
        import torch
        from PIL import Image
        from transformers import Sam3Model, Sam3Processor
        torch.set_num_threads(2)
        model = Sam3Model.from_pretrained(model_path, local_files_only=True).to(device).eval()
        processor = Sam3Processor.from_pretrained(model_path, local_files_only=True)
    while True:
        line = sys.stdin.buffer.readline(MAX_REPLY_BYTES+1)
        if not line:
            return
        request_id = None
        try:
            if len(line) > MAX_REPLY_BYTES or not line.endswith(b'\n'):
                raise ValueError('SAM3 input too large')
            request = json.loads(line)
            request_id = request['id']
            prompt = request['prompt']
            if not isinstance(prompt,str) or not 1 <= len(prompt) <= 200:
                raise ValueError('Invalid prompt')
            image = Image.open(io.BytesIO(base64.b64decode(request['jpeg'],validate=True))).convert('RGB')
            width, height = image.size
            if max(width,height) > 1024:
                raise ValueError('Input dimensions exceed budget')
            with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
                inputs = processor(images=image,text=prompt,return_tensors='pt').to(model.device)
                outputs = model(**inputs)
                result = processor.post_process_instance_segmentation(outputs,
                    threshold=float(request['threshold']), mask_threshold=.5,
                    target_sizes=inputs.get('original_sizes').tolist())[0]
            segments=[]
            for index in range(min(16,len(result['scores']))):
                x0,y0,x1,y1 = result['boxes'][index].detach().cpu().tolist()
                x0,x1 = max(0.,min(1.,x0/width)),max(0.,min(1.,x1/width))
                y0,y1 = max(0.,min(1.,y0/height)),max(0.,min(1.,y1/height))
                if x1<=x0 or y1<=y0: continue
                mask=result['masks'][index].detach().cpu().numpy().squeeze().astype(np.uint8)
                contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                polygons=[]
                for contour in sorted(contours,key=cv2.contourArea,reverse=True)[:4]:
                    contour=cv2.approxPolyDP(contour,max(1.,.003*cv2.arcLength(contour,True)),True).reshape(-1,2)
                    if len(contour)<3:continue
                    if len(contour)>128:contour=contour[np.linspace(0,len(contour)-1,128,dtype=int)]
                    polygons.append([[max(0.,min(1.,float(x)/width)),max(0.,min(1.,float(y)/height))] for x,y in contour])
                segments.append({'bbox':{'x':x0,'y':y0,'width':x1-x0,'height':y1-y0},
                                 'score':float(result['scores'][index].detach().cpu()),'polygons':polygons})
            response={'id':request_id,'segments':validate_segments(segments)}
        except Exception as exc:
            response={'id':request_id,'error':type(exc).__name__+': '+str(exc)[:200]}
        sys.stdout.write(json.dumps(response)+'\n');sys.stdout.flush()


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',action='store_true',required=True)
    parser.add_argument('--model-path',required=True)
    parser.add_argument('--device',choices=('cpu','mps','cuda'),default='cpu')
    args=parser.parse_args()
    _worker_main(args.model_path,args.device)

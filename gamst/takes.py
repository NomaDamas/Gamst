"""GAMST 240 takes as Pipeline modules: the take recorder and the after-take flip counter.

The coin-flip recipe composes them behind :class:`gamst.inputs.IPhone240Input`::

    camera (gamst240)  ->  record (TakeRecorder)  --take-->  count (TakeFlips)  ->  vision.rotation.refined

- ``record`` writes each take (Start→Stop on the phone) to its own folder: ``raw.gmst`` (every
  frame as received; replay it with ``--take``), ``summary.json`` and, when a view draws one,
  ``hud.mp4``. A finished take becomes a frame of the derived input ``take``; the take is media
  and stays on disk, never in context.
- ``count`` counts the throws of one named object in a take, after the take, in a separate process
  so the analysis never competes with capture (:func:`refine_take`, which writes ``refined.json``
  and ``refined.mp4``), and commits each throw as a ``vision.rotation.refined`` observation.

``python -m gamst.takes refine TAKE_DIR --label ...`` counts a saved take without a runtime;
``render TAKE_DIR`` re-draws ``refined.mp4`` from ``refined.json``.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .inputs import decode, read_recording
from .pipeline import Derived


# ---------------------------------------------------------------- files

class VideoSink:
    """H.264 via ffmpeg when available (upload-ready), else OpenCV mp4v."""

    def __init__(self, path, fps, size):
        self.path, self.size = str(path), size
        ffmpeg = shutil.which('ffmpeg')
        if ffmpeg:
            self.process = subprocess.Popen(
                [ffmpeg, '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                 '-s', f'{size[0]}x{size[1]}', '-r', str(fps), '-i', '-', '-c:v', 'libx264',
                 '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', self.path],
                stdin=subprocess.PIPE)
            self.writer = None
        else:
            import cv2
            self.process = None
            self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    def write(self, image):
        if self.process is not None:
            self.process.stdin.write(image.tobytes())
        else:
            self.writer.write(image)

    def close(self):
        if self.process is not None:
            self.process.stdin.close()
            self.process.wait()
        else:
            self.writer.release()


class Take:
    """One Start→Stop take: the raw stream, a real-time HUD video (when a view draws one) and a summary.

    Counting runs afterwards in a separate process (``refined.json``, ``refined.mp4``), started on
    Stop or by the HUD's ANALYZE button, so the analysis never competes with capture.
    """

    def __init__(self, root, *, hud_size=None):
        stamp = time.strftime('%Y%m%d-%H%M%S')
        self.dir = Path(root) / f'take-{stamp}'
        number = 1
        while self.dir.exists():  # two takes in one second (a replayed recording) keep both
            number += 1
            self.dir = Path(root) / f'take-{stamp}-{number}'
        self.dir.mkdir(parents=True)
        self.raw = open(self.dir / 'raw.gmst', 'wb')
        self.hud = VideoSink(self.dir / 'hud.mp4', 30, hud_size) if hud_size else None
        self.frames, self.first, self.last = 0, None, None
        self._hud_due = None

    def add(self, header, payload, seconds):
        self.raw.write(header + payload)
        self.frames += 1
        self.first = seconds if self.first is None else self.first
        self.last = seconds

    def hud_wanted(self, seconds):
        """Does ``hud.mp4`` need a picture for the frame captured at ``seconds``?"""
        return self.hud is not None and (self._hud_due is None or seconds >= self._hud_due)

    def write_hud(self, image, seconds):
        """``hud.mp4`` advances on capture time, so it plays in real time (30 fps)."""
        if self.hud is None:
            return
        if self._hud_due is None:
            self._hud_due = seconds
        while seconds >= self._hud_due:
            self.hud.write(image)
            self._hud_due += 1 / 30

    def close(self):
        self.raw.close()
        if self.hud is not None:
            self.hud.close()
        span = (self.last - self.first) if self.frames > 1 else 0
        summary = dict(frames=self.frames, seconds=round(span, 3),
                       capture_fps=round((self.frames - 1) / span, 1) if span else None)
        (self.dir / 'summary.json').write_text(json.dumps(summary, indent=2))
        return self.dir



# ---------------------------------------------------------------- Pipeline modules

class TakeRecorder:
    """Writes each take of a GAMST 240 input under ``root``; a finished take is emitted as ``take``.

    A take starts with the first recording frame (``PhoneFrame.recording``) and ends at the first
    frame without the flag, a disconnect, or the end of the input (``finish``). The emitted item is
    ``{"dir", "frames", "clock"}``: ``clock`` maps the phone's capture ms to the runtime clock, so
    counts land at the moment the throw happened. A view sets ``hud_size`` to record its screen.
    """
    name, version = 'gamst240.take_recorder', '1'

    def __init__(self, root, *, source=None, emits='take'):
        self.root, self.source, self.emits = Path(root), source, emits
        self.take = None   # the take being recorded
        self.saved = []    # finished take folders, in order
        self.hud_size = None

    def process(self, frame):
        item = frame.data
        if not getattr(item, 'recording', False):
            return self._end()
        if self.take is None:
            self.take = Take(self.root, hud_size=self.hud_size)
            print(json.dumps({'take': 'started', 'dir': str(self.take.dir)}), flush=True)
        self.take.add(item.header, item.payload, item.info.seconds)
        return []

    def _end(self):
        if self.take is None:
            return []
        take, self.take = self.take, None
        directory = take.close()
        self.saved.append(directory)
        print(json.dumps({'take': 'saved', 'dir': str(directory), 'frames': take.frames}), flush=True)
        return [Derived(self.emits, dict(dir=directory, frames=take.frames,
                                         clock=getattr(self.source, 'clock', None)), 'take')]

    def finish(self):
        return self._end()

    def close(self):
        if self.take is not None:
            self.saved.append(self.take.close())
            self.take = None
        print(json.dumps({'takes': [str(path) for path in self.saved]}), flush=True)


class _ServerJob:
    """A take queued on the refine server; finished when its progress says done or failed."""

    def __init__(self, directory, server):
        self.path, self.server = directory / 'refine.progress.json', server

    def poll(self):
        try:
            stage = json.loads(self.path.read_text()).get('stage')
        except (OSError, ValueError):
            stage = None
        return 0 if stage in ('done', 'failed') else None if self.server.poll() is None else 1


class _Finished:
    """Stands in for a refine process that already ran (a cached take)."""

    def poll(self):
        return 0


class TakeFlips:
    """Counts the throws of one named object in each take, after the take; commits ``vision.rotation.refined``.

    Consumes the ``take`` input of :class:`TakeRecorder`. Counting (:func:`refine_take`: YOLOE
    finds ``label`` in native-resolution tiles, EfficientTAM follows it, the flight's outline
    thinning is counted, :mod:`gamst.providers.rotation_refine`) runs in a separate process and
    starts when a take ends (``on="stop"``) or on :meth:`analyze` (``on="analyze"``, the HUD's
    ANALYZE). ``count_at_end`` counts the last take when the input ends (a replay; nobody presses
    ANALYZE) and waits for it, so its throws reach the Store.

    ``fast``: one long-lived refine server keeps the models loaded and follows only the most
    confidently named throw; otherwise each take gets its own process and every throw is counted.
    ``cached``: a take counted beforehand, shown at once on ANALYZE (stage backup).
    """
    name, version = 'vision.rotation.refined', '1'

    def __init__(self, human, *, label, models, device, efficienttam_source, worker_python=None,
                 expected_max_rev_s=40., on='analyze', fast=True, count_at_end=False, show=False,
                 cached=None, log_dir='gamst-takes'):
        if on not in ('analyze', 'stop'):
            raise ValueError("on must be 'analyze' or 'stop'")
        self.human, self.label, self.on, self.fast = human, label, on, bool(fast)
        self.count_at_end, self.show, self.cached = count_at_end, show, cached
        self.log_dir = Path(log_dir)
        self.refine_args = [
            '--label', label, '--efficienttam-source', str(efficienttam_source), '--models', str(models),
            '--device', device, '--worker-python', worker_python or sys.executable,
            '--expected-max-rev-s', str(expected_max_rev_s)] + (['--fast'] if self.fast else [])
        self.last = None       # the newest take
        self.started = set()   # take folders whose count was started
        self.pending = []      # takes being counted, until their refine process exits
        self.records = []      # committed vision.rotation.refined records, in order
        self.progress, self.take_frames, self.take_fps = None, 0, 0.
        self.server, self._task, self._lock = None, None, asyncio.Lock()

    def open(self):
        """With ``fast``, start the refine server now, so the models are loaded before the first ANALYZE."""
        if self.fast and self.server is None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.server = subprocess.Popen(
                [sys.executable, '-m', 'gamst.takes', 'refine-server', *self.refine_args], stdin=subprocess.PIPE,
                stdout=open(self.log_dir / 'refine-server.log', 'w'), stderr=subprocess.STDOUT, text=True,
                start_new_session=True)
        return self

    def process(self, frame):
        self.last = frame.data
        if self.on == 'stop':
            self.start(self.last)
        return None

    def _launch(self, directory):
        """Start counting ``directory``; returns something with ``poll()`` (None while it runs)."""
        if self.server is not None and self.server.poll() is None:
            self.server.stdin.write(f'{directory}\n')
            self.server.stdin.flush()
            return _ServerJob(directory, self.server)
        with open(directory / 'refine.log', 'w') as log:
            return subprocess.Popen([sys.executable, '-m', 'gamst.takes', 'refine', str(directory), *self.refine_args],
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

    def start(self, take):
        directory = Path(take['dir'])
        summary = json.loads((directory / 'summary.json').read_text())
        self.take_frames, self.take_fps, self.progress = summary['frames'], summary['capture_fps'] or 0., None
        (directory / 'refine.progress.json').unlink(missing_ok=True)
        self.started.add(directory)
        self.pending.append(dict(dir=directory, clock=take['clock'], process=self._launch(directory),
                                 started=time.monotonic(), published=False))
        print(json.dumps({'refine': 'started', 'dir': str(directory)}), flush=True)
        self._watch()

    def analyze(self):
        """ANALYZE: count the newest take (or show the cached one)."""
        if self.cached:
            cached = Path(self.cached)
            recording = read_recording(cached / 'raw.gmst')
            first, span = recording[0][0].seconds, recording[-1][0].seconds - recording[0][0].seconds
            base = time.time_ns() // 1_000_000 - round(span * 1000) - 500
            self.take_frames, self.take_fps = len(recording), (len(recording) - 1) / span if span else 0.
            self.pending.append(dict(dir=cached, clock=lambda ms: base + round((ms / 1000 - first) * 1000),
                                     process=_Finished(), started=time.monotonic(), published=False))
            self._watch()
            return
        if self.last is not None and not any(item['dir'] == Path(self.last['dir']) for item in self.pending):
            self.start(self.last)

    def _watch(self):
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._poll_until_done(.1))

    async def _poll_until_done(self, every_s):
        while self.pending:
            await self.poll()
            await asyncio.sleep(every_s)

    async def poll(self):
        """Commit finished counts (registered observations + named objects in ObjectMemory)."""
        from .providers.rotation_refine import publish
        async with self._lock:
            for item in list(self.pending):
                path = item['dir'] / 'refined.json'
                if not item['published']:
                    try:
                        self.progress = json.loads((item['dir'] / 'refine.progress.json').read_text())
                    except (OSError, ValueError):
                        pass
                if not item['published'] and path.exists():
                    item['published'] = True
                    output = json.loads(path.read_text())
                    records = await publish(self.human, output['results'], take_id=item['dir'].name,
                                            clock=item['clock'])
                    self.records += records
                    objects = await self.human.find_objects(text=output['label'])
                    print(json.dumps({'refined': str(item['dir']), 'timing': output.get('timing'),
                                      'records': [record['id'] for record in records],
                                      'objects': [dict(entity_id=entry['entity_id'], label=entry['label'],
                                                       track_id=entry['track_id']) for entry in objects]}),
                          flush=True)
                if item['process'].poll() is None:
                    continue
                self.pending.remove(item)
                video = item['dir'] / 'refined.mp4'
                if not item['published']:
                    print(json.dumps({'refine_failed': str(item['dir']), 'log': str(item['dir'] / 'refine.log')}),
                          flush=True)
                elif self.show and video.exists() and shutil.which('open'):
                    subprocess.Popen(['open', str(video)])

    def status(self):
        """What the HUD shows while counting: seconds waited, the refine stage, the take's size."""
        waited = max((time.monotonic() - item['started'] for item in self.pending if not item['published']),
                     default=0.)
        return dict(pending_s=waited, progress=self.progress, take_frames=self.take_frames, take_fps=self.take_fps,
                    ready=self.last is not None or bool(self.cached))

    async def finish(self):
        if not self.count_at_end:
            return None
        if self.cached or self.last is not None and Path(self.last['dir']) not in self.started:
            self.analyze()
        while self.pending:  # wait, so the throws reach the Store before the run ends
            await self.poll()
            await asyncio.sleep(.5)
        return None

    async def close(self):
        if self._task is not None:
            self._task.cancel()
        if self.server is not None:
            self.server.terminate()


# ---------------------------------------------------------------- counting a saved take (separate process)

async def refine_take(directory, *, label, models, device, efficienttam_source, worker_python,
                      detect_every=4, threshold=.2, fast=False, batch=16, warm=None, expected_max_rev_s=40.):
    """Re-count a saved take's throws of one named object; writes ``refined.json``.

    YOLOE (GAMST's pinned worker, text prompt ``label``) looks at every ``detect_every``-th frame
    in native-resolution tiles; its hits seed EfficientTAM, which follows that one object through
    each throw. See :mod:`gamst.providers.rotation_refine`.

    ``fast`` (for live demos, one throw per take):

    - YOLOE runs in this already separate process, ``batch`` tiles per call, on every 2nd frame
      where the picture moves.
    - Only the most confidently named throw is followed, and the remaining seeds are skipped once
      a complete free-fall count exists.

    ``refine.progress.json`` reports the current stage for the HUD.
    """
    import cv2
    import numpy as np
    from .capabilities import SensorMetadata
    from .demos.rotation import REFINE_ASSETS, detector_config
    from .halo import render_refined
    from .plugins import InputFrame
    from .providers.rotation_refine import EfficientTAMVideo, moving_frames, refine, tiles
    began = time.perf_counter()
    directory, models = Path(directory), Path(models)

    def report(stage, done, total):
        (directory / 'refine.progress.part.json').write_text(json.dumps(dict(stage=stage, done=done, total=total)))
        (directory / 'refine.progress.part.json').rename(directory / 'refine.progress.json')

    recording = read_recording(directory / 'raw.gmst')
    frames = [(info.seconds * 1000, (lambda info=info, payload=payload: decode(info, payload)))
              for info, _, payload in recording]
    if fast:
        # Quarter-resolution grayscale straight from the JPEG decoder: cheap enough for every 2nd frame.
        grays = [(index, cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_4))
                 for index, (_, _, payload) in enumerate(recording) if index % 2 == 0]
        moving = moving_frames([stamp for stamp, _ in frames], [item for item in grays if item[1] is not None])
        looked = sorted(index for index in moving if index % 2 == 0)
    else:
        looked = list(range(0, len(frames), detect_every))
    report('finding', 0, len(looked))
    config = detector_config(models, [label], device, threshold)
    detections = []

    def tile_frames(index):
        """(tile origin and size, InputFrame) for each native-resolution tile of one frame."""
        stamp, load = frames[index]
        image = load()
        if image is None:
            return []
        output = []
        for number, (x, y) in enumerate(tiles(*image.shape[:2])):
            tile = np.ascontiguousarray(image[y:y + 640, x:x + 640])
            sensor = SensorMetadata(source_id='gamst.take', sensor_id=f'tile-{number}', sequence=index,
                                    capture_sequence=index, coordinate_frame='image', stream_epoch='refine')
            output.append(((stamp, index, x, y, tile.shape[1], tile.shape[0]),
                           InputFrame(tile, 'video', int(stamp), subject_id='local-user', sensor=sensor)))
        return output

    def collect(where, value):
        stamp, index, x, y, w, h = where
        for item in value['objects']:
            box = item['bbox']
            detections.append((stamp, index, item['score'], (
                x + box['x'] * w, y + box['y'] * h, x + (box['x'] + box['width']) * w, y + (box['y'] + box['height']) * h)))

    if fast:
        detector = (warm or {}).get('detector')
        owned = detector is None
        if owned:
            from .providers.yoloe_native import load_in_process
            detector = load_in_process(config)
        try:
            pending = []
            for count, index in enumerate(looked, 1):
                pending += tile_frames(index)
                if len(pending) >= batch or count == len(looked):
                    for where, value in zip([p[0] for p in pending],
                                            detector.process_batch([p[1] for p in pending])):
                        collect(where, value)
                    pending = []
                    report('finding', count, len(looked))
        finally:
            if owned:
                detector.close()
    else:
        from . import ProcessProvider
        worker = await ProcessProvider.open(worker_python, 'yoloe.local', config)
        try:
            for count, index in enumerate(looked, 1):
                for where, frame in tile_frames(index):
                    collect(where, await worker.process(frame))
                report('finding', count, len(looked))
        finally:
            await worker.close()
    detected = time.perf_counter()
    name = 'efficienttam_ti_512x512.pt'
    segmenter = (warm or {}).get('segmenter') or EfficientTAMVideo(
        efficienttam_source, models / name, REFINE_ASSETS[name][1], device=device)
    report('tracking', 0, 1)
    results = refine(frames, detections, segmenter, label=label, stop_when_complete=fast,
                     max_throws=1 if fast else None, progress=report, expected_max_rev_s=expected_max_rev_s)
    timing = dict(frames=len(frames), detector_frames=len(looked), detect_s=round(detected - began, 1),
                  track_s=round(time.perf_counter() - detected, 1), total_s=round(time.perf_counter() - began, 1))
    output = dict(label=label, detections=len(detections), results=results, timing=timing)
    (directory / 'refined.part.json').write_text(json.dumps(output, ensure_ascii=False, indent=2))
    (directory / 'refined.part.json').rename(directory / 'refined.json')
    report('rendering', 0, 1)
    render_refined(directory)
    report('done', 1, 1)
    return output



def _refine_main(argv):
    import argparse
    from .demos.rotation import DEFAULT_MODELS
    parser = argparse.ArgumentParser(prog='python -m gamst.takes refine')
    parser.add_argument('take_dir')
    parser.add_argument('--label', required=True, help='open-vocabulary text for the object to follow')
    parser.add_argument('--efficienttam-source', required=True)
    parser.add_argument('--models', default=str(DEFAULT_MODELS))
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    parser.add_argument('--worker-python', default=sys.executable)
    parser.add_argument('--expected-max-rev-s', type=float, default=40.)
    parser.add_argument('--fast', action='store_true',
                        help='detector only where the picture moves; stop at the first complete count')
    args = parser.parse_args(argv)
    output = asyncio.run(refine_take(args.take_dir, label=args.label, models=args.models, device=args.device,
                                     efficienttam_source=args.efficienttam_source,
                                     worker_python=args.worker_python, fast=args.fast,
                                     expected_max_rev_s=args.expected_max_rev_s))
    print(json.dumps(dict(detections=output['detections'], timing=output['timing'], throws=[
        {key: event.get(key) for key in ('track_id', 'turns', 'rev_s', 'status', 'coverage', 'flight_frames')}
        | dict(trajectory=event['trajectory'].get('kind')) for event in output['results']])))


def _refine_server(argv):
    """Keep YOLOE (batched, in this process) and EfficientTAM loaded; count each take directory
    read from stdin with the fast path. Saves the model loading time on every ANALYZE."""
    import argparse
    from .demos.rotation import DEFAULT_MODELS, REFINE_ASSETS, detector_config
    from .providers.rotation_refine import EfficientTAMVideo
    from .providers.yoloe_native import load_in_process
    parser = argparse.ArgumentParser(prog='python -m gamst.takes refine-server')
    parser.add_argument('--label', required=True)
    parser.add_argument('--efficienttam-source', required=True)
    parser.add_argument('--models', default=str(DEFAULT_MODELS))
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    parser.add_argument('--worker-python', default=sys.executable)
    parser.add_argument('--expected-max-rev-s', type=float, default=40.)
    parser.add_argument('--fast', action='store_true', help='accepted for symmetry; the server is always fast')
    args = parser.parse_args(argv)
    models = Path(args.models)
    name = 'efficienttam_ti_512x512.pt'
    warm = dict(detector=load_in_process(detector_config(models, [args.label], args.device, .2)),
                segmenter=EfficientTAMVideo(args.efficienttam_source, models / name, REFINE_ASSETS[name][1],
                                            device=args.device))
    print(json.dumps({'refine_server': 'ready'}), flush=True)
    for line in sys.stdin:
        if not line.strip():
            continue
        directory = Path(line.strip())
        try:
            output = asyncio.run(refine_take(directory, label=args.label, models=models, device=args.device,
                                             efficienttam_source=args.efficienttam_source,
                                             worker_python=args.worker_python, fast=True, warm=warm,
                                             expected_max_rev_s=args.expected_max_rev_s))
            print(json.dumps({'refined': str(directory), 'timing': output['timing']}), flush=True)
        except Exception as error:  # one bad take must not stop the server
            (directory / 'refine.progress.json').write_text(json.dumps(dict(stage='failed', error=str(error))))
            print(json.dumps({'refine_failed': str(directory), 'error': str(error)}), flush=True)



def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 2 and argv[0] == 'render':
        from .halo import render_refined
        render_refined(argv[1])
    elif len(argv) > 1 and argv[0] == 'refine':
        _refine_main(argv[1:])
    elif len(argv) > 1 and argv[0] == 'refine-server':
        _refine_server(argv[1:])
    else:
        raise SystemExit('usage: python -m gamst.takes refine TAKE_DIR --label ... | '
                         'refine-server --label ... | render TAKE_DIR')


if __name__ == '__main__':
    main()

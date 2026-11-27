"""Models that run in their own Python environment, one worker process each (``ProcessProvider``).

The enrolled-person models need packages that conflict with each other (transformers 4 and 5, onnx
runtimes, vLLM), so each runs in the interpreter of its registry ``environment``. A worker here is
*not* a Pipeline module by itself: it answers one module (``SpeakerVerify``, ``TargetPerson``,
``TargetSpeech`` ...) and its answer is never committed as a record. Embeddings identify a person, joint
positions and separated audio are raw: they travel as private payloads (:mod:`gamst.private_payload`),
never as fields of a value; only the module's scores and counts become context.

=========================  ==================================================  ====================
profile                    worker (real model)                                  client (this process)
=========================  ==================================================  ====================
``speaker.embedding``      sherpa-onnx CAM++ / WeSpeaker w2v-BERT               ``RemoteEmbedder``
``face.embedding``         LVFace ONNX                                          ``RemoteEmbedder``
``body.embedding``         SOLIDER-REID                                         ``RemoteEmbedder``
``target_speech.local``    WeSep / USEF-TFGridNet / SAM-Audio spans             ``RemoteExtractor``
``pixit.sources``          pyannote PixIT (diarization + separation)            ``RemoteSeparator``
``sepformer.sources``      SpeechBrain SepFormer (two sources)                  ``RemoteSeparator``
``qwen3.forced_aligner``   Qwen3-ForcedAligner                                  ``RemoteAligner``
``faster_whisper.words``   FasterWhisperPlugin with word times                  ``RemoteWhisper``
``qwen3.asr.words``        Qwen3-ASR + Qwen3-ForcedAligner phrases (no CUDA)    ``RemoteWhisper``
``hosted.asr.words``       a hosted ASR API's phrases (audio leaves the device) ``RemoteWhisper``
``streaming.asr.words``     a local streaming recogniser (sherpa-onnx, Voxtral)  ``RemoteStreamingWords``
``faster_whisper.align``    Whisper forced alignment of a given transcript       ``RemoteAligner``
``sapiens2.pose``          Sapiens2 + DETR                                      ``RemotePose``
``rtmw.pose``              RTMW whole-body + YOLOX (rtmlib ONNX)                ``RemoteRTMW``
=========================  ==================================================  ====================

``active_speaker.lightasd`` (Light-ASD: face crops and sound -> speaking logits) answers ``RemoteActiveSpeaker``.

Every config names local files. Model identity must be known before the worker starts (an enrollment
made with another model is refused), so the weight's ``sha256`` is required.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re

from ..capabilities import CapabilitySpec, OutputField, SensorMetadata, bounded_json
from ..plugins import InputFrame

SAMPLE_RATE = 16000
EMBEDDING = (OutputField('usable', 'boolean'),)
SOURCES = (OutputField('sources', 'integer'), OutputField('samples_returned', 'integer'))
FIELDS = {
    'speaker.embedding': ('audio', EMBEDDING),
    'face.embedding': ('video', EMBEDDING),
    'body.embedding': ('video', EMBEDDING),
    'target_speech.local': ('audio', (OutputField('samples_returned', 'integer'),)),
    'pixit.sources': ('audio', SOURCES),
    'sepformer.sources': ('audio', SOURCES),
    'qwen3.forced_aligner': ('audio', (OutputField('items', 'array'),)),
    'faster_whisper.words': ('audio', (OutputField('transcripts', 'array'),)),
    'qwen3.asr.words': ('audio', (OutputField('transcripts', 'array'),)),
    'hosted.asr.words': ('audio', (OutputField('transcripts', 'array'),)),
    'streaming.asr.words': ('audio', (OutputField('transcripts', 'array'),)),
    'faster_whisper.align': ('audio', (OutputField('items', 'array'),)),
    'sapiens2.pose': ('video', (OutputField('poses', 'array'), OutputField('model', 'string'),
                                OutputField('image_width', 'integer'), OutputField('image_height', 'integer'))),
    'rtmw.pose': ('video', (OutputField('poses', 'array'), OutputField('model', 'string'),
                            OutputField('image_width', 'integer'), OutputField('image_height', 'integer'),
                            OutputField('frames', 'integer', required=False),
                            OutputField('benchmark', 'object', required=False))),
    'active_speaker.lightasd': ('video', (OutputField('scores', 'array'),)),
}
PROFILES = frozenset(FIELDS)


def model_id(profile, config):
    """The same identity the in-process embedders report, from the pinned digest alone."""
    digest = config.get('sha256')
    if type(digest) is not str or re.fullmatch('[0-9a-f]{64}', digest) is None:
        raise ValueError(f'{profile} needs the sha256 of its weight file')
    name = {'face.embedding': 'lvface', 'body.embedding': 'solider-reid', 'qwen3.forced_aligner': 'qwen3-forced-aligner',
            'faster_whisper.words': 'faster-whisper', 'sapiens2.pose': 'sapiens2', 'rtmw.pose': 'rtmw', 'qwen3.asr.words': 'qwen3-asr',
            'hosted.asr.words': 'hosted-asr', 'active_speaker.lightasd': 'lightasd', 'pixit.sources': 'pyannote-pixit',
            'sepformer.sources': 'sepformer', 'faster_whisper.align': 'whisper-align',
            'streaming.asr.words': config.get('backend')}.get(profile) or {
        'usef': 'usef-tfgridnet', 'sam-span': 'sam-audio'}.get(config.get('backend'), config.get('backend'))
    return f'{name}:{digest[:16]}'


def capability(profile, config):
    config = bounded_json(config)
    modality, fields = FIELDS[profile]
    if profile == 'target_speech.local' and config.get('backend') == 'usef' and config.get('allow_noncommercial') is not True:
        raise PermissionError('USEF-TSE is CC BY-NC 4.0: set allow_noncommercial to opt in')
    if profile == 'sapiens2.pose' and config.get('accept_license') is not True:
        raise PermissionError('read the Sapiens2 License and set accept_license to opt in')
    if profile == 'rtmw.pose':
        from .rtmw_pose import check_config as check_rtmw
        check_rtmw(config)
    if profile == 'active_speaker.lightasd':
        from .active_speaker import check_weights
        check_weights(config.get('sha256'), config.get('allow_noncommercial', False))
    if profile == 'hosted.asr.words':
        from .hosted_asr import check_config
        check_config(config)
    elif profile == 'streaming.asr.words' and config.get('backend') not in ('sherpa-zipformer', 'voxtral-mlx'):
        raise ValueError('streaming.asr.words backend is sherpa-zipformer or voxtral-mlx')
    elif not (profile == 'faster_whisper.words' and 'sha256' not in config):
        model_id(profile, config)
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    return CapabilitySpec(f'model.{profile}', f'model.{profile}', modality, f'{profile}.worker', '1.' + digest, fields,
                          'native' if modality == 'audio' else 'frame', stateful=True, interpretation='estimate',
                          devices=('cpu', 'mps', 'cuda'), license='adapter:Apache-2.0;model:see registry')


# ---------------------------------------------------------------- worker side

class _Worker:
    def reset(self):
        pass

    def close(self):
        pass


def _numpy():
    import numpy as np
    return np


class _Embedding(_Worker):
    def __init__(self, embed):
        self.embed = embed

    def query(self, frame, request):
        try:
            values = self.embed(frame)
        except ValueError:          # too short, silent or no face: no evidence, not a failure
            values = None
        return dict(usable=values is not None), _numpy().asarray([] if values is None else values, dtype='<f4')


def _speech(samples):
    samples = _numpy().asarray(samples, dtype='<f4')
    return dict(samples_returned=len(samples)), samples


class _Extraction(_Worker):
    """``{"role": "enrollment"}`` hands over the enrolled clip; every other query is a mixture block."""

    def __init__(self, extractor):
        self.extractor, self.enrollment = extractor, None

    def query(self, frame, request):
        if (request or {}).get('role') == 'enrollment':
            data = _numpy().asarray(frame.data)
            self.enrollment = data.reshape(-1)[:int(request['samples'])] if 'samples' in request else data
            return _speech([])
        if self.enrollment is None:
            raise RuntimeError('no enrolled clip')
        return _speech(self.extractor.extract(frame.data, self.enrollment))

    def close(self):
        self.enrollment = None


class _Separation(_Worker):
    """A mixture block in, one source per speaker out (speakers x samples)."""

    def __init__(self, separator):
        self.separator = separator

    def query(self, frame, request):
        sources = _numpy().asarray(self.separator.separate(frame.data), dtype='<f4')
        if sources.shape[0] == 0:
            sources = sources.reshape(0, 0)
        return dict(sources=int(sources.shape[0]), samples_returned=int(sources.shape[1])), sources


class _SpanExtraction(_Worker):
    """SAM-Audio: a mixture block with the ``anchors`` that belong to it."""

    def __init__(self, module):
        self.module = module

    def query(self, frame, request):
        return _speech(self.module.separate(frame.data, [tuple(anchor) for anchor in request['anchors']]))

    def close(self):
        self.module.close()


class _Aligner(_Worker):
    def __init__(self, aligner):
        self.aligner = aligner

    def process_regions(self, frame, evidence):
        samples = frame.data
        if getattr(samples, 'ndim', 1) == 2:     # a long segment travels as rows (the wire limits each dimension)
            samples = samples.reshape(-1)[:int(evidence['samples'])]
        return dict(items=self.aligner.align(samples, frame.sample_rate, evidence['text'], evidence['language']))


class _Whisper(_Worker):
    """The async plugin on this worker's own event loop."""

    def __init__(self, module):
        self.module, self.loop = module, asyncio.new_event_loop()

    def _words(self, signals):
        return dict(transcripts=[item['value'] for item in signals if item['name'] == 'speech.transcript_words'])

    def process(self, frame):
        result = self._words(self.loop.run_until_complete(self.module.process(frame)))
        return result if result['transcripts'] else None

    def finish(self):
        result = self._words(self.loop.run_until_complete(self.module.finish()))
        return result if result['transcripts'] else None

    def reset(self):
        self.module.reset()

    def close(self):
        self.loop.close()


class _Pose(_Worker):
    def __init__(self, plugin):
        self.plugin = plugin

    def query(self, frame, request):
        np, found = _numpy(), self.plugin.infer(frame.data)
        height, width = frame.data.shape[:2]
        joints = np.asarray([[[point['x'], point['y'], point['visibility']] for point in pose['landmarks']]
                             for pose in found], dtype='<f4').reshape(len(found), 308, 3)
        return dict(poses=[dict(bbox=pose['bbox']) for pose in found], model=self.plugin.version,
                    image_width=int(width), image_height=int(height)), joints


class _RTMW(_Worker):
    """Consecutive frames -> boxes and joints. ``frame.data`` is one image, or ``request["frames"]`` images of one size
    stacked by rows; ``request["timestamps_ms"]``: their capture times (for the detection interval)."""

    def __init__(self, models):
        self.models = models

    def query(self, frame, request):
        from .rtmw_pose import MAX_FRAMES, POINTS
        np = _numpy()
        request = request or {}
        if request.get('benchmark'):
            count = request.get('frames', 1)
            if type(count) is not int or not 1 <= count <= MAX_FRAMES or frame.data.shape[0] % count:
                raise ValueError(f'rtmw.pose benchmarks 1..{MAX_FRAMES} stacked frames')
            rows = frame.data.shape[0] // count
            images = [frame.data[index * rows:(index + 1) * rows] for index in range(count)]
            measured = self.models.benchmark(images, detect=bool(request.get('detect')))
            return dict(poses=[], model=self.models.version, image_width=int(frame.data.shape[1]), image_height=rows,
                        benchmark=measured), np.zeros((0, POINTS, 3), dtype='<f4')
        count = request.get('frames', 1)
        times = request.get('timestamps_ms') or [request.get('timestamp_ms')]
        if type(count) is not int or not 1 <= count <= MAX_FRAMES or len(times) != count \
                or frame.data.shape[0] % count:
            raise ValueError(f'rtmw.pose takes 1..{MAX_FRAMES} stacked frames with one capture time each')
        rows = frame.data.shape[0] // count
        images = [frame.data[index * rows:(index + 1) * rows] for index in range(count)]
        height, width = rows, frame.data.shape[1]
        public, joints = [], []
        for index, (boxes, keypoints, scores) in enumerate(self.models.infer_batch(images, times)):
            for box, points, values in zip(boxes, keypoints, scores):
                rows_ = np.zeros((POINTS, 3), dtype='<f4')
                rows_[:, 0], rows_[:, 1], rows_[:, 2] = points[:, 0] / width, points[:, 1] / height, np.clip(values, 0., 1.)
                joints.append(rows_)
                box = [float(value) for value in box]
                x0, y0 = max(0., box[0] / width), max(0., box[1] / height)
                x1, y1 = min(1., box[2] / width), min(1., box[3] / height)
                public.append(dict(in_batch=index, bbox=dict(x=x0, y=y0, width=x1 - x0, height=y1 - y0)
                                   if x1 > x0 and y1 > y0 else None))
        joints = np.stack(joints) if joints else np.zeros((0, POINTS, 3), dtype='<f4')
        return dict(poses=public, model=self.models.version, image_width=int(width), image_height=int(height),
                    frames=count), joints

    def reset(self):
        self.models.reset()


class _ActiveSpeaker(_Worker):
    """One window of one face: ``frame.data`` is ``active_speaker.pack_window``, the request says how many frames."""

    def __init__(self, model):
        self.model = model

    def process_regions(self, frame, request):
        from .active_speaker import unpack_window
        faces, pcm = unpack_window(frame.data, request['frames'])
        return dict(scores=[round(float(value), 4) for value in self.model.scores(faces, pcm)])


def load(profile, config):
    """Build the real model in the worker's interpreter."""
    capability(profile, config)
    config = dict(config)
    if profile == 'speaker.embedding':
        from .speaker_verify import load_embedder
        config.pop('device', None) if config['backend'] == 'sherpa-onnx' else None
        embedder = load_embedder(config.pop('backend'), config.pop('path'), **config)
        return _Embedding(lambda frame: embedder.embed(frame.data, frame.sample_rate))
    if profile == 'face.embedding':
        from .person_match import LVFaceEmbedder
        embedder = LVFaceEmbedder.from_local(config.pop('path'), **config)
        return _Embedding(lambda frame: embedder.embed(frame.data))
    if profile == 'body.embedding':
        from .person_match import SOLIDERBodyEmbedder
        embedder = SOLIDERBodyEmbedder.from_local(config.pop('source'), config.pop('checkpoint'), **config)
        return _Embedding(lambda frame: embedder.embed(frame.data))
    if profile == 'target_speech.local':
        from .target_speech import EXTRACTORS, SAMSpanBackend
        backend = config.pop('backend')
        if backend == 'sam-span':
            config.pop('sha256')
            return _SpanExtraction(SAMSpanBackend.from_local(config))
        return _Extraction(EXTRACTORS[backend].from_local(**config))
    if profile == 'pixit.sources':
        from .pixit import PixITSeparator
        return _Separation(PixITSeparator.from_local(config.pop('config'), config.pop('separation'),
                                                     config.pop('embedding'), **config))
    if profile == 'sepformer.sources':
        from .sepformer import SepFormerSeparator
        return _Separation(SepFormerSeparator.from_local(config.pop('directory'), **config))
    if profile == 'qwen3.forced_aligner':
        from .asr_words import QwenForcedAligner
        return _Aligner(QwenForcedAligner.from_local(config.pop('path'), **config))
    if profile == 'faster_whisper.align':
        from .whisper_align import load as load_aligner
        return _Aligner(load_aligner(config))
    if profile == 'streaming.asr.words':
        from .streaming_local import load as load_streaming
        return _Whisper(load_streaming(config))
    if profile == 'faster_whisper.words':
        from ..plugins import FasterWhisperPlugin
        from .asr_words import WhisperWords
        config.pop('sha256', None)
        phrases = config.pop('phrases', None)
        plugin = FasterWhisperPlugin(config.pop('model_path'), word_timestamps=True, **config)
        if phrases is not None:
            from .whisper_phrases import PhraseWorker, WhisperPhrases
            return _Whisper(PhraseWorker(WhisperPhrases(plugin, **phrases)))
        return _Whisper(WhisperWords(plugin))
    if profile == 'active_speaker.lightasd':
        from .active_speaker import LightASD
        return _ActiveSpeaker(LightASD.from_local(config.pop('source_dir'), config.pop('checkpoint_path'),
                                                  config.pop('sha256'), **config))
    if profile == 'qwen3.asr.words':
        from .qwen_words import QwenDecoder
        from .whisper_phrases import PhraseWorker, Phrases
        phrases = config.pop('phrases', None) or {}
        config.pop('sha256')
        decoder = QwenDecoder.from_local(config.pop('model_path'), config.pop('files'), config.pop('aligner_path'),
                                         config.pop('aligner_sha256'), **config)
        return _Whisper(PhraseWorker(Phrases(decoder, **phrases)))
    if profile == 'hosted.asr.words':
        from .hosted_asr import load_decoder
        from .whisper_phrases import PhraseWorker, Phrases
        phrases = config.pop('phrases', None) or {}
        return _Whisper(PhraseWorker(Phrases(load_decoder(config), **phrases)))
    if profile == 'rtmw.pose':
        from .rtmw_pose import RTMWModels
        return _RTMW(RTMWModels(**config))
    from .sapiens2_pose import Sapiens2PosePlugin
    config['pose_sha256'] = config.pop('sha256')
    return _Pose(Sapiens2PosePlugin.from_local(**config))


# ---------------------------------------------------------------- client side

class _Client:
    def __init__(self, worker, profile, config):
        self.worker, self.model_id = worker, model_id(profile, config) if 'sha256' in config else profile
        self._sequence = 0

    @classmethod
    async def open(cls, python, profile, config, *, startup_timeout=600, inference_timeout=120):
        from ..private_payload import CONTRACTS
        from ..process_provider import ProcessProvider
        worker = await ProcessProvider.open(python, profile, config, startup_timeout=startup_timeout,
                                            inference_timeout=inference_timeout, private=profile in CONTRACTS)
        return cls(worker, profile, config)

    def _frame(self, data, modality, *, sample_rate=None, timestamp_ms=0, sensor_id='input'):
        import numpy as np
        self._sequence += 1
        return InputFrame(np.ascontiguousarray(data), modality, int(timestamp_ms), 'local-user', sample_rate,
                          SensorMetadata(source_id='gamst.model', sensor_id=sensor_id, sequence=self._sequence))

    async def close(self):
        await self.worker.close()


class RemoteEmbedder(_Client):
    async def embed(self, data, sample_rate=None):
        modality = 'audio' if sample_rate else 'video'
        result, private = await self.worker.query(self._frame(data, modality, sample_rate=sample_rate))
        if not result['usable']:
            raise ValueError('too short, silent or unclear for the embedding model')
        return private.take('embedding').tolist()


class RemoteAligner(_Client):
    ROW = 16000

    async def align(self, samples, sample_rate, text, language):
        import numpy as np
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        request = dict(text=text, language=language)
        data = samples
        if len(samples) > self.ROW:      # rows of one second: any length fits the wire's per-dimension limit
            data = np.zeros(-(-len(samples) // self.ROW) * self.ROW, np.float32)
            data[:len(samples)] = samples
            data, request['samples'] = data.reshape(-1, self.ROW), len(samples)
        result = await self.worker.process_regions(self._frame(data, 'audio', sample_rate=sample_rate), request)
        return result['items']


class RemoteExtractor(_Client):
    _sent = None

    async def _query(self, samples, request):
        _, private = await self.worker.query(self._frame(samples, 'audio', sample_rate=SAMPLE_RATE), request)
        return private.take('waveform')

    ROW = 16000

    async def extract(self, mixture, enrollment):
        if self._sent is not enrollment:
            import numpy as np
            samples = np.asarray(enrollment, dtype=np.float32).reshape(-1)
            request, data = dict(role='enrollment'), samples
            if len(samples) > self.ROW:      # rows of one second: a clip of any length fits the wire's per-dimension limit
                data = np.zeros(-(-len(samples) // self.ROW) * self.ROW, np.float32)
                data[:len(samples)] = samples
                data, request['samples'] = data.reshape(-1, self.ROW), len(samples)
            await self._query(data, request)
            self._sent = enrollment
        return await self._query(mixture, dict(role='mixture'))

    async def separate(self, window, anchors):
        return await self._query(window, dict(anchors=[list(anchor) for anchor in anchors]))


class RemoteSeparator(_Client):
    """``separate(window) -> (speakers, samples)`` answered by a separation worker."""

    async def separate(self, window):
        import numpy as np
        result, private = await self.worker.query(self._frame(window, 'audio', sample_rate=SAMPLE_RATE), {})
        sources = private.take('sources')
        return sources if result['sources'] else np.zeros((0, len(window)), np.float32)


class RemoteWhisper(_Client):
    """Pipeline module: a transcript worker's utterances (Whisper windows or phrases, Qwen3-ASR or hosted
    phrases) as ``speech.transcript_words`` records."""
    owns_process = True

    def __init__(self, worker, profile, config):
        super().__init__(worker, profile, config)
        from .asr_words import words_capability
        self.spec = words_capability('1.' + self.model_id.replace(':', '.'))

    def _values(self, result):
        return [self.spec.validate(value) for value in (result or {}).get('transcripts', ())] or None

    async def process(self, frame):
        if frame.modality != 'audio' or frame.data is None:
            return None
        return self._values(await self.worker.process(InputFrame(
            frame.data, 'audio', frame.timestamp_ms, frame.subject_id, frame.sample_rate,
            frame.sensor or SensorMetadata(source_id='gamst.model', sensor_id='input', sequence=0))))

    async def finish(self):
        return self._values(await self.worker.finish())

    async def reset(self):
        await self.worker.reset()


class RemoteStreamingWords(RemoteWhisper):
    """A local streaming recogniser's records, with ``final_after_ms``: wall time from when a final's last sound
    reached this module to when its record came back (what a live microphone would wait)."""

    def __init__(self, worker, profile, config):
        super().__init__(worker, profile, config)
        from .streaming_asr import CaptureClock, streaming_capability
        self.spec = streaming_capability('1.' + self.model_id.replace(':', '.'))
        self.clock = CaptureClock(SAMPLE_RATE)

    def _values(self, result):
        from .streaming_asr import _wall_ms
        values = []
        for value in (result or {}).get('transcripts', ()):
            value = dict(value)
            arrived = self.clock.arrival_ms(value['end_ms']) if value.get('status') == 'final' else None
            if arrived is not None:
                value['final_after_ms'] = max(0, _wall_ms() - arrived)
            values.append(self.spec.validate(value))
        return values or None

    async def process(self, frame):
        if frame.modality == 'audio' and frame.data is not None:
            self.clock.push(len(frame.data), frame.timestamp_ms)
        return await super().process(frame)

    async def reset(self):
        self.clock.reset()
        await super().reset()


class RemotePose(_Client):
    """Async stand-in for ``Sapiens2PosePlugin.infer``; tracking stays in this process."""

    async def infer(self, image):
        from ..crew.modules import bounded_image
        from ..private_payload import CONTRACTS, landmarks
        result, private = await self.worker.query(self._frame(bounded_image(image), 'video'))
        columns = CONTRACTS['sapiens2.pose'].columns
        return [dict(frame_pose_index=index, bbox=pose['bbox'], keypoint_set='goliath308', track_id=None,
                     association_quality='unavailable', identity_verified=False, speaker_association='unavailable',
                     image_width=result['image_width'], image_height=result['image_height'],
                     landmarks=[dict(point, presence=1.) for point in landmarks(rows, columns)])
                for index, (pose, rows) in enumerate(zip(result['poses'], private.take('joints')))]


class RemoteRTMW(_Client):
    """Async stand-in for :class:`gamst.providers.rtmw_pose.RTMWModels`; tracking stays in this process."""

    async def infer(self, image, timestamp_ms):
        return (await self.infer_batch([image], [timestamp_ms]))[0]

    def _stacked(self, images, timestamp_ms=0):
        """One wire frame of consecutive images: through shared memory without stacking them first, else stacked."""
        import numpy as np
        from dataclasses import replace
        if len(images) == 1:
            return self._frame(images[0], 'video', timestamp_ms=timestamp_ms)
        if getattr(self.worker, '_frames', None) is not None:
            return replace(self._frame(images[0], 'video', timestamp_ms=timestamp_ms), data=tuple(images))
        return self._frame(np.concatenate(images), 'video', timestamp_ms=timestamp_ms)

    async def infer_batch(self, images, timestamps_ms):
        """Consecutive frames of one size -> one list of poses per frame (one worker call: the frames go stacked)."""
        import numpy as np
        from ..crew.modules import bounded_image
        from ..private_payload import CONTRACTS, landmarks
        images = [np.ascontiguousarray(bounded_image(image)) for image in images]
        frame = self._stacked(images, timestamps_ms[0])
        result, private = await self.worker.query(
            frame, {'frames': len(images), 'timestamps_ms': [int(ms) for ms in timestamps_ms]})
        columns = CONTRACTS['rtmw.pose'].columns
        found = [[] for _ in images]
        for pose, rows in zip(result['poses'], private.take('joints')):
            poses = found[pose.get('in_batch', 0)]
            poses.append(dict(frame_pose_index=len(poses), bbox=pose['bbox'], keypoint_set='coco_wholebody133',
                              track_id=None, association_quality='unavailable', identity_verified=False,
                              speaker_association='unavailable', image_width=result['image_width'],
                              image_height=result['image_height'],
                              landmarks=[dict(point, presence=1.) for point in landmarks(rows, columns)]))
        return found

    async def benchmark(self, *, seconds=2., frames=8, size=(720, 1280)):
        """This worker's speed now, as a live lane would get it (shared-memory transfer included): calls of ``frames``
        synthetic frames for ``seconds``; ms per frame of the pose path, and one detection's ms. Several workers
        measured at the same time show what they give together."""
        import time
        import numpy as np
        frame = self._stacked([np.full((*size, 3), 127, np.uint8) for _ in range(frames)])
        result, private = await self.worker.query(frame, {'benchmark': True, 'frames': frames, 'detect': True})
        private.take('joints')
        first, calls, started = result['benchmark'], 0, time.perf_counter()    # the first call also warms up
        while not calls or time.perf_counter() - started < seconds:
            result, private = await self.worker.query(frame, {'benchmark': True, 'frames': frames})
            private.take('joints')
            calls += 1
        wall = (time.perf_counter() - started) * 1000 / (calls * frames)
        detect = dict(result['benchmark'], detect_ms=first['detect_ms'])
        return dict(ms_per_frame=round(wall, 2), model_ms_per_frame=detect['ms_per_frame'], detect_ms=detect['detect_ms'],
                    det_every_ms=detect['det_every_ms'], frames=frames, seconds=seconds, version=detect['version'])

    async def reset(self):
        await self.worker.reset()


class RemoteActiveSpeaker(_Client):
    """Light-ASD in its own environment: (face crops, PCM16) of one face -> one speaking logit per crop."""

    async def scores(self, faces, pcm):
        from .active_speaker import pack_window
        self._sequence += 1
        frame = InputFrame(pack_window(faces, pcm), 'video', 0, 'local-user', None,
                           SensorMetadata(source_id='gamst.model', sensor_id='lips', sequence=self._sequence))
        result = await self.worker.process_regions(frame, dict(frames=len(faces)))
        if len(result['scores']) != len(faces):
            raise ValueError('Light-ASD returned another number of scores than faces')
        return result['scores']

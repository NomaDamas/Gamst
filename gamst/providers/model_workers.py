"""Models that run in their own Python environment, one worker process each (``ProcessProvider``).

The enrolled-person models need packages that conflict with each other (transformers 4 and 5, onnx
runtimes, vLLM), so each runs in the interpreter of its registry ``environment``. A worker here is
*not* a Pipeline module by itself: it answers one module (``SpeakerVerify``, ``TargetPerson``,
``TargetSpeech`` ...) and its answer is never committed as a record. Embeddings identify a person and
separated audio is raw media; only the module's scores and counts become context.

=========================  ==================================================  ====================
profile                    worker (real model)                                  client (this process)
=========================  ==================================================  ====================
``speaker.embedding``      sherpa-onnx CAM++ / WeSpeaker w2v-BERT               ``RemoteEmbedder``
``face.embedding``         LVFace ONNX                                          ``RemoteEmbedder``
``body.embedding``         SOLIDER-REID                                         ``RemoteEmbedder``
``target_speech.local``    WeSep / USEF-TFGridNet / SAM-Audio spans             ``RemoteExtractor``
``qwen3.forced_aligner``   Qwen3-ForcedAligner                                  ``RemoteAligner``
``faster_whisper.words``   FasterWhisperPlugin with word times                  ``RemoteWhisper``
``sapiens2.pose``          Sapiens2 + DETR                                      ``RemotePose``
=========================  ==================================================  ====================

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
EMBEDDING = (OutputField('usable', 'boolean'), OutputField('embedding', 'array', required=False))
FIELDS = {
    'speaker.embedding': ('audio', EMBEDDING),
    'face.embedding': ('video', EMBEDDING),
    'body.embedding': ('video', EMBEDDING),
    'target_speech.local': ('audio', (OutputField('samples_returned', 'integer'),)),
    'qwen3.forced_aligner': ('audio', (OutputField('items', 'array'),)),
    'faster_whisper.words': ('audio', (OutputField('transcripts', 'array'),)),
    'sapiens2.pose': ('video', (OutputField('poses', 'array'), OutputField('model', 'string'))),
}
PROFILES = frozenset(FIELDS)
# The only profile whose response carries waveform bytes, to its owning module alone.
PRIVATE_AUDIO = frozenset(('target_speech.local',))


def model_id(profile, config):
    """The same identity the in-process embedders report, from the pinned digest alone."""
    digest = config.get('sha256')
    if type(digest) is not str or re.fullmatch('[0-9a-f]{64}', digest) is None:
        raise ValueError(f'{profile} needs the sha256 of its weight file')
    name = {'face.embedding': 'lvface', 'body.embedding': 'solider-reid', 'qwen3.forced_aligner': 'qwen3-forced-aligner',
            'faster_whisper.words': 'faster-whisper', 'sapiens2.pose': 'sapiens2'}.get(profile) or {
        'usef': 'usef-tfgridnet', 'sam-span': 'sam-audio'}.get(config.get('backend'), config.get('backend'))
    return f'{name}:{digest[:16]}'


def capability(profile, config):
    config = bounded_json(config)
    modality, fields = FIELDS[profile]
    if profile == 'target_speech.local' and config.get('backend') == 'usef' and config.get('allow_noncommercial') is not True:
        raise PermissionError('USEF-TSE is CC BY-NC 4.0: set allow_noncommercial to opt in')
    if profile == 'sapiens2.pose' and config.get('accept_license') is not True:
        raise PermissionError('read the Sapiens2 License and set accept_license to opt in')
    if not (profile == 'faster_whisper.words' and 'sha256' not in config):
        model_id(profile, config)
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    return CapabilitySpec(f'model.{profile}', f'model.{profile}', modality, f'{profile}.worker', '1.' + digest, fields,
                          'native' if modality == 'audio' else 'frame', stateful=True, interpretation='estimate',
                          devices=('cpu', 'mps', 'cuda'), license='adapter:Apache-2.0;model:see registry')


# ---------------------------------------------------------------- worker side

class _Embedding:
    def __init__(self, embed):
        self.embed = embed

    def process(self, frame):
        try:
            values = self.embed(frame)
        except ValueError:          # too short, silent or no face: no evidence, not a failure
            values = None
        return dict(usable=False) if values is None else dict(usable=True, embedding=values)

    def reset(self):
        pass

    def close(self):
        pass


class _Extraction(_Embedding):
    """The first frame (sensor ``enrollment``) is the enrolled clip; every other one is a mixture block."""

    def __init__(self, extractor):
        self.extractor, self.enrollment = extractor, None

    def process(self, frame):
        import numpy as np
        if frame.sensor.sensor_id == 'enrollment':
            self.enrollment = frame.data
            return np.zeros(0, np.float32)
        if self.enrollment is None:
            raise RuntimeError('no enrolled clip')
        return np.asarray(self.extractor.extract(frame.data, self.enrollment), dtype=np.float32)

    def close(self):
        self.enrollment = None


class _SpanExtraction(_Embedding):
    """SAM-Audio: a JSON frame with the anchors, then the mixture block they belong to."""

    def __init__(self, module):
        self.module, self.anchors = module, []

    def process(self, frame):
        import numpy as np
        if isinstance(frame.data, dict):
            self.anchors = [tuple(anchor) for anchor in frame.data['anchors']]
            return np.zeros(0, np.float32)
        return np.asarray(self.module.separate(frame.data, self.anchors), dtype=np.float32)

    def close(self):
        self.module.close()


class _Aligner(_Embedding):
    def __init__(self, aligner):
        self.aligner = aligner

    def process_regions(self, frame, evidence):
        return dict(items=self.aligner.align(frame.data, frame.sample_rate, evidence['text'], evidence['language']))


class _Whisper(_Embedding):
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


class _Pose(_Embedding):
    def __init__(self, plugin):
        self.plugin = plugin

    def process(self, frame):
        poses = [dict(bbox=pose['bbox'], keypoints=[[round(point['x'], 5), round(point['y'], 5),
                                                     round(point['visibility'], 4)] for point in pose['landmarks']])
                 for pose in self.plugin.infer(frame.data)]
        return dict(poses=poses, model=self.plugin.version)


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
    if profile == 'qwen3.forced_aligner':
        from .asr_words import QwenForcedAligner
        return _Aligner(QwenForcedAligner.from_local(config.pop('path'), **config))
    if profile == 'faster_whisper.words':
        from ..plugins import FasterWhisperPlugin
        from .asr_words import WhisperWords
        config.pop('sha256', None)
        return _Whisper(WhisperWords(FasterWhisperPlugin(config.pop('model_path'), word_timestamps=True, **config)))
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
        from ..process_provider import ProcessProvider
        worker = await ProcessProvider.open(python, profile, config, startup_timeout=startup_timeout,
                                            inference_timeout=inference_timeout)
        return cls(worker, profile, config)

    def _frame(self, data, modality, *, sample_rate=None, timestamp_ms=0, sensor_id='input'):
        import numpy as np
        self._sequence += 1
        data = data if isinstance(data, dict) else np.ascontiguousarray(data)
        return InputFrame(data, modality, int(timestamp_ms), 'local-user', sample_rate,
                          SensorMetadata(source_id='gamst.model', sensor_id=sensor_id, sequence=self._sequence))

    async def close(self):
        await self.worker.close()


class RemoteEmbedder(_Client):
    async def embed(self, data, sample_rate=None):
        modality = 'audio' if sample_rate else 'video'
        result = await self.worker.process(self._frame(data, modality, sample_rate=sample_rate))
        if not result or not result['usable']:
            raise ValueError('too short, silent or unclear for the embedding model')
        return result['embedding']


class RemoteAligner(_Client):
    async def align(self, samples, sample_rate, text, language):
        result = await self.worker.process_regions(self._frame(samples, 'audio', sample_rate=sample_rate),
                                                   dict(text=text, language=language))
        return result['items']


class RemoteExtractor(_Client):
    _sent = None

    async def extract(self, mixture, enrollment):
        if self._sent is not enrollment:
            await self.worker.process(self._frame(enrollment, 'audio', sample_rate=SAMPLE_RATE, sensor_id='enrollment'))
            self._sent = enrollment
        return await self.worker.process(self._frame(mixture, 'audio', sample_rate=SAMPLE_RATE, sensor_id='mixture'))

    async def separate(self, window, anchors):
        await self.worker.process(self._frame(dict(anchors=[list(anchor) for anchor in anchors]), 'audio',
                                              sample_rate=SAMPLE_RATE, sensor_id='anchors'))
        return await self.worker.process(self._frame(window, 'audio', sample_rate=SAMPLE_RATE, sensor_id='mixture'))


class RemoteWhisper(_Client):
    """Pipeline module: the Whisper worker's windows as ``speech.transcript_words`` records."""

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


class RemotePose(_Client):
    """Async stand-in for ``Sapiens2PosePlugin.infer``; tracking stays in this process."""

    async def infer(self, image):
        from ..crew.modules import bounded_image
        result = await self.worker.process(self._frame(bounded_image(image), 'video'))
        return [dict(frame_pose_index=index, bbox=pose['bbox'], keypoint_set='goliath308', track_id=None,
                     association_quality='unavailable', identity_verified=False, speaker_association='unavailable',
                     landmarks=[dict(x=x, y=y, visibility=score, presence=1.) for x, y, score in pose['keypoints']])
                for index, pose in enumerate(result['poses'])]

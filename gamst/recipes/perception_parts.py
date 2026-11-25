"""Recipe parts for enrolled-person speech and video, the semantic lane and the physics modules.

Builders only: packages, weight URLs and digests live in :mod:`gamst.crew.registry`. A builder reads
the config that ``gamst recipe setup`` wrote for its role; a recipe item may instead carry its own
``"config"`` (local paths), which is the only way to use a candidate setup refuses to install
(non-commercial, licence-bound or unpinned weights).
"""
from __future__ import annotations

import importlib
import sys

from ..crew.modules import WorkerModule


def _entry(options, context, profile):
    """(config, python, profile): the item's own ``"config"``, else what setup installed for its role."""
    if isinstance(options.get('config'), dict):
        entry = dict(config=options['config'], python=options.get('python'))
    else:
        entry = context['installed'](options)
    return dict(entry['config']), context.get('worker_python') or entry.get('python') or sys.executable, profile


async def _open(client, entry):
    from ..providers import model_workers
    config, python, profile = entry
    return await getattr(model_workers, client).open(python, profile, config)


async def speaker_embedder(options, context):
    """The embedding model of a ``speaker-verify`` item in its own environment; one per recipe run."""
    entry = _entry(options, context, 'speaker.embedding')
    cache = context.setdefault('speaker_embedders', {})
    key = (entry[0].get('backend'), entry[0].get('path'))
    if key not in cache:
        cache[key] = await _open('RemoteEmbedder', entry)
    return cache[key]


async def _speaker_verify(options, context):
    from ..providers.speaker_verify import SpeakerVerify
    arguments = {name: options[name] for name in ('hysteresis', 'segment_ms', 'hop_ms', 'min_dbfs',
                                                  'min_utterance_ms', 'ring_seconds', 'segments',
                                                  'capability_id') if name in options}
    threshold = context.get('speaker_threshold')
    module = SpeakerVerify(await speaker_embedder(options, context), reference=options.get('reference', 'voice'),
                           threshold=float(options.get('threshold', .5) if threshold is None else threshold),
                           **arguments)
    return dict(module=module, spec=module.spec)


def _emits(options):
    emits = options.get('emits') or ()
    if len(emits) != 1:
        raise SystemExit(f"module {options.get('name')!r} must emit exactly one derived input")
    return emits[0]


async def _speaker_gate(options, context):
    from ..providers.target_speech import SpeakerGate
    return dict(module=SpeakerGate(reference=options.get('reference', 'voice'), emits=_emits(options),
                                   segment_ms=int(options.get('segment_ms', 1500))))


async def _target_speech(options, context):
    from ..providers.target_speech import SAMSpanSpeech, TargetSpeech
    entry = _entry(options, context, 'target_speech.local')
    arguments = dict(reference=options.get('reference', 'voice'), emits=_emits(options))
    remote = await _open('RemoteExtractor', entry)
    if entry[0].get('backend') == 'sam-span':
        return dict(module=SAMSpanSpeech(remote, **arguments))
    arguments.update({name: int(options[name]) for name in ('block_ms', 'context_ms') if name in options})
    return dict(module=TargetSpeech(remote, **arguments))


async def _whisper_words(options, context):
    config, python, profile = _entry(options, context, 'faster_whisper.words')
    config = dict(model_path=config['model_path'], device=config.get('device', 'cpu'),
                  window_seconds=float(options.get('window_seconds', 3.)))
    language = context.get('language') or options.get('language')
    if language:
        config['language'] = language
    module = await _open('RemoteWhisper', (config, python, profile))
    return dict(module=module, spec=module.spec)


async def _qwen_words(options, context):
    """Qwen3-ASR (existing worker profile) + Qwen3-ForcedAligner, each in its own environment."""
    from ..process_provider import ProcessProvider
    from ..providers.asr_words import AlignedASR
    config, python, _ = _entry(options, context, None)
    if 'asr' not in config or 'aligner' not in config:
        raise SystemExit('qwen-words needs config.asr and config.aligner ({"python", "config"} each)')
    asr, aligned = config['asr'], config['aligner']
    aligner = await _open('RemoteAligner', (aligned['config'], aligned.get('python') or python,
                                            'qwen3.forced_aligner'))
    worker = await ProcessProvider.open(asr.get('python') or python, asr.get('profile', 'qwen3.asr.endpoint'),
                                        asr['config'], startup_timeout=600, inference_timeout=120)
    module = AlignedASR(worker, aligner, language=context.get('language') or options.get('language'),
                        hop_ms=int(options.get('hop_ms', 20)))
    return dict(module=module, spec=module.spec)


async def _prosody(options, context):
    from ..providers.prosody import AudioProsody
    module = AudioProsody()
    return dict(module=module, spec=AudioProsody.capability())


async def _voice_channels(options, context):
    from ..providers.voice_style import VoiceChannels
    module = VoiceChannels(step_ms=int(options.get('step_ms', 100)))
    return dict(module=module, spec=module.spec)


async def face_embedder(options, context):
    return await _open('RemoteEmbedder', _entry(options, context, 'face.embedding'))


async def body_embedder(options, context):
    """Optional: ``"body": {"python", "config"}`` of a SOLIDER-REID environment and checkpoint."""
    body = options.get('body')
    return None if not body else await _open('RemoteEmbedder', (
        dict(body['config']), body.get('python') or sys.executable, 'body.embedding'))


async def _target_person(options, context):
    from ..providers.person_match import TargetPerson
    arguments = {name: options[name] for name in ('face_threshold', 'face_reject', 'body_threshold',
                                                  'face_every_ms', 'body_every_ms', 'hold_ms') if name in options}
    joints = options.get('joints')
    emits = [name for name in options.get('emits') or () if name != joints]
    return dict(module=TargetPerson(await face_embedder(options, context), await body_embedder(options, context),
                                    reference=options.get('reference', 'person'),
                                    emits=emits[0] if emits else None, joints=joints, **arguments))


async def _sapiens2_pose(options, context):
    from ..providers.sapiens2_pose import Sapiens2PosePlugin
    entry = _entry(options, context, 'sapiens2.pose')
    remote = await _open('RemotePose', entry)
    return dict(module=Sapiens2PosePlugin(None, None, None, None, device=None, version=remote.model_id, remote=remote,
                                          joints=options.get('joints'),
                                          tracker=options.get('tracker', 'ocsort')))


class SegmentModule(WorkerModule):
    """A worker that answers once per segment; says since when a segment is open, and closes the last one."""

    def __init__(self, worker):
        super().__init__(worker)
        self.pending_since_ms = None

    async def process(self, frame):
        if self.pending_since_ms is None:
            self.pending_since_ms = frame.timestamp_ms
        value = await super().process(frame)
        if value is not None:
            self.pending_since_ms = None
        return value

    async def finish(self):
        self.pending_since_ms = None
        return await self.worker.finish()

    async def reset(self):
        self.pending_since_ms = None
        await super().reset()


async def _semantic_scene(options, context):
    """The existing streaming VLM client, isolated in its worker so a slow reply never blocks capture.

    It interprets segments: ``segment_s`` seconds of frames taken every ``every_ms`` go to the model together.
    The lane keeps the frames of the next segments while a reply is pending.
    """
    from ..process_provider import ProcessProvider
    config, python, profile = _entry(options, context, 'streamingvlm.local')
    every_ms = int(options.get('every_ms', 500))
    frames = max(2, min(64, round(1000 * float(options.get('segment_s', 4)) / every_ms)))
    config.setdefault('window_frames', frames)
    config.setdefault('window_ms', min(60000, config['window_frames'] * every_ms))
    config.setdefault('max_gap_ms', max(2000, min(10000, 3 * every_ms)))
    module = SegmentModule(await ProcessProvider.open(python, profile, config, inference_timeout=90))
    return dict(module=module, spec=module.spec, every_ms=every_ms, queue=min(64, 3 * config['window_frames']))


def _math(path):
    """A measurement module with no weights; ``"config"`` holds its constructor arguments."""
    async def build(options, context):
        module_name, name = path.split(':')
        module = getattr(importlib.import_module(f'gamst.providers.{module_name}'), name)(**(options.get('config') or {}))
        spec = getattr(module, 'spec', None)
        if spec is None:   # OpenCVFlow declares its contract per method
            spec = type(module).capability((options.get('config') or {}).get('method', 'lucas_kanade'))
        return dict(module=module, spec=spec)
    return build


MATH = {'kinematics': 'kinematics:MetricKinematics', 'image-trajectories': 'trajectories:ImageTrajectories',
        'physical-state': 'physical_state:PhysicalState', 'rhythm': 'rhythm:RhythmEstimator',
        'angular-motion': 'angular_motion:AngularMotionCounter', 'object-spin': 'object_spin:ObjectSpinCounter',
        'optical-flow': 'motion:OpenCVFlow'}


def parts(Part):
    speech = dict(role='audio.speaker_verify', candidate='sherpa.campplus_zh_en')
    result = {
        'speaker-verify': Part('module', **speech, build=_speaker_verify),
        'speaker-gate': Part('module', build=_speaker_gate),
        'target-speech': Part('module', role='audio.target_speech', candidate='wesep.bsrnn_ecapa',
                              build=_target_speech),
        'whisper-words': Part('module', role='audio.asr', candidate='faster_whisper.small', build=_whisper_words),
        'qwen-words': Part('module', role='audio.word_alignment', candidate='qwen3.forced_aligner_0_6b',
                           build=_qwen_words),
        'prosody': Part('module', role='audio.prosody', candidate='dsp', build=_prosody),
        'voice-channels': Part('module', build=_voice_channels),
        'target-person': Part('module', role='vision.face_identity', candidate='lvface.b', build=_target_person),
        'sapiens2-pose': Part('module', role='vision.pose', candidate='sapiens2.1b', build=_sapiens2_pose),
        'semantic-scene': Part('module', role='vision.semantic_scene', candidate='qwen3.5.endpoint',
                               build=_semantic_scene),
    }
    result.update({name: Part('module', build=_math(path)) for name, path in MATH.items()})
    return result

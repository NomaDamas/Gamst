"""Recipe parts for enrolled-person speech and video, the semantic lane and the physics modules.

Builders only: packages, weight URLs and digests live in :mod:`gamst.crew.registry`. A builder reads
the config that ``gamst recipe setup`` wrote for its role; a recipe item may instead carry its own
``"config"`` (local paths), which is the only way to use a candidate setup refuses to install
(non-commercial, licence-bound or unpinned weights).
"""
from __future__ import annotations

import importlib
import sys


def _config(options, context):
    value = options.get('config')
    return dict(value) if isinstance(value, dict) else dict(context['installed'](options)['config'])


def speaker_embedder(options, context):
    """The embedding model of a ``speaker-verify`` item; one instance per model and recipe run."""
    from ..providers.speaker_verify import load_embedder
    config = _config(options, context)
    cache = context.setdefault('speaker_embedders', {})
    key = (config['backend'], config['path'])
    if key not in cache:
        arguments = {name: config[name] for name in ('sha256', 'device', 'num_threads') if name in config}
        if config['backend'] == 'sherpa-onnx':
            arguments.pop('device', None)
        cache[key] = load_embedder(config['backend'], config['path'], **arguments)
    return cache[key]


async def _speaker_verify(options, context):
    from ..providers.speaker_verify import SpeakerVerify
    arguments = {name: options[name] for name in ('hysteresis', 'segment_ms', 'hop_ms', 'min_dbfs',
                                                  'min_utterance_ms', 'ring_seconds', 'segments',
                                                  'capability_id') if name in options}
    threshold = context.get('speaker_threshold')
    module = SpeakerVerify(speaker_embedder(options, context), reference=options.get('reference', 'voice'),
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
    from ..providers.target_speech import EXTRACTORS, SAMSpanSpeech, TargetSpeech
    config = _config(options, context)
    backend = config.pop('backend')
    arguments = dict(reference=options.get('reference', 'voice'), emits=_emits(options))
    if backend == 'sam-span':
        return dict(module=SAMSpanSpeech.from_local(config, **arguments))
    for name in ('block_ms', 'context_ms'):
        if name in options:
            arguments[name] = int(options[name])
    return dict(module=TargetSpeech(EXTRACTORS[backend].from_local(**config), **arguments))


async def _whisper_words(options, context):
    from ..plugins import FasterWhisperPlugin
    from ..providers.asr_words import WhisperWords
    config = _config(options, context)
    return dict(module=WhisperWords(FasterWhisperPlugin(
        config['model_path'], window_seconds=float(options.get('window_seconds', 3.)),
        language=context.get('language') or options.get('language'), device=config.get('device', 'cpu'),
        word_timestamps=True)))


async def _qwen_words(options, context):
    """Qwen3-ASR (existing worker profile) + Qwen3-ForcedAligner; both from explicit local configs."""
    from ..process_provider import ProcessProvider
    from ..providers.asr_words import AlignedASR, QwenForcedAligner
    config = _config(options, context)
    if 'asr' not in config or 'aligner' not in config:
        raise SystemExit('qwen-words needs config.asr (a qwen3.asr.endpoint worker config) and config.aligner')
    aligner = QwenForcedAligner.from_local(**config['aligner'])
    worker = await ProcessProvider.open(context.get('worker_python') or config.get('python') or sys.executable,
                                        config.get('profile', 'qwen3.asr.endpoint'), config['asr'])
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


def face_embedder(options, context):
    from ..providers.person_match import LVFaceEmbedder
    config = _config(options, context)
    return LVFaceEmbedder.from_local(config['path'], sha256=config.get('sha256'),
                                     providers=config.get('providers'))


def body_embedder(options, context):
    """Optional: ``"body": {"source", "checkpoint", ...}`` of a SOLIDER-REID checkout and checkpoint."""
    from ..providers.person_match import SOLIDERBodyEmbedder
    return SOLIDERBodyEmbedder.from_local(**options['body']) if options.get('body') else None


async def _target_person(options, context):
    from ..providers.person_match import TargetPerson
    arguments = {name: options[name] for name in ('face_threshold', 'face_reject', 'body_threshold',
                                                  'face_every_ms', 'body_every_ms', 'hold_ms') if name in options}
    emits = options.get('emits') or ()
    return dict(module=TargetPerson(face_embedder(options, context), body_embedder(options, context),
                                    reference=options.get('reference', 'person'),
                                    emits=emits[0] if emits else None, **arguments))


async def _sapiens2_pose(options, context):
    from ..providers.sapiens2_pose import Sapiens2PosePlugin
    return dict(module=Sapiens2PosePlugin.from_local(**_config(options, context)))


async def _semantic_scene(options, context):
    """The existing streaming VLM client, isolated in its worker so a slow reply never blocks capture."""
    from ..crew.modules import open_worker
    config = _config(options, context)
    config.setdefault('window_frames', 1)
    entry = {} if isinstance(options.get('config'), dict) else context['installed'](options)
    module = await open_worker(context.get('worker_python') or entry.get('python') or sys.executable,
                               'streamingvlm.local', config, inference_timeout=90)
    return dict(module=module, spec=module.spec, every_ms=int(options.get('every_ms', 1000)), queue=1)


def _math(path):
    """A measurement module with no weights; ``"config"`` holds its constructor arguments."""
    async def build(options, context):
        module_name, name = path.split(':')
        module = getattr(importlib.import_module(f'gamst.providers.{module_name}'), name)(**(options.get('config') or {}))
        return dict(module=module, spec=module.spec)
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

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

VOICE_PARTS = ('speaker-verify', 'speaker-gate', 'target-speech')    # what ties sound to an enrolled voice


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
                                                  'capability_id', 'context_ms', 'words', 'source_gate_db')
                 if name in options}
    consumes = options.get('consumes') or ()
    if options.get('sources') in consumes:      # separated sources of overlapping voices, when that part runs
        arguments['sources'] = options['sources']
    if options.get('turns') and any(item.split(':', 1)[0] == options['turns'] for item in consumes):
        arguments['turns'] = True
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


async def _speaker_sources(options, context):
    """Overlapping voices -> one source per speaker (PixIT or SepFormer), in the model's own environment."""
    from ..providers.target_speech import SpeakerSources
    config, python, _ = _entry(options, context, None)
    profiles = {'pixit': 'pixit.sources', 'sepformer': 'sepformer.sources'}
    if config.get('backend') not in profiles:
        raise SystemExit(f"speaker-sources backend is one of {', '.join(profiles)}")
    profile = profiles[config.pop('backend')]
    from ..providers import model_workers
    remote = await model_workers.RemoteSeparator.open(python, profile, config, inference_timeout=600)
    arguments = {name: int(options[name]) for name in ('block_ms', 'context_ms') if name in options}
    return dict(module=SpeakerSources(remote, emits=_emits(options), **arguments))


async def _hosted_identify(options, context):
    """pyannoteAI voiceprint + identification (hosted, billed, audio leaves the device: explicit opt-in)."""
    from ..providers.pyannoteai import HostedIdentify, PyannoteAI
    config, _, _ = _entry(options, context, None)
    voiceprint_file = options.get('voiceprint_file') or config.pop('voiceprint_file', None)
    try:
        client = PyannoteAI(**config)
    except PermissionError as error:
        raise SystemExit(str(error)) from None
    module = HostedIdentify(client, reference=options.get('reference', 'voice'),
                            block_ms=int(options.get('block_ms', 60000)), voiceprint_file=voiceprint_file)
    return dict(module=module, spec=module.spec)


async def _whisper_words(options, context):
    config, python, profile = _entry(options, context, 'faster_whisper.words')
    config = dict(model_path=config['model_path'], device=config.get('device', 'cpu'),
                  window_seconds=float(options.get('window_seconds', 3.)))
    language = context.get('language') or options.get('language')
    if language:
        config['language'] = language
    if options.get('segmentation', 'window') == 'phrases':     # utterances end where phrases do, not at a window edge
        config['phrases'] = {key: options[key] for key in ('hop_seconds', 'margin_seconds', 'max_seconds', 'beam_size')
                             if key in options}
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


async def _audio_onsets(options, context):
    from ..providers.audio_onsets import AudioOnsets
    module = AudioOnsets(**{key: options[key] for key in ('low_hz', 'high_hz', 'sensitivity', 'min_gap_ms', 'min_dbfs')
                            if key in options})
    return dict(module=module, spec=module.spec)


async def _coincidence(options, context):
    from ..providers.coincidence import Coincidence
    module = Coincidence(options['of'], options['by'], of_event=options['of_event'], by_event=options['by_event'],
                         before_ms=int(options.get('before_ms', 30)), after_ms=int(options.get('after_ms', 150)))
    return dict(module=module, spec=module.spec)


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


TRANSCRIPTS = ('speech.transcript_words', 'speech.transcript_update')    # signal names of utterances
CONTEXT = ('transcript', 'counts')


class SegmentModule(WorkerModule):
    """A worker that answers once per segment; says since when a segment is open, and closes the last one.

    ``context`` (of ``CONTEXT``): what the Store already holds about the open segment goes beside each frame as text
    data; nothing is waited for. ``counts`` are the events and counts of ``capabilities`` (the recipe's records).
    """

    def __init__(self, worker, *, human=None, context=(), capabilities=()):
        super().__init__(worker)
        self.pending_since_ms = None
        self.human, self.declared, self.capabilities = human, tuple(context), tuple(capabilities)
        self.context = set(self.declared)

    def status(self):
        return dict(context=sorted(self.context))

    def configure(self, options):
        """Switch off (and on again) what the recipe gives the model besides the frames: ``{"transcript": false}``."""
        if not options or set(options) - set(self.declared) or any(type(value) is not bool for value in options.values()):
            raise ValueError(f"options are {', '.join(self.declared) or 'none here'}: true or false")
        self.context = (self.context | {key for key, value in options.items() if value}) - {
            key for key, value in options.items() if not value}

    def evidence(self, frame):
        """What other modules said about the open segment and is known now (current revisions only)."""
        from ..revisions import record_span
        store, start, end = self.human.store, self.pending_since_ms, frame.timestamp_ms
        result = dict(as_of_ms=store.known_now(), transcript=[], counts=[], events=[])
        replaced = store.superseded(frame.subject_id)

        def current(**arguments):
            return [record for record in store.query(frame.subject_id, kind='observation.signal', registered_only=True,
                                                     **arguments) if record['id'] not in replaced]

        if 'transcript' in self.context:
            newest = {}
            for record in current(start=start, limit=128, any_of=list(TRANSCRIPTS)):
                value, span = record['observation']['value'], record_span(record)
                if isinstance(value, dict) and value.get('text') and span and span[0] <= end and span[1] >= start:
                    newest[value.get('utterance_id') or record['id']] = dict(
                        record_id=record['id'], text=str(value['text'])[:2000], status=value.get('status', 'final'),
                        start_ms=span[0], end_ms=span[1],
                        **{key: value[key] for key in ('utterance_id', 'revision') if value.get(key) is not None})
            result['transcript'] = list(newest.values())[-16:]
        for capability in self.capabilities if 'counts' in self.context else ():
            records = current(start=start, limit=256, capability_id=capability)
            for record in records:
                value = record['observation']['value']
                for event in (value.get('events') or ()) if isinstance(value, dict) else ():
                    if isinstance(event, dict):
                        first, last = (event.get(key) if type(event.get(key)) is int else record['timestamp_ms']
                                       for key in ('start_ms', 'end_ms'))
                        if first <= end and last >= start:
                            result['events'].append(dict(
                                record_id=record['id'], capability=capability, start_ms=first, end_ms=last,
                                event=str(event.get('event', event.get('type')))[:100],
                                **({'label': str(event[key])[:100] for key in ('term', 'label')
                                    if event.get(key) is not None})))
            upto = current(start=max(0, end - 600_000), end=end, limit=1, capability_id=capability)
            if upto and type((upto[0]['observation']['value'] or {}).get('count')) is int:
                result['counts'].append(dict(record_id=upto[0]['id'], capability=capability,
                                             count=upto[0]['observation']['value']['count'],
                                             timestamp_ms=upto[0]['timestamp_ms'],
                                             **{key: value for key, value in upto[0]['observation']['value'].items()
                                                if key in ('reference_id', 'reference_revision')}))
        result['events'] = result['events'][-32:]
        return result

    async def process(self, frame):
        if self.pending_since_ms is None:
            self.pending_since_ms = frame.timestamp_ms
        value = await self._ask(self.worker, frame, self.evidence(frame) if self.declared else None)
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
    The lane keeps the frames of the next segments while a reply is pending. Unless ``"context"`` (or
    ``--vlm-context``) says otherwise, what is transcribed and counted in the same run is given to it as text data;
    a hosted endpoint gets the frames only, unless the owner consents to that separately
    (``--vlm-remote-context``, or ``allow_remote_context`` beside ``allow_remote`` in its config).
    """
    from ..process_provider import ProcessProvider
    from ..providers.streaming_semantic import is_remote
    config, python, profile = _entry(options, context, 'streamingvlm.local')
    every_ms = int(options.get('every_ms', 500))
    frames = max(2, min(64, round(1000 * float(options.get('segment_s', 4)) / every_ms)))
    config.setdefault('window_frames', frames)
    config.setdefault('window_ms', min(60000, config['window_frames'] * every_ms))
    config.setdefault('max_gap_ms', max(2000, min(10000, 3 * every_ms)))
    recipe, modules, asked = context.get('recipe') or {}, context.get('modules') or (), context.get('vlm_context')
    names = options.get('context', CONTEXT) if asked is None else [name for name in asked.split(',') if name and name != 'none']
    if set(names) - set(CONTEXT):
        raise SystemExit(f"a VLM's context is {', '.join(CONTEXT)} or none")
    if config.get('endpoint') is not None and is_remote(config['endpoint']):
        if context.get('remote_context'):
            config['allow_remote_context'] = True
        if config.get('allow_remote_context') is not True:
            if asked is not None and names:
                raise SystemExit(f"{', '.join(names)} would go to a hosted endpoint; that needs its own consent: "
                                 '--vlm-remote-context (frames only: leave --vlm-context out)')
            names = ()
            print('note: the hosted endpoint gets frames only; transcript and counts need --vlm-remote-context', file=sys.stderr)
    else:
        config.pop('allow_remote_context', None)
    from . import PARTS
    capabilities = list(recipe.get('records', ()))[:8]
    given = [name for name in CONTEXT if name in names and context.get('human') is not None and (
        capabilities if name == 'counts' else any(PARTS[item['use']].transcribes for item in modules))]
    config.pop('context', None)
    if given:
        config['context'] = True
        if config.get('allow_remote_context'):
            print(f"note: {' and '.join(given)} go to the hosted endpoint with the frames", file=sys.stderr)
    module = SegmentModule(await ProcessProvider.open(python, profile, config, inference_timeout=90),
                           human=context.get('human'), context=given, capabilities=capabilities)
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


async def _rhythm(options, context):
    """The rhythm of one value: a record's ``value``, or one named channel of a sensor input (``"channel"``)."""
    from ..providers.rhythm import RhythmEstimator
    from .parts import _sensor
    sensor, config = _sensor(options, context), dict(options.get('config') or {})
    if sensor is None:
        module = RhythmEstimator(**config)
        return dict(module=module, spec=module.spec)
    names = [item['name'] for item in sensor.columns]
    if options.get('channel') not in names:
        raise SystemExit(f"module {options['name']!r} names one \"channel\" of the sensor: {', '.join(names)}")
    module = RhythmEstimator(**{**config, 'modality': sensor.modality})
    return dict(module=_Value(module, names.index(options['channel'])), spec=module.spec)


class _Value:
    """One channel of a sensor sample as the single value a rhythm module reads; a missing one is no sample."""

    def __init__(self, module, index):
        self.module, self.index, self.spec = module, index, module.spec

    def process(self, frame):
        from dataclasses import replace
        if not frame.data['mask'][self.index]:
            return None
        return self.module.process(replace(frame, data=dict(value=frame.data['channels'][self.index])))

    def reset(self):
        self.module.reset()


MATH = {'kinematics': 'kinematics:MetricKinematics', 'image-trajectories': 'trajectories:ImageTrajectories',
        'physical-state': 'physical_state:PhysicalState', 
        'angular-motion': 'angular_motion:AngularMotionCounter', 'object-spin': 'object_spin:ObjectSpinCounter',
        'optical-flow': 'motion:OpenCVFlow'}


def parts(Part):
    speech = dict(role='audio.speaker_verify', candidate='sherpa.campplus_zh_en')
    result = {
        'speaker-verify': Part('module', **speech, build=_speaker_verify),
        'speaker-gate': Part('module', build=_speaker_gate),
        'target-speech': Part('module', role='audio.target_speech', candidate='wesep.bsrnn_ecapa',
                              build=_target_speech),
        'speaker-sources': Part('module', role='audio.speaker_sources', candidate='pyannote.pixit_ami',
                                build=_speaker_sources),
        'hosted-identify': Part('module', role='audio.speaker_turns', candidate='pyannoteai.precision2',
                                build=_hosted_identify),
        'whisper-words': Part('module', role='audio.asr', candidate='faster_whisper.small', build=_whisper_words,
                              transcribes=True),
        'qwen-words': Part('module', role='audio.word_alignment', candidate='qwen3.forced_aligner_0_6b',
                           build=_qwen_words, transcribes=True),
        'prosody': Part('module', role='audio.prosody', candidate='dsp', build=_prosody),
        'audio-onsets': Part('module', build=_audio_onsets),
        'coincidence': Part('module', build=_coincidence),
        'voice-channels': Part('module', build=_voice_channels),
        'target-person': Part('module', role='vision.face_identity', candidate='lvface.b', build=_target_person),
        'sapiens2-pose': Part('module', role='vision.pose', candidate='sapiens2.1b', build=_sapiens2_pose),
        'semantic-scene': Part('module', role='vision.semantic_scene', candidate='qwen3.5.endpoint',
                               build=_semantic_scene),
    }
    result['rhythm'] = Part('module', build=_rhythm)
    result.update({name: Part('module', build=_math(path)) for name, path in MATH.items()})
    return result

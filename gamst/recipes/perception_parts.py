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

VOICE_PARTS = ('speaker-verify', 'speaker-gate', 'target-speech', 'speaker-tracks')    # what ties sound to an enrolled voice


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


def _verified(name, modules):
    """A verification lane reads ``name``'s words, directly or through modules that pass them on (the TSE decision)."""
    readers = {item['name']: [source.split(':', 1)[0] for source in item.get('consumes', ())] for item in modules}
    seen, todo = set(), [name]
    while todo:
        current = todo.pop()
        for item in modules:
            if current in readers[item['name']] and item['name'] not in seen:
                if item.get('use') == 'word-verify':
                    return True
                if item.get('use') == 'tse-decide':
                    seen.add(item['name'])
                    todo.append(item['name'])
    return False


def heard_settings(options, context):
    """The item's ``partials`` (``true`` or PartialSpotter settings) with the command line's (``--partials``,
    ``--partial-revisions``, ``--partial-ms``); None when off."""
    given, line = options.get('partials'), context.get('partials')
    if line is False or given in (None, False) and line is None:
        return None
    return dict(given if isinstance(given, dict) else {}, **(line or {}))


def partials_wanted(options, context):
    """A recogniser emits partial records when its item says so and the command line does not turn them off."""
    return bool(options.get('partials')) and context.get('partials') is not False


def _mode(context):
    """The run's counting mode for a module that has one (``count``, the default, or ``search``: gamst.counting)."""
    mode = context.get('mode') or 'count'
    return {} if mode == 'count' else dict(mode=mode, search=context.get('search'))


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
    if _verified(options.get('name'), context.get('modules') or ()):
        # A verification lane settles these words later: each is counted at once as provisional, and the late words
        # of a catch-up worker are scored on the input's kept audio.
        arguments['provisional'] = True
        audio = [source for source in consumes if source in (context.get('inputs') or {})]
        if audio:
            arguments['audio_memory'] = audio[0]
        if arguments.get('words'):      # L1 policy of these words: the item's, then the command line's (--l1 ...)
            l1 = dict(options.get('l1') or {}, **(context.get('l1') or {}))
            if l1:
                arguments['l1'] = l1
    partials = heard_settings(options, context)
    if partials is not None and arguments.get('words'):     # counted words in the partial hypotheses, at once
        arguments['partials'] = partials
    decision = context.get('speaker_decision') or {}
    if decision.get('l1') is not None and arguments.get('words') and not arguments.get('sources') \
            and not arguments.get('turns'):     # speaker decision v2 at the live lane (per-word scores only)
        arguments.update(relative=decision['l1'], bank=decision.get('bank'))
    module = SpeakerVerify(await speaker_embedder(options, context), reference=options.get('reference', 'voice'),
                           threshold=float(options.get('threshold', .5) if threshold is None else threshold),
                           **arguments, **_mode(context))
    return dict(module=module, spec=module.spec)


VERIFY_RECOGNISER, VERIFY_DIARIZER = 'openai.gpt_live_transcribe.hosted', 'pyannoteai.precision3.offline'


async def _word_verify(options, context):
    """The verification lane (:mod:`gamst.providers.word_verification`): a second recogniser, diarization of windows
    and the few-shot voice settle the words counted at once. Hosted by default (OpenAI gpt-live-transcribe with the
    counted words as keywords + Qwen3-ForcedAligner here; pyannoteAI Precision-3): ``--allow-remote-asr`` is the
    owner's opt-in for the recogniser and ``--allow-remote-diarization`` for pyannoteAI, and each says its price first.
    More recognisers are read when the item consumes their ``speech.transcript_words``: then ``"rule"`` defaults to
    ``"vote": true, "decide": "primary"`` (the live recogniser is never outvoted; ``"majority"`` stays an option)."""
    from ..providers.pyannoteai import PyannoteAI
    from ..providers.slot_sources import HostedBatch
    from ..providers.word_verification import WordVerification
    consumes = options.get('consumes') or ()
    audio = next((source for source in consumes if source in (context.get('inputs') or {})), None)
    if audio is None:
        raise SystemExit(f"{options.get('name')}: verification reads the audio input (as ticks) and the words it settles")
    hearing = dict(name=f"{options.get('name')}.recogniser", candidate=options.get('recogniser', VERIFY_RECOGNISER),
                   bias=options.get('bias', 'terms'),
                   **{key: options[key] for key in ('aligner', 'aligner_config', 'aligner_python', 'config', *ENDPOINT_KEYS)
                      if key in options})
    recogniser = (await _stream_words(hearing, context))['module']
    diarization = dict(_registry_config('audio.diarization', options.get('diarizer', VERIFY_DIARIZER)) or {},
                       **(options.get('diarizer_config') or {}))
    if context.get('allow_remote_diarization'):
        diarization['allow_remote'] = True
    diarization.pop('exclusive', None)
    try:
        client = PyannoteAI(**diarization)
    except PermissionError:
        await recogniser.close()
        raise SystemExit(f"{options.get('name')}: {options.get('diarizer', VERIFY_DIARIZER)} sends the sound to "
                         'pyannoteAI (it leaves this device and is billed per job, 20 s minimum): add --allow-remote-diarization '
                         'to consent') from None
    print(f"note: verification diarization: windows of the sound go to pyannoteAI ({client.model}) as they become "
          f"complete; list price {client.eur_per_hour} EUR per hour of audio, 20 s minimum per job", file=sys.stderr)
    diarizer = HostedBatch(client)
    diarizer.name = f'pyannoteai:{client.model}'
    from . import PARTS
    items = {item['name']: item for item in context.get('modules') or ()}
    voters = [source for source in consumes if source.split(':', 1)[0] in items
              and PARTS[items[source.split(':', 1)[0]]['use']].transcribes]
    rule = dict(options.get('rule') or {})
    if voters:      # another recogniser is read: it adds words and may contest the primary's, never outvote it
        rule.setdefault('vote', True)
        rule.setdefault('decide', 'primary')     # owner decision 31; "majority" (two of the three) stays an option
    module = WordVerification(recogniser, diarizer, await speaker_embedder(options, context),
                              reference=options.get('reference', 'voice'), terms=options.get('terms', 'terms'),
                              audio=audio, rule=rule, voice=dict(
                                  {'threshold': float(context['speaker_threshold'])} if context.get('speaker_threshold')
                                  is not None else {}, **(options.get('voice') or {}),
                                  **((context.get('speaker_decision') or {}).get('l4') or {})),
                              bank=(context.get('speaker_decision') or {}).get('bank'), windows=options.get('windows'),
                              **({'feed_speed': options['feed_speed']} if 'feed_speed' in options else {}),
                              **_mode(context))
    return dict(module=module, spec=module.spec)


async def _tse_words(options, context):
    """The TSE lane (:mod:`gamst.providers.overlap_tse`): where the live diarization's slots say voices meet, the enrolled
    person's voice is extracted (``audio.target_speech``, default PS4), heard by a fork of the live recogniser and scored.
    ``"workers"``: extraction workers for catch-up (each its own model process)."""
    from ..providers.overlap_tse import TargetSpeechWords
    entry = _entry(options, context, 'target_speech.local')
    if entry[0].get('backend') not in ('ps4', 'wesep', 'usef'):
        raise SystemExit(f"{options.get('name')}: the TSE lane needs an enrollment-conditioned extractor (ps4, wesep, usef)")
    consumes = options.get('consumes') or ()
    audio = next((source for source in consumes if source in (context.get('inputs') or {})), None)
    if audio is None:
        raise SystemExit(f"{options.get('name')}: the TSE lane reads the audio input and the speaker slots")
    workers = int(options.get('workers', 1))

    async def open_extractor():
        return await _open('RemoteExtractor', entry)

    module = TargetSpeechWords(await open_extractor(), await speaker_embedder(dict(name=options.get('name'), use='speaker-verify'),
                                                                     context),
                               reference=options.get('reference', 'voice'), terms=options.get('terms', 'terms'),
                               audio=audio, asr=options.get('asr', 'asr'), detect=options.get('detect'),
                               window=options.get('window'), rule=dict(
                                   {'threshold': float(context['speaker_threshold'])}
                                   if context.get('speaker_threshold') is not None else {}, **(options.get('rule') or {})),
                               workers=workers, open_extractor=open_extractor if workers > 1 else None)
    return dict(module=module, spec=module.spec)


async def _tse_decide(options, context):
    """The words of ``speaker_check`` inside a TSE stretch, decided from the extracted voice (math, no model)."""
    from ..providers.overlap_tse import TSEDecision
    module = TSEDecision(tse=options.get('tse', 'tse'), reference=options.get('reference', 'voice'),
                         terms=options.get('terms', 'terms'), rule=dict(
                             {'threshold': float(context['speaker_threshold'])}
                             if context.get('speaker_threshold') is not None else {}, **(options.get('rule') or {})))
    return dict(module=module, spec=module.spec)


async def _speaker_slots(options, context):
    """Streaming speaker activity (``audio.diarization``: who is active, per slot, not who it is), in its own environment.

    ``"mode"`` and ``"threshold"`` go to the model's config; audio reaches it in whole 10 ms blocks on a sample clock.
    """
    from ..crew.modules import WorkerModule
    from ..process_provider import ProcessProvider
    from ..providers.speaker_tracks import WholeBlocks
    if isinstance(options.get('config'), dict):
        config, python = dict(options['config']), options.get('python') or sys.executable
        profile = options.get('profile')
        if not profile:
            raise SystemExit(f"{options.get('name')}: a module with its own \"config\" names its worker \"profile\"")
    else:
        entry = context['installed'](options)
        config, python, profile = dict(entry['config']), entry.get('python') or sys.executable, entry.get('provider_profile')
    python = context.get('worker_python') or python
    config.update({key: options[key] for key in ('mode', 'threshold') if key in options})
    worker = await ProcessProvider.open(python, profile, config, startup_timeout=300, inference_timeout=60)
    module = WholeBlocks(WorkerModule(worker))
    return dict(module=module, spec=module.spec)


async def _speaker_tracks(options, context):
    """Each slot's own sound against the enrolled voice; each word goes to the slot active when it was said."""
    from ..providers.speaker_tracks import SpeakerTracks
    arguments = {name: options[name] for name in ('piece_ms', 'min_ms', 'max_slot_ms', 'stable_ms', 'ring_seconds',
                                                  'margin_ms', 'wait_ms', 'min_dbfs', 'overlap', 'capability_id')
                 if name in options}
    threshold = context.get('speaker_threshold')
    module = SpeakerTracks(await speaker_embedder(options, context), reference=options.get('reference', 'voice'),
                           threshold=float(options.get('threshold', .5) if threshold is None else threshold), **arguments)
    return dict(module=module, spec=module.spec)


async def _track_verify(options, context):
    """Settles the provisional words of ``speaker-tracks`` in its own lane (``"queue"``, default 256 records)."""
    from ..providers.speaker_tracks import TrackVerification
    module = TrackVerification(reference=options.get('reference', 'voice'), **{
        name: int(options[name]) for name in ('settle_ms', 'window_ms') if name in options})
    return dict(module=module, spec=module.spec, queue=int(options.get('queue', 256)))


async def _transcript_replay(options, context):
    """Recorded ``speech.transcript_words`` given when the ASR gave them (``"file"``: :mod:`gamst.providers.transcript_replay`)."""
    from ..providers.transcript_replay import TranscriptReplay
    if not options.get('file'):
        raise SystemExit(f"{options.get('name')}: transcript-replay needs \"file\" (a words JSONL)")
    module = TranscriptReplay(options['file'], origin_ms=options.get('origin_ms'))
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


PHRASE_KEYS = ('hop_seconds', 'margin_seconds', 'max_seconds')


def bias_terms(options, context):
    """The spellings of the terms reference named by ``"bias"`` (a recipe option), or none.

    The model is told these words before it listens (Qwen3-ASR's context prompt, a hosted API's hot words or key
    terms). That can make it write one of them where it was not said: measure with and without. Words given while a
    run goes on (MCP) do not change it.
    """
    name = options.get('bias')
    if not name:
        return []
    from ..providers.term_events import parse_terms, plain
    value = (context.get('references') or {}).get(name)
    declared = {item.get('name') for item in (context.get('recipe') or {}).get('references') or ()}
    if value is None and name not in declared:
        return []       # a run that has no such reference (enrolling terms from clips): nothing to bias with
    if value is None:
        raise SystemExit(f"{options.get('name')}: \"bias\" names the reference {name!r}, which this run does not have")
    terms, _ = parse_terms(value)[:2]
    spellings = []
    for term, written in terms.items():
        for spelling in [plain(item) for item in [term, *written]]:     # no patterns; whole words as written
            if spelling and spelling not in spellings:
                spellings.append(spelling)
    return spellings[:100]


async def _qwen_words(options, context):
    """Qwen3-ASR-1.7B + Qwen3-ForcedAligner-0.6B.

    Set up by role (no ``"config"``): the transformers backend in one worker (CPU or Apple MPS), the same phrases as
    ``whisper-words`` with ``segmentation: phrases``. ``"bias": "terms"`` gives the words being counted to Qwen3-ASR as
    its context prompt; ``"pause_seconds"`` splits phrases where the words pause.

    With ``"config": {"asr": {...}, "aligner": {...}}``: the CUDA streaming engine (vLLM) and the aligner, each in its
    own environment (:class:`gamst.providers.asr_words.AlignedASR`).
    """
    from ..process_provider import ProcessProvider
    from ..providers.asr_words import AlignedASR
    if isinstance(options.get('config'), dict) and 'asr' in options['config']:
        config, python, _ = _entry(options, context, None)
        if 'aligner' not in config:
            raise SystemExit('qwen-words needs config.asr and config.aligner ({"python", "config"} each)')
        asr, aligned = config['asr'], config['aligner']
        aligner = await _open('RemoteAligner', (aligned['config'], aligned.get('python') or python,
                                                'qwen3.forced_aligner'))
        worker = await ProcessProvider.open(asr.get('python') or python, asr.get('profile', 'qwen3.asr.endpoint'),
                                            asr['config'], startup_timeout=600, inference_timeout=120)
        module = AlignedASR(worker, aligner, language=context.get('language') or options.get('language'),
                            hop_ms=int(options.get('hop_ms', 20)))
        return dict(module=module, spec=module.spec)
    asr, python, profile = _entry(options, context, 'qwen3.asr.words')
    if 'files' not in asr:
        raise SystemExit('qwen-words without its own config needs the audio.asr candidate qwen3.asr_1_7b.transformers')
    aligner = context['installed'](dict(use='role', role='audio.word_alignment',
                                        candidate=options.get('aligner', 'qwen3.forced_aligner_0_6b')))['config']
    config = dict(model_path=asr['model_path'], files=asr['files'], sha256=asr['sha256'], aligner_path=aligner['path'],
                  aligner_sha256=aligner['sha256'], device=options.get('device', asr.get('device', 'cpu')),
                  dtype=options.get('dtype', asr.get('dtype', 'float32')),
                  language=context.get('language') or options.get('language') or 'ko',
                  pause_seconds=float(options.get('pause_seconds', .5)),
                  phrases={key: options[key] for key in PHRASE_KEYS if key in options})
    terms = bias_terms(options, context)
    if terms:
        config['context'] = ' '.join(terms)
    from ..providers import model_workers
    module = await model_workers.RemoteWhisper.open(python, profile, config, startup_timeout=600, inference_timeout=300)
    return dict(module=module, spec=module.spec)


async def _hosted_words(options, context):
    """A hosted ASR API with word times (``alibaba.*.hosted``, ``elevenlabs.*.hosted``), as phrases.

    Audio leaves the device and it costs money: the config's ``allow_remote`` (or ``--allow-remote-asr``) is the
    owner's opt-in, and the cost line is printed before anything is sent. ``"bias": "terms"`` sends the counted words
    as hot words / key terms.
    """
    from ..providers.hosted_asr import check_config, cost_line
    config, python, profile = _entry(options, context, 'hosted.asr.words')
    if context.get('allow_remote_asr'):
        config['allow_remote'] = True
    for key in ('endpoint', 'api_key_env'):
        if options.get(key):
            config[key] = options[key]
    config.update(language=context.get('language') or options.get('language') or 'ko',
                  pause_seconds=float(options.get('pause_seconds', .5)),
                  phrases={key: options[key] for key in PHRASE_KEYS if key in options} or dict(hop_seconds=8.))
    terms = bias_terms(options, context)
    if terms:
        config['terms'] = terms
    try:
        check_config(config)
    except PermissionError:
        raise SystemExit(remote_asr_refused(options.get('name'), options.get('candidate', HOSTED_WORDS),
                                            (context.get('recipe') or {}).get('name'))) from None
    except ValueError as error:
        raise SystemExit(f"{options.get('name')}: {error}") from error
    print('note: ' + cost_line(config, config['phrases']), file=sys.stderr)
    from ..providers import model_workers
    module = await model_workers.RemoteWhisper.open(python, profile, config, startup_timeout=60, inference_timeout=120)
    return dict(module=module, spec=module.spec)


ENDPOINT_KEYS = ('pause_ms', 'short_pause_ms', 'min_seconds', 'soft_max_seconds', 'max_seconds', 'margin_db')
STREAM_WORDS, HOSTED_WORDS = 'openai.gpt_live_transcribe.hosted', 'elevenlabs.scribe_v2.hosted'   # the parts' defaults


def remote_asr_refused(name, candidate, recipe=None):
    """What a run says when a hosted recogniser would get the sound without the owner's opt-in."""
    from ..crew.registry import ROLES
    local = sorted(key for key, chosen in ROLES['audio.asr'].candidates.items()
                   if chosen.provider_profile == 'streaming.asr.words')
    return (f"{name}: {candidate} sends the sound to a paid hosted API (it leaves this device and is billed). "
            f"To consent, add --allow-remote-asr (or \"allow_remote\": true in the module's \"config\"). "
            f"To transcribe on this device instead, add {' or '.join('--asr ' + key for key in local)} "
            f"(set it up first: gamst recipe setup {recipe or 'RECIPE'} --asr CANDIDATE)")


def remote_asr_preflight(recipe, modules, allowed):
    """Before any input opens or worker starts: a hosted streaming recogniser without the opt-in stops the run.

    Only ``stream-words`` is checked here (its config is the registry's and the item's, known before the run);
    ``hosted-words`` reads the config setup wrote, and its builder refuses the same way.
    """
    if allowed:
        return
    from ..providers.streaming_hosted import check_config
    for item in modules:
        if item.get('use') != 'stream-words':
            continue
        candidate = item.get('candidate', STREAM_WORDS)
        own = item.get('config') if isinstance(item.get('config'), dict) else {}
        config = dict(_registry_config('audio.asr', candidate) or {}, **own)
        if item.get('endpoint'):
            config['endpoint'] = item['endpoint']
        if 'api' not in config:
            continue        # a local recogniser
        try:
            check_config(config)
        except PermissionError:
            raise SystemExit(remote_asr_refused(item['name'], candidate, recipe.get('name'))) from None
        except ValueError:
            pass            # the builder says what is wrong


def _registry_config(role, candidate):
    from ..crew.registry import ROLES
    chosen = ROLES[role].candidates.get(candidate)
    return None if chosen is None else dict(chosen.config or {})


async def _stream_words(options, context):
    """A streaming recogniser with capture-time word times (owner decision: the product path is streaming).

    Hosted (``openai.*.hosted``, ``elevenlabs.*.hosted``): a WebSocket session in this process. Audio leaves the device
    and is billed by the second: the config's ``allow_remote`` (or ``--allow-remote-asr``) is the owner's opt-in and the
    cost line is printed first; nothing needs setting up (the key comes from the environment). A recogniser that returns
    no word times (OpenAI) gets them from the ``audio.word_alignment`` candidate in its own lane (``"aligner"``, default
    ``qwen3.forced_aligner_0_6b``; ``faster_whisper.align_large_v3_turbo`` is a non-default baseline); the text stays the
    recogniser's.

    Local (``sherpa.*``, ``voxtral.*``): the recogniser in its worker, set up by role. ``"partials": true`` also emits
    partial records. ``"bias": "terms"`` gives the counted words as key terms / keywords where the API has them.
    """
    from ..providers import model_workers
    candidate = options.get('candidate', STREAM_WORDS)
    registry = _registry_config('audio.asr', candidate) or {}
    language = context.get('language') or options.get('language') or 'ko'
    if 'api' in registry or 'api' in (options.get('config') or {}):
        from ..providers.streaming_asr import Endpointer, StreamingWords
        from ..providers.streaming_hosted import check_config, cost_line, session
        config = dict(registry, **(options.get('config') or {}))
        if context.get('allow_remote_asr'):
            config['allow_remote'] = True
        for key in ('endpoint', 'api_key_env', 'model', 'commit_strategy', 'delay', 'prompt', 'time_base', 'max_delay',
                    'endpointing', 'mode', 'min_turn_silence', 'max_turn_silence', 'vad_threshold',
                    'end_of_utterance_silence_trigger'):
            if options.get(key):
                config[key] = options[key]
        config['language'] = language
        terms = bias_terms(options, context)
        if terms:
            config['terms'] = terms
        try:
            check_config(config)
            chosen = session(config)
        except PermissionError:
            raise SystemExit(remote_asr_refused(options.get('name'), candidate,
                                                (context.get('recipe') or {}).get('name'))) from None
        except ValueError as error:
            raise SystemExit(f"{options.get('name')}: {error}") from error
        seconds = None
        for source in (context.get('inputs') or {}).values():
            duration = getattr(source, 'duration_s', None)
            try:
                seconds = float(duration()) if callable(duration) else seconds
            except Exception:
                pass
        print('note: ' + cost_line(config, seconds), file=sys.stderr)
        aligner = None
        if not chosen.word_times:
            if isinstance(options.get('aligner_config'), dict):
                entry = dict(config=options['aligner_config'], python=options.get('aligner_python'))
            else:
                entry = context['installed'](dict(use='role', role='audio.word_alignment',
                                                  candidate=options.get('aligner', 'qwen3.forced_aligner_0_6b')))
            aligned = dict(entry['config'])
            profile = 'faster_whisper.align' if 'model_path' in aligned else 'qwen3.forced_aligner'
            aligner = await model_workers.RemoteAligner.open(
                context.get('worker_python') or entry.get('python') or sys.executable, profile, aligned,
                startup_timeout=300, inference_timeout=120)
        endpointer = None
        if chosen.manual_commit:
            endpointer = Endpointer(**{key: float(options[key]) for key in ENDPOINT_KEYS if key in options})
        module = StreamingWords(chosen, aligner=aligner, language=language, partials=partials_wanted(options, context),
                                endpointer=endpointer,
                                # a second recogniser of the run (a voter) commits under a capability of its own
                                **({'capability_id': options['capability_id']} if options.get('capability_id') else {}))
        return dict(module=module, spec=module.spec)
    config, python, profile = _entry(options, context, 'streaming.asr.words')
    config = dict(config, language=language, partials=partials_wanted(options, context))
    settings = {'voxtral-mlx': ('delay_ms', 'pause_seconds', 'max_seconds'),
                'sherpa-zipformer': ('num_threads', 'decoding_method')}.get(config.get('backend'), ())
    config.update({key: options[key] for key in settings if key in options})
    module = await model_workers.RemoteStreamingWords.open(python, profile, config, startup_timeout=600,
                                                           inference_timeout=120)
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
                                    emits=emits[0] if emits else None, joints=joints, **arguments, **_mode(context)))


async def _active_speaker(options, context):
    """Lips that move with the sound, per face; its ``face-points`` and ``face-identity`` stages run inside it."""
    from ..providers.active_speaker import ActiveSpeaker, RemoteFacePoints
    stages = list(context.get('stages', {}).values())
    points = next((item for item in stages if item['use'] == 'face-points'), None)
    identity = next((item for item in stages if item['use'] == 'face-identity'), None)
    if points is None:
        raise SystemExit('active-speaker needs a face-points stage ("in": "<this module>")')
    config, python, _ = _entry(points, context, 'mediapipe.face.video')
    opened = [await RemoteFacePoints.open(python, config)]
    try:
        opened.append(await _open('RemoteActiveSpeaker', _entry(options, context, 'active_speaker.lightasd')))
        opened.append(await face_embedder(identity, context) if identity is not None else None)
    except BaseException:
        for worker in opened:
            await worker.close()
        raise
    faces, lips, embedder = opened
    arguments = {name: options[name] for name in ('hop_ms', 'lookahead_ms', 'context_ms', 'min_face_px', 'max_yaw',
                                                  'min_sharpness', 'identity_every_ms') if name in options}
    module = ActiveSpeaker(faces, lips, embedder, reference=options.get('reference', 'person'), **arguments)
    return dict(module=module, spec=module.spec)


async def _speaker_evidence(options, context):
    from ..providers.speaker_evidence import SpeakerEvidence
    arguments = {name: options[name] for name in ('low', 'high', 'margin_ms', 'min_slots', 'speaking_logit',
                                                  'face_threshold', 'face_reject') if name in options}
    threshold = context.get('speaker_threshold')
    module = SpeakerEvidence(threshold=float(options.get('threshold', .5) if threshold is None else threshold),
                             **arguments)
    return dict(module=module, spec=module.spec)


async def _sapiens2_pose(options, context):
    from ..providers.sapiens2_pose import Sapiens2PosePlugin
    entry = _entry(options, context, 'sapiens2.pose')
    remote = await _open('RemotePose', entry)
    return dict(module=Sapiens2PosePlugin(None, None, None, None, device=None, version=remote.model_id, remote=remote,
                                          joints=options.get('joints'),
                                          tracker=options.get('tracker', 'ocsort')))


async def _rtmw_pose(options, context):
    """RTMW whole-body pose (rtmlib ONNX) in its own environment; joints on the derived ``joints`` input."""
    from ..providers.rtmw_pose import RTMWPosePlugin
    config, python, profile = _entry(options, context, 'rtmw.pose')
    for key in ('device', 'compute_units', 'det_every_ms', 'max_poses'):   # the item's own settings win
        if key in options:
            config[key] = options[key]
    remote = await _open('RemoteRTMW', (config, python, profile))
    return dict(module=RTMWPosePlugin(remote, joints=options.get('joints'), tracker=options.get('tracker', 'ocsort')))


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
        'speaker-slots': Part('module', role='audio.diarization', candidate='nemotron3', build=_speaker_slots),
        'speaker-tracks': Part('module', **speech, build=_speaker_tracks),
        'track-verify': Part('module', build=_track_verify),
        'tse-words': Part('module', role='audio.target_speech', candidate='ps4.bsrnn_real', build=_tse_words,
                          extra_roles=(('speaker', 'audio.speaker_verify', 'sherpa.campplus_zh_en'),)),
        'tse-decide': Part('module', build=_tse_decide),
        'word-verify': Part('module', **speech, build=_word_verify,
                            extra_roles=(('aligner', 'audio.word_alignment', 'qwen3.forced_aligner_0_6b'),)),
        'transcript-replay': Part('module', build=_transcript_replay, transcribes=True),
        'whisper-words': Part('module', role='audio.asr', candidate='faster_whisper.small', build=_whisper_words,
                              transcribes=True, serves=(None,)),
        'qwen-words': Part('module', role='audio.asr', candidate='qwen3.asr_1_7b.transformers',
                           build=_qwen_words, transcribes=True, serves=('qwen3.asr.words',),
                           extra_roles=(('aligner', 'audio.word_alignment', 'qwen3.forced_aligner_0_6b'),)),
        'hosted-words': Part('module', role='audio.asr', candidate='elevenlabs.scribe_v2.hosted', build=_hosted_words,
                             transcribes=True, serves=('hosted.asr.words',)),
        'stream-words': Part('module', role='audio.asr', candidate='openai.gpt_live_transcribe.hosted',
                             build=_stream_words, transcribes=True,
                             serves=('streaming.asr.words', 'gamst.providers.streaming_asr:StreamingWords')),
        'prosody': Part('module', role='audio.prosody', candidate='dsp', build=_prosody),
        'audio-onsets': Part('module', build=_audio_onsets),
        'coincidence': Part('module', build=_coincidence),
        'voice-channels': Part('module', build=_voice_channels),
        'target-person': Part('module', role='vision.face_identity', candidate='lvface.b', build=_target_person),
        'sapiens2-pose': Part('module', role='vision.pose', candidate='sapiens2.1b', build=_sapiens2_pose),
        'rtmw-pose': Part('module', role='vision.pose', candidate='rtmw.l_384x288', build=_rtmw_pose),
        'active-speaker': Part('module', role='vision.active_speaker', candidate='lightasd.ava', build=_active_speaker),
        'face-points': Part('module', role='vision.face', candidate='mediapipe.face'),        # a stage of active-speaker
        'face-identity': Part('module', role='vision.face_identity', candidate='lvface.b'),   # a stage of active-speaker
        'speaker-evidence': Part('module', build=_speaker_evidence),
        'semantic-scene': Part('module', role='vision.semantic_scene', candidate='qwen3.5.endpoint',
                               build=_semantic_scene),
    }
    result['rhythm'] = Part('module', build=_rhythm)
    result.update({name: Part('module', build=_math(path)) for name, path in MATH.items()})
    return result

"""Hosted streaming recognisers over a WebSocket, as sessions of :class:`gamst.providers.streaming_asr.StreamingWords`.

Audio leaves the device and is billed by the second of audio sent (silence included), so a hosted endpoint is the
owner's explicit opt-in, as for the hosted VLM and the phrase-mode hosted ASR (:mod:`gamst.providers.hosted_asr`):
``allow_remote: true``, ``wss://``, and the key only from the environment variable ``api_key_env`` names. A local
endpoint (``ws://127.0.0.1``, a stand-in server in tests) needs neither. :func:`cost_line` is printed before any
audio is sent.

APIs (read from the official documentation on 2026-11-26):

- ``openai``: ``gpt-live-transcribe`` (default; ``gpt-transcribe`` also works) over
  ``wss://api.openai.com/v1/realtime?intent=transcription`` with ``Authorization: Bearer``. ``session.update`` with
  ``session.type: transcription``, 24 kHz PCM16, ``turn_detection: null`` (this model has no server VAD, so the
  client commits each turn), ``languages`` (a list), ``keywords`` (hints, one line each, no ``<`` ``>``), optional
  ``prompt`` and ``delay`` (minimal/low/medium/high/xhigh). Answers:
  ``input_audio_buffer.committed`` (``item_id``), ``conversation.item.input_audio_transcription.delta`` /
  ``.completed`` / ``.failed`` (matched by ``item_id``; completions may come out of order). The docs say:
  "gpt-live-transcribe doesn't return word-level timestamps", so every final's text is the API's and its word times
  come from a local aligner. $0.017 per minute of audio (model page).
  https://developers.openai.com/api/docs/guides/realtime-transcription
- ``elevenlabs``: Scribe v2 Realtime (``scribe_v2_realtime``) or Scribe v3 Live (``scribe_v3_live``) over
  ``wss://api.elevenlabs.io/v1/speech-to-text/realtime`` with ``xi-api-key``; query ``model_id``,
  ``audio_format=pcm_16000``, ``language_code``, ``commit_strategy`` (``manual`` default, ``vad``,
  ``turn_prediction`` on v3 only), ``include_timestamps=true`` (word times in seconds in
  ``committed_transcript_with_timestamps``), ``keyterms``. Audio goes as ``input_audio_chunk`` with
  ``audio_base_64``, ``commit``, ``sample_rate``. $0.22 per hour of audio, keyterms +$0.05 per hour (pricing page).
  https://elevenlabs.io/docs/api-reference/speech-to-text/v-1-speech-to-text-realtime
- ``assemblyai``: AssemblyAI-STT-3.7-Realtime (``assemblyai-stt-3-7-realtime``, 32 languages incl. ``ko``) over
  ``wss://streaming.assemblyai.com/v3/ws`` with ``Authorization: <key>``; binary PCM16 frames; ``Turn`` messages with
  ``words[].start/end`` in milliseconds from the stream start, ``end_of_turn`` marks the final; ``ForceEndpoint`` ends a
  turn, ``Terminate`` the session; ``language_codes``, ``keyterms_prompt`` (JSON lists). Billed by session time,
  $0.30 per hour, keyterms included (pricing page). https://www.assemblyai.com/docs/streaming/message-sequence
- ``deepgram``: Nova-3 monolingual Korean (``model=nova-3&language=ko``) over ``wss://api.deepgram.com/v1/listen`` with
  ``Authorization: Token <key>``; binary PCM16; ``Results`` with ``is_final`` and ``words[].start/end`` in seconds from
  the stream start; ``Finalize`` flushes, ``CloseStream`` ends; ``keyterm`` (repeated). $0.0048 per minute, keyterm
  prompting +$0.0013 per minute (pricing page). https://developers.deepgram.com/docs/live-streaming-audio
- ``speechmatics``: Realtime ``enhanced`` with ``language: ko`` over ``wss://eu2.rt.speechmatics.com/v2`` with
  ``Authorization: Bearer <key>``; ``StartRecognition``, binary ``AddAudio``, ``AddPartialTranscript`` /
  ``AddTranscript`` with ``results[].start_time/end_time`` in seconds from the stream start; ``ForceEndOfUtterance``,
  ``EndOfStream``; ``additional_vocab``; ``max_delay`` (0.7-4 s, default 4; the quickstart's 0.7 here).
  $0.40 per hour (pricing page). https://docs.speechmatics.com/api-ref/realtime-transcription-websocket

Profanity: none of them masks by default for Korean. AssemblyAI ``filter_profanity`` and Deepgram ``profanity_filter``
default to false and are sent as false; Speechmatics only tags (English, Spanish, Italian); ElevenLabs redacts only
with ``entity_redaction``; OpenAI documents no filter. Measured: docs/measurements/streaming-asr-2026-11-26.md.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
from urllib.parse import urlencode, urlparse

APIS = ('openai', 'elevenlabs', 'assemblyai', 'deepgram', 'speechmatics')
DEFAULTS = {
    'openai': dict(endpoint='wss://api.openai.com/v1/realtime?intent=transcription', model='gpt-live-transcribe',
                   api_key_env='OPENAI_API_KEY', usd_per_hour=1.02),
    'elevenlabs': dict(endpoint='wss://api.elevenlabs.io/v1/speech-to-text/realtime', model='scribe_v2_realtime',
                       api_key_env='ELEVENLABS_API_KEY', usd_per_hour=.22, usd_per_hour_keyterms=.05),
    'assemblyai': dict(endpoint='wss://streaming.assemblyai.com/v3/ws', model='assemblyai-stt-3-7-realtime',
                       api_key_env='ASSEMBLYAI_API_KEY', usd_per_hour=.30, usd_per_hour_keyterms=0.),
    'deepgram': dict(endpoint='wss://api.deepgram.com/v1/listen', model='nova-3', api_key_env='DEEPGRAM_API_KEY',
                     usd_per_hour=.288, usd_per_hour_keyterms=.078),
    'speechmatics': dict(endpoint='wss://eu2.rt.speechmatics.com/v2', model='enhanced',
                         api_key_env='SPEECHMATICS_API_KEY', usd_per_hour=.40, usd_per_hour_keyterms=0.),
}


def is_local(endpoint):
    return urlparse(endpoint).hostname in ('127.0.0.1', 'localhost', '::1')


def check_config(config):
    """Refuse a hosted endpoint without the owner's opt-in, ``wss`` and a key variable name; never a key itself."""
    if config.get('api') not in APIS:
        raise ValueError(f"streaming ASR api is one of {', '.join(APIS)}")
    if any(key in config for key in ('api_key', 'key', 'token')):
        raise ValueError('a key is read from the environment only: name its variable in api_key_env')
    endpoint = config.get('endpoint') or DEFAULTS[config['api']]['endpoint']
    if not is_local(endpoint):
        if config.get('allow_remote') is not True:
            raise PermissionError('audio would leave this device for a paid hosted API: set allow_remote to opt in')
        if urlparse(endpoint).scheme != 'wss':
            raise ValueError('a hosted streaming ASR endpoint must use wss')
        if not (config.get('api_key_env') or DEFAULTS[config['api']]['api_key_env']):
            raise ValueError('a hosted streaming ASR endpoint needs api_key_env')
    for term in config.get('terms') or ():
        if any(char in str(term) for char in '<>\r\n'):
            raise ValueError(f'a term may not hold <, > or a line break: {term!r}')
    return config


def cost_line(config, seconds=None):
    """What leaves the device and what it may cost, said before any audio is sent."""
    api = config['api']
    endpoint = config.get('endpoint') or DEFAULTS[api]['endpoint']
    where = 'a local endpoint' if is_local(endpoint) else urlparse(endpoint).hostname
    rate = float(config.get('usd_per_hour', DEFAULTS[api]['usd_per_hour']))
    if config.get('terms'):
        rate += float(config.get('usd_per_hour_keyterms', DEFAULTS[api].get('usd_per_hour_keyterms', 0.)))
    model = config.get('model') or DEFAULTS[api]['model']
    expected = f'; this input ({seconds:.0f} s) about ${rate * seconds / 3600:.4f}' if seconds else ''
    return (f'streaming ASR: every second of sound (silence too) goes to {where} ({model}) as it is captured; '
            f'list price ${rate:g} per hour of audio{expected}')


class Session:
    """What :class:`gamst.providers.streaming_asr.StreamingWords` drives; answers go on ``self.answers``."""
    rate, manual_commit, word_times = 16000, True, True

    def __init__(self, config, *, connect=None):
        check_config(config)
        self.config, self.api = dict(config), config['api']
        defaults = DEFAULTS[self.api]
        self.endpoint = config.get('endpoint') or defaults['endpoint']
        self.model = config.get('model') or defaults['model']
        self.language = config.get('language', 'ko')
        self.terms = [str(term) for term in config.get('terms') or ()][:100]
        variable = config.get('api_key_env') or defaults['api_key_env']
        self.key = os.environ.get(variable)
        if not is_local(self.endpoint) and not self.key:
            raise ValueError(f'{variable} is not set')
        self.name = f'{self.api}.{self.model}'
        self.version = f"{self.api}:{self.model}:{config.get('revision', 'hosted-unversioned')}:terms.{len(self.terms)}"
        self._connect = connect
        self.answers = asyncio.Queue()
        self.ws, self.reader, self.ended, self.seconds_sent = None, None, False, 0.
        self.concurrency = int(config.get('concurrency', 1 if is_local(self.endpoint) else 2))

    def fork(self):
        """Another session of the same recogniser (a catch-up worker's); its audio is billed as well."""
        return type(self)(dict(self.config), connect=self._connect)

    def set_context(self, text):
        """Text said just before this session's first sound (the recogniser's prompt; ``config['context_key']``)."""
        if getattr(self, 'context_key', None) is None:
            return          # this API takes no preceding text: the new session starts without it
        base = self.config.setdefault('_own_' + self.context_key, self.config.get(self.context_key) or '')
        self.config[self.context_key] = f'{base} {text}'.strip()

    async def _open_socket(self, url, headers):
        if self._connect is not None:
            return await self._connect(url, headers)
        from websockets.asyncio.client import connect
        return await connect(url, additional_headers=headers, max_size=8 * 1024 * 1024, open_timeout=20)

    async def _send(self, message):
        await self.ws.send(json.dumps(message))

    async def _read(self):
        try:
            async for raw in self.ws:
                try:
                    message = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                for answer in self.translate(message):
                    await self.answers.put(answer)
        except Exception as error:     # the connection closed or failed: say so once, then stop
            if not self.ended:
                await self.answers.put(dict(kind='error', message=f'{type(error).__name__}: {error}'))
        finally:
            self.ended = True
            await self.answers.put(None)

    async def close(self):
        if self.ws is not None:
            try:
                await self.ws.close()
            except Exception:
                pass
        if self.reader is not None:
            try:
                await asyncio.wait_for(self.reader, 5)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                self.reader.cancel()


class OpenAISession(Session):
    """``gpt-live-transcribe``: 24 kHz PCM16 in, deltas and completed turns out; no word times (an aligner adds them)."""
    rate, manual_commit, word_times, context_key = 24000, True, False, 'prompt'

    async def open(self):
        headers = {'Authorization': 'Bearer ' + self.key} if self.key else {}
        self.ws = await self._open_socket(self.endpoint, headers)
        transcription = dict(model=self.model, languages=[self.language])
        if self.terms:
            transcription['keywords'] = self.terms
        for key in ('prompt', 'delay'):
            if self.config.get(key):
                transcription[key] = self.config[key]
        await self._send(dict(type='session.update', session=dict(type='transcription', audio=dict(input=dict(
            format=dict(type='audio/pcm', rate=self.rate), transcription=transcription, turn_detection=None)))))
        self._partials = {}
        self.reader = asyncio.ensure_future(self._read())

    async def send(self, pcm):
        if pcm:
            self.seconds_sent += len(pcm) / 2 / self.rate
            await self._send(dict(type='input_audio_buffer.append', audio=base64.b64encode(pcm).decode('ascii')))

    async def commit(self):
        await self._send(dict(type='input_audio_buffer.commit'))

    async def end(self):
        pass

    def translate(self, message):
        kind = message.get('type', '')
        if kind == 'input_audio_buffer.committed':
            return [dict(kind='committed', key=message['item_id'])]
        if kind == 'conversation.item.input_audio_transcription.delta':
            key = message.get('item_id')
            self._partials[key] = self._partials.get(key, '') + str(message.get('delta', ''))
            return [dict(kind='partial', key=key, text=self._partials[key])]
        if kind == 'conversation.item.input_audio_transcription.completed':
            self._partials.pop(message.get('item_id'), None)
            return [dict(kind='final', key=message['item_id'], text=str(message.get('transcript', '')), words=None)]
        if kind == 'conversation.item.input_audio_transcription.failed':
            error = message.get('error') or {}
            return [dict(kind='error', message=f"turn {message.get('item_id')} failed: {error.get('message', error)}"),
                    dict(kind='final', key=message.get('item_id'), text='', words=None)]
        if kind == 'error':
            error = message.get('error') or {}
            return [dict(kind='error', message=str(error.get('message', error)))]
        return []


class ElevenLabsSession(Session):
    """Scribe Realtime / Live: 16 kHz PCM16 in; committed segments with word times (seconds) out."""
    rate, word_times, context_key = 16000, True, 'previous_text'

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.strategy = config.get('commit_strategy', 'manual')
        if self.strategy not in ('manual', 'vad', 'turn_prediction'):
            raise ValueError('commit_strategy is manual, vad or turn_prediction')
        self.manual_commit = self.strategy == 'manual'
        # Where Scribe counts word times from (``session``: the start of the session's audio; ``segment``: the start of
        # each committed segment). Measured with the real API (docs/measurements/streaming-asr-2026-11-26.md).
        self.time_base = config.get('time_base', 'session')

    async def open(self):
        query = [('model_id', self.model), ('audio_format', 'pcm_16000'), ('language_code', self.language),
                 ('commit_strategy', self.strategy), ('include_timestamps', 'true')]
        query += [('keyterms', term) for term in self.terms]
        for key in ('vad_silence_threshold_secs', 'vad_threshold', 'min_speech_duration_ms', 'min_silence_duration_ms'):
            if key in self.config:
                query.append((key, str(self.config[key])))
        separator = '&' if '?' in self.endpoint else '?'
        headers = {'xi-api-key': self.key} if self.key else {}
        self.ws = await self._open_socket(self.endpoint + separator + urlencode(query), headers)
        self._segment, self._first, self._ending = 0, True, False
        self.reader = asyncio.ensure_future(self._read())

    async def send(self, pcm, *, commit=False):
        if not pcm and not commit:
            return
        self.seconds_sent += len(pcm) / 2 / self.rate
        message = dict(message_type='input_audio_chunk', audio_base_64=base64.b64encode(pcm).decode('ascii'),
                       commit=bool(commit), sample_rate=self.rate)
        if self._first and self.config.get('previous_text'):
            message['previous_text'] = str(self.config['previous_text'])[-50:]   # what was said just before
        self._first = False
        await self._send(message)

    async def commit(self):
        await self.send(b'', commit=True)

    async def end(self):
        self._ending = True
        if not self.manual_commit:
            await self.send(b'', commit=True)

    def translate(self, message):
        kind = message.get('message_type', '')
        if kind == 'partial_transcript':
            return [dict(kind='partial', key=str(self._segment), text=str(message.get('text', '')))]
        if kind == 'committed_transcript_with_timestamps':
            key = str(self._segment)
            self._segment += 1
            words = [(str(word.get('text', '')).strip(), float(word['start']), float(word['end']))
                     for word in message.get('words') or () if word.get('type', 'word') == 'word'
                     and str(word.get('text', '')).strip() and word.get('start') is not None
                     and word.get('end') is not None]
            out = [dict(kind='committed', key=key),
                   dict(kind='final', key=key, text=str(message.get('text', '')), words=words or None,
                        base=self.time_base)]
            if self._ending and not self.manual_commit:
                self.ended = True
            return out
        if kind in ('error', 'auth_error', 'quota_exceeded', 'rate_limited', 'input_error', 'transcriber_error',
                    'resource_exhausted', 'session_time_limit_exceeded', 'chunk_size_exceeded', 'insufficient_audio_activity',
                    'commit_throttled', 'unaccepted_terms', 'queue_overflow'):
            return [dict(kind='error', message=f"{kind}: {message.get('error', message.get('message', ''))}")]
        return []


class _BinarySession(Session):
    """Sessions that take raw PCM16 as binary frames and close the socket themselves after ``end``."""
    rate, manual_commit, word_times, closes_on_end = 16000, False, True, True

    async def send(self, pcm):
        if pcm:
            self.seconds_sent += len(pcm) / 2 / self.rate
            self.sent_chunks = getattr(self, 'sent_chunks', 0) + 1
            await self.ws.send(pcm)

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        strategy = self.config.get('commit_strategy', 'server')
        if strategy not in ('server', 'manual'):
            raise ValueError('commit_strategy is server (the service ends turns) or manual (GAMST commits)')
        self.manual_commit = strategy == 'manual'


class AssemblyAISession(_BinarySession):
    """AssemblyAI-STT-3.7-Realtime: turns with word times in ms from the stream start."""

    async def open(self):
        query = [('speech_model', self.model), ('sample_rate', str(self.rate)), ('encoding', 'pcm_s16le'),
                 ('language_codes', json.dumps([self.language])), ('filter_profanity', 'false')]
        if self.terms:
            query.append(('keyterms_prompt', json.dumps(self.terms, ensure_ascii=False)))
        for key in ('mode', 'min_turn_silence', 'max_turn_silence', 'vad_threshold', 'prompt'):
            if key in self.config:
                query.append((key, str(self.config[key])))
        separator = '&' if '?' in self.endpoint else '?'
        self.ws = await self._open_socket(self.endpoint + separator + urlencode(query),
                                          {'Authorization': self.key} if self.key else {})
        self.reader = asyncio.ensure_future(self._read())

    async def commit(self):
        await self._send(dict(type='ForceEndpoint'))

    async def end(self):
        await self._send(dict(type='Terminate'))

    def translate(self, message):
        kind = message.get('type', '')
        if kind == 'Turn':
            key = str(message.get('turn_order'))
            if not message.get('end_of_turn'):
                return [dict(kind='partial', key=key, text=str(message.get('transcript', '')))]
            words = [(str(word.get('text', '')).strip(), word['start'] / 1000, word['end'] / 1000)
                     for word in message.get('words') or () if str(word.get('text', '')).strip()
                     and type(word.get('start')) in (int, float) and type(word.get('end')) in (int, float)]
            return [dict(kind='final', key=key, text=str(message.get('transcript', '')), words=words or None,
                         base='session')]
        if kind == 'Error' or 'error' in message:
            return [dict(kind='error', message=str(message.get('error') or message))]
        return []


class DeepgramSession(_BinarySession):
    """Deepgram Nova-3: interim and final results with word times in seconds from the stream start."""

    async def open(self):
        query = [('model', self.model), ('language', self.language), ('encoding', 'linear16'),
                 ('sample_rate', str(self.rate)), ('channels', '1'), ('interim_results', 'true'), ('punctuate', 'true'),
                 ('profanity_filter', 'false'), ('endpointing', str(self.config.get('endpointing', 300)))]
        query += [('keyterm', term) for term in self.terms]
        separator = '&' if '?' in self.endpoint else '?'
        self.ws = await self._open_socket(self.endpoint + separator + urlencode(query),
                                          {'Authorization': 'Token ' + self.key} if self.key else {})
        self.reader = asyncio.ensure_future(self._read())

    async def commit(self):
        await self._send(dict(type='Finalize'))

    async def end(self):
        await self._send(dict(type='CloseStream'))

    def translate(self, message):
        kind = message.get('type', '')
        if kind == 'Results':
            alternative = ((message.get('channel') or {}).get('alternatives') or [{}])[0]
            key = str(round(1000 * float(message.get('start', 0))))
            text = str(alternative.get('transcript', ''))
            if not message.get('is_final'):
                return [dict(kind='partial', key=key, text=text)]
            words = [(str(word.get('punctuated_word') or word.get('word') or '').strip(), float(word['start']),
                      float(word['end'])) for word in alternative.get('words') or ()
                     if str(word.get('word', '')).strip() and word.get('start') is not None and word.get('end') is not None]
            return [dict(kind='final', key=key, text=text, words=words or None, base='session')]
        if kind == 'Error' or message.get('err_code'):
            return [dict(kind='error', message=str(message.get('description') or message.get('err_msg') or message))]
        return []


class SpeechmaticsSession(_BinarySession):
    """Speechmatics Realtime: final segments with word times in seconds from the stream start."""

    async def open(self):
        self.ws = await self._open_socket(self.endpoint, {'Authorization': 'Bearer ' + self.key} if self.key else {})
        config = dict(language=self.language, model=self.model, enable_partials=True,
                      max_delay=float(self.config.get('max_delay', .7)))
        if self.terms:
            config['additional_vocab'] = [dict(content=term) for term in self.terms]
        if 'end_of_utterance_silence_trigger' in self.config:
            config['conversation_config'] = dict(
                end_of_utterance_silence_trigger=float(self.config['end_of_utterance_silence_trigger']))
        await self._send(dict(message='StartRecognition', transcription_config=config,
                              audio_format=dict(type='raw', encoding='pcm_s16le', sample_rate=self.rate)))
        self.reader = asyncio.ensure_future(self._read())

    async def commit(self):
        await self._send(dict(message='ForceEndOfUtterance'))

    async def end(self):
        await self._send(dict(message='EndOfStream', last_seq_no=getattr(self, 'sent_chunks', 0)))

    @staticmethod
    def words(results):
        words = []
        for item in results or ():
            content = str(((item.get('alternatives') or [{}])[0]).get('content', '')).strip()
            if not content or item.get('start_time') is None:
                continue
            if item.get('type') == 'punctuation' and words:
                words[-1][0] += content
            elif item.get('type') != 'punctuation':
                words.append([content, float(item['start_time']), float(item['end_time'])])
        return [tuple(word) for word in words]

    def translate(self, message):
        kind = message.get('message', '')
        if kind in ('AddPartialTranscript', 'AddTranscript'):
            metadata = message.get('metadata') or {}
            self._segment = getattr(self, '_segment', 0)
            key, text = str(self._segment), str(metadata.get('transcript', '')).strip()
            if kind == 'AddPartialTranscript':
                return [dict(kind='partial', key=key, text=text)]
            self._segment += 1
            words = self.words(message.get('results'))
            return [dict(kind='final', key=key, text=text, words=words or None, base='session')]
        if kind == 'EndOfTranscript':
            self.ended = True
        if kind == 'Error':
            return [dict(kind='error', message=f"{message.get('type')}: {message.get('reason')}")]
        return []


SESSIONS = {'openai': OpenAISession, 'elevenlabs': ElevenLabsSession, 'assemblyai': AssemblyAISession,
            'deepgram': DeepgramSession, 'speechmatics': SpeechmaticsSession}


def session(config, **kwargs):
    return SESSIONS[config['api']](config, **kwargs)

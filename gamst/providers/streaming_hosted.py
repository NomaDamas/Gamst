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
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
from urllib.parse import urlencode, urlparse

APIS = ('openai', 'elevenlabs')
DEFAULTS = {
    'openai': dict(endpoint='wss://api.openai.com/v1/realtime?intent=transcription', model='gpt-live-transcribe',
                   api_key_env='OPENAI_API_KEY', usd_per_hour=1.02),
    'elevenlabs': dict(endpoint='wss://api.elevenlabs.io/v1/speech-to-text/realtime', model='scribe_v2_realtime',
                       api_key_env='ELEVENLABS_API_KEY', usd_per_hour=.22, usd_per_hour_keyterms=.05),
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
    if api == 'elevenlabs' and config.get('terms'):
        rate += float(config.get('usd_per_hour_keyterms', DEFAULTS[api]['usd_per_hour_keyterms']))
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


SESSIONS = {'openai': OpenAISession, 'elevenlabs': ElevenLabsSession}


def session(config, **kwargs):
    return SESSIONS[config['api']](config, **kwargs)

"""Hosted speech recognition with word times, as decoders of :class:`gamst.providers.whisper_phrases.Phrases`.

Audio leaves the device and every request costs money, so a hosted endpoint is the owner's explicit opt-in, the same
rule as the hosted VLM (:mod:`gamst.providers.streaming_semantic`): ``allow_remote: true``, HTTPS, and the key only
from the environment variable ``api_key_env`` names (never stored in a config, profile or record). A local endpoint
(127.0.0.1, e.g. a stand-in server in tests) needs neither.

APIs (read from their official documentation, 2026-11-26; neither has been called by GAMST with a real key):

- ``dashscope``: Alibaba Model Studio ``qwen-audio-3.1-asr-flash``, DashScope native multimodal-generation:
  base64 WAV data URI in ``input.messages``, ``parameters.language_hints``; the answer has word times in
  milliseconds (``output.sentence[s].words[].begin_time/end_time``). Hot words: ``parameters.vocabulary``
  ({word: weight 1-5}). https://www.alibabacloud.com/help/en/model-studio/qwen-audio-asr-http-api
- ``elevenlabs``: ElevenLabs Scribe (``model_id`` ``scribe_v2``), multipart ``POST /v1/speech-to-text`` with
  ``timestamps_granularity=word``; ``words[]`` of type ``word`` with ``start``/``end`` in seconds. Key terms:
  ``keyterms`` (a surcharge). https://elevenlabs.io/docs/api-reference/speech-to-text/convert

The phrase buffer decodes the held audio again every ``hop_seconds``, so the audio sent is more than the audio
heard: at most about ``max_seconds / hop_seconds`` times. :func:`cost_line` says so before the first request.
"""
from __future__ import annotations

import base64
import io
import json
import os
import urllib.request
import uuid
import wave
from urllib.parse import urlparse

from .whisper_phrases import Phrase

RATE = 16000
APIS = ('dashscope', 'elevenlabs')


def is_local(endpoint):
    return urlparse(endpoint).hostname in ('127.0.0.1', 'localhost', '::1')


def check_config(config):
    """Refuse a hosted endpoint without the owner's opt-in, HTTPS and a key variable name; never a key itself."""
    if config.get('api') not in APIS:
        raise ValueError(f"hosted ASR api is one of {', '.join(APIS)}")
    for key in ('endpoint', 'model'):
        if type(config.get(key)) is not str or not config[key]:
            raise ValueError(f'hosted ASR needs {key}')
    if any(key in config for key in ('api_key', 'key', 'token')):
        raise ValueError('a key is read from the environment only: name its variable in api_key_env')
    if '{' in config['endpoint']:
        raise ValueError(f"set the endpoint: {config['endpoint']} names a placeholder (e.g. your workspace)")
    if not is_local(config['endpoint']):
        if config.get('allow_remote') is not True:
            raise PermissionError('audio would leave this device for a paid hosted API: set allow_remote to opt in')
        if urlparse(config['endpoint']).scheme != 'https':
            raise ValueError('a hosted ASR endpoint must use https')
        if type(config.get('api_key_env')) is not str or not config['api_key_env']:
            raise ValueError('a hosted ASR endpoint needs api_key_env (the name of the variable holding the key)')
    return config


def cost_line(config, phrases):
    """What leaves the device and what it may cost, said before any request."""
    hop, longest = float(phrases.get('hop_seconds', 4.)), float(phrases.get('max_seconds', 25.))
    where = 'a local endpoint' if is_local(config['endpoint']) else urlparse(config['endpoint']).hostname
    price = config.get('price') or 'price not in the config'
    return (f"hosted ASR: audio goes to {where} ({config['model']}); price {price}; the phrase buffer sends each "
            f"second of sound up to about {longest / hop:.0f} times (every {hop:g} s, at most {longest:g} s at once)")


def wav_bytes(samples):
    import numpy as np
    pcm = (np.clip(np.asarray(samples, dtype=np.float32), -1, 1) * 32767).astype('<i2')
    handle = io.BytesIO()
    with wave.open(handle, 'wb') as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(pcm.tobytes())
    return handle.getvalue()


def pause_phrases(words, pause_seconds):
    """[(text, start_s, end_s)] -> phrases split where the words pause (text joined with spaces)."""
    phrases, current = [], []
    for word in words:
        if current and word[1] - current[-1][2] >= pause_seconds:
            phrases.append(current)
            current = []
        current.append(word)
    if current:
        phrases.append(current)
    return [Phrase(' '.join(text for text, _, _ in group), group, max(end for _, _, end in group)) for group in phrases]


class HostedDecoder:
    """One request per decode; the answer's words become phrases with times from the start of the samples."""

    def __init__(self, config, *, opener=None):
        check_config(config)
        self.config, self.api = dict(config), config['api']
        self.endpoint, self.model = config['endpoint'], config['model']
        self.language = config.get('language', 'ko')
        self.terms = [str(term) for term in config.get('terms') or ()][:100]
        self.pause_seconds = float(config.get('pause_seconds', .5))
        self.timeout = float(config.get('timeout', 60))
        variable = config.get('api_key_env')
        self.key = os.environ.get(variable) if variable else None
        if not is_local(self.endpoint) and not self.key:
            raise ValueError(f'{variable} is not set')
        self.name = f'hosted.{self.api}'
        self.version = f"{self.api}:{self.model}:{config.get('revision', 'hosted-unversioned')}:terms.{len(self.terms)}"
        self.opener = opener or urllib.request.urlopen
        self.requests, self.seconds_sent = 0, 0.

    def decode(self, samples):
        self.requests += 1
        self.seconds_sent += len(samples) / RATE
        audio = wav_bytes(samples)
        if self.api == 'dashscope':
            return self._dashscope(audio)
        return self._elevenlabs(audio)

    def _send(self, request):
        with self.opener(request, timeout=self.timeout) as response:
            return json.loads(response.read(8 * 1024 * 1024 + 1))

    def _dashscope(self, audio):
        parameters = dict(format='wav', sample_rate=str(RATE), language_hints=[self.language])
        if self.terms:
            parameters['vocabulary'] = {term: int(self.config.get('term_weight', 4)) for term in self.terms}
        content = [dict(type='input_audio', input_audio=dict(
            data='data:audio/wav;base64,' + base64.b64encode(audio).decode('ascii')))]
        body = dict(model=self.model, input=dict(messages=[dict(role='user', content=content)]), parameters=parameters)
        headers = {'Content-Type': 'application/json', 'X-DashScope-SSE': 'disable'}
        if self.key:
            headers['Authorization'] = 'Bearer ' + self.key
        answer = self._send(urllib.request.Request(self.endpoint, data=json.dumps(body).encode(), headers=headers))
        output = answer.get('output') or {}
        sentences = output.get('sentences') or ([output['sentence']] if output.get('sentence') else [])
        phrases = []
        for sentence in sentences:
            words = [(str(word['text']).strip(), word['begin_time'] / 1000, word['end_time'] / 1000)
                     for word in sentence.get('words') or () if str(word.get('text', '')).strip()
                     and type(word.get('begin_time')) in (int, float) and type(word.get('end_time')) in (int, float)]
            if words:
                text = str(sentence.get('text') or ' '.join(word for word, _, _ in words)).strip()
                phrases.append(Phrase(text, words, max(end for _, _, end in words)))
        return phrases

    def _elevenlabs(self, audio):
        boundary = uuid.uuid4().hex
        fields = [('model_id', self.model), ('language_code', self.language), ('timestamps_granularity', 'word'),
                  ('tag_audio_events', 'false')] + [('keyterms', term) for term in self.terms]
        body = io.BytesIO()
        for name, value in fields:
            body.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
        body.write(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
                   'Content-Type: audio/wav\r\n\r\n'.encode() + audio + f'\r\n--{boundary}--\r\n'.encode())
        headers = {'Content-Type': f'multipart/form-data; boundary={boundary}'}
        if self.key:
            headers['xi-api-key'] = self.key
        answer = self._send(urllib.request.Request(self.endpoint, data=body.getvalue(), headers=headers))
        words = [(str(word['text']).strip(), float(word['start']), float(word['end'])) for word in answer.get('words') or ()
                 if word.get('type', 'word') == 'word' and str(word.get('text', '')).strip()
                 and word.get('start') is not None and word.get('end') is not None]
        return pause_phrases(words, self.pause_seconds)


def load_decoder(config):
    return HostedDecoder(config)

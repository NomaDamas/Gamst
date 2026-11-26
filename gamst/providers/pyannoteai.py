"""pyannoteAI hosted speaker identification (voiceprint + diarization), as speaker-turn records.

**The audio leaves this device** and every successful job is billed, so nothing here runs unless the
config says ``allow_remote: true``, and a cost line is printed before each paid call. The API key is
read from the environment variable named by ``api_key_env`` (default ``PYANNOTEAI_API_KEY``) at the
moment of the call; it is never part of a config, a record, a report or a log line.

API (https://docs.pyannote.ai, OpenAPI read 2026-11-26), base ``https://api.pyannote.ai``, header
``Authorization: Bearer <key>``:

- ``POST /v1/media/input {"url": "media://<key>"}`` -> ``{"url": <pre-signed PUT url>}``; the WAV is PUT
  there (without the API key: it is a storage URL). pyannoteAI keeps it at least 24 h.
- ``POST /v1/voiceprint {"url", "model"}`` -> ``{"jobId"}``; at most 30 s of one speaker. Billed per
  voiceprint (list price 0.015 EUR).
- ``POST /v1/identify {"url", "model", "voiceprints": [{"label", "voiceprint"}], "matching": {"threshold",
  "exclusive"}, "turnLevelConfidence"}`` -> ``{"jobId"}``. Billed by audio duration, 20 s minimum per job.
- ``GET /v1/jobs/{jobId}`` -> ``{"status": created|pending|running|succeeded|failed|canceled, "output"}``.
  Identification output: ``identification: [{"speaker", "diarizationSpeaker", "match", "start", "end",
  "confidence"}]``; voiceprint output: ``{"voiceprint"}``. Outputs are deleted after 24 h.

Prices change and depend on the plan: ``eur_per_hour`` and ``eur_per_voiceprint`` in the config are the
numbers the cost line uses (the registry names where they were read). Only ``succeeded`` jobs are billed.

The voiceprint is a voice template of the enrolled person: it stays in memory, or in the owner's own
``voiceprint_file`` (written with mode 600) so it is created once; it is never a record.
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from urllib.parse import urlparse

from ..capabilities import CapabilitySpec, OutputField
from .target_speech import SAMPLE_RATE, Blocks

ENDPOINT = 'https://api.pyannote.ai'
MAX_VOICEPRINT_S = 30.
MIN_BILLED_S = 20.
DONE = ('succeeded', 'failed', 'canceled')


def wav_bytes(samples, rate=SAMPLE_RATE):
    """16-bit mono WAV of float samples in [-1, 1]."""
    import wave
    import numpy as np
    pcm = (np.clip(np.asarray(samples, dtype=np.float32).reshape(-1), -1., 1.) * 32767).astype('<i2')
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(int(rate))
        target.writeframes(pcm.tobytes())
    return buffer.getvalue()


class PyannoteAI:
    """A few blocking HTTP calls (``urllib``); run them off the event loop."""

    def __init__(self, *, endpoint=ENDPOINT, api_key_env='PYANNOTEAI_API_KEY', model='precision-2',
                 eur_per_hour, eur_per_voiceprint=.015, allow_remote=False, matching_threshold=0,
                 poll_s=2., timeout_s=600., out=None, **_):
        if allow_remote is not True:
            raise PermissionError('pyannoteAI is a hosted service: the audio leaves this device and each job is '
                                  'billed. Set allow_remote: true to opt in')
        parsed = urlparse(endpoint)
        local = parsed.hostname in ('127.0.0.1', 'localhost', '::1')
        if parsed.scheme != 'https' and not (parsed.scheme == 'http' and local):
            raise ValueError('the pyannoteAI endpoint must be https (http only for a local stand-in)')
        if type(api_key_env) is not str or not api_key_env:
            raise ValueError('api_key_env names the environment variable that holds the key')
        self.endpoint, self.api_key_env, self.model = endpoint.rstrip('/'), api_key_env, model
        self.eur_per_hour, self.eur_per_voiceprint = float(eur_per_hour), float(eur_per_voiceprint)
        self.matching_threshold = float(matching_threshold)
        self.poll_s, self.timeout_s = float(poll_s), float(timeout_s)
        self.out = out or sys.stderr
        self.spent_eur, self.calls = 0., []

    # ------------------------------------------------------------------ http

    def _key(self):
        key = os.environ.get(self.api_key_env)
        if not key:
            raise PermissionError(f'set {self.api_key_env} (the pyannoteAI API key) in the environment')
        return key

    def _request(self, method, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.endpoint + path, data=data, method=method, headers={
            'Authorization': f'Bearer {self._key()}', 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read() or b'{}')
        except urllib.error.HTTPError as error:
            detail = error.read()[:300].decode(errors='replace')
            raise RuntimeError(f'pyannoteAI {method} {path}: HTTP {error.code} {detail}') from None

    def upload(self, samples):
        """``media://`` URL of the samples as a WAV in pyannoteAI's temporary storage."""
        media = f'media://gamst-{uuid.uuid4().hex}'
        target = self._request('POST', '/v1/media/input', {'url': media})['url']
        put = urllib.request.Request(target, data=wav_bytes(samples), method='PUT',
                                     headers={'Content-Type': 'audio/wav'})
        with urllib.request.urlopen(put, timeout=120) as response:
            if response.status not in (200, 201, 204):
                raise RuntimeError(f'pyannoteAI upload: HTTP {response.status}')
        return media

    def _job(self, job):
        deadline = time.monotonic() + self.timeout_s
        while True:
            value = self._request('GET', f"/v1/jobs/{job['jobId']}")
            if value.get('status') in DONE:
                if value['status'] != 'succeeded':
                    raise RuntimeError(f"pyannoteAI job {value['status']}: "
                                       f"{str((value.get('output') or {}).get('error'))[:200]}")
                return value.get('output') or {}
            if time.monotonic() > deadline:
                raise TimeoutError('pyannoteAI job did not finish in time (it may still be billed if it succeeds)')
            time.sleep(self.poll_s)

    # ------------------------------------------------------------------ paid calls

    def _write(self, line):
        self.out.write(line + '\n')
        self.out.flush()

    def _say(self, kind, cost, detail):
        self._write(f'pyannoteAI {kind}: {detail}; expected cost {cost:.4f} EUR (list price, before any '
                    f'included credit); the audio leaves this device')

    def voiceprint(self, samples):
        seconds = len(samples) / SAMPLE_RATE
        if seconds > MAX_VOICEPRINT_S:
            raise ValueError('a voiceprint is made from at most 30 s of one speaker')
        self._say('voiceprint', self.eur_per_voiceprint,
                  f'one voiceprint from {seconds:.1f} s, model {self.model}, {self.eur_per_voiceprint} EUR each')
        output = self._job(self._request('POST', '/v1/voiceprint', {'url': self.upload(samples), 'model': self.model}))
        self.spent_eur += self.eur_per_voiceprint
        self.calls.append(dict(call='voiceprint', seconds=round(seconds, 2), eur=self.eur_per_voiceprint))
        return output['voiceprint']

    def identify(self, samples, voiceprints):
        seconds = len(samples) / SAMPLE_RATE
        billed = max(MIN_BILLED_S, seconds)
        cost = billed / 3600 * self.eur_per_hour
        self._say('identify', cost, f'{seconds:.1f} s of audio (billed as {billed:.1f} s), model {self.model}, '
                                    f'{self.eur_per_hour} EUR/h')
        body = {'url': self.upload(samples), 'model': self.model, 'turnLevelConfidence': True,
                'voiceprints': [dict(label=label, voiceprint=value) for label, value in voiceprints.items()],
                'matching': {'threshold': self.matching_threshold, 'exclusive': True}}
        output = self._job(self._request('POST', '/v1/identify', body))
        self.spent_eur += cost
        self.calls.append(dict(call='identify', seconds=round(seconds, 2), billed_s=round(billed, 2),
                               eur=round(cost, 6)))
        return output

    def diarize(self, samples, *, exclusive=False):
        """``/v1/diarize`` of the whole sound (offline): ``{"diarization": [{speaker, start, end}], ...}``."""
        seconds = len(samples) / SAMPLE_RATE
        billed = max(MIN_BILLED_S, seconds)
        cost = billed / 3600 * self.eur_per_hour
        self._say('diarize', cost, f'{seconds:.1f} s of audio (billed as {billed:.1f} s), model {self.model}, '
                                   f'{self.eur_per_hour} EUR/h')
        body = {'url': self.upload(samples), 'model': self.model}
        if exclusive:
            body['exclusive'] = True
        output = self._job(self._request('POST', '/v1/diarize', body))
        self.spent_eur += cost
        self.calls.append(dict(call='diarize', seconds=round(seconds, 2), billed_s=round(billed, 2),
                               eur=round(cost, 6)))
        return output

    def live_stream(self, *, heartbeat_ms=200):
        """``POST /v1/live`` -> the single-use WebSocket URL of one Live-1 stream (billed by streamed duration)."""
        self._say('live stream', MIN_BILLED_S / 3600 * self.eur_per_hour,
                  f'Live-1 at {self.eur_per_hour} EUR/h of streamed audio, {MIN_BILLED_S:.0f} s minimum per stream; '
                  f'the cost shown is that minimum')
        return self._request('POST', '/v1/live', {'heartbeat': {'enabled': True, 'interval': int(heartbeat_ms)}})['url']

    def streamed(self, seconds):
        """Book one finished stream: billed by the seconds sent, 20 s minimum."""
        if seconds <= 0:
            return
        billed = max(MIN_BILLED_S, seconds)
        cost = billed / 3600 * self.eur_per_hour
        self.spent_eur += cost
        self.calls.append(dict(call='live', seconds=round(seconds, 2), billed_s=round(billed, 2), eur=round(cost, 6)))
        self._write(f'pyannoteAI live stream ended: {seconds:.1f} s streamed (billed as {billed:.1f} s), '
                    f'about {cost:.4f} EUR at list price')


def turns(output, label, offset_ms):
    """Identification segments -> turns in capture milliseconds; ``match``: the enrolled voiceprint."""
    result = []
    for segment in output.get('identification') or ():
        start, end = segment.get('start'), segment.get('end')
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or end <= start:
            continue
        confidence = segment.get('confidence') if isinstance(segment.get('confidence'), dict) else {}
        item = dict(speaker=str(segment.get('diarizationSpeaker') or segment.get('speaker'))[:40],
                    start_ms=int(round(offset_ms + 1000 * start)), end_ms=int(round(offset_ms + 1000 * end)),
                    match=segment.get('match') == label)
        if isinstance(confidence.get(label), (int, float)):
            item['confidence'] = round(float(confidence[label]), 3)
        result.append(item)
    return result


def overlap_ms(items):
    """Milliseconds where two or more turns are active."""
    edges = sorted([(item['start_ms'], 1) for item in items] + [(item['end_ms'], -1) for item in items])
    total, active, since = 0, 0, None
    for at, step in edges:
        if active >= 2 and since is not None:
            total += at - since
        active += step
        since = at
    return int(total)


TURNS_SPEC = CapabilitySpec(
    'audio.speaker_turns', 'audio.speaker_turns', 'audio', 'gamst.pyannoteai', '1.identify',
    (OutputField('kind', 'string'), OutputField('reference', 'string'), OutputField('start_ms', 'integer'),
     OutputField('end_ms', 'integer'), OutputField('turns', 'array'), OutputField('speakers', 'integer'),
     OutputField('overlap_ms', 'integer'), OutputField('model', 'object')),
    'window', window_ms=60000, stateful=True, reset_on_gap=True, freshness_ms=120000, interpretation='estimate',
    devices=('cpu',), license='adapter Apache-2.0; hosted service per pyannoteAI terms')


class HostedIdentify:
    """Pipeline module: room audio in blocks of ``block_ms`` -> ``audio.speaker_turns`` with the enrolled match.

    The voiceprint is made once from the voice reference's clip (``{"wav"}`` or ``{"samples"}``, the first
    30 s) unless ``voiceprint_file`` already holds one. Speaker labels are local to each block.
    """

    def __init__(self, client, *, reference='voice', block_ms=60000, voiceprint_file=None):
        if block_ms < MIN_BILLED_S * 1000:
            raise ValueError('blocks shorter than 20 s are billed as 20 s: use block_ms >= 20000')
        self.client, self.reference_name, self.voiceprint_file = client, reference, voiceprint_file
        self.blocks = Blocks(block_ms=block_ms, context_ms=0, ring_seconds=2 * block_ms / 1000)
        self.spec, self._voiceprint, self._clip = TURNS_SPEC, None, None
        self.name, self.version = 'hosted_identify', f'pyannoteai.{client.model}'

    def bind(self, pipeline):
        reference = pipeline.references.get(self.reference_name)
        if reference is None or reference['kind'] != 'voice':
            raise ValueError(f'hosted identification needs a voice reference named {self.reference_name!r}')
        if self.voiceprint_file and os.path.isfile(self.voiceprint_file):
            with open(self.voiceprint_file) as handle:
                self._voiceprint = handle.read().strip()
            return
        from .target_speech import enrollment_audio
        self._clip = enrollment_audio(reference['value'])[:int(MAX_VOICEPRINT_S * SAMPLE_RATE)]

    def reset(self):
        self.blocks.reset()

    async def _ready(self):
        if self._voiceprint is None:
            if self._clip is None:
                raise RuntimeError('no voice clip bound for the voiceprint')
            self._voiceprint, self._clip = await asyncio.to_thread(self.client.voiceprint, self._clip), None
            if self.voiceprint_file:
                descriptor = os.open(self.voiceprint_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(descriptor, 'w') as handle:
                    handle.write(self._voiceprint)
        return self._voiceprint

    async def process(self, frame):
        if frame.modality != 'audio' or frame.data is None or isinstance(frame.data, dict):
            return None
        if frame.sample_rate != SAMPLE_RATE:
            raise ValueError('hosted identification needs 16 kHz mono audio')
        return await self._identify(self.blocks.push(frame.data, int(frame.timestamp_ms)))

    async def finish(self):
        return await self._identify(self.blocks.flush())

    async def _identify(self, blocks):
        outputs = []
        for start_ms, end_ms, window, _ in blocks:
            if len(window) < SAMPLE_RATE:      # under a second: nothing worth 20 billed seconds
                continue
            voiceprint = await self._ready()
            output = await asyncio.to_thread(self.client.identify, window, {self.reference_name: voiceprint})
            begin = end_ms - 1000 * len(window) / SAMPLE_RATE
            found = turns(output, self.reference_name, begin)
            outputs.append(self.spec.validate(dict(
                kind='turns', reference=self.reference_name, start_ms=int(round(begin)), end_ms=int(end_ms),
                turns=found, speakers=len({item['speaker'] for item in found}), overlap_ms=overlap_ms(found),
                model=dict(id=f'pyannoteai:{self.client.model}', matching_threshold=self.client.matching_threshold))))
        return outputs or None

"""Local streaming recognisers with their own word times, run in a worker (``streaming.asr.words``).

Each is fed the microphone chunk by chunk and answers at once: partial text while a turn goes on, and final
utterances whose words carry capture times. The worker returns the same ``speech.transcript_words`` records as the
phrase parts; :class:`gamst.providers.model_workers.RemoteStreamingWords` adds when each final arrived.

- ``sherpa-zipformer``: sherpa-onnx ``OnlineRecognizer`` with a streaming Zipformer transducer, e.g. k2-fsa
  ``sherpa-onnx-streaming-zipformer-korean-2024-06-16`` (KsponSpeech, Apache-2.0). Token times come from the
  transducer; sherpa's endpoint rules end a turn. CPU, a few per cent of real time.
- ``voxtral-mlx``: Mistral ``Voxtral-Mini-4B-Realtime-2602`` (Apache-2.0; 13 languages incl. Korean; FLEURS Korean
  WER 15.74 % at 480 ms delay per its model card) through ``mlx-audio``'s streaming session on Apple silicon. The
  decoder emits one token per 80 ms of audio, ``transcription_delay_ms`` behind it (delayed streams), so token *i*
  of the stream belongs to the audio at ``i * 80 ms``: the token clock is the word clock. ``[STREAMING_WORD]`` starts a
  word. A turn ends where no text came for ``pause_seconds`` (the model has no turns), or at ``max_seconds``.

Times on the recogniser's own clock (seconds of audio it was fed) become capture times through
:class:`gamst.providers.streaming_asr.CaptureClock`.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from ..plugins import InputFrame, _signal
from .audio_ring import mono_float
from .streaming_asr import CaptureClock

RATE = 16000
WORD_START = '▁'


def verify(root, files):
    """``files``: {name: sha256}; every one must be there and match. Returns a short digest of the set."""
    root = Path(root)
    if not isinstance(files, dict) or not files:
        raise ValueError('name the pinned files ({name: sha256})')
    for name, expected in sorted(files.items()):
        path = root / name
        if Path(name).name != name or not path.is_file():
            raise ValueError(f'{root} has no {name}')
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1 << 22), b''):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f'{path} does not match its pinned sha256')
    return hashlib.sha256(''.join(f'{name}={files[name]};' for name in sorted(files)).encode()).hexdigest()[:16]


class _Local:
    """What both recognisers share: the capture clock, utterance ids and the records."""
    partials = False

    def _start(self, language, partials):
        self.language, self.partials = language, bool(partials)
        self.clock = CaptureClock(RATE)
        self._subject, self._utterance, self._revision, self._last = 'local-user', 0, 0, None

    def _record(self, words, status, *, text=None, reason=None):
        """``words``: [(text, start_s, end_s)] on the recogniser's clock -> one record (signal dict)."""
        at = lambda seconds: self.clock.seconds_ms(seconds)
        items = [dict(text=word, start_ms=at(start), end_ms=max(at(start) + 1, at(end))) for word, start, end in words]
        text = text if text is not None else ' '.join(word for word, _, _ in words)
        start_ms = items[0]['start_ms'] if items else at(0)
        end_ms = items[-1]['end_ms'] if items else start_ms + 1
        value = dict(utterance_id=f'{self.name}:{self._epoch}:{self._utterance}', revision=self._revision, text=text,
                     status=status, start_ms=start_ms, end_ms=end_ms, language=self.language,
                     words=items if status == 'final' else [])
        if status != 'final':
            value['heard_to_ms'] = at(self.clock.sent / RATE)
        if reason:
            value['finalization_reason'] = reason
        self._revision += 1
        if status == 'final':
            self._utterance, self._revision = self._utterance + 1, 0
        return _signal(InputFrame(None, 'audio', self._last, self._subject, RATE), 'speech.transcript_words', value,
                       self.name, self.version)

    def _push(self, frame):
        if frame.sample_rate != RATE:
            raise ValueError('streaming ASR needs 16 kHz mono audio')
        samples = mono_float(frame.data)
        self._last, self._subject = frame.timestamp_ms, frame.subject_id
        self.clock.push(len(samples), frame.timestamp_ms)
        return samples


class SherpaStreaming(_Local):
    name = 'sherpa_onnx.zipformer'

    def __init__(self, recognizer, model_id, *, language='ko', partials=False):
        self.recognizer, self.version = recognizer, model_id
        self._start(language, partials)
        self._epoch = 0
        self.stream, self._partial = recognizer.create_stream(), ''

    @classmethod
    def from_local(cls, path, files, *, encoder, decoder, joiner, tokens='tokens.txt', language='ko', partials=False,
                   num_threads=2, decoding_method='greedy_search', rule1_min_trailing_silence=2.4,
                   rule2_min_trailing_silence=1.2, rule3_min_utterance_length=20.):
        import sherpa_onnx
        digest = verify(path, files)
        root = Path(path)
        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(root / tokens), encoder=str(root / encoder), decoder=str(root / decoder),
            joiner=str(root / joiner), num_threads=int(num_threads), sample_rate=RATE, feature_dim=80,
            decoding_method=decoding_method, enable_endpoint_detection=True,
            rule1_min_trailing_silence=float(rule1_min_trailing_silence),
            rule2_min_trailing_silence=float(rule2_min_trailing_silence),
            rule3_min_utterance_length=float(rule3_min_utterance_length))
        return cls(recognizer, f'sherpa-zipformer:{digest}:{decoding_method}', language=language, partials=partials)

    @staticmethod
    def words(result, segment_end_s):
        """Tokens and their times (from the turn's start) -> [(word, start_s, end_s)] from the stream start."""
        base, words = float(getattr(result, 'start_time', 0.) or 0.), []
        tokens, times = list(result.tokens), list(result.timestamps)
        for index, (token, at) in enumerate(zip(tokens, times)):
            piece = token.replace(WORD_START, ' ')
            if not words or piece.startswith(' '):
                words.append([piece.strip(), base + at, base + at])
            else:
                words[-1][0] += piece
            following = base + times[index + 1] if index + 1 < len(times) else min(segment_end_s, base + at + .24)
            words[-1][2] = max(words[-1][1] + .04, min(following, base + at + .4))
        return [(word, start, end) for word, start, end in words if word]

    def _decode(self):
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)

    def _final(self, reason):
        result = self.recognizer.get_result_all(self.stream)
        words = self.words(result, self.clock.sent / RATE)
        self.recognizer.reset(self.stream)
        self._partial = ''
        return [self._record(words, 'final', reason=reason)] if words else []

    async def process(self, frame):
        if frame.modality != 'audio' or frame.data is None:
            return []
        samples = self._push(frame)
        self.stream.accept_waveform(RATE, samples)
        self._decode()
        if self.recognizer.is_endpoint(self.stream):
            return self._final('endpoint')
        text = self.recognizer.get_result(self.stream).strip() if self.partials else ''
        if text and text != self._partial:
            self._partial = text
            return [self._record([], 'partial', text=text)]
        return []

    async def finish(self):
        if self._last is None:
            return []
        self.stream.input_finished()
        self._decode()
        out = self._final('input_end')
        self.reset()
        return out

    def reset(self):
        self.stream, self._partial = self.recognizer.create_stream(), ''
        self.clock.reset()
        self._epoch += 1
        self._utterance = self._revision = 0


class VoxtralStreaming(_Local):
    name = 'voxtral_realtime.mlx'
    TOKEN_S = .08

    def __init__(self, model, model_id, *, language='ko', partials=False, delay_ms=480, pause_seconds=.6,
                 max_seconds=15., session_tokens=3600, max_decode_tokens=64):
        self.model, self.delay_ms, self.pause_seconds = model, int(delay_ms), float(pause_seconds)
        if self.delay_ms % 80 or not 80 <= self.delay_ms <= 2400:
            raise ValueError('delay_ms is a multiple of 80 between 80 and 2400')
        self.max_seconds, self.session_tokens, self.max_decode = float(max_seconds), int(session_tokens), int(max_decode_tokens)
        self.version = f'{model_id}:delay.{self.delay_ms}:pause.{self.pause_seconds:g}'
        self._start(language, partials)
        self._epoch = -1
        self._new_session()

    @classmethod
    def from_local(cls, path, files, **options):
        from mlx_audio.stt.utils import load_model
        digest = verify(path, files)
        return cls(load_model(str(path)), f'voxtral-mini-4b-realtime-2602:{digest}', **options)

    def _new_session(self, origin_s=0.):
        self.session = self.model.create_streaming_session(transcription_delay_ms=self.delay_ms)
        self._origin_s, self._seen = origin_s, 0          # stream seconds where this session's audio starts
        self._words, self._text_at, self._epoch = [], None, self._epoch + 1
        self._utterance = self._revision = 0
        self._bytes = None

    def _tokens(self):
        """New tokens since the last call -> words of the open turn (bytes until a word is complete)."""
        out, generated = [], self.session.generated
        tokenizer = self.model._tokenizer
        while self._seen < len(generated):
            index, token = self._seen, generated[self._seen]
            self._seen += 1
            at = self._origin_s + index * self.TOKEN_S
            if token == 33 or (token >= 1000 and self._bytes is None):     # [STREAMING_WORD] or the first text
                self._close_word()
                self._bytes, self._word = b'', [at, at + self.TOKEN_S]
            if token >= 1000:
                piece = tokenizer.token_bytes(token)
                if piece.startswith(b' ') and self._bytes:
                    self._close_word()
                    self._bytes, self._word = b'', [at, at + self.TOKEN_S]
                self._bytes += piece
                self._word[1] = at + self.TOKEN_S
                self._text_at = at
            out += self._maybe_final(at)
        return out

    def _close_word(self):
        if self._bytes:
            text = self._bytes.decode('utf-8', 'replace').strip()
            if text:
                self._words.append((text, self._word[0], self._word[1]))
        self._bytes = None

    def _maybe_final(self, now_s, *, reason=None):
        open_words = self._words or self._bytes
        if not open_words:
            return []
        quiet = self._text_at is None or now_s - self._text_at >= self.pause_seconds
        long = self._words and now_s - self._words[0][1] >= self.max_seconds
        if not (quiet or long or reason):
            return []
        self._close_word()
        if not self._words:
            return []
        words, self._words = self._words, []
        return [self._record(words, 'final', reason=reason or ('pause' if quiet else 'max_seconds'))]

    def _step(self):
        while not self.session.done:
            before = len(self.session.generated)
            self.session.step(max_decode_tokens=self.max_decode)
            if len(self.session.generated) == before:
                break

    async def process(self, frame):
        if frame.modality != 'audio' or frame.data is None:
            return []
        samples = self._push(frame)
        self.session.feed(samples)
        self._step()
        out = self._tokens()
        if self.partials and (self._words or self._bytes):
            pending = self._words + ([(self._bytes.decode('utf-8', 'ignore').strip(), *self._word)] if self._bytes else [])
            text = ' '.join(word for word, _, _ in pending if word)
            if text:
                out.append(self._record([], 'partial', text=text))
        if len(self.session.generated) >= self.session_tokens and not self._words and not self._bytes:
            self.session.close()     # a long stream: a new session at a pause (its context starts again)
            self._step()
            out += self._tokens() + self._maybe_final(self.clock.sent / RATE, reason='session_end')
            self._new_session(self.clock.sent / RATE)
        return out

    async def finish(self):
        if self._last is None:
            return []
        self.session.close()
        self._step()
        out = self._tokens() + self._maybe_final(self.clock.sent / RATE, reason='input_end')
        self.reset()
        return out

    def reset(self):
        self.clock.reset()
        self._new_session()


def load(config):
    config = dict(config)
    backend, path, files = config.pop('backend'), config.pop('path'), config.pop('files')
    config.pop('sha256', None)
    if backend == 'sherpa-zipformer':
        return SherpaStreaming.from_local(path, files, **config)
    if backend == 'voxtral-mlx':
        return VoxtralStreaming.from_local(path, files, **config)
    raise ValueError('streaming.asr.words backend is sherpa-zipformer or voxtral-mlx')

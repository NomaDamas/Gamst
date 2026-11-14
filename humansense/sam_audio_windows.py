"""Complete-window SAM-Audio contracts, not a model loader or capture clock."""
from dataclasses import dataclass
import hashlib
import json
import math

SAMPLE_RATE = 48000
MAX_PROMPT_BYTES = 4096
MAX_ANCHORS = 32
SOURCE_REVISION = 'bb4c6999d2677c7402360e426afc01ddfad6dce0'


def _hop(value):
    if type(value) is not int or not 1 <= value <= SAMPLE_RATE:
        raise ValueError('explicit effective codec hop of1..48000 samples required')
    return value


def _samples(value):
    if type(value) is not int or not SAMPLE_RATE <= value <= 4*SAMPLE_RATE:
        raise ValueError('complete window must contain48000..192000 samples')
    return value


def _hash(value):
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class PreparedWindow:
    """Detached immutable PCM; array() returns a private writable copy.

    Digests bind input values only. Capture provenance and tokenizer truncation
    must be supplied by the native/context integration, never inferred here.
    """
    pcm: bytes
    prompt: str
    anchors: tuple
    feature_spans: tuple
    hop_length: int
    sample_count: int
    padded_samples: int
    pcm_sha256: str
    prompt_sha256: str

    def array(self):
        import numpy as np
        return np.frombuffer(self.pcm, dtype='<f4').copy()


def prepare_window(pcm, *, sample_rate, prompt, hop_length, anchors=None):
    """Validate batch1 mono input; never resample, mix, pad or truncate it.

    Anchors are +/- intervals in source-window seconds. Feature indices follow
    pinned upstream ceil(seconds*rate/hop). Overlapping intervals or empty
    quantized spans are rejected instead of silently overwriting conditioning.
    """
    import numpy as np
    hop = _hop(hop_length)
    if type(sample_rate) is not int or sample_rate != SAMPLE_RATE:
        raise ValueError('explicit48000Hz input required; no implicit resampling')
    if (type(pcm) is not np.ndarray or pcm.dtype != np.dtype('float32') or
            pcm.ndim != 1):
        raise ValueError('plain mono one-dimensional float32 ndarray required')
    count = _samples(len(pcm))
    # Validation and hashing must examine the same owned snapshot, not a
    # caller buffer that can change between a finite check and serialization.
    pcm = pcm.copy(order='C')
    if pcm.ndim != 1 or len(pcm) != count or pcm.dtype != np.dtype('float32'):
        raise ValueError('input layout changed while copying')
    if not np.isfinite(pcm).all() or (np.abs(pcm) > 1).any():
        raise ValueError('input PCM must be finite and normalized within[-1,1]')
    if type(prompt) is not str or not prompt.strip() or len(prompt) > MAX_PROMPT_BYTES:
        raise ValueError('nonempty bounded prompt required')
    try:
        encoded_prompt = prompt.encode('utf-8')
    except UnicodeError as error:
        raise ValueError('prompt must be valid UTF8') from error
    if len(encoded_prompt) > MAX_PROMPT_BYTES:
        raise ValueError('prompt exceeds4096 UTF8 bytes')
    if anchors is None:
        anchors = []
    if type(anchors) not in (list, tuple) or len(anchors) > MAX_ANCHORS:
        raise ValueError('at most32 explicit manual anchors required')
    anchors = tuple(anchors)
    if len(anchors) > MAX_ANCHORS:
        raise ValueError('manual anchor list changed beyond its bound')
    spans, originals = [], []
    duration = count / SAMPLE_RATE
    for anchor in anchors:
        if type(anchor) not in (tuple, list) or len(anchor) != 3:
            raise ValueError('anchor must be(token,start_seconds,end_seconds)')
        token, start, end = anchor
        if type(token) is not str or token not in ('+', '-'):
            raise ValueError('anchor token must be+or-')
        if any(type(v) not in (int, float) or not 0 <= v <= duration for v in (start, end)):
            raise ValueError('anchor times must be finite numbers')
        if not 0 <= start < end <= duration:
            raise ValueError('anchor must lie strictly within the complete window')
        start, end = float(start), float(end)
        first, last = math.ceil(start*SAMPLE_RATE/hop), math.ceil(end*SAMPLE_RATE/hop)
        if first >= last or last > (count+hop-1)//hop:
            raise ValueError('anchor disappears or exceeds the actual codec feature window')
        for (_, old_start, old_end), (_, old_first, old_last) in zip(originals, spans):
            if max(start, old_start) < min(end, old_end) or max(first, old_first) < min(last, old_last):
                raise ValueError('overlapping manual anchors are unsupported')
        originals.append((token, start, end))
        spans.append((token, first, last))
    # Convert only byte order/contiguity, after all shape and size validation.
    payload = pcm.astype('<f4', copy=False).tobytes(order='C')
    declaration = dict(schema_version=1, prompt=prompt, anchors=originals,
                       sample_rate=SAMPLE_RATE, hop_length=hop, source_revision=SOURCE_REVISION)
    prompt_digest = _hash(json.dumps(declaration, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=True, allow_nan=False).encode())
    return PreparedWindow(payload, prompt, tuple(originals), tuple(spans), hop, count,
                          ((count+hop-1)//hop)*hop, _hash(payload), prompt_digest)


def validate_outputs(target, residual, *, sample_count, hop_length):
    """Validate NumPy-converted official one-item lists and trim codec tail.

    Both estimates must have the exact codec-aligned length. SDK output bound
    is +/-16, not an amplitude calibration; never clip or renormalize estimates.
    Returned arrays are detached; no additive reconstruction is asserted.
    """
    import numpy as np
    count, hop = _samples(sample_count), _hop(hop_length)
    padded = ((count+hop-1)//hop)*hop
    results = []
    for values in (target, residual):
        if type(values) is not list or len(values) != 1:
            raise ValueError('native output must contain exactly one waveform per list')
        waveform = values[0]
        if (type(waveform) is not np.ndarray or waveform.dtype != np.dtype('float32') or
                waveform.ndim != 1 or len(waveform) != padded):
            raise ValueError('native output must be float32 with exact codec-aligned length')
        waveform = waveform.copy(order='C')
        if waveform.ndim != 1 or len(waveform) != padded or waveform.dtype != np.dtype('float32'):
            raise ValueError('native output layout changed while copying')
        if not np.isfinite(waveform).all() or (np.abs(waveform) > 16).any():
            raise ValueError('native output exceeds finite SDK amplitude bound16')
        results.append(waveform[:count].astype('<f4', copy=True))
    return tuple(results)

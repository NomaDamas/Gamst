# Revision-aware speech context

The runtime now reduces registered ASR updates into one latest entry per utterance. A model adapter owns recognition, acoustic state and endpoint detection; this layer owns revision ordering, evidence, bounded retention and LLM context. No ASR model is loaded by this feature. It does not turn windowed inference into native streaming.

```python
from humansense import HumanSense, SensorMetadata, transcript_capability

async with HumanSense(consent=True) as human:
    spec = transcript_capability('asr.example', 'your.asr', 'checkpoint.version',
        streaming_mode='native', window_ms=320, lookahead_ms=240)
    human.register_capability(spec)
    # Capture times must be aligned Unix milliseconds, no later than now.
    await human.push_observation(spec.id, {
        'utterance_id': 'utterance1', 'revision': 0, 'text': '안녕',
        'status': 'partial', 'start_ms': capture_start, 'end_ms': capture_end,
        'language': 'ko-KR',
    }, sensor=SensorMetadata('microphone', 'mono', sequence=0,
                             stream_epoch='connection1'), timestamp_ms=capture_end)
    # revision=1 supplies the ENTIRE replacement text, for example '안녕하세요'.
    # status='final' seals the utterance; it does not assert transcription truth.
    context = await human.get_context()
    utterances = context['transcripts']['utterances']
```

The helper's window/lookahead/mode must describe the actual chosen backend. The example values are illustrative, not validated Nemotron performance. SensorStreams providers can return this same update object from `process`; their declared capability must use the helper or the equivalent complete schema and reserved output name `speech.transcript_update`. The runtime validates semantic ordering before adding the record. If storage fails, the reducer does not commit. Raw PCM never enters this context.

## Update rules

- The key is subject, source, sensor, capability, stream epoch and utterance ID. Equal local IDs from different microphones or restarted streams cannot overwrite each other.
- First revision is zero; changed updates increment by exactly one. Full replacement supports corrections, punctuation, language changes and empty-text retraction. Text is bounded to 4096 characters; no automatic concatenation or arbitrary prefix commitment occurs.
- Capture sequence strictly increases across a stream namespace, even between utterances. Capture timestamps cannot move backwards. Utterance start is fixed, end cannot move backwards, and each utterance spans at most five minutes. Rotate to a new utterance explicitly for longer speech.
- An exact payload retry with a new capture sequence advances the stream watermark but preserves the original retained evidence and update time. Same-sequence replay is rejected. A finalized utterance cannot be changed; use a new ID for a separate correction. Final may be the first update.
- A gap, provider error, stream clear or runtime pause/reset marks affected partial entries `interrupted`. This is a runtime status, never a fake final transcript. Confirmed final entries remain historical evidence. A late final cannot revive an interrupted utterance; the new stream epoch starts at revision zero.

## Context and bounds

`get_context()['transcripts']` returns chronological latest utterance entries with original capture bounds, update time, revision, status, source/epoch, models, evidence ID and freshness. The dedicated section replaces generic latest-signal presentation for the reserved update name; obsolete partial hypotheses are retained only in the bounded audit store, not concatenated into the LLM transcript. Prior finalized utterances remain available while a newer utterance is partial. Other sources, legacy string transcripts and text inputs retain their existing behavior.

Defaults: at most 256 utterances, 128 retained stream namespaces and 1 MiB encoded entry storage; retention is the lesser of runtime retention and 24 hours. Namespace watermarks survive entry eviction to reject delayed updates. A stream namespace limit rejects new namespaces until expiry or clear, rather than silently losing ordering protection. Evicted utterances cannot resume at a nonzero revision. Context returns at most 16 entries under a 32 KiB entry budget and explicitly reports `retained_count` and `truncated`; text is not silently shortened. Container metadata adds a small amount beyond the entry budget.

The reducer is in-memory and session-local even if the audit store uses a SQLite file. Restart does not reconstruct a live transcript from old partials. Closing clears transcript memory; deleting a subject clears its entries and watermarks. Reconnects require new epochs. Freshness refers to the latest accepted update, not whether the recognized words were correct or whether an LLM understood them.

## Verification and remaining model work

`python -m unittest discover -s tests -p test_transcripts.py -v` exercises replacement, Korean text, finality, retraction, retry semantics, malformed/out-of-order updates, epoch separation, gap/clear/pause interruption, byte/count/retention bounds and storage failure atomicity. These are contract and integrated-runtime tests with synthetic ASR updates, not recognition-quality evidence.

The user deferred model installation on 2026-11-14. Native Nemotron/Qwen adapters and actual Korean/English recognition, partial/final/revision timing and accuracy checks remain required under #23 and #6; weights will not be downloaded now. Existing model evidence remains identified separately. The overall release and VLM cost/quality gates are not satisfied by these tests.

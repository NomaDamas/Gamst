# Transcript-interval speaking-rate estimates

The ASR adapter owns recognition and capture boundaries; `TranscriptMemory` owns
accepted revisions and interruption. Its query projection now computes an
orthographic rate over each retained utterance's exact latest text and interval.
The result reaches Python `get_transcripts()`, `get_context()['transcripts']`, the
same MCP tools and user-cadenced live context subscriptions. It needs no new model,
VLM call, package, worker or audio capture. It is not a separate registered
acoustic provider and cannot replace acoustic-rate or recognition validation.

```python
utterances = (await human.get_transcripts())['utterances']
for utterance in utterances:
    estimate = utterance['speaking_rate']
    # Check source/evidence, utterance['fresh'], status, reason and unit first.
    # Both provisional/estimated values remain based on unverified ASR text.
```

## Counting contract

`method='asr_text_interval_v1'` records its Unicode database version. Normalize a
copy of the bounded text to NFC; the original transcript/evidence is unchanged.
Supported languages are `en`, `ko`, or one two-letter region suffix such as
`en-US`/`ko-KR`, case-insensitive. Other tags produce `unsupported_language`.

- English: count runs of Unicode Latin letters. Straight/curly-right apostrophes
  (`'`, `’`) and ASCII hyphens join runs only when surrounded by letters. Other
  punctuation/whitespace separates runs. `We're well-known—really. Café!` counts
  four orthographic words. Unit: `orthographic_words_per_minute`;
  `value = count * 60000 / duration_ms`.
- Korean: count NFC Hangul syllable blocks U+AC00..U+D7A3, ignoring punctuation and
  whitespace. `안녕하세요, 반가워요!` counts nine blocks. Unit:
  `hangul_blocks_per_second`; `value = count * 1000 / duration_ms`.
  Decomposed Jamo that NFC composes into complete blocks are supported. Remaining
  isolated Jamo, Latin code-switching and Hanja are not silently counted.

Unicode number categories produce `numeric_text`: a written number's spoken
expansion is unknown. Other scripts, symbols, emoji, unsupported combining marks
or controls produce `unsupported_script_or_symbol`. No numerical rate is emitted
for those cases; punctuation-only/empty text returns `no_lexical_text`, not a
claim of zero speech. These conservative exclusions are explicit coverage limits,
not recognition failures. No language is inferred from text.

The interval uses ASR `end_ms - start_ms`, calculated as an integer difference.
At least1000ms is required; this is a fixed engineering reporting threshold,
not an empirically validated reliability guarantee. Shorter/zero intervals give
`interval_under_1000ms`, retain an available count and return `value=null`.
The existing reducer bounds each utterance to five minutes and4096characters.
All rates include pauses and any silence inside the ASR capture interval:
`pause_time_included=True`. There is no word alignment, VAD duration subtraction
or articulation-rate measurement. Decoder chunks and endpoint decisions can
change the denominator independently of actual speech tempo.

## Revisions, provenance and streaming

The output has method/Unicode version, status/reason, exact interval/duration,
revision, count, value, unit and `acoustic_rate_verified=False`.
`status='provisional'` means a partial transcript; `estimated` means a finalized
transcript. Neither establishes recognition correctness or acoustic truth.
`unavailable` accompanies a null value and explicit reason. Rate magnitude is not
a confidence score. Do not compare English words/minute numerically with Korean
blocks/second or use either as a calibrated cross-speaker measure.

Each full replacement recomputes from scratch; revisions are never summed.
A partial changing from four words to two over the same two seconds changes
120 to60words/minute. An empty final retracts the value. Exact-payload retries
retain the original evidence and value; no new inferred count is accumulated.
Runtime interruption from a gap/error/pause/clear returns unavailable with
`interrupted_utterance`. Final history remains valid as historical text-derived
evidence even when stale; the enclosing `fresh=False` must not be ignored.

Subject, source, sensor, capability, epoch, utterance and evidence IDs come from
the enclosing retained entry. There is no aggregation across utterances, no
speaker identity inference, and no way to attribute mixed-speaker speech to one
person. Clear/delete follows transcript memory ownership and removes the same
rate projections. Context cadence stays user-controlled; querying never advances
ASR or refreshes old capture timestamps.

The projection is standard-library-only and linear in bounded text length. It
retains no extra state and is computed before the existing query byte-limit
check; returned entries, including rate metadata, stay inside the entry budget.
It adds context bytes, so a tight budget can return fewer entries with explicit
`truncated=True`. It does not shorten text to make a rate fit. Audit records and
raw `speech.transcript_update` signal notifications retain their existing schema;
use transcript/context queries or context subscriptions for this derived field.

## Validation and limitations

```sh
python3 -m unittest discover -s tests -p test_speaking_rate.py -v
```

Tests use fixed strings with known orthographic counts, Unicode normalization,
large integer capture epochs, interval thresholds, ambiguity, revisions/retry/
retraction, source isolation, interruption, clear and exact encoded query bounds.
The real SensorStreams/reducer/MCP path runs with deterministic ASR updates. When
the existing optional WebSocket runtime is available, an actual loopback context
subscription checks partial-to-final replacement at a requested50ms cadence.
Neither test is trained recognition or acoustic speech-rate evidence.

[Required child#114](https://github.com/NomaDamas/Gamst/issues/114) tracks real
Korean/English speech, manual counts/pauses, ASR timing/count errors, mixed speech,
paced memory/latency and end-to-end model execution. Parent#25 and ASR#68/#76 remain
open. No SOTA accuracy, cost benchmark or physical speaking-rate claim follows
from these arithmetic tests.

For an acoustic alternative, [de Jong and Wempe's original method](https://sites.google.com/site/speechrate)
uses intensity peaks and voicing to estimate syllable nuclei without transcripts.
That is a separate algorithm with its own script license and acoustic acceptance;
this change does not implement it or inherit its published accuracy.

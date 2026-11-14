# Original transport/native crosscheck (implementation in progress)

`benchmarks.billing_transport_crosscheck.crosscheck_transport` compares original
collector-native3 entries with the concrete Responses evidence inventory and
references a freshly recomputed original account-aggregate report. It performs no
network request and assigns no per-request price.

The native input must come from `read_native3_evidence`, using the original eleven
collector files, their expected byte lengths/SHA256 values, and selected sidecar
ordinal. Standalone `EvaluationSession.snapshot()` dictionaries are unsupported.
Seven declaration files are read by fixed names beneath the selected directory;
recorded `bootstrap.paths` are never opened. Original source traces are streamed
(up to 3,000 lines, 512 KiB each). Metadata, JSON expansion, events and reports are
bounded. A relocated copy remains valid if every original byte still matches.

The original source chain, declaration identities, sidecar chain and completion
bindings are checked before the selected native journal/report/summary accounting
is replayed through the same pure validators used by the live channel. Unknown
billing can leave a native report's instance field null: the actual entry instance
comes from its original admission event, linked through the finished attempt ID.
An admission without an actual attempt does not fabricate a callback.

Supply an independent exact `expected_binding` declaration with `owner_kind`,
`owner`, `provider`, `model_revision`, `role`, `projection_version` and
`transport_version`. Its owner/provider/model must match the original scope and
rate-card declaration. The whole selected lane is enumerated before transport
matching, so an empty transport inventory cannot conceal an existing invocation.
The declaration hash is reported; native3 does not prove that transport selection
was frozen before execution.

`local_binding_complete` means only that this declared lane has a complete
structural one-to-one mapping. It is not proof of complete instrumentation,
exclusive account use, provider acceptance, invoice authenticity or real cost.
Prepared requests, native entries, possible sends and observed responses retain
separate counts. Host UTC samples have unknown absolute accuracy, so nominal
query-time relations never verify bucket assignment. Declared account/filter
mismatches remain visible without changing original aggregate amounts.

Original wire envelopes were not retained in native3 collector snapshots. Stored
hello/EOF/channel-prefix assertions remain producer-reported; this reader never
certifies them by reconstructing guessed frames. All actual-acceptance and
monetary flags remain false/incomplete.

Current tests execute the actual collector, Text/Visual SDK, loopback Responses
HTTP, original evidence sink and readers. They use explicit synthetic origins.
Native4 reader integration, additional adversarial coverage and final independent
review/full-suite/CI remain required
before #276 can close. The aggregate importer itself was merged separately in
PR #278. Original sources and prior completed bundles are never amended.

## Separate persisted output and offline command

`persist_crosscheck` writes a new private directory containing `crosscheck.json`
and a completion file written last. Reuse recomputes the crosscheck and compares
all expected bytes. Rehashing a changed report and its completion does not make
it accepted. Neither original transport files nor prior aggregate bundles are
modified. A supplied reader-issued native index represents its original verified
snapshot; the CLI below reopens and verifies original native files on every run.

```sh
python3 -m benchmarks.billing_crosscheck_bundle \
  --manifest /absolute/input/acquisition.json \
  --assets /absolute/input \
  --transport /absolute/original-transport \
  --transport-inventory-sha256 ORIGINAL_INVENTORY_SHA256 \
  --native-root /absolute/original-collector \
  --native-files /absolute/input/native-file-identities.json \
  --selection /absolute/input/transport-selection.json \
  --sidecar-ordinal 0 \
  --output /absolute/new-crosscheck
```

`native-file-identities.json` maps each of the eleven fixed original filenames
to its exact `bytes` and `sha256`; it is not a replacement native report.
`transport-selection.json` contains the explicit declaration described above.
All original files remain necessary. The command makes no provider calls and
prints no raw responses, prompts, credentials or monetary estimate.

Integration tests include a real HTTP body received before cancellation, and a
native entry whose connection fails or reaches its own deadline before any send.
Both retain unknown per-attempt cost. A source-collector timeout is not substituted
for a completed transport failure record.

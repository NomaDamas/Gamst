# Original account aggregate evidence

The offline importer preserves original OpenAI Organization Costs and Completions
Usage JSON alongside a bounded deterministic projection. It does not fetch data,
install dependencies, authenticate an account, generate invoice line IDs, or
allocate account totals to individual SDK requests. Actual financial acceptance
remains incomplete.

Run from the repository root with explicit absolute paths:

```sh
python3 -m benchmarks.billing_bundle \
  --manifest /absolute/input/acquisition.json \
  --assets /absolute/input \
  --output /absolute/new-evidence-directory
```

The caller explicitly authorizes retention of the supplied original financial
JSON by selecting the output directory. It is independent of SDK observation
retention. No raw financial JSON is published into LLM Context.

## Acquisition declaration

The JSON object contains exactly these fields:

- `schema_version`: integer `1`.
- `kind`: `openai_aggregate_acquisition`.
- `account_scope`: bounded caller-declared account label, not verified ownership.
- `origin`: `synthetic_fixture` or `provider_export_supplied`; the latter does not
  prove authenticity.
- `acquired_at`: integer UTC epoch seconds, at or after all query end times.
- `queries`: one to sixteen query declarations.

Each query contains exactly `id`, `endpoint` (`costs` or `completions_usage`),
`start_time`, `end_time` (UTC epoch seconds), `bucket_width`, `group_by`, `filters`,
and `pages`. Costs support `1d`; usage supports `1m`, `1h`, and `1d`. Groups and
array filters must be sorted and unique. Supported field inventories are defined
in `benchmarks/billing_aggregate.py`; unsupported declarations fail validation.

Each page contains exactly `request_cursor`, `path`, `bytes`, and `sha256`. The
first request cursor is null. Later cursors must match the original previous
page's `next_page`. Paths are confined relative paths beneath the absolute asset
root. All path components reject symlinks. Files must be regular, bounded and
match their declared size and SHA256. A missing final page is an error, not a
zero-cost tail.

## Meaning of the projection

Original numeric lexemes are retained. USD values use exact integer units of
one billionth of a dollar; nonrepresentable values are rejected, never rounded.
Negative adjustments remain separate from positive amounts. Missing or unknown
currency is unresolved. Usage counters and their subsets are not added together.
Missing coverage intervals and unknown projection fields remain visible.

Missing and null selected grouping dimensions share one unknown identity for
duplicate detection. Identical query/page inventories become aliases. Changed
page inventories for the same logical query quarantine both revisions. Overlapping
queries with different grouping are independent, nonadditive views. There is no
cross-query grand total or per-attempt price.

The bundle contains `acquisition.json`, `report.json`, content-addressed original
page files, and a `completion.json` written last. Directory permissions are 0700;
files are 0600. A failed write leaves an incomplete prefix, never a success marker.
Existing output is reused only after recomputing from the supplied originals and
comparing every expected byte, file inventory, permission and link constraint.
A self-consistent forged report plus completion hash is insufficient. Different
input or parser results require a new output directory; existing evidence is
never repaired or overwritten.

Limits include 64 page references, 4 MiB per page, 64 MiB originals, 10,000
projected rows, 16 MiB encoded report and 1 MiB declaration. Parsing bounds nesting,
nodes, numeric digits and exponents before expensive arithmetic. Diagnostic counts
retain at most sixteen pointer examples per code.

## Remaining integration and acceptance

Issue #275 must provide actual transport evidence before #276 can reconcile
request inventories. SDK callback entry is not proof that a request reached the
provider. UTC bucket assignment needs measured monotonic/UTC anchors and clock
uncertainty; boundary ambiguity remains incomplete. Provider usage aggregates
are not request invoices, and matching counts alone cannot prove exclusive
account attribution. Issue #277 tracks the original financial-source acceptance
contract. This importer alone closes none of those gates.

Focused tests use synthetic original JSON and filesystem failures. They establish
parser and persistence behavior, not real cloud cost or model quality.

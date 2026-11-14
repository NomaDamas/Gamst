"""Original OpenAI account aggregate evidence; never a request invoice adapter."""
import hashlib
import json
from pathlib import PurePosixPath

from .billing_json import SourceNumber, decode_original
from .recorded_source import read_capture

GROUPS = {
    'costs': ('api_key_id', 'api_source', 'line_item', 'project_id', 'user_id'),
    'completions_usage': ('api_key_id', 'api_source', 'batch', 'model', 'project_id', 'service_tier', 'user_id'),
}
FILTERS = {
    'costs': {'api_key_ids': 'api_key_id', 'line_items': 'line_item', 'project_ids': 'project_id'},
    'completions_usage': {'api_key_ids': 'api_key_id', 'batch': 'batch', 'models': 'model',
                          'project_ids': 'project_id', 'user_ids': 'user_id'},
}
USAGE = ('input_tokens', 'output_tokens', 'num_model_requests', 'input_cached_tokens',
         'input_cache_write_tokens', 'input_audio_tokens', 'output_audio_tokens',
         'input_cache_write_12h_tokens', 'input_cache_write_30m_tokens', 'input_uncached_tokens',
         'input_text_tokens', 'output_text_tokens', 'input_cached_text_tokens',
         'input_cached_audio_tokens', 'input_cached_image_tokens', 'input_image_tokens', 'output_image_tokens')
MAX_REPORT = 16 * 1024**2


def encoded(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def exact(value, fields):
    if type(value) is not dict or set(value) != set(fields):
        raise ValueError('exact aggregate declaration keys required')


def integer(value, low=0, high=2**53-1):
    if type(value) is not int or not low <= value <= high:
        raise ValueError('bounded exact aggregate integer required')
    return value


def identifier(value):
    if type(value) is not str or not 1 <= len(value) <= 256 or len(value.encode('utf-8')) > 1024 or any(ord(c) < 32 for c in value):
        raise ValueError('bounded nonempty aggregate identifier required')
    return value


def source_integer(value):
    if type(value) is not SourceNumber:
        raise ValueError('original integer token required')
    return value.integer()


def validate_manifest(value):
    exact(value, ('schema_version', 'kind', 'account_scope', 'origin', 'acquired_at', 'queries'))
    if integer(value['schema_version'], 1, 1) != 1 or value['kind'] != 'openai_aggregate_acquisition':
        raise ValueError('aggregate acquisition schema1 required')
    identifier(value['account_scope'])
    if value['origin'] not in ('synthetic_fixture', 'provider_export_supplied'):
        raise ValueError('explicit supplied evidence origin required')
    integer(value['acquired_at'])
    if type(value['queries']) is not list or not 1 <= len(value['queries']) <= 16:
        raise ValueError('one to sixteen original queries required')
    ids = set(); files = {}; page_count = 0
    for query in value['queries']:
        exact(query, ('id', 'endpoint', 'start_time', 'end_time', 'bucket_width', 'group_by', 'filters', 'pages'))
        qid = identifier(query['id'])
        if qid in ids: raise ValueError('unique query IDs required')
        ids.add(qid)
        endpoint = query['endpoint']
        if type(endpoint) is not str or endpoint not in GROUPS:
            raise ValueError('supported original endpoint required')
        start = integer(query['start_time']); end = integer(query['end_time'], start+1)
        if end > value['acquired_at']: raise ValueError('query exceeds declared acquisition time')
        if query['bucket_width'] not in (('1d',) if endpoint == 'costs' else ('1m', '1h', '1d')):
            raise ValueError('supported original bucket width required')
        groups = query['group_by']
        if type(groups) is not list or any(type(g) is not str or g not in GROUPS[endpoint] for g in groups) or groups != sorted(set(groups)):
            raise ValueError('unique sorted supported grouping required')
        filters = query['filters']
        if type(filters) is not dict or set(filters)-set(FILTERS[endpoint]):
            raise ValueError('supported original query filters required')
        for key, vals in filters.items():
            if key == 'batch':
                if type(vals) is not bool: raise ValueError('batch filter must be boolean')
            else:
                if type(vals) is not list or not 1 <= len(vals) <= 256:
                    raise ValueError('bounded nonempty query filter required')
                for item in vals: identifier(item)
                if vals != sorted(set(vals)): raise ValueError('unique sorted filter values required')
        pages = query['pages']
        if type(pages) is not list or not pages: raise ValueError('original pages required')
        page_count += len(pages)
        if page_count > 64: raise ValueError('aggregate page reference bound exceeded')
        for index, page in enumerate(pages):
            exact(page, ('request_cursor', 'path', 'bytes', 'sha256'))
            if index == 0:
                if page['request_cursor'] is not None: raise ValueError('first request cursor must be null')
            else: identifier(page['request_cursor'])
            path = page['path']; digest = page['sha256']
            if type(path) is not str or not 1 <= len(path) <= 4096 or '\0' in path or PurePosixPath(path).is_absolute() or any(p in ('', '.', '..') for p in path.split('/')):
                raise ValueError('relative confined original path required')
            size = integer(page['bytes'], 1, 4*1024**2)
            if type(digest) is not str or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('original SHA256 required')
            if path in files and files[path] != (size, digest): raise ValueError('conflicting original file declarations')
            files[path] = (size, digest)
    if sum(v[0] for v in files.values()) > 64*1024**2: raise ValueError('joint original byte bound exceeded')
    raw = encoded(value)
    if len(raw) > 1024**2: raise ValueError('aggregate manifest exceeds bound')
    return json.loads(raw)


class Diagnostics:
    def __init__(self): self.values = {}
    def add(self, code, pointer):
        row = self.values.setdefault(code, {'count': 0, 'examples': []})
        row['count'] += 1
        if len(row['examples']) < 16: row['examples'].append(pointer)


def unknown_fields(value, supported, diagnostics, pointer):
    if type(value) is not dict: raise ValueError('original JSON object required')
    if set(value)-set(supported): diagnostics.add('unknown_projection_fields', pointer)


def grouped(row, query, diagnostics, pointer):
    values = {}; key = []
    for name in GROUPS[query['endpoint']]:
        state = 'missing' if name not in row else 'null' if row[name] is None else 'reported'
        value = row.get(name)
        if value is not None:
            if name == 'batch':
                if type(value) is not bool: raise ValueError('original batch grouping must be boolean')
            else: identifier(value)
            if name not in query['group_by']: diagnostics.add('unexpected_nonnull_grouping', pointer)
            if name == 'api_source' and value not in ('agents_api', 'unlabeled'):
                diagnostics.add('unknown_api_source', pointer)
        if name in query['group_by']:
            if value is None: diagnostics.add('unknown_selected_grouping', pointer)
            key.append(('unknown',) if value is None else (type(value).__name__, value))
        values[name] = dict(state=state, value=value)
    for name, expected in query['filters'].items():
        actual = row.get(FILTERS[query['endpoint']][name])
        if actual is None: diagnostics.add('unverified_filter_scope', pointer)
        elif (actual is not expected if name == 'batch' else actual not in expected):
            raise ValueError('original result contradicts query filter')
    return values, tuple(key)


def numeric_field(row, name, diagnostics, pointer, *, counter=False):
    if name not in row or row[name] is None:
        diagnostics.add('unknown_numeric_field', pointer)
        return dict(state='missing' if name not in row else 'null', lexeme=None, value=None)
    number = row[name]
    if type(number) is not SourceNumber: raise ValueError('original numeric token required')
    value = number.integer(signed=True) if counter else dict(numerator=number.value.numerator, denominator=number.value.denominator)
    if number.value < 0: diagnostics.add('adjustment_activity', pointer)
    return dict(state='reported', lexeme=number.lexeme, value=value)


def row_projection(row, query, diagnostics, pointer):
    endpoint = query['endpoint']
    if type(row) is not dict or row.get('object') != ('organization.costs.result' if endpoint == 'costs' else 'organization.usage.completions.result'):
        raise ValueError('unsupported original aggregate result type')
    fields = GROUPS[endpoint] + ('object',) + (('amount', 'quantity', 'quantity_unit') if endpoint == 'costs' else USAGE)
    unknown_fields(row, fields, diagnostics, pointer)
    groups, group_key = grouped(row, query, diagnostics, pointer)
    result = dict(source_pointer=pointer, groups=groups, per_attempt_cost=None)
    if endpoint == 'costs':
        amount = row.get('amount'); atoms = None
        if amount is None:
            diagnostics.add('unknown_amount', pointer)
            reported = dict(state='missing' if 'amount' not in row else 'null', currency=None, lexeme=None)
        else:
            unknown_fields(amount, ('currency', 'value'), diagnostics, pointer)
            currency = amount.get('currency')
            if currency is not None: identifier(currency)
            value = amount.get('value')
            if value is not None and type(value) is not SourceNumber: raise ValueError('original amount token required')
            reported = dict(state='reported', currency=currency, lexeme=None if value is None else value.lexeme)
            if currency == 'usd' and value is not None:
                atoms = value.usd_atoms()
                if atoms < 0: diagnostics.add('negative_cost_adjustment', pointer)
            else: diagnostics.add('unknown_or_unsupported_currency_amount', pointer)
        result.update(amount=reported, usd_atoms=atoms, quantity=numeric_field(row, 'quantity', diagnostics, pointer))
        unit = row.get('quantity_unit')
        if unit is not None: identifier(unit)
        else: diagnostics.add('unknown_quantity_unit', pointer)
        result['quantity_unit'] = unit
    else:
        result['usage'] = {name:numeric_field(row, name, diagnostics, pointer, counter=True) for name in USAGE}
        # Subset values are retained, never blindly added to input/output totals.
        total, cached = row.get('input_tokens'), row.get('input_cached_tokens')
        if isinstance(total, SourceNumber) and isinstance(cached, SourceNumber) and total.value >= 0 and cached.value > total.value:
            diagnostics.add('inconsistent_usage_subset', pointer)
    return result, group_key


def _parse_query(query, originals, budget):
    diagnostics = Diagnostics(); rows = []; intervals = set(); keys = set(); hashes = set(); cursors = set()
    prior = None; expected_cursor = None
    for page_index, reference in enumerate(query['pages']):
        cursor = reference['request_cursor']
        if cursor != expected_cursor or cursor in cursors:
            raise ValueError('missing, repeated or cyclic original page cursor')
        cursors.add(cursor)
        page_hash = reference['sha256']
        if page_hash in hashes: raise ValueError('duplicate page in original pagination chain')
        hashes.add(page_hash)
        page = decode_original(originals[page_hash])
        pointer = page_hash + '#'
        unknown_fields(page, ('object', 'data', 'has_more', 'next_page'), diagnostics, pointer)
        if page.get('object') != 'page' or type(page.get('data')) is not list or type(page.get('has_more')) is not bool or 'next_page' not in page:
            raise ValueError('original page structural fields required')
        expected_cursor = page['next_page']
        if page['has_more']:
            identifier(expected_cursor)
            if page_index == len(query['pages'])-1: raise ValueError('incomplete original pagination')
        elif expected_cursor is not None or page_index != len(query['pages'])-1:
            raise ValueError('terminal page and declared inventory disagree')
        for bi, bucket in enumerate(page['data']):
            bp = pointer + '/data/' + str(bi)
            unknown_fields(bucket, ('object', 'start_time', 'end_time', 'results'), diagnostics, bp)
            if bucket.get('object') != 'bucket' or type(bucket.get('results')) is not list:
                raise ValueError('original bucket structural fields required')
            start, end = source_integer(bucket.get('start_time')), source_integer(bucket.get('end_time'))
            if not query['start_time'] <= start < end <= query['end_time']:
                raise ValueError('original bucket lies outside query interval')
            interval = (start, end)
            if prior is not None and interval != prior and start < prior[1]:
                raise ValueError('unordered or partially overlapping original buckets')
            prior = interval; intervals.add(interval)
            if len(intervals) > 10000: raise ValueError('original bucket bound exceeded')
            for ri, row in enumerate(bucket['results']):
                budget['rows'] += 1
                if budget['rows'] > 10000: raise ValueError('joint original row bound exceeded')
                rp = bp + '/results/' + str(ri)
                projected, grouping = row_projection(row, query, diagnostics, rp)
                key = (start, end, grouping)
                if key in keys: raise ValueError('duplicate logical aggregate group, including unknown grouping')
                keys.add(key)
                projected.update(start_time=start, end_time=end, original_page_sha256=page_hash)
                budget['bytes'] += len(encoded(projected)) + 1
                if budget['bytes'] > MAX_REPORT: raise ValueError('expanded aggregate report exceeds bound')
                rows.append(projected)
    gaps = []; position = query['start_time']
    for start, end in sorted(intervals):
        if start > position: gaps.append(dict(start_time=position, end_time=start))
        position = end
    if position < query['end_time']: gaps.append(dict(start_time=position, end_time=query['end_time']))
    for gap in gaps: diagnostics.add('unknown_coverage_interval', str(gap['start_time'])+':'+str(gap['end_time']))
    positive = negative = 0; unknown_amounts = 0
    if query['endpoint'] == 'costs':
        for row in rows:
            atoms = row['usd_atoms']
            if atoms is None: unknown_amounts += 1
            elif atoms < 0: negative += atoms
            else: positive += atoms
            if max(positive.bit_length(), abs(negative).bit_length()) > 256:
                raise ValueError('aggregate sum exceeds exact rational bound')
    return dict(query_id=query['id'], rows=rows, diagnostics=diagnostics.values,
        coverage_complete=not gaps, missing_intervals=gaps,
        projection_complete='unknown_projection_fields' not in diagnostics.values,
        known_positive_usd_atoms=positive if query['endpoint']=='costs' else None,
        negative_adjustment_usd_atoms=negative if query['endpoint']=='costs' else None,
        unknown_amount_rows=unknown_amounts, state='unresolved_aggregate',
        per_attempt_cost=None, eligible_for_existing_native3_money_join=False)


def build_report(manifest, asset_directory):
    """Read/hash actual original bytes once; return report and immutable originals.

    This function neither calls a provider nor assigns aggregate money to calls.
    Different query groupings/revisions are independent evidence views, not bills.
    """
    from pathlib import Path
    manifest = validate_manifest(manifest)
    originals = {}; file_inventory = []; visited = set()
    for query in manifest['queries']:
        for page in query['pages']:
            if page['path'] in visited: continue
            visited.add(page['path'])
            payload = read_capture(dict(payload_path=page['path'], payload_bytes=page['bytes'],
                                        payload_sha256=page['sha256']), asset_directory)
            originals.setdefault(page['sha256'], payload)
            file_inventory.append(dict(path=page['path'], bytes=len(payload), sha256=sha(payload)))
    budget = dict(rows=0, bytes=0); views = []; aliases = []; logical = {}; exact_views = {}
    for query in manifest['queries']:
        identity = {k:v for k,v in query.items() if k not in ('id', 'pages')}
        identity['account_scope'] = manifest['account_scope']
        logical_digest = sha(encoded(identity))
        signature = sha(encoded([dict(sha256=p['sha256'], request_cursor=p['request_cursor']) for p in query['pages']]))
        exact_key = (logical_digest, signature)
        if exact_key in exact_views:
            aliases.append(dict(query_id=query['id'], canonical_query_id=exact_views[exact_key]))
            continue
        view = _parse_query(query, originals, budget)
        view.update(query_sha256=logical_digest, original_page_inventory_sha256=signature,
                    nonadditive_alternative_query_ids=[])
        exact_views[exact_key] = query['id']
        same = logical.setdefault(logical_digest, [])
        if same:
            for other in same + [view]:
                other['state'] = 'revision_quarantined'
                other['known_positive_usd_atoms'] = None
                other['negative_adjustment_usd_atoms'] = None
        same.append(view); views.append(view)
    declarations = {q['id']:q for q in manifest['queries']}
    for index, view in enumerate(views):
        left = declarations[view['query_id']]
        for other in views[index+1:]:
            right = declarations[other['query_id']]
            if left['endpoint']==right['endpoint'] and max(left['start_time'], right['start_time']) < min(left['end_time'], right['end_time']):
                view['nonadditive_alternative_query_ids'].append(other['query_id'])
                other['nonadditive_alternative_query_ids'].append(view['query_id'])
    parser = sha(Path(__file__).read_bytes() + Path(__file__).with_name('billing_json.py').read_bytes())
    report = dict(schema_version=1, kind='openai_aggregate_evidence', parser_sha256=parser,
        acquisition_manifest_sha256=sha(encoded(manifest)), account_scope=manifest['account_scope'],
        origin=manifest['origin'], acquired_at=manifest['acquired_at'], files=file_inventory,
        views=views, query_aliases=aliases, usd_atom_scale='0.000000001',
        cross_query_total=None, per_attempt_cost=None, eligible_for_existing_native3_money_join=False,
        actual_acceptance='incomplete', receipt_authenticity_verified=False,
        account_isolation_verified=False, transport_inventory_verified=False,
        raw_financial_retention_separate_from_sdk=True)
    if len(encoded(report)) > MAX_REPORT: raise ValueError('expanded aggregate report exceeds bound')
    return report, originals

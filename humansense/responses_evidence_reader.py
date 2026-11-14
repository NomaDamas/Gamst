"""Bounded original transport-artifact reader, never a native or invoice verifier."""
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from dataclasses import dataclass
from .native_entry import encoded,_hash,_id,_issue_entry_binding
from .openai_responses_codec import strict_json,response_projection,normalize,PROJECTION


def _keys(value,keys):
    if type(value) is not dict or set(value)!=set(keys.split()):raise ValueError('exact sealed evidence fields required')


def _open_directory(directory):
    path=Path(os.path.abspath(os.fspath(directory)));fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            following=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
            os.close(fd);fd=following
        result=fd;fd=None;return result
    finally:
        if fd is not None:os.close(fd)


def _read(fd,name,limit):
    handle=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=fd)
    try:
        before=os.fstat(handle)
        if not stat.S_ISREG(before.st_mode) or before.st_size>limit:raise ValueError('bounded regular evidence artifact required')
        parts=[];size=0
        while True:
            part=os.read(handle,min(65536,limit-size+1))
            if not part:break
            size+=len(part)
            if size>limit:raise ValueError('evidence grew beyond bound')
            parts.append(part)
        after=os.fstat(handle);named=os.stat(name,dir_fd=fd,follow_symlinks=False)
        attrs=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        if attrs(before)!=attrs(after) or attrs(after)!=attrs(named) or size!=before.st_size:raise ValueError('evidence changed during read')
        return b''.join(parts)
    finally:os.close(handle)


_READER=object()
@dataclass(frozen=True,init=False)
class ValidatedTransportEvidence:
    """Immutable inspected bytes; no native ownership or receipt authenticity claim."""
    inventory_sha256: str
    _rows: tuple
    def __init__(self,inventory_sha256,rows,*,_reader=None):
        if _reader is not _READER:raise TypeError('use the bounded original evidence reader')
        object.__setattr__(self,'inventory_sha256',inventory_sha256);object.__setattr__(self,'_rows',rows)
    def entries(self):
        for manifest,raw in self._rows:
            yield dict(manifest=json.loads(manifest),raw_response=raw,
                original_bytes_verified=True,native_binding_verified=False,receipt_authenticity_verified=False,
                count_bucket_verified=False,actual_acceptance='incomplete')


def read_responses_evidence(directory,*,expected_inventory_sha256):
    _hash(expected_inventory_sha256);fd=_open_directory(directory)
    try:
        inventory_raw=_read(fd,'inventory.json',1048576)
        if hashlib.sha256(inventory_raw).hexdigest()!=expected_inventory_sha256:raise ValueError('sealed inventory digest differs')
        inventory=strict_json(inventory_raw)
        _keys(inventory,'schema_version kind owner_kind owner_sha256 entries incomplete actual_acceptance receipt_authenticity_verified whole_pipeline_cost_verified')
        if (type(inventory['schema_version']) is not int or inventory['schema_version']!=1 or inventory['kind']!='openai_responses_inventory' or inventory['incomplete'] is not False or
            inventory['actual_acceptance']!='incomplete' or inventory['receipt_authenticity_verified'] is not False or inventory['whole_pipeline_cost_verified'] is not False):raise ValueError('complete structural inventory with unverified truth flags required')
        _hash(inventory['owner_sha256']);refs=inventory['entries']
        if type(refs) is not list or len(refs)>1024:raise ValueError('bounded transport inventory required')
        rows=[];total=len(inventory_raw);tokens=set();request_ids=set();provider_ids=set()
        for ordinal,ref in enumerate(refs):
            _keys(ref,'ordinal financial_token manifest_path manifest_sha256 manifest_bytes tombstone_path tombstone_sha256 tombstone_bytes raw_path raw_sha256 raw_bytes state')
            if type(ref['ordinal']) is not int or ref['ordinal']!=ordinal or ref['state']!='sealed' or any(ref[k] is not None for k in ('tombstone_path','tombstone_sha256')) or ref['tombstone_bytes']!=0:raise ValueError('ordered complete evidence entries required')
            if type(ref['financial_token']) is not str or re.fullmatch('[0-9a-f]{32}',ref['financial_token']) is None or ref['financial_token'] in tokens:raise ValueError('unique opaque financial token required')
            tokens.add(ref['financial_token'])
            if ref['manifest_path']!=f'call-{ordinal:04d}.json' or ref['raw_path']!=f'raw-{ordinal:04d}.bin':raise ValueError('fixed owned artifact names required')
            manifest_raw=_read(fd,ref['manifest_path'],65536);raw=_read(fd,ref['raw_path'],1048576);total+=len(manifest_raw)+len(raw)
            if total>64*1024*1024:raise ValueError('evidence session exceeds64MiB')
            for data,size,sha in ((manifest_raw,ref['manifest_bytes'],ref['manifest_sha256']),(raw,ref['raw_bytes'],ref['raw_sha256'])):
                if type(size) is not int or len(data)!=size or hashlib.sha256(data).hexdigest()!=sha:raise ValueError('artifact inventory identity differs')
            m=strict_json(manifest_raw)
            _keys(m,'schema_version kind origin projection_version transport_version opaque_financial_token entry_binding declared_account_scope declared_project provider requested_model returned_model sdk_request_sha256 wire_sha256 wire_bytes client_request_id server_request_id response_id response_status http_status selected_headers raw_response usage_raw_locator normalized_usage milestones terminal failure_category remote_acceptance consumption_basis money_basis actual_cost receipt_authenticity_verified whole_pipeline_cost_verified created_at clock_error_bound_ns billing_time_assignment count_bucket_verified clock_jump_detected')
            if m['schema_version']!=1 or m['kind']!='openai_responses_transport' or m['opaque_financial_token']!=ref['financial_token'] or m['origin'] not in ('synthetic_fixture','direct_transport_observed'):raise ValueError('transport manifest identity differs')
            _validate_manifest(m,raw)
            if m['actual_cost'] is not None or m['money_basis']!='unknown' or m['receipt_authenticity_verified'] is not False or m['whole_pipeline_cost_verified'] is not False or m['count_bucket_verified'] is not False or m['clock_error_bound_ns'] is not None:raise ValueError('unverified monetary/time facts required')
            binding=m['entry_binding']
            if binding is not None:
                _issue_entry_binding(binding) # Shape validation only; do not export an entry capability.
                if (binding['owner_kind']!=inventory['owner_kind'] or hashlib.sha256(encoded(binding['owner'])).hexdigest()!=inventory['owner_sha256'] or
                    (binding['provider'],binding['model_revision'],binding['prompt_sha256'])!=(m['provider'],m['requested_model'],m['sdk_request_sha256'])):raise ValueError('entry/transport owner differs')
            elif m['terminal']!='not_entered':raise ValueError('entered transport is missing native binding')
            descriptor=m['raw_response']
            _keys(descriptor,'path sha256 bytes complete truncated')
            if (descriptor['path'],descriptor['sha256'],descriptor['bytes'])!=(ref['raw_path'],ref['raw_sha256'],len(raw)):raise ValueError('raw original binding differs')
            if m['client_request_id'] in request_ids:raise ValueError('physical request identity reused')
            request_ids.add(m['client_request_id'])
            for key in ('server_request_id','response_id'):
                if m[key] is not None:
                    identity=(m['provider'],m['declared_account_scope'],key,m[key])
                    if identity in provider_ids:raise ValueError('provider request identity reused')
                    provider_ids.add(identity)
            rows.append((manifest_raw,raw))
        return ValidatedTransportEvidence(expected_inventory_sha256,tuple(rows),_reader=_READER)
    finally:os.close(fd)


def _validate_manifest(m,raw):
    from .openai_responses import TRANSPORT,_SELECTED
    if type(m['schema_version']) is not int or m['projection_version']!=PROJECTION or m['transport_version']!=TRANSPORT:raise ValueError('known projection and transport required')
    for key in ('declared_account_scope','declared_project','provider','requested_model','client_request_id'):_id(m[key])
    if m['provider']!='openai':raise ValueError('direct OpenAI provider required')
    for key in ('returned_model','server_request_id','response_id','response_status'):
        if m[key] is not None:_id(m[key])
    for key in ('sdk_request_sha256','wire_sha256'):_hash(m[key])
    if type(m['wire_bytes']) is not int or not 1<=m['wire_bytes']<=16*1024*1024:raise ValueError('wire byte bound required')
    if m['http_status'] is not None and (type(m['http_status']) is not int or not 100<=m['http_status']<=599):raise ValueError('bounded HTTP status required')
    headers=m['selected_headers']
    if type(headers) is not dict or set(headers)-_SELECTED or len(encoded(headers))>8192:raise ValueError('selected headers exceed projection')
    for value in headers.values():
        if type(value) is not str or len(value)>1024 or any(ord(c)<32 or ord(c)>126 for c in value):raise ValueError('bounded printable selected header required')
    if headers.get('x-request-id')!=m['server_request_id']:raise ValueError('server identity differs from original selected header')
    if m['terminal'] not in ('not_entered','completed','cancelled','timeout','error') or m['failure_category'] not in (None,'cancelled','timeout','error'):raise ValueError('known transport outcome required')
    if m['remote_acceptance'] not in ('unknown','response_observed') or m['consumption_basis'] not in ('unknown','response_reported'):raise ValueError('known evidence strength required')
    if m['billing_time_assignment']!='ambiguous_unverified_host_clock' or type(m['clock_jump_detected']) is not bool:raise ValueError('host time uncertainty required')
    marks=m['milestones']
    if type(marks) is not list or not 2<=len(marks)<=8:raise ValueError('bounded ordered milestones required')
    order=('prepared','entered','possible_send','headers','body_complete','terminal');previous=-1;last=-1;events=[]
    for mark in marks:
        _keys(mark,'event clock')
        if mark['event'] not in order or order.index(mark['event'])<=previous:raise ValueError('milestone order invalid')
        previous=order.index(mark['event']);events.append(mark['event']);clock=mark['clock'];_keys(clock,'monotonic_before_ns utc_ns monotonic_after_ns')
        if any(type(v) is not int or not 0<=v<=2**63-1 for v in clock.values()) or not last<=clock['monotonic_before_ns']<=clock['monotonic_after_ns']:raise ValueError('bounded monotonic bracket required')
        last=clock['monotonic_after_ns']
    if events[0]!='prepared' or events[-1]!='terminal':raise ValueError('prepared and terminal evidence required')
    if (m['entry_binding'] is not None)!=('entered' in events) or ('headers' in events and 'possible_send' not in events):raise ValueError('entry/send/header milestones disagree')
    descriptor=m['raw_response']
    if type(descriptor.get('complete')) is not bool or type(descriptor.get('truncated')) is not bool or descriptor['complete'] and descriptor['truncated']:raise ValueError('raw response completeness required')
    if descriptor['complete']!=('body_complete' in events):raise ValueError('body-complete milestone differs')
    if m['usage_raw_locator'] not in (None,'/usage'):raise ValueError('original usage pointer required')
    if descriptor['complete']:
        try:value,metadata=response_projection(raw,m['requested_model'])
        except ValueError:
            if m['terminal']=='completed':raise
        else:
            for key in ('response_id','returned_model','response_status','created_at','usage_raw_locator'):
                if m[key]!=metadata[key]:raise ValueError('projection differs from original response bytes')
            if m['terminal']=='completed':
                expected=normalize(value,m['requested_model'])
                if m['normalized_usage']!=expected['usage']:raise ValueError('normalized usage differs from original bytes')
    if m['terminal']=='completed' and (m['http_status']!=200 or not descriptor['complete'] or 'entered' not in events):raise ValueError('completed transport lacks original response')
    if m['origin']=='synthetic_fixture' and not (m['declared_account_scope'].startswith('fixture') and m['declared_project'].startswith('fixture')):raise ValueError('synthetic account identity required')

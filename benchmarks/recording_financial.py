"""Strict synthetic original-line evidence for recording ledger v2.

This module recognizes a deliberately synthetic authority only. Hash integrity,
an HTTP response, a token count, and a declared settlement are not real receipts.
No legacy ledger/native version is promoted by these validators.
"""
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

from gamst.capabilities import identifier
from gamst.recording_settlement import (AUTHORITY,ADAPTER,INVOICE_FIELDS,LINE_FIELDS,
    FINALITY_FIELDS,EVIDENCE_FIELDS,INPUT_GRAMMAR,OUTPUT_GRAMMAR,validate_synthetic_invoice)

N63 = 2**63 - 1
N256 = 2**256 - 1
PAGE_BYTES = 65536
SOURCE_BYTES = 1024**2
MAX_ROWS = 131072
MAX_PAGES = 1024
MAX_METADATA = 64*1024**2
PAGE_KINDS = {
    'source': 'recording_financial_source_page',
    'evidence': 'recording_financial_evidence_page',
    'operations': 'recording_financial_operation_page',
    'lanes': 'recording_financial_lane_page',
    'finality': 'recording_financial_finality_page',
    'line-locators': 'recording_cost_line_locator_page',
}
_RUN_ISSUER = object()
RUN_VIOLATIONS = ('source_incomplete','clock_discontinuity','operational_incomplete',
    'external_work_unknown','reservation_overrun','missing_native_prefix','unsupported_entry_coverage')


@dataclass(frozen=True,slots=True,init=False)
class FinancialRunView:
    """Issued by the runtime's complete source/native/receipt structural join.

    This is a correlation boundary, not an authenticity or isolation claim.
    Every missing actual lifecycle timestamp remains None.
    """
    execution_sha256: str
    run_inventory_sha256: str
    execution_inventory_sha256: str
    parent_receipt_inventory_sha256: str
    invocation_ordinal: int
    stream_id: str
    recording_id: str
    arm: str
    source_manifest_sha256: str
    source_coverage_sha256: str
    source_comparable: bool
    task_row_inventory_sha256: str
    clock_domain: str
    clock_consistent: bool
    activation_started_ns: int
    child_spawn_started_ns: object
    playback_anchor_ns: object
    source_finished_ns: object
    child_cleanup_finished_ns: object
    process_reaped_ns: object
    parent_resources_closed_ns: object
    operational_closed: bool
    external_work_closed: bool
    worker_inventory_sha256: str
    native_terminal_sha256: object
    controller_settlement_inventory_sha256: str
    semantic_unknown_instances: tuple
    violations: tuple

    def __init__(self,*,_issuer=None,**values):
        if _issuer is not _RUN_ISSUER:
            raise TypeError('FinancialRunView must be issued by the complete recording join')
        keys(values,self.__dataclass_fields__,'financial run view')
        for name in ('execution_sha256','run_inventory_sha256','source_manifest_sha256',
                     'source_coverage_sha256','task_row_inventory_sha256','worker_inventory_sha256',
                     'execution_inventory_sha256','parent_receipt_inventory_sha256',
                     'controller_settlement_inventory_sha256'):
            sha(values[name])
        nullable(values['native_terminal_sha256'],sha)
        for name in ('stream_id','recording_id','clock_domain'): identifier(values[name])
        enum(values['arm'],'ABC'); integer(values['invocation_ordinal'],maximum=95)
        for name in ('source_comparable','clock_consistent','operational_closed','external_work_closed'):
            if type(values[name]) is not bool: raise ValueError('explicit observed completeness flag required')
        integer(values['activation_started_ns'])
        for name in ('child_spawn_started_ns','playback_anchor_ns','source_finished_ns',
                     'child_cleanup_finished_ns','process_reaped_ns','parent_resources_closed_ns'):
            nullable(values[name],integer)
        unknown = tuple(values['semantic_unknown_instances'])
        if len(unknown) > 16384 or len(set(unknown)) != len(unknown): raise ValueError('bounded unique unknown native instances required')
        for item in unknown: sha(item)
        violations = tuple(values['violations'])
        if len(violations) > len(RUN_VIOLATIONS) or len(set(violations)) != len(violations): raise ValueError('finite unique run violations required')
        for item in violations: enum(item,RUN_VIOLATIONS)
        if values['native_terminal_sha256'] is None and (
                'missing_native_prefix' not in violations or any(values[name] for name in
                ('source_comparable','operational_closed','external_work_closed'))):
            raise ValueError('missing actual terminal requires an explicitly incomplete failed arm')
        values['semantic_unknown_instances'] = unknown; values['violations'] = violations
        for name,value in values.items(): object.__setattr__(self,name,value)


def _issue_financial_run_view(**values):
    """Internal trusted runtime-join seam, never a JSON deserializer."""
    return FinancialRunView(_issuer=_RUN_ISSUER,**values)


def keys(value, fields, label='object'):
    if type(value) is not dict or set(value) != set(fields):
        raise ValueError('exact '+label+' fields required')


def integer(value, minimum=0, maximum=N63):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError('bounded exact integer required')
    return value


def sha(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('lowercase SHA256 required')
    return value


def enum(value, options):
    if type(value) is not str or value not in options:
        raise ValueError('unsupported enum')
    return value


def array(value, maximum, minimum=0):
    if type(value) is not list or not minimum <= len(value) <= maximum:
        raise ValueError('bounded list required')
    return value


def nullable(value, validator):
    return None if value is None else validator(value)


def _shape(value, *, limit, depth=0, nodes=None):
    if nodes is None:
        nodes = [max(8192, limit)]
    nodes[0] -= 1
    if nodes[0] < 0 or depth > 64:
        raise ValueError('financial JSON tree exceeds bound')
    kind = type(value)
    if value is None or kind is bool:
        return
    if kind is int:
        if len(str(value)) > 78 or not -N256 <= value <= N256:
            raise ValueError('financial integer exceeds 256-bit bound')
    elif kind is str:
        if len(value.encode('utf-8')) > limit:
            raise ValueError('financial string exceeds document bound')
    elif kind is list:
        if len(value) > MAX_ROWS:
            raise ValueError('financial array exceeds bound')
        for item in value:
            _shape(item, limit=limit, depth=depth+1, nodes=nodes)
    elif kind is dict:
        if len(value) > MAX_ROWS:
            raise ValueError('financial object exceeds bound')
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError('string JSON keys required')
            _shape(key, limit=limit, depth=depth+1, nodes=nodes)
            _shape(item, limit=limit, depth=depth+1, nodes=nodes)
    else:
        raise ValueError('exact finite financial JSON required')


def encode(value, limit=PAGE_BYTES, *, newline=False):
    _shape(value, limit=limit)
    data = bytearray()
    for part in json.JSONEncoder(sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                                 allow_nan=False).iterencode(value):
        part = part.encode('ascii')
        if len(data)+len(part)+int(newline) > limit:
            raise ValueError('financial document exceeds byte bound')
        data.extend(part)
    if newline:
        data.extend(b'\n')
    return bytes(data)


def digest(value):
    return hashlib.sha256(encode(value, 128*1024**2)).hexdigest()


def snapshot(value, limit=PAGE_BYTES):
    return decode(encode(value, limit), limit)


def decode(data, limit=PAGE_BYTES):
    if type(data) is not bytes or len(data) > limit:
        raise ValueError('bounded original JSON bytes required')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def number(text):
        if len(text.lstrip('-')) > 78:
            raise ValueError('financial integer lexeme exceeds bound')
        result = int(text)
        if not -N256 <= result <= N256:
            raise ValueError('financial integer exceeds 256-bit bound')
        return result
    def forbidden(text):
        raise ValueError('floating/nonfinite financial JSON number forbidden')
    try:
        value = json.loads(data.decode('utf-8'), object_pairs_hook=pairs, parse_int=number,
                           parse_float=forbidden, parse_constant=forbidden)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError('strict bounded UTF-8 JSON required') from error
    _shape(value, limit=limit)
    return value


def fraction(value):
    value = Fraction(value)
    if not 0 <= value.numerator <= N256 or not 1 <= value.denominator <= N256:
        raise ValueError('nonnegative reduced 256-bit rational required')
    return value


def rational(value):
    keys(value, ('numerator','denominator'), 'rational')
    numerator = integer(value['numerator'], maximum=N256)
    denominator = integer(value['denominator'], 1, N256)
    result = fraction(Fraction(numerator, denominator))
    if (result.numerator, result.denominator) != (numerator, denominator):
        raise ValueError('reduced exact rational required')
    return result


def ratio(value):
    value = fraction(value)
    return dict(numerator=value.numerator, denominator=value.denominator)


def claim_sha256(provider, account_scope, invoice_id, line_id):
    for item in (provider, account_scope, invoice_id, line_id):
        identifier(item)
    return digest(['gamst.recording.claim.v1',provider,account_scope,invoice_id,line_id])


def reference(value):
    keys(value, ('path','file_sha256','bytes'), 'artifact reference')
    path = value['path']
    if (type(path) is not str or len(path) > 256 or path.startswith('/') or
            re.fullmatch(r'[a-zA-Z0-9_.-]+(?:/[a-zA-Z0-9_.-]+)*', path) is None or
            any(part in ('.','..') for part in path.split('/'))):
        raise ValueError('fixed relative artifact path required')
    sha(value['file_sha256']); integer(value['bytes'], 1, 128*1024**2)
    return value


def service_window(value):
    start, end, event = (value[k] for k in ('service_start_ns','service_end_ns','service_event_ns'))
    if event is None:
        integer(start); integer(end, 1)
        if start >= end:
            raise ValueError('positive service interval required')
    elif start is not None or end is not None:
        raise ValueError('service event cannot also have an interval')
    else:
        integer(event)


def operation_key(value):
    keys(value, ('invocation_ordinal','binding_index','call_ordinal'), 'fixture operation key')
    integer(value['invocation_ordinal'], maximum=95)
    integer(value['binding_index'], maximum=63)
    integer(value['call_ordinal'], maximum=16383)
    return tuple(value[k] for k in ('invocation_ordinal','binding_index','call_ordinal'))


def validate_invoice(value):
    value = snapshot(value)
    value = validate_synthetic_invoice(value)
    keys(value, INVOICE_FIELDS, 'synthetic invoice')
    if (type(value['schema_version']) is not int or value['schema_version'] != 1 or
            value['kind'] != 'synthetic_recording_invoice' or
            value['evidence_class'] != 'fixture' or value['authority'] != AUTHORITY):
        raise ValueError('recognized synthetic original invoice required')
    for key in ('provider','revision','account_scope','invoice_id','currency','currency_unit'):
        identifier(value[key])
    seen = set()
    for line in array(value['lines'],64):
        keys(line,LINE_FIELDS,'original synthetic line')
        for key in ('line_id','stream_id','component_id','resource_id','lane_id'):
            identifier(line[key])
        if line['line_id'] in seen:
            raise ValueError('unique original invoice line required')
        seen.add(line['line_id'])
        nullable(line['operation_key'],operation_key)
        service_window(line); rational(line['source_quantity'])
        integer(line['quantity']); integer(line['billed_atomic'])
    return value


def validate_finality(value):
    value = snapshot(value)
    keys(value,FINALITY_FIELDS,'synthetic finality')
    if (type(value['schema_version']) is not int or value['schema_version'] != 1 or
            value['kind'] != 'synthetic_recording_finality' or value['evidence_class'] != 'fixture' or
            value['authority'] != AUTHORITY or value['state'] != 'final_no_further_charges'):
        raise ValueError('recognized synthetic finality required')
    for key in ('execution_sha256','accounting_policy_sha256','operation_inventory_sha256',
                'lane_closure_inventory_sha256'):
        sha(value[key])
    identifier(value['stream_id'])
    hashes = array(value['source_document_sha256'],131072)
    if hashes != sorted(set(map(sha, hashes))):
        raise ValueError('unique sorted original document hashes required')
    if integer(value['issued_ns']) < integer(value['service_end_ns']):
        raise ValueError('finality cannot precede service closure')
    return value


@dataclass(frozen=True,init=False)
class FinancialAdapters:
    digest: str
    _encoded_adapters: tuple

    def __init__(self,digest,adapters):
        object.__setattr__(self,'digest',digest)
        object.__setattr__(self,'_encoded_adapters',tuple(encode(a) for a in adapters))

    @property
    def adapters(self): return tuple(decode(a) for a in self._encoded_adapters)

    def get(self, adapter_id):
        for adapter in self.adapters:
            if adapter['adapter_id'] == adapter_id:
                return dict(adapter)
        raise ValueError('financial adapter not in frozen inventory')


def validate_adapters(value, *, expected_sha256):
    value = snapshot(value)
    if digest(value) != sha(expected_sha256):
        raise ValueError('financial inventory differs from frozen digest')
    keys(value,('schema_version','kind','adapters'),'financial adapter inventory')
    if type(value['schema_version']) is not int or value['schema_version'] != 1 or value['kind'] != 'recording_financial_adapters':
        raise ValueError('financial adapters v1 required')
    seen = set()
    adapters = []
    for adapter in array(value['adapters'],64,1):
        keys(adapter,('adapter_id','implementation','implementation_sha256','provider','revision',
                      'evidence_class','input_schema_sha256','output_schema_sha256','closure_capability'),'financial adapter')
        for key in ('adapter_id','provider','revision'):
            identifier(adapter[key])
        if adapter['adapter_id'] in seen:
            raise ValueError('unique financial adapter required')
        seen.add(adapter['adapter_id'])
        if (adapter['implementation'] != ADAPTER or adapter['evidence_class'] != 'fixture' or
                adapter['input_schema_sha256'] != digest(INPUT_GRAMMAR) or
                adapter['output_schema_sha256'] != digest(OUTPUT_GRAMMAR)):
            raise ValueError('only frozen synthetic financial grammar is recognized')
        sha(adapter['implementation_sha256'])
        enum(adapter['closure_capability'],('explicit_original_lines_and_finality','none'))
        adapters.append(dict(adapter))
    return FinancialAdapters(expected_sha256,tuple(adapters))


def validate_financial_evidence(value, *, adapters, invoice, source_reference):
    if type(adapters) is not FinancialAdapters:
        raise TypeError('validated frozen financial adapters required')
    value = snapshot(value)
    invoice = validate_invoice(invoice)
    keys(value,EVIDENCE_FIELDS,'financial evidence')
    if (type(value['schema_version']) is not int or value['schema_version'] != 1 or
            value['kind'] != 'recording_financial_evidence' or value['origin'] != 'synthetic_fixture' or
            value['adapter_inventory_sha256'] != adapters.digest):
        raise ValueError('frozen synthetic financial evidence required')
    adapter = adapters.get(value['adapter_id'])
    if (value['provider'],value['revision']) != (adapter['provider'],adapter['revision']):
        raise ValueError('original invoice provider/revision mismatch')
    reference(value['source_document']); reference(source_reference)
    if value['source_document'] != source_reference:
        raise ValueError('original source document reference differs')
    for key in ('provider','revision','account_scope','invoice_id','currency','currency_unit'):
        if value[key] != invoice[key]:
            raise ValueError('financial evidence changed original invoice field')
    candidates = [line for line in invoice['lines'] if line['line_id'] == value['line_id']]
    if len(candidates) != 1:
        raise ValueError('exact original invoice line required')
    line = candidates[0]
    for key in ('stream_id','billed_atomic','service_start_ns','service_end_ns','service_event_ns'):
        if value[key] != line[key]:
            raise ValueError('financial evidence changed original invoice line')
    keys(value['quantities'],(line['component_id'],),'original line quantities')
    if value['quantities'][line['component_id']] != line['source_quantity']:
        raise ValueError('original source quantity differs')
    if value['claim_sha256'] != claim_sha256(*(value[k] for k in ('provider','account_scope','invoice_id','line_id'))):
        raise ValueError('original logical claim identity differs')
    nullable(value['owner_instance'],sha)
    if value['finality_evidence_sha256'] is not None:
        raise ValueError('initial F1 finality must join downstream without a digest cycle')
    integer(value['observed_received_ns']); integer(value['billed_atomic'])
    return value


class FinancialArtifacts:
    """Exclusive fixed-path files under an owned directory, with shared precharge.

    `charge` is an in-process budget object's reserve(byte_count), never JSON.
    Failed/partial writes retain their charge and are not silently replaced.
    """
    def __init__(self, root, *, budget=None):
        self.root = Path(root).absolute()
        if len(str(self.root)) > 4096 or len(self.root.parts) > 128:
            raise ValueError('bounded owned financial root path required')
        self._fd = os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_NONBLOCK)
        try:
            for part in self.root.parts[1:]:
                if part in ('.','..'): raise ValueError('no root path traversal')
                child = os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=self._fd)
                os.close(self._fd); self._fd = child
        except BaseException:
            os.close(self._fd)
            raise
        self._identity = os.fstat(self._fd)
        self._budget = budget
        self._closed = False
        self._metadata = self._source = self._pages = self._rows = self._total_bytes = 0
        self._files = set()
        self._section_files = {}
        self._read_files = {}; self._read_source = self._read_metadata = self._read_total = 0
        self._index_bytes = 0; self._claimed_chains = set(); self._claimed_pages = set();self._receipt_ordinal=0

    def allocate_receipt_ordinal(self):
        self._check()
        if self._receipt_ordinal >= 16384: raise ValueError('collection receipt artifact capacity exhausted')
        ordinal=self._receipt_ordinal;self._receipt_ordinal+=1
        return ordinal

    def reserve_index(self,value):
        """Conservative actual Python join-index bound, shared by this reader.

        Shared references are deliberately charged again rather than trusted to
        make an independently maximum combination fit. One bounded decoded
        page is temporary working memory outside the retained index.
        """
        def size(item,depth=0):
            if depth > 64: raise ValueError('financial index depth exceeded')
            amount = sys.getsizeof(item)
            if isinstance(item,dict):
                amount += sum(size(k,depth+1)+size(v,depth+1)+32 for k,v in item.items())
            elif isinstance(item,(list,tuple)): amount += sum(size(v,depth+1)+16 for v in item)
            return amount
        amount = size(value)+256
        if self._index_bytes+amount > 128*1024**2:
            raise ValueError('joint financial/native retained join index exceeds 128 MiB')
        self._index_bytes += amount

    def close(self):
        if not self._closed:
            self._closed = True
            os.close(self._fd)

    def __enter__(self): return self
    def __exit__(self,*args): self.close()

    def _check(self):
        if self._closed:
            raise RuntimeError('financial artifact owner is closed')
        current = os.stat(self.root,follow_symlinks=False)
        if not stat.S_ISDIR(current.st_mode) or (current.st_dev,current.st_ino) != (self._identity.st_dev,self._identity.st_ino):
            raise ValueError('owned financial root was replaced')

    def _open_parent(self,path,create=False):
        self._check()
        parts = path.split('/')
        fd = os.dup(self._fd)
        try:
            for part in parts[:-1]:
                if create:
                    try: os.mkdir(part,mode=0o700,dir_fd=fd)
                    except FileExistsError: pass
                child = os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=fd)
                os.close(fd); fd = child
            return fd,parts[-1]
        except BaseException:
            os.close(fd)
            raise

    @staticmethod
    def _path(path):
        if type(path) is not str or re.fullmatch(
                r'(?:financial/(?:source/source-[0-9a-f]{64}\.json|(?:source|evidence|operations|lanes|finality|line-locators)/(?:page|index)-[0-9]{6}\.json|settlements/receipt-[0-9]{6}\.json|evidence/evidence-[0-9]{6}\.json|finality/finality-[0-9]{6}\.json)|ledger/(?:lifetime-[0-9]{3}|coverage-[0-9]{3}|cost-[0-9]{6}|closure-[0-9]{3}|validity-[0-9]{6}|late-[0-9]{6}|financial-inventory|closures|ledger|checkpoint)\.json)',path) is None:
            raise ValueError('fixed financial writer path required')
        return path

    def read(self,ref,*,maximum=PAGE_BYTES):
        reference(ref); self._path(ref['path'])
        if ref['path'].startswith('financial/source/source-') and ref['path'] != 'financial/source/source-'+ref['file_sha256']+'.json':
            raise ValueError('original source filename must preserve exact file hash')
        if ref['bytes'] > maximum:
            raise ValueError('financial reference exceeds file bound')
        identity = (ref['file_sha256'],ref['bytes'])
        previous = self._read_files.get(ref['path'])
        if previous is not None and previous != identity:
            raise ValueError('retained artifact path rebound to different bytes')
        if previous is None:
            source = ref['path'].startswith('financial/source/source-')
            metadata = ref['path'].startswith('financial/') and not source
            if (self._read_total+ref['bytes'] > 4*1024**3 or
                    source and self._read_source+ref['bytes'] > MAX_METADATA or
                    metadata and self._read_metadata+ref['bytes'] > MAX_METADATA):
                raise ValueError('global financial read inventory byte cap exceeded')
            self._read_files[ref['path']] = identity; self._read_total += ref['bytes']
            if source: self._read_source += ref['bytes']
            if metadata: self._read_metadata += ref['bytes']
        directory,name = self._open_parent(ref['path'])
        try:
            fd = os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=directory)
            try:
                before = os.fstat(fd)
                if not stat.S_ISREG(before.st_mode) or before.st_size != ref['bytes']:
                    raise ValueError('exact regular financial file required')
                chunks = bytearray()
                while len(chunks) <= maximum:
                    part = os.read(fd,min(65536,maximum+1-len(chunks)))
                    if not part: break
                    chunks.extend(part)
                after = os.fstat(fd)
                identity = lambda info: (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)
                if identity(before) != identity(after) or len(chunks) != ref['bytes'] or hashlib.sha256(chunks).hexdigest() != ref['file_sha256']:
                    raise ValueError('original financial file changed or hash differs')
                value = decode(bytes(chunks),maximum)
                if bytes(chunks) != encode(value,maximum,newline=True):
                    raise ValueError('exact canonical financial artifact bytes plus LF required')
                return value
            finally: os.close(fd)
        finally: os.close(directory)

    def write(self,path,value,*,source=False):
        self._path(path)
        actual_source=path.startswith('financial/source/source-')
        if source and not actual_source: raise ValueError('original source flag requires fixed source path')
        source=actual_source
        maximum = 128*1024**2 if path == 'ledger/ledger.json' else PAGE_BYTES
        data = encode(value,maximum,newline=True)
        if path in self._files:
            raise ValueError('financial artifact cannot be rewritten')
        if source:
            if path != 'financial/source/source-'+hashlib.sha256(data).hexdigest()+'.json':
                raise ValueError('source filename must preserve exact file hash')
            if self._source+len(data) > MAX_METADATA:
                raise ValueError('financial original source bytes exhausted')
            self._source += len(data)
        elif path.startswith('financial/'):
            if self._metadata+len(data) > MAX_METADATA:
                raise ValueError('financial metadata bytes exhausted')
            self._metadata += len(data)
        if self._total_bytes+len(data) > 4*1024**3:
            raise ValueError('collection financial/ledger artifact bytes exhausted')
        self._total_bytes += len(data)
        if self._budget is not None:
            self._budget.reserve(len(data))
        self._files.add(path)
        directory,name = self._open_parent(path,True)
        try:
            fd = os.open(name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=directory)
            try:
                view = memoryview(data)
                while view:
                    count = os.write(fd,view)
                    if count <= 0: raise OSError('financial write made no progress')
                    view = view[count:]
                os.fsync(fd)
            finally: os.close(fd)
            os.fsync(directory)
        finally: os.close(directory)
        return dict(path=path,file_sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))

    def write_invoice(self,value):
        value = validate_invoice(value)
        data = encode(value,PAGE_BYTES,newline=True)
        return self.write('financial/source/source-'+hashlib.sha256(data).hexdigest()+'.json',value,source=True)

    def write_index(self,section,rows):
        enum(section,PAGE_KINDS)
        pages = []
        count = 0
        previous = None
        batch = []
        first_page = self._section_files.get((section,'page'),0)
        first_index = self._section_files.get((section,'index'),0)
        def flush():
            nonlocal previous
            value = dict(schema_version=1,kind=PAGE_KINDS[section],page_ordinal=len(pages),
                         previous_page_sha256=previous,rows=list(batch))
            ref = self.write('financial/'+section+'/page-%06d.json'%(first_page+len(pages)),value)
            pages.append(ref); previous = ref['file_sha256']; batch.clear()
            self._pages += 1
        for row in rows:
            if self._rows+1 > MAX_ROWS:
                raise ValueError('financial row capacity exhausted')
            candidate = dict(schema_version=1,kind=PAGE_KINDS[section],page_ordinal=len(pages),
                             previous_page_sha256=previous,rows=batch+[row])
            try: encode(candidate,PAGE_BYTES,newline=True)
            except ValueError:
                if not batch: raise
                flush()
            if len(batch) == 128: flush()
            batch.append(row); count += 1; self._rows += 1
            if self._pages >= MAX_PAGES:
                raise ValueError('financial page capacity exhausted')
        if batch: flush()
        previous = None; index_count = 0; tail = None
        for start in range(0,len(pages),128):
            if self._pages >= MAX_PAGES:
                raise ValueError('financial data/index page capacity exhausted')
            value = dict(schema_version=1,kind='recording_financial_index_page',page_kind=PAGE_KINDS[section],
                         page_ordinal=index_count,previous_page_sha256=previous,pages=pages[start:start+128])
            tail = self.write('financial/'+section+'/index-%06d.json'%(first_index+index_count),value)
            previous = tail['file_sha256']; index_count += 1; self._pages += 1
        self._section_files[section,'page'] = first_page+len(pages)
        self._section_files[section,'index'] = first_index+index_count
        return dict(row_count=count,page_count=len(pages),index_page_count=index_count,index_tail=tail)

    def assert_financial_inventory(self):
        """Known retained financial files cannot be omitted to obtain a pass."""
        self._check(); found = set(); count = 0
        try: financial = os.open('financial',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=self._fd)
        except FileNotFoundError:
            if any(path.startswith('financial/') for path in self._read_files):
                raise ValueError('retained financial directory disappeared')
            return
        try:
            sections = []
            with os.scandir(financial) as listing:
                for item in listing:
                    if len(sections) >= 7: raise ValueError('unknown financial directory section')
                    sections.append(item)
            for section in sections:
                if section.name not in (*PAGE_KINDS,'settlements') or not section.is_dir(follow_symlinks=False):
                    raise ValueError('unknown/nonowned financial directory entry')
                fd = os.open(section.name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=financial)
                try:
                    with os.scandir(fd) as listing:
                        for item in listing:
                            count += 1
                            if count > 524288: raise ValueError('financial file inventory capacity exceeded')
                            path = 'financial/'+section.name+'/'+item.name
                            self._path(path)
                            if not item.is_file(follow_symlinks=False): raise ValueError('regular original financial artifact required')
                            found.add(path)
                finally: os.close(fd)
        finally: os.close(financial)
        if found != {path for path in self._read_files if path.startswith('financial/')}:
            raise ValueError('orphan/omitted original financial evidence or late invoice')

    def assert_validity_inventory(self,notice_refs,validity_refs):
        expected={reference(ref)['path'] for ref in (*notice_refs,*validity_refs)}
        found=set();count=0
        directory,name=self._open_parent('ledger/checkpoint.json')
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    count+=1
                    if count>524288:raise ValueError('bounded retained ledger file inventory required')
                    if entry.name.startswith(('late-','validity-')):
                        path='ledger/'+entry.name;self._path(path)
                        if not entry.is_file(follow_symlinks=False):raise ValueError('regular retained validity/notice file required')
                        found.add(path)
        finally:os.close(directory)
        if found!=expected:raise ValueError('known retained late notice/validity entry omitted from reviewed chain')


def iter_financial_rows(store,section,index):
    """Validate bounded index-chain and stream rows in original committed order."""
    enum(section,PAGE_KINDS)
    keys(index,('row_count','page_count','index_page_count','index_tail'),'financial index')
    rows = integer(index['row_count'],maximum=MAX_ROWS)
    pages = integer(index['page_count'],maximum=MAX_PAGES)
    count = integer(index['index_page_count'],maximum=MAX_PAGES)
    if not rows:
        if pages or count or index['index_tail'] is not None:
            raise ValueError('empty financial index must have zero counts/null tail')
        return
    if not pages or count != (pages+127)//128 or pages+count > MAX_PAGES:
        raise ValueError('financial index page/count mismatch')
    tail = reference(index['index_tail'])
    chain = (section,tail['path'],tail['file_sha256'])
    if chain in store._claimed_chains: raise ValueError('financial index aliased by another inventory')
    store._claimed_chains.add(chain)
    match = re.fullmatch('financial/'+section+r'/index-([0-9]{6})\.json',tail['path'])
    if match is None: raise ValueError('fixed financial index tail path required')
    final_file_ordinal = int(match.group(1))
    if final_file_ordinal < count-1: raise ValueError('financial index file ordinal underflow')
    # Index pages are bounded before retention; <=1024 refs, never all rows.
    blocks = []
    for ordinal in range(count-1,-1,-1):
        file_ordinal = final_file_ordinal-(count-1-ordinal)
        if tail['path'] != 'financial/'+section+'/index-%06d.json'%file_ordinal:
            raise ValueError('fixed contiguous financial index path required')
        page = store.read(tail)
        keys(page,('schema_version','kind','page_kind','page_ordinal','previous_page_sha256','pages'),'financial index page')
        if (type(page['schema_version']) is not int or page['schema_version'] != 1 or
                page['kind'] != 'recording_financial_index_page' or page['page_kind'] != PAGE_KINDS[section] or
                type(page['page_ordinal']) is not int or page['page_ordinal'] != ordinal):
            raise ValueError('financial index page identity mismatch')
        refs = array(page['pages'],128,1)
        if ordinal < count-1 and len(refs) != 128:
            raise ValueError('nonfinal financial index pages must be full')
        for ref in refs: reference(ref)
        blocks.append(refs)
        if ordinal == 0:
            if page['previous_page_sha256'] is not None:
                raise ValueError('first financial index previous must be null')
        else:
            previous_hash = sha(page['previous_page_sha256'])
            previous_path = 'financial/'+section+'/index-%06d.json'%(file_ordinal-1)
            # Find exact bounded predecessor length without following a caller path.
            directory,name = store._open_parent(previous_path)
            try:
                info = os.stat(name,dir_fd=directory,follow_symlinks=False)
                if not stat.S_ISREG(info.st_mode) or not 1 <= info.st_size <= PAGE_BYTES:
                    raise ValueError('bounded regular financial index predecessor required')
            finally: os.close(directory)
            tail = dict(path=previous_path,file_sha256=previous_hash,bytes=info.st_size)
    refs = [ref for block in reversed(blocks) for ref in block]
    if len(refs) != pages:
        raise ValueError('financial data page count mismatch')
    previous = None; seen = set(); actual_rows = 0; first_file_ordinal = None
    for ordinal,ref in enumerate(refs):
        match = re.fullmatch('financial/'+section+r'/page-([0-9]{6})\.json',ref['path'])
        if match is None: raise ValueError('fixed financial data page path required')
        if first_file_ordinal is None: first_file_ordinal = int(match.group(1))
        path = 'financial/'+section+'/page-%06d.json'%(first_file_ordinal+ordinal)
        if ref['path'] != path or ref['file_sha256'] in seen:
            raise ValueError('duplicate/aliased financial page')
        if ref['path'] in store._claimed_pages: raise ValueError('financial page reused across independent indexes')
        store._claimed_pages.add(ref['path'])
        seen.add(ref['file_sha256'])
        page = store.read(ref)
        keys(page,('schema_version','kind','page_ordinal','previous_page_sha256','rows'),'financial data page')
        if (type(page['schema_version']) is not int or page['schema_version'] != 1 or
                page['kind'] != PAGE_KINDS[section] or type(page['page_ordinal']) is not int or
                page['page_ordinal'] != ordinal or page['previous_page_sha256'] != previous):
            raise ValueError('financial data page chain mismatch')
        batch = array(page['rows'],128,1)
        actual_rows += len(batch)
        if actual_rows > rows:
            raise ValueError('financial row count exceeds index')
        yield from batch
        previous = ref['file_sha256']
    if actual_rows != rows:
        raise ValueError('financial row count differs from index')

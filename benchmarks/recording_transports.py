"""Finite recording completions: explicit fixture or one-shot loopback HTTP.

These transports require an actual native-entry binding. They never infer a
server/GPU entry, a monetary receipt, or closure of an external HTTP server.
"""
from abc import ABC, abstractmethod
import asyncio
import base64
from dataclasses import dataclass
import hashlib
import json
import os
import struct
import time
import zlib

from gamst.native_entry import BoundCompletion, PreparedCall, NativeEntryBinding
from gamst.capabilities import bounded_json, identifier
from gamst.streams import _finish
from .owned_capture import CapturePacket
from .recorded_source import encoded, digest, _keys, _integer
from .recording_runtime_contract import validate_adapter, validate_fixture_responses, sha, text
from .recording_dependencies import VerifiedDependencyInventory

MAX_WIRE = 16*1024**2
MAX_HEADERS = 16*1024
MAX_PNG = 8*1024**2
MAX_PIXELS = 1920*1080*3


class TransportOutcomeSink(ABC):
    """Trusted synchronous bounded output seam, never selected by JSON."""
    @abstractmethod
    def record(self, row): pass


class FixtureReceiptRouter(ABC):
    """Explicit synthetic receipt scenario, not a provider-money fallback."""
    @abstractmethod
    async def settle(self, binding, scenario_sha256, *, call_ordinal): pass


@dataclass(frozen=True, slots=True)
class TransportCapacity:
    max_request_bytes: int
    max_png_bytes: int = 0
    max_decoded_bytes: int = 0

    def __post_init__(self):
        _integer(self.max_request_bytes, 1, 12*1024**2)
        _integer(self.max_png_bytes, 0, MAX_PNG)
        _integer(self.max_decoded_bytes, 0, MAX_PIXELS)
        if bool(self.max_png_bytes) != bool(self.max_decoded_bytes):
            raise ValueError('visual encoded/decoded capacity must be jointly reserved')

    def workspace_bytes(self, response_bytes):
        # Detached SDK bytes + parsed strings + multipart URL + wire encoding,
        # PNG decode + digest copy, response copy and bounded receive chunk.
        wire = min(MAX_WIRE, self.max_request_bytes*2+1024)
        return (self.max_request_bytes*8 + wire*3 + self.max_png_bytes*2
                + self.max_decoded_bytes*3 + response_bytes*32 + 65536 + 16*1024**2 + MAX_HEADERS*2)


class TransportMemoryBudget:
    def __init__(self, capacity):
        self.capacity = _integer(capacity, 1, 256*1024**2)
        self._reserved = 0

    @property
    def reserved_bytes(self): return self._reserved

    def _reserve(self, amount):
        if self._reserved+amount > self.capacity:
            raise ValueError('joint completion workspace capacity exhausted before activation')
        self._reserved += amount

    def _release(self, amount):
        if not 0 <= amount <= self._reserved: raise RuntimeError('workspace reservation mismatch')
        self._reserved -= amount


def _sdk_request(request, capacity):
    if type(request) is not dict or set(request) not in (
            {'provider', 'model_revision', 'messages'}, {'provider', 'model_revision', 'messages', 'images'}):
        raise ValueError('completed SDK Text/Visual request required')
    identifier(request['provider']); identifier(request['model_revision'])
    messages = request['messages']
    if type(messages) is not list or len(messages) != 2: raise ValueError('exact SDK system/user messages required')
    for message, role in zip(messages, ('system', 'user')):
        _keys(message, ('role', 'content'))
        if message['role'] != role or type(message['content']) is not str or not message['content']:
            raise ValueError('exact original SDK text messages required')
        if len(message['content']) > capacity.max_request_bytes or len(message['content'].encode('utf-8')) > capacity.max_request_bytes:
            raise ValueError('SDK text exceeds frozen workspace reservation')
    if 'images' in request:
        if not capacity.max_png_bytes: raise ValueError('visual request has no frozen workspace reservation')
        images = request['images']
        if type(images) is not list or len(images) != 1: raise ValueError('one actual SDK PNG required')
        _keys(images[0], ('mime_type', 'data_base64', 'sha256', 'source_frame_sha256', 'capture_ms', 'shape'))
        image = images[0]
        if image['mime_type'] != 'image/png': raise ValueError('actual SDK PNG required')
        sha(image['sha256']); sha(image['source_frame_sha256'])
        _integer(image['capture_ms'], 0, 2**53-1)
        shape = image['shape']
        if type(shape) is not list or len(shape) != 3: raise ValueError('bounded BGR shape required')
        _integer(shape[0], 32, 1080); _integer(shape[1], 32, 1920); _integer(shape[2], 3, 3)
        if type(image['data_base64']) is not str or len(image['data_base64']) > ((capacity.max_png_bytes+2)//3)*4:
            raise ValueError('SDK base64 exceeds frozen workspace reservation')
    raw = encoded(request)
    if len(raw) > capacity.max_request_bytes: raise ValueError('SDK request exceeds preflight workspace reservation')
    return raw


def _response(raw, model):
    if type(raw) is not bytes or len(raw)>1048576:raise ValueError('bounded raw HTTP response bytes required')
    # Bound object allocation BEFORE json.loads. Count both keys and values;
    # this is intentionally stricter than the downstream SDK tree walker.
    source = raw.decode('utf-8')
    depth = nodes = 0; quoted = escaped = primitive = False; string_units = 0
    for char in source:
        if quoted:
            string_units += 1
            if string_units > 49152: raise ValueError('HTTP JSON string lexeme exceeds bounded workspace')
            if escaped: escaped = False
            elif char == '\\': escaped = True
            elif char == '"': quoted = False
            continue
        if char == '"':
            quoted = True; string_units = 0; primitive = False; nodes += 1
        elif char in '[{':
            depth += 1; nodes += 1; primitive = False
            if depth > 12: raise ValueError('HTTP JSON exceeds structural depth')
        elif char in ']}': depth -= 1; primitive = False
        elif char in ',: \t\r\n': primitive = False
        elif not primitive: nodes += 1; primitive = True
        if nodes > 8192: raise ValueError('HTTP JSON exceeds preparse node bound')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result: raise ValueError('duplicate HTTP response key')
            result[key] = value
        return result
    def invalid(_): raise ValueError('nonfinite HTTP response number')
    def integer(token):
        if len(token.lstrip('-')) > 19: raise ValueError('HTTP integer lexeme exceeds N63')
        number = int(token)
        if not -2**63+1 <= number <= 2**63-1: raise ValueError('HTTP integer exceeds N63')
        return number
    try:
        value = json.loads(source, object_pairs_hook=pairs,
                           parse_constant=invalid, parse_int=integer)
    except (UnicodeError, RecursionError) as error:
        raise ValueError('strict HTTP UTF-8 JSON required') from error
    value = bounded_json(value, max_bytes=1048576)
    allowed = {'id', 'object', 'created', 'model', 'choices', 'usage', 'system_fingerprint'}
    if type(value) is not dict or not {'model', 'choices'} <= set(value) or set(value)-allowed:
        raise ValueError('exact bounded chat completion response required')
    if value['model'] != model: raise ValueError('response model differs from frozen model')
    if 'id' in value: text(value['id'], 128, printable=True)
    if 'object' in value and value['object'] != 'chat.completion': raise ValueError('chat.completion object required')
    if 'created' in value: _integer(value['created'], 0, 2**63-1)
    if value.get('system_fingerprint') is not None: text(value['system_fingerprint'], 128, printable=True)
    choices = value['choices']
    if type(choices) is not list or len(choices) != 1: raise ValueError('exact one completed choice required')
    choice = choices[0]; _keys(choice, ('index', 'message', 'finish_reason'))
    _integer(choice['index'], 0, 0)
    if choice['finish_reason'] != 'stop': raise ValueError('only stop-finished response accepted')
    message = choice['message']; _keys(message, ('role', 'content'))
    if message['role'] != 'assistant' or not text(message['content'], 8192).strip():
        raise ValueError('nonempty assistant content required')
    usage = value.get('usage')
    if usage is not None:
        _keys(usage, ('prompt_tokens', 'completion_tokens', 'total_tokens'))
        for count in usage.values(): _integer(count, 0, 10**12)
        if usage['total_tokens'] != usage['prompt_tokens']+usage['completion_tokens']:
            raise ValueError('exact total token sum required')
        usage = dict(input_tokens=usage['prompt_tokens'], output_tokens=usage['completion_tokens'], cached_input_tokens=0)
    return dict(text=message['content'], usage=usage), value.get('id')


def _bounded_http_transport(response_limit):
    """Raw admission BEFORE h11 creates header objects; no private pool patch."""
    import httpcore
    import httpx
    import ssl
    import certifi

    class Stream(httpcore.AsyncNetworkStream):
        def __init__(self, stream): self.stream=stream; self.head=bytearray(); self.admitted=False
        async def read(self, max_bytes, timeout=None):
            if self.admitted:return await self.stream.read(min(max_bytes,65536),timeout)
            while True:
                chunk=await self.stream.read(min(max_bytes,MAX_HEADERS+1-len(self.head)),timeout)
                if not chunk:raise ValueError('incomplete bounded HTTP response head')
                self.head.extend(chunk)
                end=self.head.find(b'\r\n\r\n')
                if end>=0:
                    end+=4
                    if end>MAX_HEADERS:raise ValueError('raw HTTP response headers exceed16KiB')
                    lines=bytes(self.head[:end-4]).split(b'\r\n')
                    if len(lines)>129:raise ValueError('raw HTTP response exceeds128 header fields')
                    status=lines[0].split(b' ',2)
                    if len(status)<2 or status[0] not in (b'HTTP/1.0',b'HTTP/1.1') or len(status[1])!=3 or not status[1].isdigit() or not 200<=int(status[1])<=599:
                        raise ValueError('only one final HTTP response supported')
                    lengths=[]
                    for line in lines[1:]:
                        key,sep,value=line.partition(b':')
                        if not sep:raise ValueError('exact HTTP response field required')
                        if key.lower()==b'transfer-encoding':raise ValueError('chunked response outside bounded transport')
                        if key.lower()==b'content-length':lengths.append(value.strip())
                    if len(lengths)!=1 or not lengths[0].isdigit() or len(lengths[0])>19 or int(lengths[0])>response_limit:
                        raise ValueError('one bounded Content-Length response required')
                    self.admitted=True;result=bytes(self.head);self.head.clear();return result
                if len(self.head)>=MAX_HEADERS or self.head.count(b'\r\n')>129:
                    raise ValueError('raw HTTP response headers exceed bounded workspace')
        async def write(self, buffer, timeout=None):return await self.stream.write(buffer,timeout)
        async def aclose(self):return await self.stream.aclose()
        async def start_tls(self, ssl_context, server_hostname=None, timeout=None):
            return Stream(await self.stream.start_tls(ssl_context,server_hostname,timeout))
        def get_extra_info(self, info):return self.stream.get_extra_info(info)

    class Backend(httpcore.AsyncNetworkBackend):
        def __init__(self):self.backend=httpcore.AnyIOBackend()
        async def connect_tcp(self,*args,**kwargs):return Stream(await self.backend.connect_tcp(*args,**kwargs))
        async def connect_unix_socket(self,*args,**kwargs):raise ValueError('Unix socket is not an admitted loopback endpoint')
        async def sleep(self, seconds):return await asyncio.sleep(seconds)

    class Body(httpx.AsyncByteStream):
        def __init__(self,stream):self.stream=stream
        async def __aiter__(self):
            async for part in self.stream:yield part
        async def aclose(self):await self.stream.aclose()

    class Transport(httpx.AsyncBaseTransport):
        def __init__(self):
            self.pool=httpcore.AsyncConnectionPool(ssl_context=ssl.create_default_context(cafile=certifi.where()),
                max_connections=1,max_keepalive_connections=0,http1=True,http2=False,
                retries=0,network_backend=Backend())
        async def handle_async_request(self,request):
            response=await self.pool.handle_async_request(httpcore.Request(method=request.method,
                url=httpcore.URL(scheme=request.url.raw_scheme,host=request.url.raw_host,
                    port=request.url.port,target=request.url.raw_path),headers=request.headers.raw,
                content=request.stream,extensions=request.extensions))
            return httpx.Response(response.status,headers=response.headers,stream=Body(response.stream),extensions=response.extensions)
        async def aclose(self):await self.pool.aclose()
    return Transport()


def _validate_png(image, capacity, capture):
    if image['mime_type'] != 'image/png': raise ValueError('actual SDK PNG required')
    sha(image['sha256']); sha(image['source_frame_sha256'])
    _integer(image['capture_ms'], 0, 2**53-1)
    shape = image['shape']
    if type(shape) is not list or len(shape) != 3:
        raise ValueError('plain BGR shape required')
    _integer(shape[0], 32, 1080); _integer(shape[1], 32, 1920); _integer(shape[2], 3, 3)
    if shape[0]*shape[1]*3 > capacity.max_decoded_bytes: raise ValueError('decoded PNG exceeds reserved bytes')
    data = image['data_base64']
    if not data.isascii() or len(data) > ((capacity.max_png_bytes+2)//3)*4:
        raise ValueError('PNG base64 exceeds reservation')
    png = base64.b64decode(data, validate=True)
    if not png or len(png) > capacity.max_png_bytes or base64.b64encode(png).decode('ascii') != data:
        raise ValueError('canonical bounded PNG base64 required')
    if hashlib.sha256(png).hexdigest() != image['sha256']:
        raise ValueError('actual PNG differs from SDK digest')
    if len(png) < 33 or png[:8] != b'\x89PNG\r\n\x1a\n' or png[8:16] != b'\0\0\0\rIHDR':
        raise ValueError('bounded PNG IHDR required before decoding')
    width, height = struct.unpack('!II', png[16:24])
    if (width, height) != (shape[1], shape[0]): raise ValueError('PNG dimensions differ from SDK shape')
    # Only the actual SDK encoder's plain noninterlaced 8-bit truecolor PNG
    # grammar. Reject ancillary compressed profiles/APNG before native decoding.
    position=8; kinds=[]; inflated=0; expected=height*(width*3+1); inflater=zlib.decompressobj()
    while position<len(png):
        if len(png)-position<12:raise ValueError('complete bounded PNG chunks required')
        length=struct.unpack('!I',png[position:position+4])[0];kind=png[position+4:position+8]
        if length>len(png)-position-12 or kind not in (b'IHDR',b'IDAT',b'IEND'):
            raise ValueError('only bounded actual SDK PNG chunks supported')
        data=png[position+8:position+8+length]
        crc=struct.unpack('!I',png[position+8+length:position+12+length])[0]
        if zlib.crc32(kind+data)&0xffffffff!=crc:raise ValueError('PNG chunk CRC differs')
        kinds.append(kind)
        if len(kinds)>2048:raise ValueError('bounded PNG chunk count required')
        if kind==b'IHDR' and (len(kinds)!=1 or data[8:]!=b'\x08\x02\x00\x00\x00'):
            raise ValueError('plain noninterlaced 8-bit truecolor PNG required')
        if kind==b'IDAT':
            if b'IEND' in kinds:raise ValueError('PNG IDAT after terminal chunk')
            inflated+=len(inflater.decompress(data,expected+1-inflated))
            if inflated>expected or inflater.unused_data:raise ValueError('PNG inflate exceeds bounded pixel layout')
        if kind==b'IEND' and (length or position+12!=len(png)):
            raise ValueError('one exact terminal PNG chunk required')
        position+=length+12
    if kinds[0]!=b'IHDR' or kinds[-1]!=b'IEND' or b'IDAT' not in kinds or inflated!=expected or not inflater.eof:
        raise ValueError('complete exact bounded PNG pixel stream required')
    import cv2
    import numpy as np
    from gamst.smolvlm_call import smolvlm_frame_sha256
    pixels = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if type(pixels) is not np.ndarray or pixels.dtype != np.uint8 or pixels.shape != tuple(shape):
        raise ValueError('lossless PNG must decode to original plain BGR pixels')
    if smolvlm_frame_sha256(pixels) != image['source_frame_sha256']:
        raise ValueError('decoded PNG differs from source pixel digest')
    if capture is None or (capture['pixels_sha256'], capture['capture_ms'], capture['shape']) != (
            image['source_frame_sha256'], image['capture_ms'], shape):
        raise ValueError('SDK image differs from owned original capture')


class _Completion(BoundCompletion):
    @property
    def adapter(self): return json.loads(self._adapter)

    def _initialize(self, adapter, *, invocation_ordinal, lane_id, binding_index, context, capacity, memory):
        self._adapter = encoded(validate_adapter(adapter))
        self.invocation = _integer(invocation_ordinal, 0, 95); self.lane_id = identifier(lane_id)
        self.binding_index = _integer(binding_index, 0, 63)
        if type(capacity) is not TransportCapacity or type(memory) is not TransportMemoryBudget:
            raise TypeError('typed joint completion workspace reservation required')
        # Native context validation is reused; no user-authored current scope.
        from gamst.native_entry import validate_owner
        from .recording_native_schema import snapshot
        context = snapshot(context, maximum=4096)
        validate_owner('recording_native4', dict(context=context, scope_index=0, binding_index=binding_index))
        if context['invocation_ordinal'] != invocation_ordinal: raise ValueError('completion invocation mismatch')
        self._context = encoded(context); self.capacity = capacity; self._memory = memory
        self._closed = False; self._prepared = None; self._close_task = None
        self._captures = {}; self._entries = {}; self._instances = set(); self._prompts = set(); self._call_ordinal = 0

    def entry_for_decision(self, request_decision_sha256):
        """Return only the actual observer-issued immutable entry, never JSON."""
        sha(request_decision_sha256)
        return self._entries.get(request_decision_sha256)

    def release_entry(self, request_decision_sha256):
        """Trusted driver calls after reserving immutable downstream exports."""
        sha(request_decision_sha256)
        if request_decision_sha256 not in self._entries: raise ValueError('no retained actual entry')
        del self._entries[request_decision_sha256]

    def release_capture(self, request_decision_sha256):
        """Explicitly retire capture correlation when dispatch was refused."""
        sha(request_decision_sha256)
        self._captures.pop(request_decision_sha256, None)

    def register_capture(self, request_decision_sha256, packet):
        if self._closed or type(packet) is not CapturePacket or packet.format != 'bgr8_hwc':
            raise ValueError('actual owned BGR capture required before dispatch')
        sha(request_decision_sha256)
        if request_decision_sha256 in self._captures or len(self._captures) >= 256:
            raise ValueError('unique bounded immutable request/capture mapping required')
        header = b'gamst.smolvlm.uint8.HWC.BGR.v1\0'+struct.pack('>III', *packet.shape)
        hashed = hashlib.sha256(header); hashed.update(packet.payload)
        self._captures[request_decision_sha256] = dict(pixels_sha256=hashed.hexdigest(),
            capture_ms=packet.sdk_timestamp_ms, shape=list(packet.shape),
            capture_ref=dict(kind='capture', source_index=packet.source_index, capture_ordinal=packet.record_ordinal,
                             record_sha256=packet.record_sha256, mapping_sha256=packet.mapping_sha256))

    def prepare(self, request):
        if self._closed or self._prepared is not None: raise RuntimeError('completion closed/busy; no hidden queue')
        raw = _sdk_request(request, self.capacity)
        if (request['provider'], request['model_revision']) != (self.adapter['provider'], self.adapter['model_revision']):
            raise ValueError('actual SDK provider differs from frozen adapter')
        prepared = _Prepared(self, raw); self._prepared = prepared
        return prepared

    def _binding(self, binding, raw):
        if type(binding) is not NativeEntryBinding: raise TypeError('actual observer-issued entry required')
        value = binding.snapshot()
        if (value['owner_kind'] != 'recording_native4' or encoded(value['owner']['context']) != self._context
                or value['owner']['binding_index'] != self.binding_index
                or (value['provider'], value['model_revision']) != (self.adapter['provider'], self.adapter['model_revision'])
                or value['prompt_sha256'] != hashlib.sha256(raw).hexdigest()):
            raise ValueError('actual native entry differs from immutable completion binding')
        instance = value['entry']['instance_sha256']
        if instance in self._instances or len(self._instances) >= 16384: raise ValueError('entry alias/capacity exhausted')
        self._instances.add(instance)
        ordinal = self._call_ordinal; self._call_ordinal += 1
        decision = value['entry']['request_decision_sha256']
        if decision in self._entries or len(self._entries) >= 256:
            raise ValueError('actual entry retention capacity exhausted; export/release required')
        self._entries[decision] = binding
        capture = self._captures.pop(decision, None)
        return value, ordinal, capture

    async def close(self):
        if self._prepared is not None and asyncio.current_task() is self._prepared._task:
            raise RuntimeError('completion cannot close its own callback')
        if self._close_task is None:
            self._closed = True; self._close_task = asyncio.create_task(self._close())
        try: await asyncio.shield(self._close_task)
        except asyncio.CancelledError:
            await _finish(self._close_task); raise

    async def _close(self):
        if self._prepared is not None: await self._prepared.release()
        self._captures.clear(); self._entries.clear(); self._instances.clear(); self._prompts.clear()


class _Prepared(PreparedCall):
    def __init__(self, owner, raw):
        self.owner = owner; self._raw = raw; self._task = None; self._released = False

    async def invoke(self, binding):
        if self._released or self._task is not None or self.owner._closed:
            raise RuntimeError('prepared completion is single-use')
        value, ordinal, capture = self.owner._binding(binding, self._raw)
        task = asyncio.create_task(self.owner._invoke(self._raw, binding, value, ordinal, capture))
        self._task = task
        try: return await asyncio.shield(task)
        except asyncio.CancelledError:
            task.cancel(); await _finish(asyncio.gather(task, return_exceptions=True)); raise

    async def release(self):
        if self._task is not None and not self._task.done():
            if asyncio.current_task() is self._task: raise RuntimeError('prepared call cannot join itself')
            self._task.cancel(); await _finish(asyncio.gather(self._task, return_exceptions=True))
        self._released = True; self._raw = None; self._task = None
        if self.owner._prepared is self: self.owner._prepared = None


class LocalChatCompletions(_Completion):
    @classmethod
    async def open(cls, adapter, *, invocation_ordinal, lane_id, binding_index, context,
                   dependency_inventory, sink, capacity, memory):
        checked = validate_adapter(adapter)
        if checked['implementation'] != 'local_chat_completions_v1': raise ValueError('explicit real HTTP adapter required')
        if type(dependency_inventory) is not VerifiedDependencyInventory or dependency_inventory.sha256 != checked['dependency_inventory_sha256']:
            raise ValueError('actual issued dependency inventory required before activation')
        dependency_inventory.require('httpx')
        for dependency in ('httpcore','h11','anyio','certifi','idna','sniffio'):
            dependency_inventory.require(dependency)
        if capacity.max_png_bytes:
            dependency_inventory.require('numpy'); dependency_inventory.require('opencv-python')
        if not isinstance(sink, TransportOutcomeSink): raise TypeError('typed bounded transport outcome sink required')
        self = cls(); self._initialize(checked, invocation_ordinal=invocation_ordinal, lane_id=lane_id,
            binding_index=binding_index, context=context, capacity=capacity, memory=memory)
        self._sink = sink; self._workspace = capacity.workspace_bytes(checked['config']['response_bytes'])
        memory._reserve(self._workspace)
        try:
            import httpx
            dependency_inventory.require('httpx', httpx.__version__)
            self._client = httpx.AsyncClient(timeout=None, trust_env=False, follow_redirects=False,
                http2=False, limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
                transport=_bounded_http_transport(checked['config']['response_bytes']))
        except BaseException:
            memory._release(self._workspace); raise
        return self

    async def _invoke(self, raw, binding, value, ordinal, capture):
        started = time.monotonic_ns(); deadline = asyncio.get_running_loop().time()+self.adapter['config']['timeout_ms']/1000
        request = json.loads(raw); messages = request['messages']
        if 'images' in request:
            image = request['images'][0]
            messages[-1]['content'] = [dict(type='text', text=messages[-1]['content']),
                dict(type='image_url', image_url=dict(url='data:image/png;base64,'+image['data_base64'], detail='auto'))]
        body = encoded(dict(model=self.adapter['model_id'], messages=messages, temperature=0, stream=False))
        if len(body) > MAX_WIRE: raise ValueError('HTTP wire body exceeds16MiB')
        prompt = value['prompt_sha256']; wire_sha = hashlib.sha256(body).hexdigest()
        row = dict(binding_index=self.binding_index, prompt_sha256=prompt, wire_request_sha256=wire_sha,
            response_body_sha256=None, provider_request_id=None, http_status=None, started_ns=started,
            finished_ns=started, outcome='transport_error', request_bytes=len(body), response_bytes=0, attempts=1)
        try:
            if prompt in self._prompts: raise ValueError('duplicate binding/prompt transport correlation is unsupported')
            self._prompts.add(prompt)
            if 'images' in request:
                decoder = asyncio.create_task(asyncio.to_thread(_validate_png, image, self.capacity, capture))
                try: await asyncio.shield(decoder)
                except asyncio.CancelledError:
                    await _finish(asyncio.gather(decoder, return_exceptions=True)); raise
            elif capture is not None:
                raise ValueError('text-only completion cannot consume a visual capture mapping')
            remaining = deadline-asyncio.get_running_loop().time()
            if remaining <= 0: raise asyncio.TimeoutError('completion deadline expired before HTTP')
            result = await asyncio.wait_for(self._post(body, row), remaining)
            if asyncio.get_running_loop().time() >= deadline: raise asyncio.TimeoutError('completion exceeded frozen deadline')
            row['outcome'] = 'ok'
            return result
        except asyncio.CancelledError:
            row['outcome'] = 'cancelled'; raise
        except asyncio.TimeoutError:
            row['outcome'] = 'timeout'; raise
        except ValueError:
            row['outcome'] = 'invalid_response'; raise
        finally:
            row['finished_ns'] = time.monotonic_ns()
            self._sink.record(row)

    async def _post(self, body, row):
        headers = {'Content-Type': 'application/json', 'Accept-Encoding': 'identity'}
        name = self.adapter['config']['api_key_env']
        if name is not None:
            secret = os.environ.get(name)
            if type(secret) is not str or not 1 <= len(secret.encode()) <= 8192 or any(c in secret for c in '\r\n\x00'):
                raise ValueError('explicit named credential is missing or malformed')
            headers['Authorization'] = 'Bearer '+secret
        request = self._client.build_request('POST', self.adapter['config']['endpoint'], content=body, headers=headers)
        if sum(len(k)+len(v)+4 for k, v in request.headers.raw) > MAX_HEADERS:
            raise ValueError('HTTP request headers exceed16KiB')
        response = await self._client.send(request, stream=True)
        try:
            row['http_status'] = response.status_code
            if sum(len(k)+len(v)+4 for k, v in response.headers.raw) > MAX_HEADERS:
                raise ValueError('HTTP response headers exceed16KiB')
            if response.headers.get('content-encoding', 'identity').lower() != 'identity':
                raise ValueError('compressed response is outside the frozen transport')
            limit = self.adapter['config']['response_bytes']; length = response.headers.get('content-length')
            if length is not None and (not length.isascii() or not length.isdecimal() or len(length) > 19 or int(length) > limit):
                raise ValueError('HTTP declared body exceeds response bound')
            data = bytearray()
            async for chunk in response.aiter_raw(chunk_size=65536):
                row['response_bytes'] += len(chunk)
                if len(data)+len(chunk) > limit: raise ValueError('HTTP body exceeds frozen response bound')
                data.extend(chunk)
            raw = bytes(data); row['response_body_sha256'] = hashlib.sha256(raw).hexdigest()
            if response.status_code != 200:
                row['outcome'] = 'http_error'; raise RuntimeError('non200 completion response')
            result, request_id = _response(raw, self.adapter['model_id'])
            row['provider_request_id'] = request_id
            return result
        finally:
            cleanup = asyncio.create_task(response.aclose())
            try: await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await _finish(cleanup); raise

    async def _close(self):
        try:
            await super()._close()
            await self._client.aclose()
        finally:
            self._memory._release(self._workspace)


class FixtureCompletion(_Completion):
    @classmethod
    async def open(cls, adapter, responses, *, invocation_ordinal, lane_id, binding_index,
                   context, capacity, memory, receipt_router=None):
        checked = validate_adapter(adapter)
        if checked['implementation'] != 'fixture_completion_v1': raise ValueError('explicit synthetic fixture adapter required')
        responses = validate_fixture_responses(responses)
        if digest(responses) != checked['config']['responses_sha256']:
            raise ValueError('fixture response table differs from frozen hash')
        if receipt_router is not None and (not isinstance(receipt_router, FixtureReceiptRouter)
                or checked['config']['settlement'] != 'synthetic_explicit'):
            raise ValueError('only an explicit typed synthetic receipt router is supported')
        self = cls(); self._initialize(checked, invocation_ordinal=invocation_ordinal, lane_id=lane_id,
            binding_index=binding_index, context=context, capacity=capacity, memory=memory)
        self._responses = {(r['invocation_ordinal'], r['lane_id'], r['call_ordinal']): r for r in responses['rows']}
        if any(r['synthetic_receipt_sha256'] is not None for r in self._responses.values()) and receipt_router is None:
            raise ValueError('declared synthetic receipt requires explicit authoritative router')
        self._router = receipt_router; self._workspace = capacity.workspace_bytes(65536)
        memory._reserve(self._workspace)
        return self

    async def _invoke(self, raw, binding, value, ordinal, capture):
        row = self._responses.get((self.invocation, self.lane_id, ordinal))
        if row is None: raise ValueError('frozen fixture response exhausted; no generated fallback')
        if row['synthetic_receipt_sha256'] is not None:
            await self._router.settle(binding, row['synthetic_receipt_sha256'], call_ordinal=ordinal)
        request = json.loads(raw)
        if 'images' in request:
            decoder = asyncio.create_task(asyncio.to_thread(_validate_png, request['images'][0], self.capacity, capture))
            try: await asyncio.shield(decoder)
            except asyncio.CancelledError:
                await _finish(asyncio.gather(decoder, return_exceptions=True)); raise
        elif capture is not None:
            raise ValueError('text fixture cannot consume visual capture mapping')
        await asyncio.sleep(self.adapter['config']['delay_ms']/1000)
        return dict(text=row['text'], usage=None if row['usage'] is None else dict(row['usage']))

    async def _close(self):
        try: await super()._close()
        finally:
            self._responses.clear(); self._memory._release(self._workspace)

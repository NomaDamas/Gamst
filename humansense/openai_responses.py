"""Opt-in direct OpenAI Responses HTTP/1.1; exactly one physical attempt."""
import asyncio
import hashlib
import json
import re
import ssl
import time
import uuid
from .native_entry import BoundCompletion,PreparedCall,NativeEntryBinding,encoded,_id
from .openai_responses_codec import prepare_body,response_projection,normalize,PROJECTION,MAX_RESPONSE
from .responses_evidence import anchor
from .streams import _finish

TRANSPORT='gamst.asyncio.http1.responses.v1'
_SELECTED={'x-request-id','openai-organization','openai-processing-ms','openai-version','content-type','retry-after',
    'x-ratelimit-limit-requests','x-ratelimit-limit-tokens','x-ratelimit-remaining-requests','x-ratelimit-remaining-tokens',
    'x-ratelimit-reset-requests','x-ratelimit-reset-tokens','x-ratelimit-limit-project-tokens',
    'x-ratelimit-remaining-project-tokens','x-ratelimit-reset-project-tokens'}


class OpenAIResponsesClient(BoundCompletion):
    def __init__(self,*,api_key,model_revision,account_scope,project_id,evidence_session,max_output_tokens=1024,timeout=30):
        for v in (model_revision,account_scope,project_id):_id(v)
        if type(api_key) is not str or not 1<=len(api_key)<=4096 or not api_key.isascii() or any(ord(c)<33 or ord(c)>126 for c in api_key):raise ValueError('explicit bounded transport credential required')
        if type(max_output_tokens) is not int or not 1<=max_output_tokens<=4096:raise ValueError('output token bound1..4096 required')
        if type(timeout) not in (int,float) or not .01<=timeout<=300:raise ValueError('timeout0.01..300 required')
        self.provider='openai';self.model_revision=model_revision;self.account_scope=account_scope;self.project_id=project_id
        self._key=api_key;self._sink=evidence_session;self._tokens=max_output_tokens;self._timeout=timeout
        self._host='api.openai.com';self._port=443;self._fixture=False
        self._lease=None;self._closed=False;self._close_task=None

    @classmethod
    def _loopback_fixture(cls,*,port,**kwargs):
        if type(port) is not int or not 1<=port<=65535:raise ValueError('explicit loopback port required')
        result=cls(**kwargs);result._host='127.0.0.1';result._port=port;result._fixture=True
        if not result.account_scope.startswith('fixture') or not result.project_id.startswith('fixture'):
            raise ValueError('explicit synthetic account/project required')
        return result

    def prepare(self,request):
        if self._closed or self._lease is not None:raise RuntimeError('client closed or occupied; no hidden queue')
        body,sdk_hash=prepare_body(request,self.provider,self.model_revision,self._tokens)
        lease=_Prepared(self,body,sdk_hash);self._lease=lease
        return lease

    async def _close(self):
        lease=self._lease
        if lease is not None:lease.invalidate();await lease.release()
        self._key=None

    async def close(self):
        self._closed=True
        if self._lease is not None and self._lease.owns_task(asyncio.current_task()):
            self._lease.invalidate();raise RuntimeError('transport cannot join its own client')
        if self._close_task is None:self._close_task=asyncio.create_task(self._close())
        try:await asyncio.shield(self._close_task)
        except asyncio.CancelledError:await _finish(self._close_task);raise


class _Prepared(PreparedCall):
    def __init__(self,client,body,sdk_hash):
        self.client=client;self.body=body;self.sdk_hash=sdk_hash;self.token=uuid.uuid4().hex
        self._task=None;self._release_task=None;self._invalid=False;self._started=False;self._terminal=False;self.possible_send=False
        self.writer=None;self._connect=None;self.raw=bytearray()
        self.ordinal,self.raw_path,self.raw_fd=client._sink.reserve(self)
        self.manifest=dict(schema_version=1,kind='openai_responses_transport',origin='synthetic_fixture' if client._fixture else 'direct_transport_observed',
            projection_version=PROJECTION,transport_version=TRANSPORT,opaque_financial_token=self.token,entry_binding=None,
            declared_account_scope=client.account_scope,declared_project=client.project_id,provider=client.provider,
            requested_model=client.model_revision,returned_model=None,sdk_request_sha256=sdk_hash,wire_sha256=hashlib.sha256(body).hexdigest(),wire_bytes=len(body),
            client_request_id=('fixture_' if client._fixture else '')+uuid.uuid4().hex,server_request_id=None,response_id=None,response_status=None,http_status=None,
            selected_headers={},raw_response=dict(path=None,sha256=None,bytes=0,complete=False,truncated=False),usage_raw_locator=None,
            normalized_usage=None,milestones=[],terminal=None,failure_category=None,remote_acceptance='unknown',consumption_basis='unknown',money_basis='unknown',
            actual_cost=None,receipt_authenticity_verified=False,whole_pipeline_cost_verified=False,created_at=None,
            clock_error_bound_ns=None,billing_time_assignment='ambiguous_unverified_host_clock',count_bucket_verified=False,clock_jump_detected=False)
        self.mark('prepared')

    def owns_task(self,task):return task is not None and task in (self._task,self._connect,self._release_task)

    def mark(self,event):
        sample=anchor();previous=self.manifest['milestones']
        if previous:
            first=previous[0]['clock'];lo=first['utc_ns']-first['monotonic_after_ns'];hi=first['utc_ns']-first['monotonic_before_ns']
            if sample['utc_ns']-sample['monotonic_after_ns']>hi+1000000 or sample['utc_ns']-sample['monotonic_before_ns']<lo-1000000:self.manifest['clock_jump_detected']=True
        if len(previous)>=8:raise RuntimeError('bounded transport milestones exhausted')
        previous.append(dict(event=event,clock=sample))

    def invalidate(self):
        self._invalid=True
        if self._task is not None:self._task.cancel()
        if self._task is None:self.body=b''

    async def invoke(self,binding):
        if self._started or self._invalid or self.client._closed or self.client._sink._closing:raise RuntimeError('prepared call admission closed')
        if type(binding) is not NativeEntryBinding:raise ValueError('actual typed native entry required')
        value=binding.snapshot()
        if (value['owner_kind']!=self.client._sink.owner_kind or encoded(value['owner'])!=self.client._sink._owner or
            (value['provider'],value['model_revision'],value['prompt_sha256'])!=(self.client.provider,self.client.model_revision,self.sdk_hash)):
            raise ValueError('native entry owner/provider/request differs from prepared send')
        if value['role'] not in ('text_llm','small_vlm','strong_vlm'):raise ValueError('semantic native entry required')
        self._started=True;self.manifest['entry_binding']=value;self.mark('entered')
        self._task=asyncio.create_task(self._run())
        try:return await asyncio.shield(self._task)
        except asyncio.CancelledError:
            self._task.cancel();await _finish(asyncio.gather(self._task,return_exceptions=True));raise

    async def _run(self):
        outcome='error'
        try:
            task=asyncio.create_task(self._exchange())
            try:
                done,_=await asyncio.wait((task,),timeout=self.client._timeout)
                if not done:raise TimeoutError('transport deadline exceeded')
                result=task.result();outcome='completed';return result
            finally:
                if not task.done():task.cancel()
                await _finish(asyncio.gather(task,return_exceptions=True))
        except asyncio.CancelledError:outcome='cancelled';raise
        except TimeoutError:outcome='timeout';raise
        except Exception:
            # Do not surface arbitrary peer errors, echoed bodies or credentials.
            raise ValueError('Responses transport or projection failed') from None
        finally:
            self.manifest['terminal']=outcome
            if outcome!='completed':self.manifest['failure_category']=outcome
            self.mark('terminal');self.body=b''
            self._terminal=True
            try:self.client._sink.finish(self,self.manifest,bytes(self.raw))
            finally:self.raw.clear()

    async def _exchange(self):
        client=self.client;context=None
        if not client._fixture:
            context=ssl.create_default_context();context.set_alpn_protocols(['http/1.1'])
        self._connect=asyncio.create_task(asyncio.open_connection(client._host,client._port,
            ssl=context,server_hostname='api.openai.com' if context else None,limit=16385))
        try:
            try:reader,self.writer=await asyncio.shield(self._connect)
            except BaseException:
                # Late handles must be collected and closed, never retried.
                try:
                    await _finish(self._connect)
                    reader,self.writer=self._connect.result()
                except BaseException:pass
                raise
            if self._invalid or client._closed or client._sink._closing:raise asyncio.CancelledError()
            headers=(f'POST /v1/responses HTTP/1.1\r\nHost: api.openai.com\r\nAuthorization: Bearer {client._key}\r\n'
                f'OpenAI-Project: {client.project_id}\r\nContent-Type: application/json\r\nAccept: application/json\r\nAccept-Encoding: identity\r\n'
                f'Connection: close\r\nContent-Length: {len(self.body)}\r\nX-Client-Request-Id: {self.manifest["client_request_id"]}\r\n\r\n').encode('ascii')
            if len(headers)>16384:raise ValueError('request headers exceed bound')
            self.possible_send=True;self.mark('possible_send')
            self.writer.write(headers)
            for offset in range(0,len(self.body),65536):self.writer.write(self.body[offset:offset+65536]);await self.writer.drain()
            await self.writer.drain()
            await self._read(reader)
            value,metadata=response_projection(bytes(self.raw),client.model_revision);self.manifest.update(metadata)
            if self.manifest['http_status']!=200:raise ValueError('HTTP status not supported')
            if client._fixture and (not (metadata['response_id'] or '').startswith('resp_fixture_') or self.manifest['server_request_id'] is not None and not self.manifest['server_request_id'].startswith('req_fixture_')):
                raise ValueError('fixture response identity must be visibly synthetic')
            result=normalize(value,client.model_revision)
            self.manifest['normalized_usage']=result['usage']
            if result['usage'] is not None:self.manifest['consumption_basis']='response_reported'
            return result
        finally:
            if self.writer is not None:
                self.writer.close()
                try:await _finish(asyncio.create_task(self.writer.wait_closed()))
                except (ConnectionError,OSError):pass
            self._connect=None;self.writer=None

    async def _read(self,reader):
        block=await reader.readuntil(b'\r\n\r\n')
        if len(block)>16384:raise ValueError('response headers exceed16KiB')
        lines=block[:-4].split(b'\r\n')
        if len(lines)>65 or any(len(line)>8192 for line in lines):raise ValueError('header line bound exceeded')
        if re.fullmatch(rb'HTTP/1\.1 [1-5][0-9][0-9](?: [\x20-\x7e]*)?',lines[0]) is None:raise ValueError('HTTP1.1 status required')
        status=int(lines[0][9:12]);headers={}
        for line in lines[1:]:
            name,sep,value=line.partition(b':')
            if not sep or re.fullmatch(rb'[!#$%&\x27*+.^_`|~0-9A-Za-z-]+',name) is None:raise ValueError('invalid response header')
            name=name.decode().lower();value=value.strip(b' \t')
            if any(x<32 or x>126 for x in value) or name in headers:raise ValueError('duplicate or invalid response header')
            headers[name]=value.decode('ascii')
        self.manifest['http_status']=status
        selected={k:v for k,v in headers.items() if k in _SELECTED}
        if any(len(v)>1024 for v in selected.values()) or len(encoded(selected))>8192:raise ValueError('selected header quota exceeded')
        self.manifest['selected_headers']=selected
        rid=headers.get('x-request-id')
        if rid is not None:_id(rid)
        self.manifest['server_request_id']=rid;self.manifest['remote_acceptance']='response_observed';self.mark('headers')
        if status<200 or 300<=status<400:raise ValueError('interim responses and redirects unsupported')
        if headers.get('content-encoding','identity')!='identity':raise ValueError('response compression unsupported')
        length=headers.get('content-length');transfer=headers.get('transfer-encoding')
        if transfer is not None and length is not None:raise ValueError('ambiguous HTTP framing')
        async def append(count):
            while count:
                piece=await reader.read(min(count,65536))
                if not piece:raise ValueError('truncated HTTP body')
                count-=len(piece);self.raw.extend(piece)
        if transfer is not None:
            if transfer!='chunked':raise ValueError('unsupported transfer encoding')
            chunks=0
            while True:
                line=await reader.readuntil(b'\r\n');chunks+=1
                if len(line)>18 or chunks>16384 or re.fullmatch(rb'[0-9a-fA-F]{1,8}\r\n',line) is None:raise ValueError('bounded plain chunk size required')
                count=int(line[:-2],16)
                if not count:
                    if await reader.readexactly(2)!=b'\r\n':raise ValueError('response trailers unsupported')
                    break
                if len(self.raw)+count>MAX_RESPONSE:
                    await append(MAX_RESPONSE-len(self.raw));self.manifest['raw_response']['truncated']=True;raise ValueError('response exceeds1MiB')
                await append(count)
                if await reader.readexactly(2)!=b'\r\n':raise ValueError('chunk delimiter required')
        elif length is not None:
            if re.fullmatch('[0-9]{1,10}',length) is None:raise ValueError('bounded content length required')
            count=int(length)
            await append(min(count,MAX_RESPONSE))
            if count>MAX_RESPONSE:self.manifest['raw_response']['truncated']=True;raise ValueError('response exceeds1MiB')
        else:
            while True:
                piece=await reader.read(min(65536,MAX_RESPONSE-len(self.raw)+1))
                if not piece:break
                if len(self.raw)+len(piece)>MAX_RESPONSE:
                    self.raw.extend(piece[:MAX_RESPONSE-len(self.raw)]);self.manifest['raw_response']['truncated']=True;raise ValueError('response exceeds1MiB')
                self.raw.extend(piece)
        self.manifest['raw_response']['complete']=True;self.mark('body_complete')

    async def _release(self):
        self.invalidate()
        if self._task is not None:await asyncio.gather(self._task,return_exceptions=True)
        if not self._terminal:
            self.manifest['terminal']='not_entered';self.mark('terminal');self.body=b'';self._terminal=True
            self.client._sink.finish(self,self.manifest,b'')
        if self.client._lease is self:self.client._lease=None

    async def release(self):
        if self.owns_task(asyncio.current_task()):self.invalidate();raise RuntimeError('prepared call cannot join itself')
        if self._release_task is None:self._release_task=asyncio.create_task(self._release())
        try:await asyncio.shield(self._release_task)
        except asyncio.CancelledError:await _finish(self._release_task);raise

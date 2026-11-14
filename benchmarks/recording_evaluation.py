"""Recording-owned actual callback entry; all financial exposure remains unknown."""
import asyncio
from collections import Counter
import threading
import struct
import hashlib
import time

from humansense.recording_scope import RecordingScope, RecordingScopeOwner
from humansense.streams import _finish
from .recording_native_schema import (validate_recording_scope_protocol, run_context,
    digest, sha, integer, keys, snapshot, REASONS, preflight_recording_collection, preflight_native_capacity)


class RecordingEvaluationSession(RecordingScopeOwner):
    @classmethod
    def from_frozen(cls, declarations, writer, *, run_nonce, execution_ordinal,
                    retain_evaluation_hashes=False, retain_financial_documents=False, collection_declarations=None):
        if retain_evaluation_hashes is not True:
            raise PermissionError('separate external evaluation hash retention consent required')
        self=cls(); self.declarations=validate_recording_scope_protocol(declarations)
        bundles=[self.declarations] if collection_declarations is None else collection_declarations
        self.preflight=(preflight_native_capacity(self.declarations['collection'])
            if collection_declarations is None and len(self.declarations['collection']['recordings'])>1
            else preflight_recording_collection(bundles))
        if self.preflight['collection_sha256']!=digest(self.declarations['collection']):
            raise ValueError('collection preflight belongs to another frozen collection')
        if not any(digest(v)==digest(self.declarations) for v in bundles):
            raise ValueError('current original recording absent from collection preflight')
        self.context=run_context(self.declarations,run_nonce,execution_ordinal)
        if writer.context!=self.context: raise ValueError('writer belongs to another recording execution')
        writer.bind_capacity(self.declarations['collection']['capacity'])
        q=self.declarations['scope_protocol']
        if type(retain_financial_documents) is not bool or retain_financial_documents!=q['retain_financial_documents']:
            raise PermissionError('financial retention must match explicit frozen consent')
        self._writer=writer;self._lock=threading.RLock();self._loop=asyncio.get_running_loop()
        c=self.declarations['collection'];self._caps=c['capacity'];self._inv=c['invocations'][execution_ordinal]
        self._bindings=next(a['provider_bindings'] for a in c['arms'] if a['arm']==self._inv['arm'])
        self._scopes=[s for s in q['scopes'] if s['invocation_ordinal']==execution_ordinal]
        self._handles={};self._lanes={};self._backends={};self._workers={};self._worker_ids={}
        self._tickets={};self._work={};self._counter=Counter();self._incomplete=Counter()
        self._reservations=self._entries=self._released=self._finished=self._published=0
        self._work_ordinal=0;self._mapping=None;self._activated=False;self._revoked=False
        self._closed=False;self._terminal=False;self._close_task=None;self._running=set()
        self._entry_table=bytearray(self._caps['max_entries_per_run']*128)
        self._entry_digests=bytearray(self._caps['max_entries_per_run']*32)
        self._claim_states=bytearray(self._caps['max_entries_per_run']);self._settlement_jobs=set()
        return self

    def _handle(self,index,binding):
        integer(index,0,len(self._scopes)-1)
        if binding is not None: integer(binding,0,len(self._bindings)-1)
        key=(index,binding)
        with self._lock:
            if key not in self._handles:
                self._handles[key]=RecordingScope(self,index,
                    digest(['gamst.recording.scope.v1',self.context['nonce'],self.context['scope_protocol_sha256'],index]),
                    object(),binding)
            return self._handles[key]

    def stream_scope(self,binding_index=None): return self._handle(0,binding_index)

    def task_scope(self,original_task_id,binding_index=None):
        for scope in self._scopes:
            if scope['task_id']==original_task_id: return self._handle(scope['scope_index'],binding_index)
        raise ValueError('original task belongs to another recording')

    def validate_scope(self,scope):
        with self._lock:
            if (type(scope) is not RecordingScope or scope.owner is not self or
                    self._handles.get((scope.scope_index,scope.binding_index)) is not scope):
                raise ValueError('owner-issued immutable recording scope required')

    def _binding(self,scope):
        self.validate_scope(scope)
        if scope.binding_index is None: raise ValueError('explicit provider binding handle required')
        return self._bindings[scope.binding_index]

    def binding_owner(self,scope):
        self._binding(scope)
        return dict(context=dict(self.context),scope_index=scope.scope_index,binding_index=scope.binding_index)

    def validate_worker_scope(self,scope):
        binding=self._binding(scope)
        if scope.scope_index!=0 or binding['entry_boundary']!='process_adapter':
            raise ValueError('explicit process-adapter stream binding required')

    def validate_worker_configuration(self,scope,provider,revision,config):
        self.validate_worker_scope(scope)
        binding=self._binding(scope)
        if (provider,revision,digest(config))!=(binding['provider'],binding['model_revision'],binding['config_sha256']):
            raise ValueError('actual pre-spawn provider/revision/config differs from frozen binding')

    def bind_semantic_lane(self,controller,binding_index):
        binding=self._binding(self.stream_scope(binding_index))
        if binding['operation_kinds']!=['semantic']: raise ValueError('semantic provider binding required')
        with self._lock:
            if self._activated or self._closed: raise RuntimeError('lane bindings freeze before activation')
            lane=binding['semantic_lane'];key=id(controller)
            previous=self._lanes.get(binding_index)
            if previous is not None and previous is not controller: raise ValueError('controller binding cannot change')
            if any(self._bindings[b]['semantic_lane']==lane and value is not controller for b,value in self._lanes.items()):
                raise ValueError('semantic lane cannot change controller')
            if len({id(v) for v in self._lanes.values()}|{key})>8: raise ValueError('eight semantic controllers maximum')
            self._lanes[binding_index]=controller

    def _emit(self,event,fields,token=None):
        try:return self._writer.emit(event,fields,reservation=token)
        except BaseException:
            self._incomplete['journal_failed']+=1;self._revoked=True
            raise

    def bind_backend(self,binding_index,backend,kind):
        integer(binding_index,0,len(self._bindings)-1)
        if kind not in ('callback','in_process','process'): raise ValueError('explicit backend lifetime kind required')
        with self._lock:
            if self._activated or self._closed: raise RuntimeError('backend identity freezes before activation')
            key=(binding_index,id(backend))
            if key in self._backends: return self._backends[key][1]
            if any(b==binding_index for b,_ in self._backends): raise ValueError('backend binding cannot be rebound')
            native_id=getattr(backend,'_worker_id',None) if kind=='process' else None
            if native_id is not None and native_id in self._worker_ids:
                ordinal=self._worker_ids[native_id]
                if self._workers[ordinal]['binding']!=binding_index: raise ValueError('worker binding differs from actual startup')
            else:
                ordinal=self._register_worker(binding_index,kind,native_id)
                if kind=='callback': self._worker_event(ordinal,'callback_attached')
                self._worker_event(ordinal,'ready')
            self._backends[key]=(backend,ordinal)
            return ordinal

    def validate_backend(self,scope,backend):
        self._binding(scope)
        with self._lock:
            match=self._backends.get((scope.binding_index,id(backend)))
            if match is None or match[0] is not backend: raise ValueError('actual backend differs from frozen binding')
            worker=self._workers[match[1]]
            if worker['closed'] or not worker['ready']: raise ValueError('backend lifetime is not ready')
            return match[1]

    def _worker_for(self,binding):
        matches=[v[1] for (b,_),v in self._backends.items() if b==binding]
        if len(matches)!=1: raise ValueError('one explicitly bound actual backend required')
        worker=self._workers[matches[0]]
        if not worker['ready'] or worker['closed']: raise ValueError('ready open backend required')
        return matches[0]

    def bind_mapping(self,mapping,expected_sha256):
        mapping=snapshot(mapping)
        keys(mapping,('schema_version','source_manifest_sha256','recording_id','origin_capture_ns','duration_ns',
            'sdk_anchor_ms','wall_anchor_ns','monotonic_anchor_ns','scale_numerator','scale_denominator',
            'rounding','max_wall_monotonic_drift_ns'))
        source=self.declarations['source_manifest']
        if (sha(expected_sha256)!=digest(mapping) or mapping['schema_version']!=1 or
                mapping['source_manifest_sha256']!=digest(source) or mapping['recording_id']!=source['recording_id'] or
                mapping['origin_capture_ns']!=source['origin_capture_ns'] or mapping['duration_ns']!=source['duration_ns'] or
                mapping['scale_numerator']!=1 or mapping['scale_denominator']!=1 or mapping['rounding']!='floor_to_sdk_milliseconds'):
            raise ValueError('original one-times source mapping required')
        for field in ('sdk_anchor_ms','wall_anchor_ns','monotonic_anchor_ns','max_wall_monotonic_drift_ns'): integer(mapping[field])
        if mapping['sdk_anchor_ms']!=mapping['wall_anchor_ns']//1_000_000: raise ValueError('mapping clock anchors differ')
        with self._lock:
            if self._mapping is not None or self._closed: raise ValueError('mapping binds exactly once')
            self._emit('mapping_bound',dict(mapping_sha256=expected_sha256));self._mapping=mapping;self._activated=True

    def _admit(self,scope,kind,request,evidence,prompt=None,subject=None,controller=None,request_id=None,refs=(),relationship='root',predecessor=None):
        binding=self._binding(scope)
        with self._lock:
            if self._closed or self._revoked or self._mapping is None: raise RuntimeError('recording admission unavailable')
            if kind not in binding['operation_kinds']: raise ValueError('operation outside frozen provider binding')
            if len(self._tickets)+len(self._work)>=self._caps['max_active_tickets']:
                raise RuntimeError('active recording admission bound reached')
            if self._entries+len([t for t in self._tickets.values() if t.entry is None])>=self._caps['max_entries_per_run']:
                raise RuntimeError('native entry capacity exhausted')
            if self._reservations>=self._caps['max_reservations_per_run']: raise RuntimeError('reservation capacity exhausted')
            sha(request);sha(evidence)
            if relationship not in ('root','retry','escalation'):raise ValueError('explicit operation relationship required')
            predecessor_hash=None
            if relationship=='root':
                if predecessor is not None:raise ValueError('root has no predecessor')
            else:
                from humansense.native_entry import NativeEntryBinding
                if type(predecessor) is not NativeEntryBinding:raise TypeError('original actual predecessor binding required')
                original=predecessor.snapshot();ei=integer(original['entry']['entry_ordinal'],0,self._entries-1)
                row=struct.unpack_from('<IIIIQII32s32s32s',self._entry_table,ei*128)
                predecessor_hash=digest(['gamst.recording.instance.v1',self.context['nonce'],ei])
                if (original['owner_kind']!='recording_native4' or original['owner']['context']!=self.context
                        or original['owner']['scope_index']!=scope.scope_index or row[1]!=scope.scope_index
                        or row[7].hex()!=evidence or original['entry']['instance_sha256']!=predecessor_hash
                        or original['entry']['entered_event_sha256']!=self._entry_digests[ei*32:(ei+1)*32].hex()):
                    raise ValueError('retry/escalation predecessor crosses original owner/scope/evidence')
            if prompt is not None: sha(prompt)
            # Ref records consume explicit extra credits, never eat terminal reserve.
            token=self._writer.reserve(8+len(refs));ordinal=self._reservations;self._reservations+=1
            ticket=_Call(self,ordinal,scope,token,subject,controller,request_id)
            ticket.request_sha256=request;ticket.evidence_sha256=evidence;ticket.prompt_sha256=prompt
            self._tickets[ordinal]=ticket
            try:
                self._emit('reserved',dict(ticket=ordinal,scope=scope.scope_index,binding=scope.binding_index,
                    operation_kind=kind,phase='unknown',relationship=relationship,predecessor=predecessor_hash,
                    request_sha256=request,evidence_sha256=evidence,prompt_sha256=prompt),token)
                for ref in refs:self._emit('evidence_ref',dict(ticket=ordinal,reference=ref),token)
            except BaseException:
                ticket._revoke();raise
            return ticket

    def reserve_call(self,controller,request_id,*,provider,model_revision,prompt_sha256,evaluation_scope,relationship='root',predecessor=None):
        binding=self._binding(evaluation_scope)
        if (binding['provider'],binding['model_revision'])!=(provider,model_revision) or self._lanes.get(evaluation_scope.binding_index) is not controller:
            raise ValueError('actual controller/provider differs from explicit binding')
        state=controller.snapshot();pending=next((r for r in state['requests'] if r['id']==request_id),None)
        if state['closed'] or pending is None or pending['state']!='pending': raise ValueError('original pending controller request required')
        decision=pending['decision'];evidence=decision['evidence']
        if decision['desired_route']!=binding['role']: raise ValueError('provider role differs from original route')
        return self._admit(evaluation_scope,'semantic',decision['decision_sha256'],evidence['record_sha256'],
            prompt_sha256,evidence['subject_id'],controller,request_id,relationship=relationship,predecessor=predecessor)

    def reserve_operation(self,binding_index,evaluation_scope,operation_kind,capture_refs,request_sha256,evidence_sha256):
        if evaluation_scope.binding_index!=binding_index or evaluation_scope.scope_index!=0:
            raise ValueError('direct source operation requires explicit stream binding')
        refs=snapshot(capture_refs,16384)
        if type(refs) is not list or len(refs)>16: raise ValueError('bounded capture locators required')
        records=self.declarations['source_manifest']['records'];sources=self.declarations['source_manifest']['sources']
        for ref in refs:
            keys(ref,('kind','source_index','capture_ordinal','record_sha256','mapping_sha256'))
            si=integer(ref['source_index'],0,len(sources)-1);ri=integer(ref['capture_ordinal'],0,len(records)-1)
            if (ref['kind']!='capture' or records[ri]['source_id']!=sources[si]['source_id'] or
                    ref['record_sha256']!=digest(records[ri]) or self._mapping is None or ref['mapping_sha256']!=digest(self._mapping)):
                raise ValueError('capture locator differs from immutable source')
        expected=digest(dict(binding=binding_index,operation_kind=operation_kind,capture_refs=refs))
        if request_sha256!=expected: raise ValueError('source request descriptor hash differs')
        return self._admit(evaluation_scope,operation_kind,request_sha256,evidence_sha256,refs=refs)

    async def attach_declared_receipt(self,entry_binding,receipt,source_document):
        from humansense.native_entry import NativeEntryBinding
        if type(entry_binding) is not NativeEntryBinding:raise TypeError('actual immutable native entry binding required')
        value=entry_binding.snapshot();receipt=snapshot(receipt,65536)
        if self.declarations['scope_protocol']['retain_financial_documents'] is not True:
            raise PermissionError('separate financial document retention not consented')
        if type(source_document) is not bytes or not 1<=len(source_document)<=1024**2:
            raise ValueError('bounded exact original receipt source required')
        with self._lock:
            if self._closed or self._terminal or len(self._settlement_jobs)>=4:raise RuntimeError('declared receipt admission closed/bounded')
            ei=integer(value['entry']['entry_ordinal'],0,self._entries-1)
            row=struct.unpack_from('<IIIIQII32s32s32s',self._entry_table,ei*128)
            scope=self._handle(row[1],row[2]);binding=self._binding(scope)
            expected=dict(schema_version=1,owner_kind='recording_native4',owner=self.binding_owner(scope),
                entry=dict(ticket_ordinal=row[0],entry_ordinal=ei,instance_sha256=digest(['gamst.recording.instance.v1',self.context['nonce'],ei]),
                    request_decision_sha256=row[8].hex(),entered_event_sha256=self._entry_digests[ei*32:(ei+1)*32].hex()),
                provider=binding['provider'],model_revision=binding['model_revision'],role=binding['role'],
                prompt_sha256=None if row[9]==b'\0'*32 else row[9].hex())
            if value!=expected or self._claim_states[ei]:raise ValueError('original entry binding or unique declared receipt required')
            keys(receipt,('schema_version','kind','origin','money_basis','provider','revision','account_scope','invoice_id',
                'source_document_sha256','source_document_bytes','lines'))
            if (receipt['schema_version']!=1 or type(receipt['schema_version']) is not int or receipt['kind']!='recording_declared_receipt'
                    or receipt['origin'] not in ('application_declared','synthetic_fixture') or receipt['money_basis']!='unknown'
                    or (receipt['provider'],receipt['revision'])!=(binding['provider'],binding['model_revision'])
                    or receipt['source_document_sha256']!=hashlib.sha256(source_document).hexdigest()
                    or receipt['source_document_bytes']!=len(source_document)):
                raise ValueError('exact declared source/provider evidence required')
            card=self.declarations.get('cost_card');cost=self.declarations['plan']['comparison']['cost']
            if cost.get('mode')!='itemized' or card!=cost.get('rate_card'):raise ValueError('original complete cost card required')
            required={rid for rid,rate in card.items() if (rate['provider'],rate['revision'])==(binding['provider'],binding['model_revision'])}
            if not required or type(receipt['lines']) is not list or not 1<=len(receipt['lines'])<=64:raise ValueError('bounded complete declared provider lines required')
            for line in receipt['lines']:
                keys(line,('line_id','quantities','claim_sha256'))
                if type(line['quantities']) is not dict or set(line['quantities'])!=required:raise ValueError('complete original rate quantities required')
                for quantity in line['quantities'].values():integer(quantity,0,10**12)
            token=self._writer.reserve(1);self._claim_states[ei]=1
            task=asyncio.create_task(self._attach(ei,value,receipt,source_document,token));self._settlement_jobs.add(task)
        try:
            try:return await asyncio.shield(task)
            except asyncio.CancelledError:await _finish(task);raise
        finally:self._settlement_jobs.discard(task)

    async def _attach(self,ei,value,receipt,source_document,token):
        try:
            await self._writer.write_receipt_source(source_document)
            artifact=await self._writer.write_receipt_attachment(receipt)
            page=dict(schema_version=1,kind='recording_declared_claims',receipt_sha256=artifact['sha256'],
                owner_instance=value['entry']['instance_sha256'],claims=[dict(claim_sha256=line['claim_sha256'],line_index=n) for n,line in enumerate(receipt['lines'])])
            claim=await self._writer.write_claims_page(page)
            self._emit('settlement_ref',dict(entry_ordinal=ei,receipt_sha256=artifact['sha256'],claims_page_sha256=claim['sha256']),token)
            return dict(receipt_sha256=artifact['sha256'],claims_page_sha256=claim['sha256'],unknown_liability=True)
        except BaseException:
            self._incomplete['journal_failed']+=1;raise
        finally:self._writer.release(token)

    def invalidate(self,reason,subject_id=None):
        if reason not in ('pause','reset','delete','consent_revoked','close'): raise ValueError('known privacy lifecycle required')
        with self._lock:
            if self._terminal:return
            if subject_id is None:self._revoked=True
            affected=0
            for ticket in list(self._tickets.values()):
                if ticket.subject is not None and (subject_id is None or ticket.subject==subject_id):
                    affected+=1;ticket.subject=None;ticket.controller=None;ticket.request_id=None
                    if ticket.entry is None:ticket._revoke()
            self._emit('private_invalidated',dict(reason=reason,affected=affected))

    def validate_stream_binding(self,scope,source_id,sensor_id,spec):
        binding=self._binding(scope)
        if scope.scope_index!=0 or binding['entry_boundary']!='in_process_backend' or 'semantic' in binding['operation_kinds']:
            raise ValueError('explicit in-process stream observation binding required')
        source=next((v for v in self.declarations['source_manifest']['sources'] if v['source_id']==source_id),None)
        if (source is None or source['sensor_id']!=sensor_id or source_id not in binding['source_ids'] or
                (spec.provider,spec.version)!=(binding['provider'],binding['model_revision'])):
            raise ValueError('actual source/provider differs from frozen stream binding')
        with self._lock:
            if self._activated or self._closed or self._revoked:raise RuntimeError('source bindings freeze before activation')

    def validate_stream_configuration(self,scope,config):
        binding=self._binding(scope)
        config=snapshot(config,65536)
        if type(config) is not dict or digest(config)!=binding['config_sha256']:
            raise ValueError('explicit source configuration differs from frozen binding')
        return config

    def validate_stream_capture(self,scope,metadata,reference):
        self._binding(scope);keys(metadata,('source_id','sensor_id','modality','sequence','capture_ms','sample_rate','payload_bytes','payload_sha256','shape'))
        reference=snapshot(reference,2048)
        keys(reference,('kind','source_index','capture_ordinal','record_sha256','mapping_sha256'))
        source=self.declarations['source_manifest'];si=integer(reference['source_index'],0,len(source['sources'])-1)
        ri=integer(reference['capture_ordinal'],0,len(source['records'])-1);record=source['records'][ri];sensor=source['sources'][si]
        with self._lock:
            if self._mapping is None or self._closed or self._revoked:raise RuntimeError('mapped source admission unavailable')
            if (reference['kind']!='capture' or reference['mapping_sha256']!=digest(self._mapping) or
                    reference['record_sha256']!=digest(record) or record['source_id']!=sensor['source_id'] or
                    metadata!=dict(source_id=sensor['source_id'],sensor_id=sensor['sensor_id'],modality=sensor['modality'],
                        sequence=record['sequence'],capture_ms=self._mapping['sdk_anchor_ms']+record['offset_ns']//1000000,
                        sample_rate=record.get('sample_rate'),payload_bytes=record['payload_bytes'],payload_sha256=record['payload_sha256'],shape=record.get('shape')) or
                    sensor['source_id'] not in self._bindings[scope.binding_index]['source_ids']):
                raise ValueError('actual capture metadata differs from original immutable record')
        return reference

    def reserve_stream_lifecycle(self,scope):
        binding=self._binding(scope)
        if scope.scope_index!=0 or binding['entry_boundary']!='in_process_backend':
            raise ValueError('actual in-process source lifecycle required')
        with self._lock:
            if self._activated or self._closed or self._revoked:raise RuntimeError('source startup admission closed')
            token=self._writer.reserve(8)
            try:ordinal=self._register_worker(scope.binding_index,'in_process',token=token)
            except BaseException:self._writer.release(token);raise
            return _StreamLifecycle(self,scope,ordinal,token)

    def reserve_work(self,kind,metadata,*,evaluation_scope):
        self.validate_worker_scope(evaluation_scope)
        metadata=snapshot(metadata,4096)
        allowed={'worker_id','profile','configuration_sha256','operation','parent_operation_id'}
        if type(metadata) is not dict or set(metadata)-allowed or type(metadata.get('worker_id')) is not str:
            raise ValueError('explicit bounded actual worker metadata required')
        if len(metadata['worker_id'])>128: raise ValueError('worker identity bound')
        binding=evaluation_scope.binding_index
        with self._lock:
            if self._terminal or self._closed: raise RuntimeError('recording work admission closed')
            if len(self._tickets)+len(self._work)+2>self._caps['max_active_tickets']:
                raise RuntimeError('active worker ticket bound')
            native_id=metadata['worker_id']
            if kind=='worker_lifecycle':
                if native_id in self._worker_ids or metadata.get('configuration_sha256')!=self._bindings[binding]['config_sha256']:
                    raise ValueError('actual worker configuration/lifetime mismatch')
                # Startup and residency are distinct eight-credit work tickets.
                token=self._writer.reserve(16)
                try:ordinal=self._register_worker(binding,'process',native_id,token)
                except BaseException:self._writer.release(token);raise
            elif kind=='worker_operation':
                if native_id not in self._worker_ids: raise ValueError('unregistered actual worker lifetime')
                ordinal=self._worker_ids[native_id]
                if self._workers[ordinal]['binding']!=binding: raise ValueError('worker changed binding')
                if metadata.get('operation') not in ('start','process','reset','finish','endpoint','close'):
                    raise ValueError('known worker operation required')
                token=self._writer.reserve(8)
            else: raise ValueError('known owned worker ticket kind required')
            parent=metadata.get('parent_operation_id')
            if parent is not None:
                try:ticket=self._tickets[int(parent)]
                except (ValueError,KeyError,TypeError):
                    self._writer.release(token);raise ValueError('original active native parent required')
                if ticket.id!=parent or ticket.entry is None or ticket.scope.binding_index!=binding:
                    self._writer.release(token);raise ValueError('worker operation rebound to foreign native entry')
                parent=ticket.entry
            return _ProcessWork(self,kind,metadata,ordinal,binding,parent,token)

    def _register_worker(self,binding,kind,native_id=None,token=None):
        if len(self._workers)>=1024 or sum(not w['closed'] for w in self._workers.values())>=256:
            raise RuntimeError('bounded worker lifetime inventory exhausted')
        ordinal=len(self._workers)
        self._emit('worker_registered',dict(worker_ordinal=ordinal,binding=binding,kind=kind),token)
        self._workers[ordinal]=dict(binding=binding,kind=kind,ready=False,closed=False,milestones=set())
        if native_id is not None:self._worker_ids[native_id]=ordinal
        return ordinal

    def _worker_event(self,ordinal,milestone,token=None):
        worker=self._workers[ordinal]
        if milestone in worker['milestones']: raise ValueError('duplicate worker milestone')
        self._emit('worker_milestone',dict(worker_ordinal=ordinal,milestone=milestone),token)
        worker['milestones'].add(milestone)
        if milestone=='ready':worker['ready']=True
        if milestone in ('callback_detached','provider_closed','reap_confirmed'):worker['closed']=True
        if milestone=='startup_failed' and 'handle_acquired' not in worker['milestones']:worker['closed']=True

    def backend_closed(self,binding_index,backend):
        with self._lock:
            key=(binding_index,id(backend));match=self._backends.get(key)
            if match is None or match[0] is not backend: raise ValueError('owned bound backend required')
            ordinal=match[1];worker=self._workers[ordinal]
            if worker['kind']=='process': raise ValueError('actual process reap hook required')
            if any(t.worker==ordinal for t in self._tickets.values()): raise RuntimeError('backend still owns active operations')
            self._worker_event(ordinal,'callback_detached' if worker['kind']=='callback' else 'provider_closed')

    async def close(self,cleanup_finished=True):
        if self._close_task is None:self._close_task=asyncio.create_task(self._close(cleanup_finished))
        try:return await asyncio.shield(self._close_task)
        except asyncio.CancelledError:
            await _finish(asyncio.gather(self._close_task,return_exceptions=True));raise

    async def _close(self,cleanup_finished):
        try:
            return await self._close_impl(cleanup_finished)
        except BaseException:
            try:await self._writer.abort()
            finally:self._terminal=True
            raise

    async def _close_impl(self,cleanup_finished):
        with self._lock:
            self._closed=True
            settlement_jobs=tuple(self._settlement_jobs)
            for ticket in list(self._tickets.values()):
                if ticket.entry is None:ticket._revoke()
        if settlement_jobs:
            await _finish(asyncio.gather(*settlement_jobs,return_exceptions=True))
        # Active callbacks remain owned by their SDK adapter/stream; caller must
        # close those before claiming cleanup, rather than cancelling threads here.
        with self._lock:
            if self._tickets or self._work or not cleanup_finished:self._incomplete['cleanup_incomplete']+=1
            if any(not w['closed'] for w in self._workers.values()):self._incomplete['cleanup_incomplete']+=1
            summary=dict(mapping_sha256=None if self._mapping is None else digest(self._mapping),
                reservation_count=self._reservations,released_count=self._released,
                outstanding_tickets=sum(t.entry is None for t in self._tickets.values()),
                outstanding_entries=self._entries-self._finished,outstanding_work=len(self._work),
                registered_workers=len(self._workers),unclosed_workers=sum(not w['closed'] for w in self._workers.values()),
                unsettled_entries=self._entries,counter_pages=[],incomplete_counts=dict(self._incomplete))
        counters=[]
        for scope in self._scopes:
            for b in range(len(self._bindings)):
                n=scope['scope_index'];counters.append(dict(scope=n,binding=b,
                    entered=self._counter[n,b,'entered'],finished=self._counter[n,b,'finished'],
                    published=self._counter[n,b,'published'],unsettled=self._counter[n,b,'entered']))
        try:
            for offset in range(0,len(counters),512):
                page=dict(schema_version=1,kind='recording_counters',nonce=self.context['nonce'],
                    ordinal=offset//512,counters=counters[offset:offset+512])
                ref=await self._writer.write_counter_page(page);summary['counter_pages'].append(ref)
            result=await self._writer.close(summary,cleanup_finished=bool(cleanup_finished))
            self._terminal=True;return result
        except BaseException:
            await self._writer.abort();self._terminal=True;raise


class _Call:
    def __init__(self,session,ordinal,scope,token,subject,controller,request_id):
        self.session=session;self.ordinal=ordinal;self.id=str(ordinal);self.scope=scope;self.token=token
        self.subject=subject;self.controller=controller;self.request_id=request_id
        self.entry=None;self.worker=None;self.instance_id=None;self._dispatched=False;self._revoked=False
        self._finished=False;self._released=False;self._result=None

    def dispatched(self):
        s=self.session
        with s._lock:
            if self._dispatched or self._revoked or self._released: raise RuntimeError('unique live dispatch required')
            if self.controller is not None:
                pending=next((r for r in self.controller.snapshot()['requests'] if r['id']==self.request_id),None)
                if pending is None or pending['state']!='dispatched': raise ValueError('controller did not dispatch original request')
            s._emit('dispatched',dict(ticket=self.ordinal),self.token);self._dispatched=True
            self.controller=None;self.request_id=None

    def _enter(self):
        s=self.session
        with s._lock:
            if self._revoked or self._released or s._closed or s._revoked or self.entry is not None:
                raise RuntimeError('revoked recording entry cannot invoke backend')
            if not self._dispatched:self.dispatched()
            worker=s._worker_for(self.scope.binding_index);ordinal=s._entries
            instance=digest(['gamst.recording.instance.v1',s.context['nonce'],ordinal])
            event=s._emit('entered',dict(ticket=self.ordinal,entry_ordinal=ordinal,instance=instance,worker_ordinal=worker),self.token)
            struct.pack_into('<IIIIQII32s32s32s',s._entry_table,ordinal*128,self.ordinal,self.scope.scope_index,
                self.scope.binding_index,worker,event['sequence'],event['chunk_ordinal'],0,
                bytes.fromhex(self.evidence_sha256),bytes.fromhex(self.request_sha256),
                b'\0'*32 if self.prompt_sha256 is None else bytes.fromhex(self.prompt_sha256))
            s._entry_digests[ordinal*32:(ordinal+1)*32]=bytes.fromhex(event['sha256'])
            self.entry=ordinal;self.worker=worker;self.instance_id=instance;s._entries+=1
            s._counter[self.scope.scope_index,self.scope.binding_index,'entered']+=1
            return event

    def _terminal(self,outcome,result=None):
        s=self.session
        with s._lock:
            if self._finished:return
            self._result=None if result is None else digest(snapshot(result,1024**2))
            try:s._emit('finished',dict(entry_ordinal=self.entry,outcome=outcome,result_sha256=self._result),self.token)
            finally:
                self._finished=True;s._finished+=1;s._counter[self.scope.scope_index,self.scope.binding_index,'finished']+=1

    async def invoke(self,callback):
        self._enter();outcome='error';result=None
        try:
            result=await callback();outcome='ok';return result
        except asyncio.CancelledError:outcome='cancelled';raise
        except asyncio.TimeoutError:outcome='timeout';raise
        finally:self._terminal(outcome,result)

    async def invoke_bound(self,callback):
        from humansense.native_entry import _issue_entry_binding
        if self.prompt_sha256 is None:raise ValueError('bound completion requires explicit semantic prompt identity')
        event=self._enter();outcome='error';result=None
        try:
            s=self.session;binding=s._binding(self.scope)
            value=_issue_entry_binding(dict(schema_version=1,owner_kind='recording_native4',owner=s.binding_owner(self.scope),
                entry=dict(ticket_ordinal=self.ordinal,entry_ordinal=self.entry,instance_sha256=self.instance_id,
                    request_decision_sha256=self.request_sha256,entered_event_sha256=event['sha256']),
                provider=binding['provider'],model_revision=binding['model_revision'],role=binding['role'],
                prompt_sha256=self.prompt_sha256))
            result=await callback(value);outcome='ok';return result
        except asyncio.CancelledError:outcome='cancelled';raise
        except asyncio.TimeoutError:outcome='timeout';raise
        finally:self._terminal(outcome,result)

    def invoke_sync(self,callback):
        self._enter();outcome='error';result=None
        try:result=callback();outcome='ok';return result
        finally:self._terminal(outcome,result)

    def finish(self,outcome):
        if self.entry is None:self.release(outcome)
        elif not self._released:self.published(outcome in ('completed','published','accepted','success'))

    def published(self,accepted):
        s=self.session
        with s._lock:
            if self._released:return
            if not self._finished: raise RuntimeError('callback terminal must precede publication')
            try:s._emit('published',dict(entry_ordinal=self.entry,outcome='accepted' if accepted else 'discarded'),self.token)
            finally:
                s._published+=1;s._counter[self.scope.scope_index,self.scope.binding_index,'published']+=1
                self._retire()

    def _retire(self):
        self._released=True;self.subject=None;self.controller=None;self.request_id=None
        self.session._tickets.pop(self.ordinal,None);self.session._writer.release(self.token)

    def _revoke(self):
        if self.entry is not None or self._released:return
        self._revoked=True
        try:self.session._emit('released',dict(ticket=self.ordinal,reason='revoked_before_entry'),self.token)
        finally:self.session._released+=1;self._retire()

    def release(self,reason='unadmitted'):
        s=self.session
        with s._lock:
            if self._released:return
            if self.entry is not None:
                if self._finished:self.published(False)
                return
            self._revoked=True
            value='cancelled_before_entry' if reason=='cancelled' else 'admission_refused'
            try:s._emit('released',dict(ticket=self.ordinal,reason=value),self.token)
            finally:s._released+=1;self._retire()


class _ProcessWork:
    def __init__(self,session,kind,metadata,worker,binding,parent,token):
        self.session=session;self.kind=kind;self.metadata=metadata;self.worker=worker
        self.binding=binding;self.parent=parent;self.token=token;self.active=None
        self.marks=set();self.done=False

    def _start(self,kind):
        s=self.session;ordinal=s._work_ordinal;s._work_ordinal+=1
        s._emit('work_started',dict(work_ordinal=ordinal,worker_ordinal=self.worker,scope=0,
            parent_entry_ordinal=self.parent,binding=self.binding,kind=kind),self.token)
        s._work[ordinal]=self;self.active=ordinal

    def _end(self,outcome):
        if self.active is None:return
        s=self.session;ordinal=self.active
        try:s._emit('work_finished',dict(work_ordinal=ordinal,outcome=outcome),self.token)
        finally:s._work.pop(ordinal,None);self.active=None

    def event(self,name,fields=None):
        s=self.session
        with s._lock:
            if self.done or name in self.marks or fields not in (None,{}):
                raise ValueError('unique structural worker milestone required')
            self.marks.add(name)
            if self.kind=='worker_lifecycle':
                mapping={'spawn_begin':'spawn_started','handle_acquired':'handle_acquired',
                    'handshake_ready':'ready','handshake_failed':'startup_failed','reap_confirmed':'reap_confirmed'}
                if name not in mapping:raise ValueError('known lifecycle milestone required')
                if name=='spawn_begin':self._start('startup')
                s._worker_event(self.worker,mapping[name],self.token)
                if name=='handshake_ready':self._end('ok');self._start('residency')
                if name=='handshake_failed':self._end('error')
                if name=='reap_confirmed':self._end('ok')
            elif name=='operation_started':
                operation=self.metadata['operation']
                self._start('reset' if operation=='reset' else 'close' if operation=='close' else 'transport')
            elif name not in ('operation_finished','invalidated'):
                raise ValueError('known worker operation milestone required')

    def finish(self,outcome):
        s=self.session
        with s._lock:
            if self.done:return
            value={'completed':'ok','closed':'ok','success':'ok','error':'error','failed':'error',
                'cancelled':'cancelled','timeout':'timeout'}.get(outcome)
            if value is None:raise ValueError('known worker outcome required')
            try:self._end(value)
            finally:self.done=True;s._writer.release(self.token)


class _StreamLifecycle:
    def __init__(self,session,scope,worker,token):
        self.session=session;self.scope=scope;self.worker=worker;self.token=token
        self.provider=None;self.startup=None;self.residency=None;self.idle_ticket=None
        self._failed=False;self._closed=False;self._startup_invoked=False

    def _startup_entry(self):
        s=self.session
        with s._lock:
            if self._startup_invoked or self._failed or s._closed or s._revoked:
                raise RuntimeError('revoked source factory cannot enter')
            self._startup_invoked=True;self.startup=_Interval(s,self.scope,self.worker,'startup')
            self.startup.start()

    async def invoke_startup(self,factory):
        self._startup_entry();outcome='error'
        try:self.provider=await factory();outcome='ok';return self.provider
        except asyncio.CancelledError:outcome='cancelled';raise
        finally:self.startup.finish(outcome)

    def invoke_startup_sync(self,factory):
        self._startup_entry();outcome='error'
        try:self.provider=factory();outcome='ok';return self.provider
        finally:self.startup.finish(outcome)

    def ready(self,provider):
        s=self.session
        with s._lock:
            if self._failed or self._closed or not self._startup_invoked or self.provider is not provider:
                raise ValueError('original returned provider required for readiness')
            if any(b==self.scope.binding_index for b,_ in s._backends):raise ValueError('source provider binding already owned')
            s._backends[self.scope.binding_index,id(provider)]=(provider,self.worker)
            s._worker_event(self.worker,'ready',self.token)
            self.residency=_Interval(s,self.scope,self.worker,'residency');self.residency.start()

    def work(self,kind):
        if kind not in ('reset','close'):raise ValueError('explicit provider reset/close work required')
        return _Interval(self.session,self.scope,self.worker,kind)

    def idle(self,active):
        if type(active) is not bool:raise ValueError('explicit idle transition required')
        with self.session._lock:
            if active:
                if self.idle_ticket is not None:return
                if self._closed or self._failed:return
                self.idle_ticket=_Interval(self.session,self.scope,self.worker,'idle');self.idle_ticket.start()
            elif self.idle_ticket is not None:
                self.idle_ticket.finish('ok');self.idle_ticket=None

    def thread_joined(self):
        with self.session._lock:
            if 'thread_joined' not in self.session._workers[self.worker]['milestones']:
                self.session._worker_event(self.worker,'thread_joined',self.token)

    def closed(self):
        s=self.session
        with s._lock:
            if self._closed:return
            self.idle(False)
            if self.residency is not None:self.residency.finish('ok')
            s._worker_event(self.worker,'provider_closed',self.token)
            self._closed=True;s._writer.release(self.token)

    def startup_failed(self):
        s=self.session
        with s._lock:
            if self._failed or self._closed:return
            if self.provider is not None:
                # A returned provider must actually be closed by its source owner.
                # Do not turn cancellation of the factory waiter into no-provider proof.
                return
            self._failed=True
            if self.startup is not None:self.startup.finish('error')
            s._worker_event(self.worker,'startup_failed',self.token)
            s._writer.release(self.token)


class _Interval:
    def __init__(self,session,scope,worker,kind):
        self.session=session;self.scope=scope;self.worker=worker;self.kind=kind
        self.token=session._writer.reserve(8);self.ordinal=None;self.done=False

    def start(self):
        s=self.session
        with s._lock:
            if self.ordinal is not None or self.done:raise RuntimeError('work interval entered twice')
            if s._terminal:raise RuntimeError('immutable native terminal already frozen')
            if len(s._work)+len(s._tickets)>=s._caps['max_active_tickets']:raise RuntimeError('active native work bound')
            n=s._work_ordinal;s._work_ordinal+=1
            s._emit('work_started',dict(work_ordinal=n,worker_ordinal=self.worker,scope=0,parent_entry_ordinal=None,
                binding=self.scope.binding_index,kind=self.kind),self.token)
            s._work[n]=self;self.ordinal=n

    async def invoke(self,callback):
        self.start();outcome='error'
        try:result=await callback();outcome='ok';return result
        except asyncio.CancelledError:outcome='cancelled';raise
        finally:self.finish(outcome)

    def invoke_sync(self,callback):
        self.start();outcome='error'
        try:result=callback();outcome='ok';return result
        finally:self.finish(outcome)

    def finish(self,outcome='ok'):
        s=self.session
        with s._lock:
            if self.done:return
            value={'ok':'ok','completed':'ok','closed':'ok','error':'error','cancelled':'cancelled','timeout':'timeout'}.get(outcome)
            if value is None:raise ValueError('bounded work outcome required')
            try:
                if self.ordinal is not None:s._emit('work_finished',dict(work_ordinal=self.ordinal,outcome=value),self.token)
            finally:
                if self.ordinal is not None:s._work.pop(self.ordinal,None)
                self.done=True;s._writer.release(self.token)

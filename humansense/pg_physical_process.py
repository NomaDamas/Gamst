"""Opt-in original PG native process provider; estimates only, never physical sensors."""
import asyncio
from collections import OrderedDict
from dataclasses import asdict
import hashlib
import math
import uuid

from .pg_artifact import validate_manifest,manifest_digest
from .pg_canonical import validate_canonical
from .pg_canonical_process import PGCanonicalProcess,_deadline
from .pg_generation import PROFILE,PROMPT_SHA,PROMPTS,ANSWERS,HEADER_LIMIT,policy,layout,crop_digest,validate_result
from .pg_physical_worker import header_value
from .sam_audio_artifact import _keys,_json,_encoded,_absolute
from .capabilities import CapabilitySpec,OutputField,SensorMetadata,identifier
from .plugins import InputFrame
from .streams import _finish


def configuration(config):
    config=_json(_encoded(config),8*1024**2)
    _keys(config,('graph','graph_sha','lock','lock_sha','canonical','canonical_sha','stage_root','generation','max_gap_ms'),'PG provider configuration')
    config['graph']=validate_manifest(config['graph'],expected_sha256=config['graph_sha'])
    config['canonical']=validate_canonical(config['canonical'],expected_sha256=config['canonical_sha'])
    if (config['canonical']['graph_manifest_sha256']!=config['graph_sha'] or
        config['canonical']['runtime_lock_sha256']!=config['lock_sha'] or manifest_digest(config['lock'])!=config['lock_sha']):
        raise ValueError('canonical runtime/graph binding mismatch')
    _absolute(config['stage_root']);config['generation']=policy(config['generation'],config['graph']['execution'])
    n=config['max_gap_ms']
    if type(n) is not int or not 1<=n<=600000:raise ValueError('bounded explicit capture gap required')
    return config


def capability(config):
    value=configuration(config)
    return CapabilitySpec('vision.pg_physical','vision.physical_concepts','video',PROFILE,'1.'+manifest_digest(value),
        (OutputField('physical','object'),OutputField('evidence','object'),OutputField('model','object')),
        'frame',freshness_ms=1000,stateful=True,reset_on_gap=False,interpretation='estimate',
        devices=(value['graph']['execution']['device'],),sensor_requirements=('explicit-rgb-object-crop','capture-time-source-metadata'),
        license='adapter:Apache-2.0;PG/assets/canonical-export:operator-must-verify')


class PGPhysicalProcess:
    @classmethod
    async def open(cls,python,config,*,startup_timeout=120,inference_timeout=30):
        if type(inference_timeout) not in (int,float) or not math.isfinite(inference_timeout) or not .01<=inference_timeout<=600:
            raise ValueError('inference timeout must be0.01..600s')
        config=configuration(config);self=cls();self.config=config;self.spec=capability(config)
        self.timeout=inference_timeout;self.closed=False;self._busy=False;self._sequence=0
        self._epoch=uuid.uuid4().hex;self._reset_generation=0;self._states=OrderedDict();self._owner=None
        self._native=await PGCanonicalProcess.open(python,config['graph'],config['graph_sha'],config['lock'],config['lock_sha'],
            config['canonical'],config['canonical_sha'],stage_root=config['stage_root'],startup_timeout=startup_timeout,generation=config['generation'])
        return self

    async def reset(self):
        if self.closed:raise RuntimeError('PG provider closed')
        self._reset_generation+=1;self._states.clear();self._owner=None
        # Reset invalidates an active response without claiming to cancel native
        # compute. The one-inflight guard remains until that operation drains.

    async def finish(self):
        if self.closed:raise RuntimeError('PG provider closed')
        return None

    async def close(self):
        self.closed=True;self._reset_generation+=1;self._states.clear()
        await self._native.close()

    def _capture(self,frame):
        if not isinstance(frame,InputFrame) or frame.modality!='video' or frame.sample_rate is not None or not isinstance(frame.sensor,SensorMetadata):
            raise ValueError('video InputFrame with SensorMetadata required')
        from .pg_generation import unpack_capture
        data,payload=unpack_capture(frame.data)
        shape,crop=data['shape'],data['crop']
        sensor=asdict(frame.sensor);sensor['channels']=list(sensor['channels'])
        binding=dict(timestamp_ms=frame.timestamp_ms,subject_id=frame.subject_id,sensor=sensor,
            worker_epoch=self._epoch,track_provenance=data['track_provenance'])
        header=dict(schema_version=1,operation='infer',request_id=uuid.uuid4().hex,
            generation_sha256=manifest_digest(self.config['generation']),binding=binding,concept=data['concept'],shape=shape,crop=crop,
            pixels_sha256=hashlib.sha256(payload).hexdigest(),crop_sha256=crop_digest(payload,shape,crop),payload_bytes=len(payload))
        return header_value(_json(_encoded(header),HEADER_LIMIT),header['generation_sha256']),payload

    def _admit(self,header):
        b=header['binding'];s=b['sensor']
        owner=(s['source_id'],s['sensor_id'],s['stream_epoch'],s['coordinate_frame'],s['calibration_id'],tuple(header['shape']))
        if self._owner!=owner:self._states.clear();self._owner=owner
        subject=b['subject_id'];concept=header['concept'];stamp=(b['timestamp_ms'],s['capture_sequence'])
        group=self._states.get(subject,{})
        previous=group.get(concept)
        if previous and (stamp[0]<=previous['stamp'][0] or stamp[1]<=previous['stamp'][1]):raise ValueError('duplicate or stale concept capture')
        if previous and stamp[0]-previous['stamp'][0]>self.config['max_gap_ms']:group={}
        if subject not in self._states and len(self._states)>=128:self._states.popitem(last=False)
        self._states[subject]=group;self._states.move_to_end(subject)
        group[concept]=dict(stamp=stamp,status='pending',value=None)
        return subject,concept,stamp

    async def _exchange(self,header,payload):
        process=self._native._process
        if process is None:raise RuntimeError('PG worker unavailable')
        process.stdin.write(_encoded(header)+b'\n');await process.stdin.drain()
        for offset in range(0,len(payload),65536):
            process.stdin.write(payload[offset:offset+65536]);await process.stdin.drain()
        raw=await process.stdout.readline()
        if not raw.endswith(b'\n') or len(raw)>HEADER_LIMIT:raise ValueError('bounded PG response required')
        response=_json(raw,HEADER_LIMIT)
        _keys(response,('status','request_id','request_sha256','canonical_sha256','generation_sha256','result'),'PG response')
        if (response['status']!='generated' or response['request_id']!=header['request_id'] or response['request_sha256']!=manifest_digest(header) or
            response['canonical_sha256']!=self.config['canonical_sha'] or response['generation_sha256']!=header['generation_sha256']):
            raise ValueError('PG response capture/policy mismatch')
        return validate_result(response['result'],header['concept'],self.config['graph']['execution'])

    async def process(self,frame):
        if self.closed:raise RuntimeError('PG provider closed')
        if self._busy:raise RuntimeError('PG accepts one in-flight request, no queue')
        # Admission and ownership are synchronous up to task creation.
        header,payload=self._capture(frame);key=self._admit(header);generation=self._reset_generation
        self._busy=True;task=None
        try:
            loop=asyncio.get_running_loop();deadline=loop.time()+self.timeout
            task=asyncio.create_task(self._exchange(header,payload))
            result,answer=await _deadline(task,deadline-loop.time())
            if loop.time()>=deadline:raise TimeoutError('PG inference deadline exceeded')
            if self.closed or generation!=self._reset_generation:return None
            self._sequence+=1
            subject,concept,stamp=key
            self._states[subject][concept]=dict(stamp=stamp,status='unknown' if answer is None else 'estimated',value=answer)
            output=self._observation(header,result,answer)
            if loop.time()>=deadline:raise TimeoutError('PG inference validation deadline exceeded')
            return output
        except BaseException as initial:
            # A partial write/read/protocol error poisons the only worker. Reap
            # before completing cancellation; late bytes cannot serve a new call.
            try:await _finish(asyncio.create_task(self.close()))
            except BaseException as cleanup:raise initial from cleanup
            finally:
                if task is not None:
                    if not task.done():task.cancel()
                    await _finish(asyncio.create_task(_drain(task)))
            raise
        finally:self._busy=False

    def _observation(self,header,result,answer):
        return self.spec.validate(dict(
            physical=dict(concept=header['concept'],status='unknown' if answer is None else 'estimated',value=answer,
                alternatives=result['alternatives'],confidence=None,interpretation='untrusted_visual_estimate',
                measured=False,quality_validated=False,current_capture=True),
            evidence=dict(**header['binding'],shape=header['shape'],crop=header['crop'],pixels_sha256=header['pixels_sha256'],
                crop_sha256=header['crop_sha256'],request_sha256=manifest_digest(header),delivery_sequence=self._sequence,
                identity_verified=False,clock_verified=False),
            model=dict(canonical_sha256=self.config['canonical_sha'],graph_sha256=self.config['graph_sha'],
                runtime_lock_sha256=self.config['lock_sha'],prompt_sha256=PROMPT_SHA,generation_sha256=header['generation_sha256'],
                input_tokens=result['input_tokens'],output_tokens=None,inference_ms=result['inference_ms'],
                raw_score_kind='uncalibrated_beam_score',cost=None,resource_admission='unavailable',actual_acceptance='incomplete',
                trained_identity_verified=False,reference_parity_verified=False,runtime_compatibility_verified=False,
                capture_timestamp_preserved=True,cache_reused_across_frames=False)))


async def _drain(task):
    await asyncio.gather(task,return_exceptions=True)

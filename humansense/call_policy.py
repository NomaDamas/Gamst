"""Evidence-bound call recommendations; no execution, queue or budget mutation."""
from dataclasses import dataclass
import hashlib
import json
import math
import re

from .capabilities import bounded_json,identifier


def _ms(value):
    if type(value) is not int or not 0<=value<=2**53-1:raise ValueError('bounded integer milliseconds required')
    return value


def _sha(value):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{64}',value):raise ValueError('lowercase SHA256 required')
    return value


def _canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()


@dataclass(frozen=True)
class CallPolicy:
    """Pure policy: caller owns verified lifecycle state and applies recommendations.

    A recommendation is NOT dispatch/success, budget approval, sensor control or
    evidence that a blindspot was inspected. See the required controller follow-up.
    """
    mode: str = 'context_only'
    semantic_interval_ms: int = 5000
    cooldown_ms: int = 1000
    max_age_ms: int = 1000
    change_enter: float = .7
    change_exit: float = .3

    def __post_init__(self):
        if self.mode not in ('context_only','periodic','sentinel','escalation'):raise ValueError('explicit supported call-policy mode required')
        for v,low,high in ((self.semantic_interval_ms,1,3600000),(self.cooldown_ms,0,600000),(self.max_age_ms,1,60000)):
            if type(v) is not int or not low<=v<=high:raise ValueError('invalid bounded policy time interval')
        for v in (self.change_enter,self.change_exit):
            if type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1:raise ValueError('finite change thresholds0..1 required')
        if self.change_exit>=self.change_enter:raise ValueError('exit threshold must be below enter threshold')

    def recommend(self,record,*,now_ms,change_score=0.,change_active=False,
                  last_dispatch_ms=None,last_semantic_capture_ms=None,
                  query=False,needs_visual=False,unknown_candidate=False,
                  conflict_rule=None,visual_reference=None):
        """Recommend against one registered observation plus explicit rule inputs.

        ``conflict_rule`` is an externally validated task-rule identifier, never
        inferred from model self-confidence or text. Change scores are caller's
        measurable rule scores, not probabilities. None means no known success.
        ``last_*`` and hysteresis state must belong to this exact source/epoch;
        this pure evaluator cannot authenticate caller-supplied state.
        """
        now=_ms(now_ms)
        for v in (change_active,query,needs_visual,unknown_candidate):
            if type(v) is not bool:raise ValueError('explicit boolean policy inputs required')
        if type(change_score) not in (int,float) or not math.isfinite(change_score) or not 0<=change_score<=1:raise ValueError('finite measured change score0..1 required')
        for t in (last_dispatch_ms,last_semantic_capture_ms):
            if t is not None and _ms(t)>now:raise ValueError('lifecycle state cannot be in the future')
        if conflict_rule is not None:identifier(conflict_rule,'validated task rule')
        # Hash the bounded record, not its untrusted words or model score meanings.
        record=bounded_json(record,max_bytes=131072)
        if type(record) is not dict or record.get('type')!='observation.signal':raise ValueError('registered observation.signal record required')
        for key in ('id','subject_id'):identifier(record.get(key),key)
        capture=_ms(record.get('timestamp_ms'));observation=record.get('observation')
        if type(observation) is not dict:raise ValueError('observation metadata required')
        capability=observation.get('capability');sensor=observation.get('sensor')
        if type(capability) is not dict or type(sensor) is not dict:raise ValueError('registered capability and source required')
        identifier(capability.get('id'));identifier(sensor.get('source_id'));identifier(sensor.get('sensor_id'))
        epoch=sensor.get('stream_epoch');identifier(epoch,'active stream epoch')
        processing=observation.get('processing_epoch');identifier(processing,'processing epoch')
        for key in ('coordinate_frame','calibration_id'):
            if sensor.get(key) is not None:identifier(sensor[key],key)
        modality=observation.get('modality')
        if modality not in ('video','audio','text','thermal','radar','csi','imu','tactile','environment','events'):raise ValueError('supported observation modality required')
        freshness=capability.get('freshness_ms')
        if type(freshness) is not int or not 0<=freshness<=3600000:raise ValueError('bounded declared freshness required')
        quality=record.get('quality')
        if type(quality) is not dict or quality.get('status') not in ('sufficient','insufficient','unavailable'):raise ValueError('declared observation quality required')
        binding=dict(subject_id=record['subject_id'],source_id=sensor['source_id'],sensor_id=sensor['sensor_id'],
            capability_id=capability['id'],stream_epoch=epoch,processing_epoch=processing,
            coordinate_frame=sensor.get('coordinate_frame'),calibration_id=sensor.get('calibration_id'),
            record_id=record['id'],capture_ms=capture,record_sha256=hashlib.sha256(_canonical(record)).hexdigest())
        visual=None
        if visual_reference is not None:
            visual=bounded_json(visual_reference,max_bytes=4096)
            required={'source_id','sensor_id','stream_epoch','capture_ms','sha256'}
            if type(visual) is not dict or set(visual)!=required:raise ValueError('exact visual evidence binding required')
            _sha(visual['sha256']);_ms(visual['capture_ms'])
            if modality!='video' or any(visual[k]!=binding[k] for k in ('source_id','sensor_id','stream_epoch','capture_ms')):
                raise ValueError('visual evidence must match this video capture/source/epoch')
        active=change_score>self.change_exit if change_active else change_score>=self.change_enter
        entered=active and not change_active
        due=last_semantic_capture_ms is None or now-last_semantic_capture_ms>=self.semantic_interval_ms
        desired='none';reason='no_trigger'
        if self.mode=='context_only':reason='context_only'
        elif self.mode=='periodic':
            if due:desired,reason='small_vlm','periodic_due'
        else:
            if conflict_rule is not None and self.mode=='escalation':desired,reason='strong_vlm','validated_rule_conflict'
            elif unknown_candidate:desired,reason='small_vlm','unknown_candidate'
            elif query:desired,reason=('small_vlm' if needs_visual else 'text_llm'),'explicit_query'
            elif conflict_rule is not None:desired,reason='small_vlm','rule_requires_inspection'
            elif entered:desired,reason=('small_vlm' if needs_visual else 'text_llm'),'change_threshold_entered'
            elif due and modality=='video':desired,reason='small_vlm','blindspot_audit_due'
        # Always report overdue visual coverage even when context-only or blocked.
        unresolved_visual=modality=='video' and due
        blocked=[]
        age=now-capture
        if desired!='none':
            if age<0:blocked.append('future_capture')
            elif age>min(self.max_age_ms,freshness):blocked.append('stale_capture')
            if quality['status']!='sufficient':blocked.append('insufficient_observation_quality')
            if desired in ('small_vlm','strong_vlm') and visual is None:blocked.append('visual_evidence_required')
            if last_dispatch_ms is not None and now-last_dispatch_ms<self.cooldown_ms:blocked.append('cooldown')
        # No hidden upgrade: untrusted observed text/semantic confidence is never read.
        policy=dict(mode=self.mode,semantic_interval_ms=self.semantic_interval_ms,cooldown_ms=self.cooldown_ms,
            max_age_ms=self.max_age_ms,change_enter=self.change_enter,change_exit=self.change_exit)
        decision=dict(desired_route=desired,recommended_route='none' if blocked else desired,
            status='blocked' if blocked else ('recommendation' if desired!='none' else 'context_only'),reason=reason,
            blockers=blocked,evidence=binding,visual_reference=visual,
            proposed_change_active=active,change_entered=entered,
            semantic_audit_due=unresolved_visual,unknown_candidate_unresolved=unknown_candidate,
            validated_rule=conflict_rule,quality_preservation_proven=False,
            executes_model=False,changes_sensor_cadence=False,budget_authorized=False,
            dispatch_recorded=False,semantic_success_recorded=False,
            evaluated_ms=now,age_ms=age,policy=policy)
        decision['decision_sha256']=hashlib.sha256(_canonical(decision)).hexdigest()
        return bounded_json(decision)

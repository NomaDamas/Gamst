"""General tracked-object rotation counter from feature-point phase.

An upstream tracker supplies one ROI center and corresponding feature points
for one object track. This provider measures signed angular phase around that
center, unwraps it over time, and emits turns, angular velocity and quality.
It is reusable for coins, wheels, tires, fans and rotating tools; it does not
know the object class and does not open a camera.
"""
from __future__ import annotations
import hashlib
import math
import threading
from ..capabilities import CapabilitySpec, OutputField, SensorMetadata, bounded_json, identifier
from ..plugins import InputFrame
class AngularMotionCounter:
    def __init__(self, *, max_gap_ms=500, min_radius=2.0, max_points=128, min_quality=0.25, history_s=10):
        if type(max_gap_ms) is not int or not 1 <= max_gap_ms <= 10000: raise ValueError("max_gap_ms must be 1..10000")
        if type(min_radius) not in (int,float) or not math.isfinite(min_radius) or min_radius <= 0: raise ValueError("min_radius must be positive")
        if type(max_points) is not int or not 4 <= max_points <= 512: raise ValueError("max_points must be 4..512")
        if type(min_quality) not in (int,float) or not 0 <= min_quality <= 1: raise ValueError("min_quality must be 0..1")
        if type(history_s) not in (int,float) or not 1 <= history_s <= 120: raise ValueError("history_s must be 1..120")
        self.max_gap_ms=max_gap_ms; self.min_radius=float(min_radius); self.max_points=max_points; self.min_quality=float(min_quality)
        version=hashlib.sha256(repr((max_gap_ms,min_radius,max_points,min_quality,history_s)).encode()).hexdigest()[:16]
        self.spec=self.capability(version=version,max_gap_ms=max_gap_ms); self._lock=threading.RLock(); self.closed=False; self.reset()
    @staticmethod
    def capability(*,version="installed",max_gap_ms=500):
        identifier(version,"model version")
        return CapabilitySpec("vision.object_rotation","vision.object_rotation","video","gamst.angular_motion",version,(OutputField("track_id","string"),OutputField("rotation_count","number"),OutputField("phase_degrees","number"),OutputField("angular_velocity_deg_s","number"),OutputField("direction","string"),OutputField("confidence","number"),OutputField("sample_count","integer")),"frame",stateful=True,reset_on_gap=True,freshness_ms=max_gap_ms,interpretation="measurement",devices=("cpu",),sensor_requirements=("tracked-object-center","corresponding-feature-points"),license="Apache-2.0")
    def reset(self): self.owner=None;self.last_ts=None;self.last_sequence=None;self.track_id=None;self.phase=0.;self.samples=0;self.previous_angles=None;self.previous_ts=None
    def close(self):
        with self._lock: self.reset();self.closed=True
    @staticmethod
    def _point(value,name):
        if type(value) is not list or len(value)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) for v in value): raise ValueError(f"{name} must be a finite [x,y] point")
        return float(value[0]),float(value[1])
    def _output(self,velocity,direction,confidence,samples):
        return self.spec.validate({"track_id":self.track_id or "unknown","rotation_count":round(abs(self.phase)/(2*math.pi),4),"phase_degrees":round(math.degrees(self.phase),3),"angular_velocity_deg_s":round(velocity,3),"direction":direction,"confidence":round(float(confidence),4),"sample_count":samples})
    def process(self,frame: InputFrame):
        with self._lock:
            if self.closed: raise RuntimeError("angular motion provider is closed")
            if not isinstance(frame,InputFrame) or frame.modality!="video" or not isinstance(frame.sensor,SensorMetadata): raise ValueError("structured video InputFrame with sensor metadata required")
            if type(frame.timestamp_ms) is not int or frame.timestamp_ms<0: raise ValueError("integer capture timestamp required")
            data=bounded_json(frame.data,max_bytes=32768)
            if not isinstance(data,dict) or set(data)-{"track_id","center","points","roi","feature_scores"}: raise ValueError("rotation data contains unsupported fields")
            track_id=data.get("track_id")
            if not isinstance(track_id,str) or not track_id: raise ValueError("track_id is required")
            center=self._point(data.get("center"),"center"); raw_points=data.get("points")
            if type(raw_points) is not list or len(raw_points)>self.max_points: raise ValueError(f"points must be a list of at most {self.max_points}")
            points=[self._point(p,"point") for p in raw_points]; scores=data.get("feature_scores")
            if scores is not None and (type(scores) is not list or len(scores)!=len(points)): raise ValueError("feature_scores must match points")
            scores=[1.]*len(points) if scores is None else [float(s) for s in scores]
            owner=(frame.subject_id,frame.sensor.source_id,frame.sensor.sensor_id,frame.sensor.stream_epoch);sequence=frame.sensor.capture_sequence if frame.sensor.capture_sequence is not None else frame.sensor.sequence
            reset=owner!=self.owner or track_id!=self.track_id or self.last_ts is None or frame.timestamp_ms-self.last_ts>self.max_gap_ms or self.last_sequence is not None and sequence<=self.last_sequence
            if reset:
                self.owner,self.track_id=owner,track_id;self.last_ts,self.last_sequence=frame.timestamp_ms,sequence;self.previous_angles=None;self.previous_ts=frame.timestamp_ms;self.phase=0.;self.samples=0
                return self._output(0.,"unknown",0.,len(points))
            angles=[math.atan2(y-center[1],x-center[0]) for (x,y),score in zip(points,scores) if math.hypot(x-center[0],y-center[1])>=self.min_radius and score>0]
            previous=self.previous_angles;previous_ts=self.previous_ts;self.last_ts,self.last_sequence=frame.timestamp_ms,sequence;self.previous_angles=angles;self.previous_ts=frame.timestamp_ms
            if len(angles)<4 or previous is None or len(previous)!=len(angles): return self._output(0.,"unknown",0.,len(angles))
            deltas=[(after-before+math.pi)%(2*math.pi)-math.pi for before,after in zip(previous,angles)];median=sorted(deltas)[len(deltas)//2];confidence=sum(abs(delta-median)<=.35 for delta in deltas)/len(deltas)
            if confidence<self.min_quality: median=0.
            self.phase+=median;self.samples+=1;direction="clockwise" if median>0 else "counterclockwise" if median<0 else "unknown";dt=max(1,frame.timestamp_ms-previous_ts)
            return self._output(math.degrees(median)*1000/dt,direction,confidence,len(angles))

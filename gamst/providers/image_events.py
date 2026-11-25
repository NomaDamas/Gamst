"""Internal bounded event rules over the provider's observed path histories."""
import hashlib
import json
import math

from ..capabilities import bounded_json, identifier

_EPS = 1e-6


def _coordinates(value, length):
    if (type(value) is not list or len(value) != length or any(
            type(x) not in (int,float) or not math.isfinite(x) or not 0 <= x <= 1 for x in value)):
        raise ValueError('rule coordinates must be finite normalized numbers')
    return value


def _cross(a,b,p):
    return (b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0])


def _side(value):
    return 1 if value > _EPS else -1 if value < -_EPS else 0


class ImageEventRules:
    """Used only with internally generated histories; not an observation validator."""
    def __init__(self, rules):
        rules = bounded_json(rules,max_bytes=8192)
        if type(rules) is not list or not 1 <= len(rules) <= 4:
            raise ValueError('event_rules must contain one to four rules')
        seen = set()
        for rule in rules:
            if type(rule) is not dict or not {'id','kind'} <= set(rule):
                raise ValueError('rule id and kind required')
            name = identifier(rule['id'])
            if name in seen: raise ValueError('duplicate event rule id')
            seen.add(name)
            if rule['kind'] == 'roi':
                if set(rule) != {'id','kind','bounds'}: raise ValueError('ROI requires bounds only')
                x0,y0,x1,y1 = _coordinates(rule['bounds'],4)
                if x1-x0 <= 2*_EPS or y1-y0 <= 2*_EPS:
                    raise ValueError('ROI must exceed the boundary deadband')
            elif rule['kind'] == 'line':
                if set(rule) != {'id','kind','start','end'}: raise ValueError('line requires start/end only')
                a,b = _coordinates(rule['start'],2),_coordinates(rule['end'],2)
                if math.hypot(b[0]-a[0],b[1]-a[1]) <= 2*_EPS:
                    raise ValueError('line must exceed the boundary deadband')
            elif rule['kind'] == 'reversal':
                if set(rule) != {'id','kind','axis','min_displacement'} or rule['axis'] not in ('x','y'):
                    raise ValueError('reversal requires axis and minimum displacement')
                value = rule['min_displacement']
                if type(value) not in (int,float) or not math.isfinite(value) or not _EPS <= value <= 1:
                    raise ValueError('minimum displacement must be 1e-6..1')
            else: raise ValueError('unsupported image event kind')
        self._rules = rules
        self.sha256 = hashlib.sha256(json.dumps(rules,sort_keys=True,separators=(',',':')).encode()).hexdigest()

    def describe(self):
        return bounded_json(self._rules)

    @staticmethod
    def _roi(point,bounds):
        x,y = point; x0,y0,x1,y1 = bounds
        if x0+_EPS < x < x1-_EPS and y0+_EPS < y < y1-_EPS: return 1
        if x < x0-_EPS or x > x1+_EPS or y < y0-_EPS or y > y1+_EPS: return -1
        return 0

    def evaluate(self, paths):
        # Internal providers guarantee bounded, increasing, same-segment observed histories.
        events = []
        for oi,obj in enumerate(paths['objects']):
            observations = obj['observations']
            if len(observations) < 2: continue
            p,q = (sample['centroid'] for sample in observations[-2:])
            for ri,rule in enumerate(self._rules):
                kind = None; count = 2
                if rule['kind'] == 'roi':
                    before,after = self._roi(p,rule['bounds']),self._roi(q,rule['bounds'])
                    if before == -1 and after == 1: kind = 'roi_entry'
                    elif before == 1 and after == -1: kind = 'roi_exit'
                elif rule['kind'] == 'line':
                    a,b = rule['start'],rule['end']
                    line_length = math.hypot(b[0]-a[0],b[1]-a[1])
                    travel_length = math.hypot(q[0]-p[0],q[1]-p[1])
                    if travel_length > 2*_EPS:
                        # Strict intersection of finite segments, excluding endpoint touches.
                        before,after = _side(_cross(a,b,p)/line_length),_side(_cross(a,b,q)/line_length)
                        if (before*after == -1 and
                            _side(_cross(p,q,a)/travel_length)*_side(_cross(p,q,b)/travel_length) == -1):
                            kind = 'line_negative_to_positive' if after == 1 else 'line_positive_to_negative'
                elif len(observations) >= 3:
                    axis = 0 if rule['axis'] == 'x' else 1
                    r = observations[-3]['centroid']
                    u,v = p[axis]-r[axis],q[axis]-p[axis]
                    if abs(u) > rule['min_displacement'] and abs(v) > rule['min_displacement'] and u*v < 0:
                        kind = 'axis_positive_to_negative' if v < 0 else 'axis_negative_to_positive'
                        count = 3
                if kind is not None:
                    events.append(dict(kind=kind,rule_index=ri,object_index=oi,
                        observation_indices=list(range(len(observations)-count,len(observations))),
                        interpretation='candidate'))
        return dict(events=events,event_summary=dict(rules=self.describe(),rules_sha256=self.sha256,
            boundary_deadband_image=_EPS,truncated=False,event_count=len(events),
            input_snapshot_complete=paths['geometry']['complete_snapshot'],coverage='observed_endpoints_only',
            between_observations='unobserved',interpolation_assumption='straight_segment_for_line_candidates',
            exact_event_time_known=False,identity_verified=False,task_semantics_verified=False))

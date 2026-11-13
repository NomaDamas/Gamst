"""Bounded, JSON-serializable sensor context for any LLM/VLM client."""
import math
from .capabilities import bounded_json


def compact_observation(name, value):
    if isinstance(name, str) and name.startswith('classification.'):
        from .extensions import classification
        return classification(value['labels'], score_kind=value.get('score_kind','score'), exclusive=value.get('exclusive',False))
    if name == 'scene.faces' and isinstance(value, list):
        return [{'frame_face_index': face.get('frame_face_index'),
                 'blendshapes': dict(sorted(face.get('blendshapes', {}).items(),
                                           key=lambda item: item[1], reverse=True)[:8],
                                    **{k: v for k, v in face.get('blendshapes', {}).items()
                                       if k in ('mouthSmileLeft', 'mouthSmileRight', 'eyeBlinkLeft',
                                                'eyeBlinkRight', 'jawOpen', 'browInnerUp')})}
                for face in value[:4]]
    if name == 'scene.poses' and isinstance(value, list):
        return [{**{k: pose.get(k) for k in ('frame_pose_index', 'track_id', 'bbox', 'geometry', 'association_quality')},
                 'landmark_count': len(pose.get('landmarks', []))} for pose in value[:4]]
    if name == 'scene.segments' and isinstance(value, list):
        return [{k: item.get(k) for k in ('bbox', 'score', 'prompt', 'track_id')} for item in value[:16]]
    if isinstance(value, str):
        return value[:2000]
    if isinstance(value, (bool, int)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, list):
        return [compact_observation('', item) for item in value[:8]]
    if isinstance(value, dict):
        return {str(k)[:100]: compact_observation('', v) for k, v in list(value.items())[:16]
                if k not in ('landmarks', 'raw', 'frame', 'samples', 'jpeg', 'image', 'audio')}
    return None


def observation_packet(records, now):
    # One newest record per name, then prioritize fresh observations. Limits are
    # independent of frame rate and exclude raw media/large landmark arrays.
    latest = {}
    for record in sorted(records, key=lambda r: r['timestamp_ms']):
        name = record.get('observation', {}).get('name')
        value = record.get('observation', {}).get('value')
        is_classification = isinstance(name,str) and name.startswith('classification.')
        if is_classification:
            try:
                compact_observation(name,value)
            except (ValueError,TypeError,KeyError):
                continue
        registered = record.get('observation', {}).get('capability')
        safe = registered or is_classification or type(value) in (int, float, bool) or name in ('text.utterance', 'text.transcript', 'speech.transcript', 'scene.description', 'scene.window', 'scene.faces', 'scene.person_boxes', 'scene.person_tracks', 'scene.poses', 'scene.body_tracks', 'scene.segments')
        if name and safe and record.get('type') == 'observation.signal':
            sensor = record['observation'].get('sensor', {})
            key = (name, registered['id'], sensor.get('source_id'), sensor.get('sensor_id')) if registered else (name,)
            latest[key] = record
    output = {}
    for key, record in sorted(latest.items(), key=lambda item: item[1]['timestamp_ms'], reverse=True)[:32]:
        name = key[0]
        observation = record['observation']
        contract = observation.get('capability')
        age = max(0, now - record['timestamp_ms'])
        # JSON-encoded tuple avoids collisions between arbitrary source identifiers.
        if contract:
            import json
            output_key = '@capability:' + json.dumps(key, separators=(',', ':'))
        else:
            output_key = name
        output[output_key] = {'value': bounded_json(observation['value'], forbid_raw=True) if contract else compact_observation(name, observation.get('value')),
                        'timestamp_ms': record['timestamp_ms'], 'age_ms': age,
                        'fresh': age <= (contract['freshness_ms'] if contract else 5000), 'evidence_id': record['id'],
                        'models': record.get('models', [])[:4],
                        'interpretation': 'model_description_unverified' if name in ('scene.description', 'scene.window') else 'model_classification_score' if name.startswith('classification.') else 'observation_not_inner_state'}
        if contract:
            output[output_key].update(name=name, capability=bounded_json(contract),
                sensor=bounded_json(observation['sensor']),
                interpretation=contract['interpretation'], quality=record.get('quality'))
    return output

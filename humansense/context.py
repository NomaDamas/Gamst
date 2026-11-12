"""Bounded, JSON-serializable sensor context for any LLM/VLM client."""
import math


def compact_observation(name, value):
    if name == 'scene.faces' and isinstance(value, list):
        return [{'frame_face_index': face.get('frame_face_index'),
                 'blendshapes': dict(sorted(face.get('blendshapes', {}).items(),
                                           key=lambda item: item[1], reverse=True)[:8])}
                for face in value[:4]]
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
        safe = type(value) in (int, float, bool) or name in ('text.utterance', 'text.transcript', 'speech.transcript', 'scene.description', 'scene.faces', 'scene.person_boxes', 'scene.person_tracks')
        if name and safe and record.get('type') == 'observation.signal':
            latest[name] = record
    output = {}
    for name, record in sorted(latest.items(), key=lambda item: item[1]['timestamp_ms'], reverse=True)[:32]:
        age = max(0, now - record['timestamp_ms'])
        output[name] = {'value': compact_observation(name, record['observation'].get('value')),
                        'timestamp_ms': record['timestamp_ms'], 'age_ms': age,
                        'fresh': age <= 5000, 'evidence_id': record['id'],
                        'models': record.get('models', [])[:4],
                        'interpretation': 'model_description_unverified' if name == 'scene.description' else 'observation_not_inner_state'}
    return output

"""Bounded display geometry, separate from compact agent context."""
from copy import deepcopy


def guide_packet(state):
    result = {}
    now = state['timestamp_ms']
    for name in ('scene.faces', 'scene.poses', 'scene.person_tracks', 'scene.body_tracks', 'scene.segments'):
        record = state.get('signals', {}).get(name)
        if not record:
            continue
        value = record['observation']['value']
        if not isinstance(value, list):
            continue
        age = max(0, now - record['timestamp_ms'])
        # Raw pixels/audio are never included. Preserve geometry needed to draw.
        allowed = {'scene.faces': ('frame_face_index', 'landmarks', 'blendshapes'),
                   'scene.poses': ('frame_pose_index', 'landmarks', 'bbox', 'geometry', 'track_id', 'association_quality'),
                   'scene.segments': ('bbox', 'score', 'polygons', 'prompt', 'track_id'),
                   'scene.person_tracks': ('bbox', 'track_id', 'visible', 'association_quality'),
                   'scene.body_tracks': ('bbox', 'track_id', 'visible', 'association_quality')}[name]
        geometry = [{k: deepcopy(item[k]) for k in allowed if k in item} for item in value[:16 if name == 'scene.segments' else 4] if isinstance(item, dict)]
        result[name] = {'value': geometry,
                        'timestamp_ms': record['timestamp_ms'], 'age_ms': age,
                        'fresh': age <= 5000, 'evidence_id': record['id']}
    return result

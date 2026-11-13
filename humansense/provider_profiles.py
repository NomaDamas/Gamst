"""Dependency-light concrete worker discovery; never starts a model or child."""
from .capabilities import bounded_json


def capability(profile, config):
    config = bounded_json(config)
    if type(config) is not dict:
        raise ValueError('provider config must be an object')
    if profile == 'rfdetr.nano':
        from .providers.detection import RFDETRObjects
        import re
        if not isinstance(config.get('sha256'), str) or not re.fullmatch('[0-9a-f]{64}', config['sha256']):
            raise ValueError('RF-DETR needs a lowercase checkpoint SHA256')
        return RFDETRObjects.capability('1.12.1.eager.sha256.'+config['sha256'])
    if profile == 'audio.prosody':
        from .providers.prosody import AudioProsody
        return AudioProsody.capability(window_ms=config.get('window_ms', 80))
    if profile == 'opencv.flow':
        from .providers.motion import OpenCVFlow
        return OpenCVFlow.capability(config.get('method', 'lucas_kanade'))
    if profile == 'yamnet.events':
        from .providers.environment_audio import YAMNetEvents
        return YAMNetEvents.capability()
    raise ValueError('unsupported worker profile; use rfdetr.nano, audio.prosody, opencv.flow or yamnet.events')

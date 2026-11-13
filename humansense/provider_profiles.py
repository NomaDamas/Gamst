"""Dependency-light concrete worker discovery; never starts a model or child."""
from .capabilities import bounded_json


def capability(profile, config):
    config = bounded_json(config)
    if type(config) is not dict:
        raise ValueError('provider config must be an object')
    if profile == 'qwen3.asr':
        from .qwen_artifact import artifact_version
        from .providers.qwen_asr import QwenStreamingASR
        if set(config) != {'path', 'files'} or type(config['path']) is not str:
            raise ValueError('qwen3.asr requires only local path and file hashes')
        return QwenStreamingASR.capability(artifact_version(config['files']))
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
    if profile == 'silero.activity':
        from .providers.vad import SileroActivity
        return SileroActivity.capability(SileroActivity.artifact_version(config.get('sha256')))
    if profile == 'yamnet.events':
        from .providers.environment_audio import YAMNetEvents
        return YAMNetEvents.capability()
    raise ValueError('unsupported worker profile; use rfdetr.nano, audio.prosody, opencv.flow, yamnet.events silero.activity or qwen3.asr')

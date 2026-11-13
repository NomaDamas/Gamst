"""Dependency-light concrete worker discovery; never starts a model or child."""
from .capabilities import bounded_json


def capability(profile, config):
    config = bounded_json(config)
    if type(config) is not dict:
        raise ValueError('provider config must be an object')
    if profile == 'mediapipe.hand':
        from .providers.hand_context import MediaPipeHandContext
        if set(config)-{'path','sha256','max_hands'} or not {'path','sha256'} <= set(config) or type(config['path']) is not str:
            raise ValueError('hand profile requires local path, sha256 and optional max_hands')
        return MediaPipeHandContext.capability(config['sha256'],max_hands=config.get('max_hands',4))
    if profile == 'mediapipe.body':
        from .providers.body_context import MediaPipeBodyContext
        if set(config)-{'path','sha256','max_poses'} or not {'path','sha256'} <= set(config) or type(config['path']) is not str:
            raise ValueError('body profile requires local path, sha256 and optional max_poses')
        return MediaPipeBodyContext.capability(config['sha256'],max_poses=config.get('max_poses',4))
    if profile == 'qwen3.asr.endpoint':
        from .providers.endpoint_asr import endpoint_capability
        from .providers.vad import SileroActivity
        if set(config) != {'asr', 'vad', 'minimum_speech_ms', 'silence_ms'}:
            raise ValueError('endpoint profile requires asr, vad and both endpoint durations')
        vad = config['vad']
        if type(vad) is not dict or set(vad) != {'path', 'sha256'} or type(vad['path']) is not str:
            raise ValueError('endpoint vad requires only local path and checkpoint hash')
        return endpoint_capability(capability('qwen3.asr', config['asr']),
            SileroActivity.capability(SileroActivity.artifact_version(vad['sha256'])),
            minimum_speech_ms=config['minimum_speech_ms'], silence_ms=config['silence_ms'])
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
    raise ValueError('unsupported worker profile; use rfdetr.nano, audio.prosody, opencv.flow, yamnet.events silero.activity, qwen3.asr qwen3.asr.endpoint mediapipe.body or mediapipe.hand')

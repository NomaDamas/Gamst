"""Dependency-light concrete worker discovery; never starts a model or child."""
from .capabilities import bounded_json


def capability(profile, config):
    if profile == 'pg.physical.local':
        from .pg_physical_process import capability as pg_capability
        return pg_capability(config)
    if profile == 'opencv.rfdetr.stereo':
        from .providers.stereo_context import StereoMetricContext
        return StereoMetricContext.capability(**config)
    if profile == 'sam_audio.text_temporal.local':
        from .providers.sam_audio_context import SAMAudioContext
        return SAMAudioContext.capability(config)
    config = bounded_json(config)
    if type(config) is not dict:
        raise ValueError('provider config must be an object')
    if profile == 'orbslam3.monocular':
        from .providers.orbslam_context import ORBSLAMContext
        return ORBSLAMContext.capability(**config)
    if profile == 'smolvlm2.500m.local':
        from .providers.smolvlm_context import SmolVLMContext
        return SmolVLMContext.capability(**config)
    if profile == 'food.segformer_b0':
        from .providers.food import FoodContext
        return FoodContext.capability(**config)
    if profile == 'visual_microphone.phase':
        from .providers.visual_microphone_context import VisualMicrophoneContext
        return VisualMicrophoneContext.capability(**config)
    if profile == 'openeb.eventcd':
        from .providers.openeb import OpenEBContext
        return OpenEBContext.capability(**config)
    if profile == 'sparsh.slip.local':
        from .providers.sparsh_context import SparshSlipContext, configuration
        return SparshSlipContext.capability(**configuration(config))
    if profile == 'efficientphys.local':
        from .providers.efficientphys_context import EfficientPhysContext, configuration
        return EfficientPhysContext.capability(**configuration(config))
    if profile == 'bosch.bsec2':
        from .providers.bsec2 import BoschBSEC2
        return BoschBSEC2.capability(**config)
    if profile == 'efficienttam.tiny512':
        from .providers.efficienttam_native import configuration
        from .providers.efficienttam_context import EfficientTAMContext
        return EfficientTAMContext.capability(**configuration(config))
    if profile == 'yoloe.local':
        from .providers.yoloe_native import configuration
        from .providers.yoloe import YOLOEContext
        args,_,_ = configuration(config)
        return YOLOEContext.capability(**args)
    if profile == 'ti.radar':
        from .providers.radar import TIRadar
        return TIRadar.capability(**config)
    if profile == 'esp32.csi':
        from .providers.csi import ESP32CSI
        return ESP32CSI.capability(**config)
    if profile == 'talknet.local':
        from .providers.talknet import TalkNetContext
        if (not {'source_dir', 'checkpoint_path', 'checkpoint_sha256'} <= set(config) or
                set(config)-{'source_dir', 'checkpoint_path', 'checkpoint_sha256', 'window_frames', 'device'} or
                any(type(config[k]) is not str for k in ('source_dir', 'checkpoint_path'))):
            raise ValueError('TalkNet requires explicit local sources/checkpoint/hash and optional window/device')
        return TalkNetContext.capability(**{k:v for k,v in config.items() if k not in ('source_dir', 'checkpoint_path')})
    if profile == 'nemotron3.diarization':
        from .providers.diarization import NemotronDiarization
        if 'path' not in config or type(config['path']) is not str or set(config)-{'path','mode','threshold','device'}:
            raise ValueError('diarization requires local bundle path and optional mode/threshold/device')
        return NemotronDiarization.capability(**{k:v for k,v in config.items() if k != 'path'})
    if profile == 'emotion2vec.seed.onnx':
        from .providers.vocal_expression import Emotion2VecContext
        if not {'model_path','head_path','model_sha256','head_sha256'} <= set(config) or set(config)-{'model_path','head_path','model_sha256','head_sha256','window_ms','hop_ms'} or any(type(config[k]) is not str for k in ('model_path','head_path')):
            raise ValueError('emotion2vec requires local model/head paths and hashes with optional window/hop')
        return Emotion2VecContext.capability(**{k:v for k,v in config.items() if k not in ('model_path','head_path')})
    if profile == 'opencv.fill':
        from .providers.fill import CalibratedColorFill
        return CalibratedColorFill.capability(**config)
    if profile == 'openpiv.flow':
        from .providers.piv import OpenPIVFlow
        return OpenPIVFlow.capability(**config)
    if profile == 'webrtc.aec3':
        from .providers.duplex import WebRTCDuplex
        if set(config)-{'sample_rate','stream_delay_ms','ambient_sensor_id','speech_sensor_id'}:
            raise ValueError('unknown duplex profile option')
        return WebRTCDuplex.capability(**config)
    if profile == 'dinov3.regions':
        from .providers.region_appearance import RegionAppearance
        if (not {'path','sha256'} <= set(config) or set(config)-{'path','sha256','capacity','retention_ms','max_regions','top_k'} or type(config['path']) is not str):
            raise ValueError('region profile requires local path/hash and bounded gallery options')
        return RegionAppearance.capability(config['sha256'],**{k:v for k,v in config.items() if k not in ('path','sha256')})
    if profile == 'dinov3.retrieval':
        from .providers.visual_retrieval import DINOv3Retrieval
        if (not {'path','sha256'} <= set(config) or set(config)-{'path','sha256','capacity','retention_ms','top_k'} or
                type(config['path']) is not str):
            raise ValueError('retrieval requires local path/hash and optional capacity/retention_ms/top_k')
        return DINOv3Retrieval.capability(config['sha256'],**{k:v for k,v in config.items() if k not in ('path','sha256')})
    if profile == 'ppocr.mobile':
        from .providers.ocr import PPOCRContext
        required={'det_path','det_sha256','rec_path','rec_sha256'}
        if (not required<=set(config) or set(config)-required-{'language','max_regions'} or
                type(config['det_path']) is not str or type(config['rec_path']) is not str):
            raise ValueError('OCR requires local detector/recognizer paths and hashes, optional language/max_regions')
        return PPOCRContext.capability(config['det_sha256'],config['rec_sha256'],
            language=config.get('language','en'),max_regions=config.get('max_regions',16))
    if profile == 'depth_anything_v2.small':
        from .providers.depth import DepthAnythingContext
        if (not {'path','sha256'} <= set(config) or set(config)-{'path','sha256','grid_size'} or
                type(config['path']) is not str):
            raise ValueError('depth profile requires path, sha256 and optional grid_size')
        return DepthAnythingContext.capability(config['sha256'],grid_size=config.get('grid_size',3))
    if profile == 'l2cs.gaze':
        from .providers.gaze import L2CSGaze
        required = {'path','sha256','face_path','face_sha256'}
        if (not required <= set(config) or set(config)-required-{'arch','max_faces'} or
                type(config['path']) is not str or type(config['face_path']) is not str):
            raise ValueError('gaze profile requires local gaze/face paths and hashes, optional arch/max_faces')
        return L2CSGaze.capability(config['sha256'],config['face_sha256'],
            arch=config.get('arch','ResNet50'),max_faces=config.get('max_faces',4))
    if profile in ('mediapipe.body.video', 'mediapipe.hand.video', 'mediapipe.face.video'):
        from .providers.landmarker_video import MediaPipeVideoContext
        if set(config)-{'path','sha256','max_entities','max_gap_ms'} or not {'path','sha256'} <= set(config) or type(config['path']) is not str:
            raise ValueError('VIDEO profile requires path, sha256 and optional max_entities/max_gap_ms')
        return MediaPipeVideoContext.capability(profile.split('.')[1], config['sha256'],
            max_entities=config.get('max_entities',4), max_gap_ms=config.get('max_gap_ms',1000))
    if profile == 'mediapipe.face':
        from .providers.face_context import MediaPipeFaceContext
        if set(config)-{'path','sha256','max_faces'} or not {'path','sha256'} <= set(config) or type(config['path']) is not str:
            raise ValueError('face profile requires local path, sha256 and optional max_faces')
        return MediaPipeFaceContext.capability(config['sha256'],max_faces=config.get('max_faces',4))
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
    if profile == 'rfdetr.trajectories':
        from .providers.detection_trajectories import RFDETRTrajectories, configuration
        args = configuration(config)
        return RFDETRTrajectories.capability('1.12.1.eager.sha256.'+args['sha256'],
            **{k:v for k,v in args.items() if k not in ('path','sha256')})
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
    raise ValueError('unsupported worker profile; use rfdetr.nano, audio.prosody, opencv.flow, yamnet.events silero.activity, qwen3.asr qwen3.asr.endpoint mediapipe.body, mediapipe.hand, mediapipe.face or their .video profiles, l2cs.gaze, depth_anything_v2.small, ppocr.mobile')

"""Owned local inference worker; invoked only by explicit ProcessProvider.open."""
import os
import sys

from .provider_profiles import capability
from .provider_wire import envelope, sizes, parse_header, decode_frame, spec_value


def _construct(profile, config):
    spec = capability(profile, config)
    if profile == 'bosch.bsec2':
        from .providers.bsec2 import BoschBSEC2
        provider = BoschBSEC2(**config)
    elif profile == 'efficienttam.tiny512':
        from .providers.efficienttam_native import load_worker
        provider = load_worker(config)
    elif profile == 'yoloe.local':
        from .providers.yoloe_native import load_worker
        provider = load_worker(config)
    elif profile == 'ti.radar':
        from .providers.radar import TIRadar
        provider = TIRadar(**config)
    elif profile == 'esp32.csi':
        from .providers.csi import ESP32CSI
        provider = ESP32CSI(**config)
    elif profile == 'talknet.local':
        from .providers.talknet import TalkNetContext
        provider = TalkNetContext.from_local(**config)
    elif profile == 'nemotron3.diarization':
        from .providers.diarization import NemotronDiarization
        provider = NemotronDiarization.from_local(**config)
    elif profile == 'emotion2vec.seed.onnx':
        from .providers.vocal_expression import Emotion2VecContext
        provider = Emotion2VecContext.from_local(**config)
    elif profile == 'opencv.fill':
        from .providers.fill import CalibratedColorFill
        provider = CalibratedColorFill(**config)
    elif profile == 'openpiv.flow':
        from .providers.piv import OpenPIVFlow
        provider = OpenPIVFlow.from_installed(**config)
    elif profile == 'webrtc.aec3':
        from .providers.duplex import WebRTCDuplex
        provider = WebRTCDuplex.from_installed(**config)
    elif profile == 'dinov3.regions':
        from .providers.region_appearance import RegionAppearance
        provider = RegionAppearance.from_local(**config)
    elif profile == 'dinov3.retrieval':
        from .providers.visual_retrieval import DINOv3Retrieval
        provider = DINOv3Retrieval.from_local(**config)
    elif profile == 'ppocr.mobile':
        from .providers.ocr import PPOCRContext
        provider = PPOCRContext.from_local(**config)
    elif profile == 'depth_anything_v2.small':
        from .providers.depth import DepthAnythingContext
        provider = DepthAnythingContext.from_local(**config)
    elif profile == 'l2cs.gaze':
        from .providers.gaze import L2CSGaze
        provider = L2CSGaze.from_local(**config)
    elif profile in ('mediapipe.body.video', 'mediapipe.hand.video', 'mediapipe.face.video'):
        from .providers.landmarker_video import MediaPipeVideoContext
        provider = MediaPipeVideoContext.from_local(profile.split('.')[1], **config)
    elif profile == 'mediapipe.face':
        from .providers.face_context import MediaPipeFaceContext
        provider = MediaPipeFaceContext.from_local(**config)
    elif profile == 'mediapipe.hand':
        from .providers.hand_context import MediaPipeHandContext
        provider = MediaPipeHandContext.from_local(**config)
    elif profile == 'mediapipe.body':
        from .providers.body_context import MediaPipeBodyContext
        provider = MediaPipeBodyContext.from_local(**config)
    elif profile == 'qwen3.asr.endpoint':
        from .providers.endpoint_asr import SileroEndpointASR
        provider = SileroEndpointASR.from_local(**config)
    elif profile == 'qwen3.asr':
        from .providers.qwen_asr import QwenStreamingASR
        provider = QwenStreamingASR.from_local(**config)
    elif profile == 'rfdetr.nano':
        from .providers.detection import RFDETRObjects
        provider = RFDETRObjects.from_local(**config)
    elif profile == 'audio.prosody':
        from .providers.prosody import AudioProsody
        provider = AudioProsody(**config)
    elif profile == 'silero.activity':
        from .providers.vad import SileroActivity
        provider = SileroActivity.from_local(**config)
    elif profile == 'yamnet.events':
        from .providers.environment_audio import YAMNetEvents
        provider = YAMNetEvents.from_local(**config)
    else:
        from .providers.motion import OpenCVFlow
        provider = OpenCVFlow(**config)
    return provider, spec


def _read_exact(source, size):
    result = bytearray()
    while len(result) < size:
        data = source.read(size-len(result))
        if not data:
            raise EOFError('provider input closed')
        result.extend(data)
    return bytes(result)


def main():
    os.environ['GAMST_PROVIDER_WORKER'] = '1'
    # Preserve a private protocol descriptor before redirecting Python AND native
    # stdout to stderr. Library print/logging cannot corrupt protocol frames.
    output = os.fdopen(os.dup(sys.stdout.fileno()), 'wb', buffering=0)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    source = sys.stdin.buffer
    provider = spec = None
    try:
        while True:
            head_size, payload_size = sizes(_read_exact(source, 8))
            request = parse_header(_read_exact(source, head_size))
            payload = _read_exact(source, payload_size)
            number = request['id']
            if type(number) is not int or number < 1:
                raise ValueError('invalid request ID')
            operation = request['operation']
            try:
                if operation == 'start' and provider is None and not payload:
                    provider, spec = _construct(request['profile'], request['config'])
                    result = dict(spec=spec_value(spec))
                elif operation == 'process' and provider is not None:
                    result = provider.process(decode_frame(request['frame'], payload))
                    if result is not None:
                        result = spec.validate(result)
                elif operation == 'process_regions' and provider is not None:
                    result = spec.validate(provider.process_regions(decode_frame(request['frame'], payload), request['evidence']))
                elif operation in ('finish', 'endpoint') and provider is not None and not payload:
                    finish = getattr(provider, operation, None)
                    if operation == 'endpoint' and not callable(finish):
                        raise ValueError('provider does not support nonterminal endpoints')
                    result = finish() if callable(finish) else None
                    if result is not None:
                        result = spec.validate(result)
                elif operation == 'reset' and provider is not None and not payload:
                    provider.reset()
                    result = None
                elif operation == 'close' and provider is not None and not payload:
                    provider.close()
                    provider = None
                    output.write(envelope(dict(id=number, ok=True, result=None)))
                    return
                else:
                    raise ValueError('invalid provider operation/state')
                output.write(envelope(dict(id=number, ok=True, result=result)))
            except Exception as exc:
                # No exception text or native logs are allowed into context:
                # third-party exceptions may contain raw input or local secrets.
                output.write(envelope(dict(id=number, ok=False, error_type=type(exc).__name__[:128])))
                return  # Native failures invalidate temporal state; explicit reopen only.
    except (EOFError, BrokenPipeError):
        pass
    finally:
        if provider is not None:
            provider.close()
        output.close()


if __name__ == '__main__':
    main()

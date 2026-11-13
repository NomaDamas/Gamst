"""Owned local inference worker; invoked only by explicit ProcessProvider.open."""
import os
import sys

from .provider_profiles import capability
from .provider_wire import envelope, sizes, parse_header, decode_frame, spec_value


def _construct(profile, config):
    spec = capability(profile, config)
    if profile == 'rfdetr.nano':
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
                elif operation == 'finish' and provider is not None and not payload:
                    finish = getattr(provider, 'finish', None)
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

"""Explicit finite replay profiles; declarations cannot import arbitrary code."""
import asyncio
import importlib.util

from humansense.capabilities import CapabilitySpec, OutputField
from humansense.plugins import AudioLevelPlugin
from .recorded_source import _keys, digest, _integer


class PCMLevel:
    spec = CapabilitySpec('audio.replay.level', 'audio.features', 'audio',
        'gamst.pcm_level', '1', (OutputField('level', 'object'),), 'frame',
        interpretation='measurement', sensor_requirements=('pcm-s16le-mono',))

    async def process(self, frame):
        signals = await AudioLevelPlugin().process(frame)
        return {'level': {s['name']:s['value'] for s in signals}}


class FixtureBGR:
    """Test-only declared pixel summary, explicitly not semantic inference."""
    def __init__(self, delay_ms):
        self.delay_ms = delay_ms
        self.spec = self.capability(delay_ms)

    @staticmethod
    def capability(delay_ms):
        return CapabilitySpec('vision.replay.fixture', 'vision.fixture', 'video',
            'gamst.synthetic_pixel_fixture', '1.'+str(delay_ms),
            (OutputField('fixture', 'object'),), 'frame', interpretation='estimate')

    async def process(self, frame):
        await asyncio.sleep(self.delay_ms/1000)
        return {'fixture':dict(synthetic=True, semantics_verified=False,
            bgr_mean=[float(x) for x in frame.data.mean(axis=(0,1))])}


def profiles_for(manifest, profiles):
    """Preflight dependencies and bounded copies before opening any provider."""
    sources = {s['source_id']:s for s in manifest['sources']}
    if type(profiles) is not dict or set(profiles) != set(sources):
        raise ValueError('one explicit profile per declared source required')
    result = {}; copied_bytes = 0; capabilities = {}
    for sid, declaration in profiles.items():
        _keys(declaration, ('profile', 'delay_ms'))
        delay = _integer(declaration['delay_ms'], 0, 500)
        name = declaration['profile']; form = sources[sid]['format']
        maximum = max((r['payload_bytes'] for r in manifest['records'] if r['source_id']==sid),default=0)
        # Two pending, one processing, submit's transient snapshots, source read
        # and owned-byte joins. Provider/model memory is separate, explicit below.
        copied_bytes += maximum*8
        if form == 'pcm_s16le_mono' and name == 'pcm_level' and delay == 0:
            spec, factory = PCMLevel.spec, PCMLevel
        elif form == 'bgr8_hwc' and name in ('fixture_bgr', 'opencv_lk'):
            if importlib.util.find_spec('numpy') is None:
                raise RuntimeError('selected BGR replay requires existing NumPy')
            if name == 'fixture_bgr':
                spec = FixtureBGR.capability(delay)
                factory = lambda d=delay: FixtureBGR(d)
            else:
                if delay != 0 or importlib.util.find_spec('cv2') is None:
                    raise RuntimeError('selected actual OpenCV profile requires existing cv2 and no fixture delay')
                if any(min(r['shape'][:2]) < 16 for r in manifest['records'] if r['source_id']==sid):
                    raise ValueError('OpenCV profile requires dimensions at least16')
                from humansense.providers.motion import OpenCVFlow
                spec, factory = OpenCVFlow.capability(), OpenCVFlow
        else:
            raise ValueError('unsupported explicit source profile')
        if spec.id in capabilities and capabilities[spec.id] != spec:
            raise ValueError('same capability cannot have conflicting profile configurations')
        capabilities[spec.id] = spec
        result[sid] = (spec, factory)
    if copied_bytes > 256*1024**2:
        raise ValueError('joint replay snapshot memory admission exceeds256MiB')
    return result, dict(profile_sha256=digest(profiles),
        admitted_capture_copy_bytes=copied_bytes, additional_provider_memory='not_measured',
        total_rss_verified=False)

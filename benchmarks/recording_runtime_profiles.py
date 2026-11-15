"""Finite owned runtime profiles; no module/factory is selected by JSON."""
from copy import deepcopy
from .recorded_profiles import PCMLevel,FixtureBGR
from .recorded_source import digest,_keys


class _PCM(PCMLevel):
    config={}
    def __init__(self):self.closed=False
    async def process(self,frame):
        if self.closed:raise RuntimeError('PCM provider closed')
        return await super().process(frame)
    async def close(self):self.closed=True


class _Fixture(FixtureBGR):
    def __init__(self,config):
        super().__init__(config['delay_ms']);self.config=deepcopy(config);self.closed=False
    async def process(self,frame):
        if self.closed:raise RuntimeError('fixture provider closed')
        return await super().process(frame)
    async def close(self):self.closed=True


def builtin_profile(delivery,config):
    """Return registered spec, finite owned factory, and its explicit config.

    Calling this function does not construct a provider or execute a model.
    Dependencies are checked by the upstream owned asset/dependency admission.
    """
    name=delivery['profile']
    if name=='acquire_only':
        if config is not None:raise ValueError('acquire-only has no provider configuration')
        return None
    if type(config) is not dict or digest(config)!=delivery['config_sha256']:
        raise ValueError('original exact provider configuration required')
    if name=='pcm_level':
        _keys(config,());return _PCM.spec,_PCM,{}
    if name=='fixture_bgr':
        _keys(config,('delay_ms',));return FixtureBGR.capability(config['delay_ms']),lambda:_Fixture(config),deepcopy(config)
    if name=='opencv_lk':
        _keys(config,())
        from gamst.providers.motion import OpenCVFlow
        import cv2
        spec=OpenCVFlow.capability(version=cv2.__version__)
        class OwnedFlow(OpenCVFlow):
            def __init__(self):
                super().__init__();self.config={};self.spec=spec
        return spec,OwnedFlow,{}
    if name=='opencv_fill':
        from gamst.providers.fill import CalibratedColorFill,_configuration
        normalized=_configuration(**config)
        if digest(normalized)!=digest(config):raise ValueError('effective fill configuration required')
        spec=CalibratedColorFill.capability(**config)
        return spec,lambda:CalibratedColorFill(**config),deepcopy(config)
    raise ValueError('unsupported built-in runtime profile')

"""Explicit plugin discovery and portable classification output contracts."""
from dataclasses import dataclass, asdict
from importlib.metadata import entry_points
import math
import re

from .plugins import _signal as signal

ENTRY_POINT_GROUP = 'catch_your_mind.plugins'


@dataclass(frozen=True)
class PluginManifest:
    name: str
    version: str
    modalities: tuple[str, ...]
    outputs: tuple[str, ...] = ()
    description: str = ''
    api_version: int = 1


def describe_plugin(plugin):
    inner = getattr(plugin,'plugin',plugin)
    manifest = getattr(inner,'manifest',None)
    if manifest is not None:
        return asdict(manifest)
    return asdict(PluginManifest(inner.name, getattr(inner,'version','unspecified'),
                                 tuple(inner.modalities)))


def discover_plugins():
    """List metadata only. Discovery does not import or execute plugin code."""
    return [{'name':e.name,'target':e.value,'distribution':e.dist.name if e.dist else None}
            for e in entry_points(group=ENTRY_POINT_GROUP)]


def load_plugin(name, **options):
    entries = [e for e in entry_points(group=ENTRY_POINT_GROUP) if e.name == name]
    if len(entries) != 1:
        raise ValueError('Plugin must resolve to exactly one installed entry point')
    plugin = entries[0].load()(**options)
    if not callable(getattr(plugin,'process',None)) or not getattr(plugin,'modalities',None):
        raise TypeError('Plugin must define modalities and process(InputFrame)')
    manifest = getattr(plugin,'manifest',None)
    if manifest is not None and (not isinstance(manifest,PluginManifest) or manifest.api_version != 1):
        raise ValueError('Unsupported plugin manifest API version')
    return plugin


def classification(labels, *, score_kind='score', exclusive=False):
    """Scores retain their semantics; no automatic probability calibration."""
    if score_kind not in ('score','probability') or type(exclusive) is not bool:
        raise ValueError('Invalid classification semantics')
    if not isinstance(labels,dict) or not 1 <= len(labels) <= 32:
        raise ValueError('Classification requires 1..32 labels')
    for label,value in labels.items():
        if not isinstance(label,str) or not 1 <= len(label) <= 80:
            raise ValueError('Labels must contain 1..80 characters')
        if type(value) not in (int,float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Scores must be finite numbers from 0 to 1')
    if score_kind == 'probability' and exclusive and not math.isclose(sum(labels.values()),1,abs_tol=.01):
        raise ValueError('Exclusive probabilities must sum to one')
    return {'labels':dict(labels),'score_kind':score_kind,'exclusive':exclusive}


def classification_name(name):
    if not isinstance(name,str) or not re.fullmatch(r'classification\.[a-zA-Z0-9_.-]{1,80}',name):
        raise ValueError('Classification signal names must start with classification.')
    return name


def register_config(human, config):
    """Explicit startup config; never automatically imports discovered packages."""
    if not isinstance(config,list) or len(config)>16:
        raise ValueError('Plugin configuration must be a list of at most 16 entries')
    from .realtime import QueuedPlugin
    plugins = []
    for item in config:
        if not isinstance(item,dict) or set(item)-{'provider','options','capacity'}:
            raise ValueError('Each plugin needs provider, options and optional capacity')
        capacity = item.get('capacity',1)
        if type(capacity) is not int or not 1 <= capacity <= 16:
            raise ValueError('Plugin capacity must be 1..16')
        options = item.get('options',{})
        if not isinstance(options,dict):
            raise ValueError('Plugin options must be an object')
        if item.get('provider') == 'huggingface':
            from .providers.huggingface import HuggingFaceClassifier
            plugin = HuggingFaceClassifier.from_local(**options)
        else:
            plugin = load_plugin(item.get('provider'),**options)
        plugins.append(QueuedPlugin(human,plugin,capacity=capacity))
    for plugin in plugins:
        human.register(plugin)
    return [describe_plugin(p) for p in plugins]

"""Authenticate and import immutable official source snapshots in one worker."""
import importlib.abc
import importlib.util
from pathlib import Path
import sys

from .sam_audio_artifact import _file
from .sam_audio_source_hashes import SOURCE_HASHES


class VerifiedSources(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Never execute caller-declared source or stale bytecode as official code.

    The fixed SHA256 inventory comes from the three official pinned Git trees.
    Third-party distributions/binaries remain the declared runtime boundary.
    """
    def __init__(self, manifest):
        for name in sys.modules:
            if self.owns(name):
                raise RuntimeError('upstream package already imported; use a fresh worker')
        self.modules = {}
        for group, files in SOURCE_HASHES.items():
            source = manifest['sources'][group]
            if set(source['files']) != set(files):
                raise ValueError('complete fixed official package inventory required')
            root = Path(source['root'])
            for name, digest in files.items():
                if source['files'][name] != digest:
                    raise ValueError('source declaration is not the pinned official source')
                data = _file(root/name, digest, 4*1024*1024, retain=True, allow_empty=True)
                package = name.endswith('/__init__.py')
                module = name[:-12] if package else name[:-3]
                self.modules[module.replace('/', '.')] = (str(root/name), data, package)

    @staticmethod
    def owns(name):
        return name.split('.')[0] in ('sam_audio', 'core', 'dacvae')

    def find_spec(self, fullname, path=None, target=None):
        if not self.owns(fullname):
            return None
        if fullname not in self.modules:
            raise ModuleNotFoundError('module is outside the fixed official source inventory')
        filename, _, package = self.modules[fullname]
        spec = importlib.util.spec_from_loader(fullname, self, origin=filename, is_package=package)
        spec.has_location = True
        return spec

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        filename, data, package = self.modules[module.__name__]
        module.__file__ = filename
        if package:
            module.__path__ = [str(Path(filename).parent)]
        exec(compile(data, filename, 'exec'), module.__dict__)

    def install(self):
        sys.meta_path.insert(0, self)

    def remove(self):
        if self in sys.meta_path:
            sys.meta_path.remove(self)


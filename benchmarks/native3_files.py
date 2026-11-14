"""Bounded original collector files for native3 structural evidence readers."""
import hashlib
import os
from pathlib import Path
import stat

from .perception_comparison import _digest
from .three_arm_replay import MAX_TRACE_BYTES

DECLARATIONS = dict(plan='comparison-plan.json', collection='collection-spec.json',
    inputs='inputs.json', stream_protocol='stream-protocol.json',
    allocation_protocol='allocation-protocol.json', reconciliation_protocol='reconciliation-protocol.json',
    instrumentation_protocol='instrumentation-protocol.json')
LIMITS = {name: 16*1024**2 for name in DECLARATIONS.values()}
LIMITS.update({'runtime-sidecars.jsonl':16*1024**2, 'runtime-complete.json':65536,
               'source-traces.jsonl':3000*MAX_TRACE_BYTES, 'complete.json':65536})


class OriginalNativeFiles:
    """Hash owned no-follow descriptors; never open stored bootstrap paths.

    Source traces are iterated instead of accumulated. A file is marked verified
    only after complete consumption and a final descriptor/path identity check.
    """
    def __init__(self, directory, inventory):
        if type(inventory) is not dict or set(inventory) != set(LIMITS):
            raise ValueError('exact original native collector inventory required')
        self.inventory = {}
        for name, limit in LIMITS.items():
            item = inventory[name]
            if type(item) is not dict or set(item) != {'bytes', 'sha256'}:
                raise ValueError('exact original size and digest required')
            size = item['bytes']
            if type(size) is not int or not 0 < size <= limit:
                raise ValueError('original native file exceeds bound')
            self.inventory[name] = dict(bytes=size, sha256=_digest(item['sha256']))
        if sum(self.inventory[name]['bytes'] for name in DECLARATIONS.values()) > 16*1024**2:
            raise ValueError('joint original declarations exceed16MiB')
        path = Path(directory)
        if not path.is_absolute() or '..' in path.parts:
            raise ValueError('absolute confined original collector directory required')
        fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for part in path.parts[1:]:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd); fd = child
        except BaseException:
            os.close(fd); raise
        self._fd = fd
        self.verified = set()

    def close(self):
        if self._fd is not None:
            os.close(self._fd); self._fd = None

    def __enter__(self): return self
    def __exit__(self, *args): self.close()

    def _open(self, name):
        if self._fd is None: raise RuntimeError('original directory closed')
        if name not in self.inventory: raise ValueError('undeclared original file')
        self.verified.discard(name)
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self._fd)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size != self.inventory[name]['bytes']:
            os.close(fd); raise ValueError('exact bounded regular original required')
        return fd, before

    @staticmethod
    def _identity(info):
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns

    def _finish(self, name, fd, before, count, digest):
        after = os.fstat(fd)
        named = os.stat(name, dir_fd=self._fd, follow_symlinks=False)
        if (self._identity(before) != self._identity(after) or self._identity(after) != self._identity(named)
                or count != self.inventory[name]['bytes'] or digest.hexdigest() != self.inventory[name]['sha256']):
            raise ValueError('original collector file changed or hash differs')
        self.verified.add(name)

    def read(self, name):
        if name == 'source-traces.jsonl': raise ValueError('source traces require bounded streaming')
        fd, before = self._open(name)
        try:
            parts = []; count = 0; digest = hashlib.sha256()
            while count <= before.st_size:
                part = os.read(fd, min(65536, before.st_size + 1 - count))
                if not part: break
                count += len(part); digest.update(part); parts.append(part)
            self._finish(name, fd, before, count, digest)
            return b''.join(parts)
        finally: os.close(fd)

    def source_lines(self):
        name = 'source-traces.jsonl'
        fd, before = self._open(name)
        with os.fdopen(fd, 'rb') as stream:
            count = lines = 0; digest = hashlib.sha256()
            while True:
                line = stream.readline(MAX_TRACE_BYTES + 1)
                if not line: break
                count += len(line); lines += 1; digest.update(line)
                if count > before.st_size or lines > 3000 or len(line) > MAX_TRACE_BYTES or not line.endswith(b'\n'):
                    raise ValueError('source original line/count/byte bound exceeded')
                yield line
            self._finish(name, stream.fileno(), before, count, digest)

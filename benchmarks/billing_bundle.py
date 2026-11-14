"""Exclusive, bounded offline aggregate evidence bundles; no provider calls."""
import os
from pathlib import Path
import stat

from .billing_aggregate import build_report, encoded, sha, validate_manifest


def _directory(path):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('absolute non-traversing bundle path required')
    fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def _verify_file(directory, name, expected):
    fd = os.open(name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=directory)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) != 0o600 or before.st_nlink != 1 or before.st_size != len(expected):
            raise ValueError('bundle file type, permissions, link count or size mismatch')
        chunks = []; remaining = len(expected) + 1
        while remaining:
            chunk = os.read(fd, min(remaining, 65536))
            if not chunk: break
            chunks.append(chunk); remaining -= len(chunk)
        after = os.fstat(fd)
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns) or b''.join(chunks) != expected:
            raise ValueError('bundle differs from recomputed original evidence')
    finally:
        os.close(fd)


def _write_file(directory, name, data):
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
    try:
        os.fchmod(fd, 0o600)
        payload = memoryview(data)
        while payload:
            count = os.write(fd, payload)
            if count <= 0: raise OSError('incomplete evidence write')
            payload = payload[count:]
        os.fsync(fd)
    finally:
        os.close(fd)


def persist_bundle(manifest, asset_directory, output_directory):
    """Recompute before reuse; completion is written last, partial bundles stay partial.

    Existing outputs are never repaired or overwritten. Original financial data
    retention is explicit here and independent of the SDK's observation storage.
    """
    manifest = validate_manifest(manifest)
    report, originals = build_report(manifest, asset_directory)
    output = Path(output_directory)
    if not output.is_absolute() or output.name in ('', '.', '..') or '..' in output.parts:
        raise ValueError('absolute output directory required')
    files = {'acquisition.json': encoded(manifest), 'report.json': encoded(report)}
    files.update({digest + '.original.json': data for digest, data in originals.items()})
    completion = encoded(dict(schema_version=1, kind='aggregate_bundle_complete',
        files={name: dict(bytes=len(data), sha256=sha(data)) for name, data in sorted(files.items())}))
    files['completion.json'] = completion
    parent = _directory(output.parent)
    directory = None
    try:
        try:
            os.mkdir(output.name, 0o700, dir_fd=parent)
            created = True
        except FileExistsError:
            created = False
        directory = os.open(output.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        info = os.fstat(directory)
        if stat.S_IMODE(info.st_mode) != 0o700 or info.st_uid != os.getuid():
            raise ValueError('private owned bundle directory required')
        if not created:
            if set(os.listdir(directory)) != set(files):
                raise ValueError('incomplete or unexpected existing bundle inventory')
            for name, data in files.items(): _verify_file(directory, name, data)
        else:
            for name, data in files.items(): _write_file(directory, name, data)
            os.fsync(directory)
            os.fsync(parent)
        return report
    finally:
        if directory is not None: os.close(directory)
        os.close(parent)


def load_acquisition(path):
    """Read one bounded regular declaration with duplicate-key rejection."""
    from .billing_json import SourceNumber, decode_original
    path = Path(path)
    if not path.is_absolute() or not path.name:
        raise ValueError('absolute acquisition manifest path required')
    directory = _directory(path.parent)
    try:
        fd = os.open(path.name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=directory)
    finally:
        os.close(directory)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or not 1 <= info.st_size <= 1024**2:
            raise ValueError('bounded regular acquisition manifest required')
        data = bytearray()
        while len(data) <= info.st_size:
            chunk = os.read(fd, min(65536, info.st_size + 1 - len(data)))
            if not chunk: break
            data.extend(chunk)
        after = os.fstat(fd)
        if len(data) != info.st_size or (info.st_size, info.st_mtime_ns, info.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError('acquisition manifest changed during read')
    finally:
        os.close(fd)
    def declared(value):
        if isinstance(value, SourceNumber): return value.integer()
        if isinstance(value, list): return [declared(item) for item in value]
        if isinstance(value, dict): return {key: declared(item) for key, item in value.items()}
        return value
    return validate_manifest(declared(decode_original(bytes(data))))


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Preserve original account aggregates offline; does not assign request costs.')
    parser.add_argument('--manifest', required=True, help='Absolute acquisition declaration JSON path')
    parser.add_argument('--assets', required=True, help='Absolute original JSON directory')
    parser.add_argument('--output', required=True, help='Absolute new private evidence directory')
    args = parser.parse_args()
    report = persist_bundle(load_acquisition(args.manifest), args.assets, args.output)
    print('Aggregate evidence preserved: %d views; actual acceptance remains incomplete.' % len(report['views']))


if __name__ == '__main__':
    main()

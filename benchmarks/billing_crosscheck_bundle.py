"""Private immutable crosscheck output, separate from every original evidence set."""
import os
from pathlib import Path
import stat

from .billing_aggregate import encoded, sha
from .billing_bundle import _directory, _write_file, _verify_file
from .billing_transport_crosscheck import crosscheck_transport


def persist_crosscheck(manifest, asset_directory, transport_directory, *,
                       expected_inventory_sha256, native_index, expected_binding, output_directory):
    """Recompute original-derived output before reuse; never modify source bundles."""
    _reject_source_output(output_directory, (native_index.directory, transport_directory))
    report=crosscheck_transport(manifest,asset_directory,transport_directory,
        expected_inventory_sha256=expected_inventory_sha256,native_index=native_index,expected_binding=expected_binding)
    return _persist_report(report, output_directory)


def _persist_report(report, output_directory):
    """Publish already recomputed evidence; completion remains the last write."""
    payload=encoded(report)
    files={'crosscheck.json':payload,'completion.json':encoded(dict(schema_version=1,
        kind='transport_crosscheck_bundle_complete',files={'crosscheck.json':dict(bytes=len(payload),sha256=sha(payload))}))}
    output=Path(output_directory)
    if not output.is_absolute() or not output.name or '..' in output.parts:
        raise ValueError('absolute new crosscheck directory required')
    parent=_directory(output.parent);directory=None
    try:
        try:
            os.mkdir(output.name,0o700,dir_fd=parent);created=True
        except FileExistsError:created=False
        directory=os.open(output.name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent)
        info=os.fstat(directory)
        if stat.S_IMODE(info.st_mode)!=0o700 or info.st_uid!=os.getuid():
            raise ValueError('private owned crosscheck directory required')
        if created:
            for name,data in files.items():_write_file(directory,name,data)
            os.fsync(directory);os.fsync(parent)
        else:
            if set(os.listdir(directory))!=set(files):raise ValueError('incomplete or unexpected crosscheck inventory')
            for name,data in files.items():_verify_file(directory,name,data)
        return report
    finally:
        if directory is not None:os.close(directory)
        os.close(parent)


def _reject_source_output(output_directory, original_directories):
    output=Path(output_directory).resolve()
    for directory in original_directories:
        original=Path(directory).resolve()
        if output==original or original in output.parents:
            raise ValueError('crosscheck output must be outside original evidence directories')


def _load_declaration(path, limit):
    from .native3_evidence import _decode
    path=Path(path)
    if not path.is_absolute() or not path.name:raise ValueError('absolute declaration file required')
    directory=_directory(path.parent)
    try:fd=os.open(path.name,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW,dir_fd=directory)
    finally:os.close(directory)
    try:
        before=os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0<before.st_size<=limit:raise ValueError('bounded regular declaration required')
        with os.fdopen(fd,'rb',closefd=False) as stream:data=stream.read(limit+1)
        after=os.fstat(fd)
        if len(data)!=before.st_size or (before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns):
            raise ValueError('declaration changed during owned read')
        return _decode(data)
    finally:os.close(fd)


def main():
    import argparse
    from .billing_bundle import load_acquisition
    from .native3_evidence import read_native3_evidence
    parser=argparse.ArgumentParser(description='Offline structural native/transport crosscheck; no request price or actual billing certification.')
    for name in ('manifest','assets','transport','transport-inventory-sha256','native-root','native-files','selection','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--sidecar-ordinal',type=int,required=True)
    args=parser.parse_args()
    index=read_native3_evidence(args.native_root,expected_files=_load_declaration(args.native_files,65536),sidecar_ordinal=args.sidecar_ordinal)
    report=persist_crosscheck(load_acquisition(args.manifest),args.assets,args.transport,
        expected_inventory_sha256=args.transport_inventory_sha256,native_index=index,
        expected_binding=_load_declaration(args.selection,8192),output_directory=args.output)
    print('Structural selected-lane match: %s; actual acceptance remains incomplete.' % report['local_binding_complete'])


if __name__=='__main__':main()

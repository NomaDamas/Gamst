"""Offline native4 crosscheck persistence; original artifacts remain untouched."""
from pathlib import Path

from .billing_crosscheck_bundle import _persist_report, _load_declaration
from .billing_native4_crosscheck import crosscheck_recording_transport


def persist_recording_crosscheck(manifest, asset_directory, transport_directory, *,
        expected_inventory_sha256, native_index, expected_binding, declarations_path,
        declarations_sha256, terminal_sha256, output_directory):
    report = crosscheck_recording_transport(manifest, asset_directory, transport_directory,
        expected_inventory_sha256=expected_inventory_sha256, native_index=native_index,
        expected_binding=expected_binding, declarations_path=declarations_path,
        declarations_sha256=declarations_sha256, terminal_sha256=terminal_sha256)
    return _persist_report(report, output_directory)


def main():
    import argparse
    from .billing_bundle import load_acquisition
    from .recording_native_schema import decode, MAX_DECLARATIONS, sha
    from .recording_native_validation import _read, validate_recording_native_inventory
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','assets','transport','transport-inventory-sha256','native-root',
                 'declarations','declarations-sha256','terminal-sha256','selection','output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    path = Path(args.declarations)
    if not path.is_absolute(): raise ValueError('absolute original declarations required')
    declarations = decode(_read(path.parent, path.name, MAX_DECLARATIONS, sha(args.declarations_sha256)))
    index = validate_recording_native_inventory(declarations, args.native_root,
        sha(args.terminal_sha256), source_inventory=declarations['source_manifest'])
    report = persist_recording_crosscheck(load_acquisition(args.manifest), args.assets, args.transport,
        expected_inventory_sha256=args.transport_inventory_sha256, native_index=index,
        expected_binding=_load_declaration(args.selection,8192), declarations_path=path,
        declarations_sha256=args.declarations_sha256, terminal_sha256=args.terminal_sha256,
        output_directory=args.output)
    print('Structural native4 lane match: %s; actual acceptance remains incomplete.' % report['local_binding_complete'])


if __name__ == '__main__': main()

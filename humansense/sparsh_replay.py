"""Explicit local tactile recording to Sparsh Context; no capture or downloads."""
import argparse
import json
from pathlib import Path
from .providers.tactile import recording_frames, _strict_json
from .providers.sparsh_context import SparshSlipContext, configuration


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    path=Path(args.config)
    if not path.is_absolute():raise ValueError('absolute local configuration path required')
    with path.open('rb') as handle:payload=handle.read(16385)
    if len(payload)>16384:raise ValueError('configuration exceeds bound')
    config=_strict_json(payload);configuration(config)
    provider=SparshSlipContext.from_local(**config)
    try:
        for frame in recording_frames(args.manifest):
            print(json.dumps(provider.process(frame),sort_keys=True,allow_nan=False))
    finally:provider.close()


if __name__=='__main__':main()

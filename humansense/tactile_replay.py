"""Inspect hash-bound tactile recordings without opening devices or models."""
import argparse
import json
from .providers.tactile import TactileInputContext, recording_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--max-gap-ms', type=int, default=100)
    args = parser.parse_args()
    provider = TactileInputContext(max_gap_ms=args.max_gap_ms)
    try:
        for frame in recording_frames(args.manifest):
            print(json.dumps(provider.process(frame), sort_keys=True, allow_nan=False))
    finally:
        provider.close()


if __name__ == '__main__': main()

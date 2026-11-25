#!/usr/bin/env python3
"""Verify the canonical temporal packet checksum across GAMST and Gamst-bench.

This is a contract probe, not a model or host acceptance run.  It deliberately
uses one bounded EventPacket-shaped mapping so GAMST computes the envelope,
gamst.crew (formerly the GamstCrew repository) parses and verifies it, and
Gamst-bench verifies the same canonical UTF-8 bytes.  The output records exact
repository heads and the shared digest.
"""
from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path


def _head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-root", type=Path, required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    for root in (repo_root, args.bench_root):
        if not root.is_dir():
            raise SystemExit(f"repository root does not exist: {root}")

    sys.path.insert(0, str(repo_root))
    gamst_temporal = importlib.import_module("gamst.temporal")
    crew_packet = importlib.import_module("gamst.crew.packet")
    sys.path.insert(0, str(args.bench_root))
    bench_contracts = importlib.import_module("gamst_bench.contracts")

    body = {
        "event_id": "cross-repo-event-1",
        "stream_id": "camera-1",
        "subject_id": "object-1",
        "modality": "vision",
        "event_type": "object_state_change",
        "source_clock": "monotonic",
        "capture_timestamp_ms": 10000,
        "receive_timestamp_ms": 10035,
        "processing_start_ms": 10036,
        "processing_end_ms": 10040,
        "start_ms": 9800,
        "end_ms": 10000,
        "sequence": 42,
        "gap": False,
        "revision": 1,
        "provider": "mediapipe",
        "model_revision": "pinned-revision",
        "value": {"state": "present"},
        "uncertainty": {"confidence": 0.98},
        "freshness_ms": 5000,
        "evidence_ids": ["frame-42"],
        "source_spans": [],
        "query_cutoff_ms": 10000,
        "monotonic_capture_ms": 10000,
        "clock_id": "camera-clock",
        "clock_epoch": "epoch-1",
        "supersedes": None,
    }
    body["checksum"] = gamst_temporal.packet_checksum(body)
    core_digest = body["checksum"]["sha256"]
    if not crew_packet.verify_packet_checksum(body):
        raise SystemExit("Crew checksum verification returned false")
    parsed = crew_packet.EventPacket.from_dict(body)
    if parsed.to_dict()["checksum"]["sha256"] != core_digest:
        raise SystemExit("Crew re-serialization changed the checksum")
    if not bench_contracts.verify_packet_checksum(body):
        raise SystemExit("bench checksum verification returned false")
    if gamst_temporal.packet_sha256(body) != bench_contracts.packet_sha256(body):
        raise SystemExit("Core and bench canonical digests differ")

    print(json.dumps({
        "core_head": _head(repo_root),
        "bench_head": _head(args.bench_root),
        "algorithm": body["checksum"]["algorithm"],
        "version": body["checksum"]["version"],
        "sha256": core_digest,
        "crew_verified": True,
        "bench_verified": True,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

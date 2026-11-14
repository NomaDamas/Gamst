"""Explicit native EventCD source -> isolated context -> SensorStreams/MCP.

Run from an installed gamst environment with python -m examples.openeb_stream.
No SDK installation, camera discovery or clock calibration is performed.
"""
import argparse
import asyncio
import json

from humansense import HumanSense, OpenEBProcess, ProcessProvider, SensorStreams
from humansense.mcp import MCPServer


async def run(args):
    selection = dict(kind=args.kind, source=args.source, source_id=args.source_id,
                     sensor_id=args.sensor_id, sensor_origin_us=args.sensor_origin_us,
                     capture_origin_ms=args.capture_origin_ms, calibration_id=args.calibration_id,
                     coordinate_frame=args.coordinate_frame)
    source = await OpenEBProcess.open(args.capture_python, selection,
                                      read_timeout=args.read_timeout)
    try:
        print(json.dumps({'selection': source.selection, 'context_config': source.config,
                          'epoch': source.epoch, 'pacing': args.pacing,
                          'native_acceptance_verified': False}))
        async with HumanSense(consent=True) as human:
            streams = SensorStreams(human)
            spec = ProcessProvider.capability('openeb.eventcd', source.config)
            async def factory():
                return await ProcessProvider.open(args.context_python, 'openeb.eventcd', source.config)
            streams.register(spec, factory)
            try:
                await streams.open(args.source_id, args.sensor_id, [spec.id])
                server = MCPServer(human)
                loop = asyncio.get_running_loop()
                anchor = loop.time()
                next_emit = anchor
                for _ in range(args.max_frames):
                    try:
                        frame = await source.next_frame(managed=True)
                    except StopAsyncIteration:
                        break
                    if args.pacing == 'capture-relative':
                        # Pace playback from the declared origin, never rewrite capture time.
                        due = anchor + (frame.timestamp_ms - args.capture_origin_ms) / 1000
                        await asyncio.sleep(max(0, due - loop.time()))
                    await streams.submit(frame)
                    await streams.drain()
                    if loop.time() >= next_emit:
                        context = await server.call_tool('get_observations', {'capability_id': spec.id})
                        print(json.dumps({'context': context}))
                        next_emit = loop.time() + args.context_interval_ms / 1000
            finally:
                await streams.clear()
    finally:
        await source.close()
        print(json.dumps({'capture_diagnostics': source.diagnostics()}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('camera', 'recording'), required=True)
    parser.add_argument('--source', required=True, help='Exact camera serial or absolute existing .raw path')
    parser.add_argument('--capture-python', required=True, help='Absolute Python with gamst and pinned OpenEB SDK')
    parser.add_argument('--context-python', required=True, help='Absolute Python with gamst and NumPy')
    parser.add_argument('--source-id', required=True)
    parser.add_argument('--sensor-id', required=True)
    parser.add_argument('--sensor-origin-us', required=True)
    parser.add_argument('--capture-origin-ms', type=int, required=True)
    parser.add_argument('--calibration-id', required=True)
    parser.add_argument('--coordinate-frame', required=True)
    parser.add_argument('--max-frames', type=int, required=True)
    parser.add_argument('--context-interval-ms', type=int, default=1000)
    parser.add_argument('--read-timeout', type=float, default=5)
    parser.add_argument('--pacing', choices=('none', 'capture-relative'), required=True)
    args = parser.parse_args()
    if not 1 <= args.max_frames <= 1000000:
        parser.error('--max-frames must be 1..1000000')
    if not 10 <= args.context_interval_ms <= 3600000:
        parser.error('--context-interval-ms must be 10..3600000')
    if args.kind == 'camera' and args.pacing != 'none':
        parser.error('camera capture requires --pacing none')
    asyncio.run(run(args))


if __name__ == '__main__':
    main()

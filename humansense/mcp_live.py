"""MCP stdio bridge to an already running local dashboard/WebSocket session."""
import argparse
import asyncio
import json
import sys
from urllib.parse import urlsplit


async def bridge(url):
    from websockets.asyncio.client import connect
    parsed = urlsplit(url)
    if parsed.scheme != 'ws' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('The live MCP bridge only connects to loopback WebSocket servers')
    async with connect(url, max_size=1048576) as socket:
        # MCP clients do not consume our optional custom event notifications.
        await socket.send(json.dumps({'jsonrpc': '2.0', 'id': '__bridge_subscription__',
                                      'method': 'humansense/subscribe', 'params': {'types': []}}))
        async def receive():
            async for raw in socket:
                message = json.loads(raw)
                if message.get('id') == '__bridge_subscription__':
                    continue
                sys.stdout.write(json.dumps(message, ensure_ascii=False) + '\n')
                sys.stdout.flush()
        async def transmit():
            while True:
                line = await asyncio.to_thread(sys.stdin.readline)
                if not line:
                    return
                if len(line) > 1048576:
                    raise ValueError('MCP request exceeds 1 MiB')
                await socket.send(line)
        tasks = [asyncio.create_task(receive()), asyncio.create_task(transmit())]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            task.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='ws://127.0.0.1:8765')
    args = parser.parse_args()
    try:
        asyncio.run(bridge(args.url))
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()

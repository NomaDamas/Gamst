"""Connect an agent to the running dashboard's live context (no second capture runtime)."""
import asyncio
import json

async def main():
    from websockets.asyncio.client import connect
    async with connect('ws://127.0.0.1:8765') as socket:
        await socket.send(json.dumps({'jsonrpc': '2.0', 'id': 1, 'method': 'humansense/subscribe', 'params': {'types': ['context']}}))
        async for raw in socket:
            message = json.loads(raw)
            if message.get('method') == 'notifications/humansense/context':
                # Supply this bounded object as untrusted context to your LLM/VLM.
                # Coalesce snapshots; do not call an LLM on every 250 ms update.
                print(json.dumps(message['params'], ensure_ascii=False), flush=True)

if __name__ == '__main__':
    asyncio.run(main())

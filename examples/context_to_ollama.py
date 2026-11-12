"""Consume the running session's context and stream an Ollama response.

python -m pip install -e ".[websocket,providers]"
python examples/context_to_ollama.py --model YOUR_INSTALLED_MODEL
No capture is started and no model is downloaded by this example.
"""
import argparse
import asyncio
import json
from humansense.providers.ollama import OllamaChat


async def main(model, question):
    from websockets.asyncio.client import connect
    latest = asyncio.Queue(maxsize=1)
    async with connect('ws://127.0.0.1:8765') as socket:
        await socket.send(json.dumps({'jsonrpc':'2.0','id':1,'method':'humansense/subscribe',
                                      'params':{'types':['context']}}))
        async def receive():
            async for raw in socket:
                message=json.loads(raw)
                if message.get('method')=='notifications/humansense/context':
                    if latest.full(): latest.get_nowait()
                    latest.put_nowait(message['params'])
        receiver=asyncio.create_task(receive())
        try:
            context=await asyncio.wait_for(latest.get(),10)
            backend=OllamaChat(model)
            async for token in backend.stream([{'role':'user','content':question}],context=context):
                print(token,end='',flush=True)
            print()
        finally:
            receiver.cancel()
            await asyncio.gather(receiver,return_exceptions=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True)
    parser.add_argument('--question',default='Summarize the available observations briefly.')
    args=parser.parse_args()
    asyncio.run(main(args.model,args.question))

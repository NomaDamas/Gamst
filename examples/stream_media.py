"""Replay a mono PCM16 WAV and/or JPEG into the running WebSocket session.

python examples/stream_media.py --wav fixture.wav --image fixture.jpg --consent
The source files are explicitly chosen fixtures, not device recordings.
"""
import argparse
import asyncio
import base64
import json
import struct
import wave
from pathlib import Path


async def main(args):
    from websockets.asyncio.client import connect
    if not args.consent or not (args.wav or args.image):
        raise ValueError('Supply --consent and --wav and/or --image')
    async with connect('ws://127.0.0.1:8765') as socket:
        counter=0
        async def rpc(method,params):
            nonlocal counter
            counter+=1
            await socket.send(json.dumps(dict(jsonrpc='2.0',id=counter,method=method,params=params)))
            while True:
                response=json.loads(await socket.recv())
                if response.get('id')==counter:
                    if 'error' in response: raise RuntimeError(response['error'])
                    return response['result']
        await rpc('humansense/stream/start',{'consent':True})
        try:
            if args.image:
                encoded=base64.b64encode(Path(args.image).read_bytes()).decode()
                print(await rpc('humansense/media',dict(modality='video',jpeg=encoded,sequence=0)))
            if args.wav:
                with wave.open(args.wav,'rb') as audio:
                    if audio.getnchannels()!=1 or audio.getsampwidth()!=2:
                        raise ValueError('Fixture must be mono PCM16 WAV')
                    rate=audio.getframerate();sequence=0
                    while data:=audio.readframes(rate//2):
                        samples=[v[0]/32768 for v in struct.iter_unpack('<h',data)]
                        print(await rpc('humansense/media',dict(modality='audio',samples=samples,sample_rate=rate,sequence=sequence)))
                        sequence+=1
                        await asyncio.sleep(len(samples)/rate)
            # Allow bounded model work to finish; inspect router metrics on another client.
            await asyncio.sleep(3)
            print(await rpc('tools/call',{'name':'get_context','arguments':{}}))
        finally:
            await rpc('humansense/stream/stop',{})

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wav');parser.add_argument('--image');parser.add_argument('--consent',action='store_true')
    asyncio.run(main(parser.parse_args()))

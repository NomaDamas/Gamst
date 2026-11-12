import unittest
from unittest.mock import patch
from types import SimpleNamespace
from humansense.providers.ollama import OllamaChat


class OllamaTests(unittest.IsolatedAsyncioTestCase):
    async def test_incremental_ndjson_and_early_close(self):
        state={'closed':False,'chunks':0}
        class Response:
            async def __aenter__(self):return self
            async def __aexit__(self,*a):state['closed']=True
            def raise_for_status(self):pass
            async def aiter_bytes(self):
                for chunk in [b'{"message":{"content":"one"},"done":false}\n',b'{"message":{"content":"two"},"done":true}\n']:
                    state['chunks']+=1;yield chunk
        class Client:
            def __init__(self,**kw):
                self.kw=kw
                assert not kw['trust_env'] and not kw['follow_redirects']
            async def __aenter__(self):return self
            async def __aexit__(self,*a):pass
            def stream(self,method,url,json):
                assert json['stream'] is True
                assert json['messages'][0]['role']=='system'
                return Response()
        with patch.dict('sys.modules',{'httpx':SimpleNamespace(AsyncClient=Client)}):
            provider=OllamaChat('fixture')
            stream=provider.stream([{'role':'user','content':'hi'}],context={'observations':{}})
            self.assertEqual(await anext(stream),'one')
            self.assertEqual(state['chunks'],1)
            await stream.aclose()
            self.assertTrue(state['closed'])
            parts=[p async for p in provider.stream([],context={})]
            self.assertEqual(parts,['one','two'])

    def test_remote_requires_explicit_opt_in(self):
        with self.assertRaises(PermissionError):OllamaChat('fixture','https://example.com/api/chat')

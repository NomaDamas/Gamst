"""The real FasterWhisperPlugin class (a stand-in model object) through a real Gamst: stand-in plugins hid both bugs."""
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from gamst import Gamst
from gamst.pipeline import Pipeline
from gamst.schema import timestamp

HAS_NUMPY = importlib.util.find_spec('numpy') is not None


class Model:
    def __init__(self, *args, **kwargs):
        pass

    def transcribe(self, samples, **kwargs):
        word = types.SimpleNamespace(word=' 대박', start=.5, end=1., probability=.83)
        return [types.SimpleNamespace(text='대박', words=[word], avg_logprob=-.1, no_speech_prob=.1,
                                      compression_ratio=1.)], None


@unittest.skipUnless(HAS_NUMPY, 'needs numpy')
class WhisperContractTest(unittest.IsolatedAsyncioTestCase):
    async def test_word_records_of_the_real_plugin_are_accepted_and_counted(self):
        import numpy as np
        from gamst.providers.asr_words import WhisperWords
        from gamst.providers.term_events import TermEvents
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules, {
                'faster_whisper': types.SimpleNamespace(WhisperModel=Model)}):
            (Path(folder) / 'model.bin').write_bytes(b'x')
            from gamst.plugins import FasterWhisperPlugin
            module = WhisperWords(FasterWhisperPlugin(folder, window_seconds=1., word_timestamps=True))
            self.assertIn('sha256:', module.version)
            async with Gamst(consent=True) as human:
                pipeline = Pipeline(human, 'words')
                pipeline.add_reference('terms', 'terms', ['대박'])
                pipeline.add_input('microphone', modality='audio')
                pipeline.add_module('asr', module, consumes='microphone')
                terms = TermEvents()
                pipeline.add_module('terms', terms, consumes='asr:speech.transcript_words', spec=terms.spec)
                start = timestamp() - 5000
                for index in range(20):
                    await pipeline.push('microphone', np.zeros(1600, np.float32), timestamp_ms=start + 100 * index,
                                        sample_rate=16000)
                await pipeline.drain()
                counted = await human.count_events('speech.term_events', event='term.spoken')
                self.assertEqual(counted['count'], 2)
                self.assertTrue(counted['evidence'][0]['event']['word_timed'])
                words = [word for record in await human.get_history()
                         if record['observation'].get('name') == 'speech.transcript_words'
                         for word in record['observation']['value']['words']]
                self.assertEqual([set(word) for word in words], [{'text', 'start_ms', 'end_ms'}] * 2)   # no uncalibrated probability


if __name__ == '__main__':
    unittest.main()

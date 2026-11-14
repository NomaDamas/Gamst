import hashlib
import unittest

try:
    import numpy as np
except ImportError:
    np = None

from humansense.sam_audio_windows import prepare_window, validate_outputs


@unittest.skipIf(np is None, 'optional numpy unavailable')
class WindowContract(unittest.TestCase):
    def prepare(self, pcm=None, **options):
        config = dict(sample_rate=48000, prompt='voice on the left', hop_length=1920)
        config.update(options)
        return prepare_window(np.zeros(48000, dtype=np.float32) if pcm is None else pcm, **config)

    def test_immutable_original_and_private_array(self):
        source = np.linspace(-1, 1, 48001, dtype=np.float32)
        anchors = [['+', 0, .04], ['-', .08, .16]]
        result = self.prepare(source, anchors=anchors)
        original = source.tobytes()
        source[:] = 0
        anchors[0][1] = .2
        self.assertEqual(result.pcm, original)
        self.assertEqual(result.pcm_sha256, hashlib.sha256(original).hexdigest())
        self.assertEqual(result.anchors[0], ('+', 0.0, .04))
        copy = result.array(); copy[:] = 0
        self.assertEqual(result.pcm, original)
        self.assertEqual(result.sample_count, 48001)
        self.assertEqual(result.padded_samples, 49920)

    def test_complete_window_limits_and_noncontiguous_copy(self):
        for count in (48000, 192000):
            source = np.ones(count*2, dtype=np.float32)[::2]
            result = self.prepare(source)
            self.assertEqual(len(result.pcm), count*4)
            self.assertTrue(result.array().flags.c_contiguous)
        for count in (0, 47999, 192001):
            with self.assertRaises(ValueError): self.prepare(np.zeros(count, np.float32))

    def test_mono_rate_dtype_and_subclass_are_explicit(self):
        for source in (np.zeros((1,48000),np.float32), np.zeros(48000,np.float64),
                       np.zeros(48000,np.int16), np.ma.array(np.zeros(48000,np.float32)),
                       [0]*48000):
            with self.subTest(kind=type(source)), self.assertRaises(ValueError): self.prepare(source)
        for rate in (16000, 48000.0, True, None):
            with self.assertRaises(ValueError): self.prepare(sample_rate=rate)
        for hop in (0, 48001, 1920.0, True):
            with self.assertRaises(ValueError): self.prepare(hop_length=hop)

    def test_invalid_sample_never_normalized(self):
        for value in (float('nan'),float('inf'),-float('inf'),1.01,-1.01):
            source = np.zeros(48000,np.float32); source[-1] = value
            with self.assertRaises(ValueError): self.prepare(source)
            if np.isnan(value): self.assertTrue(np.isnan(source[-1]))
            else: self.assertEqual(source[-1], np.float32(value))

    def test_utf8_prompt_bound_and_digest_binding(self):
        base = self.prepare(prompt='가'*1365)
        self.assertEqual(len(base.prompt.encode()),4095)
        for prompt in ('', '   ', '가'*1366, '\ud800', None, 'x'*4097):
            with self.assertRaises(ValueError): self.prepare(prompt=prompt)
        variants = [self.prepare(prompt='other'), self.prepare(hop_length=960),
                    self.prepare(anchors=[('+',0,.2)])]
        for value in variants: self.assertNotEqual(value.prompt_sha256, self.prepare().prompt_sha256)
        self.assertEqual(self.prepare(anchors=None).prompt_sha256, self.prepare(anchors=[]).prompt_sha256)

    def test_analytic_hop_edges_and_order(self):
        # Official pinned formula uses ceil for both endpoints; no nearest-floor.
        result = self.prepare(anchors=[('-',.08,.12),('+',.00001,.08),('+',.12,1)])
        self.assertEqual(result.feature_spans, (('-',2,3),('+',1,2),('+',3,25)))

    def test_disappearing_and_overlapping_spans_rejected(self):
        for anchors in ([('+',.001,.002)], [('+',0,.2),('-',.1,.3)],
                        [('+',0,.2),('+',.1,.3)], [('+',0,.2),('+',0,.2)]):
            with self.assertRaises(ValueError): self.prepare(anchors=anchors)
        result = self.prepare(anchors=[('+',0,.04),('-',.04,.08)])
        self.assertEqual(result.feature_spans,(('+',0,1),('-',1,2)))

    def test_malformed_and_unbounded_anchors(self):
        cases = [[('?',0,.1)], [('+',True,.1)], [('+',0,float('nan'))],
                 [('+',0,float('inf'))], [('+',0,10**500)], [('+',-.1,.2)],
                 [('+',.2,.2)], [('+',0,1.1)], [('+',0)], {'+':[0,.1]},
                 [('+',0,.1)]*33]
        for anchors in cases:
            with self.subTest(anchors=repr(anchors)[:80]), self.assertRaises(ValueError):
                self.prepare(anchors=anchors)

    def test_output_exact_padding_trim_and_detached_unclipped_estimates(self):
        target = np.full(49920,2,np.float32); residual = np.full(49920,-3,np.float32)
        out = validate_outputs([target],[residual],sample_count=48001,hop_length=1920)
        self.assertEqual([len(x) for x in out],[48001,48001])
        np.testing.assert_array_equal(out[0],np.full(48001,2,np.float32))
        np.testing.assert_array_equal(out[1],np.full(48001,-3,np.float32))
        out[0][0] = 7
        self.assertEqual(target[0],2)
        self.assertTrue(out[0].flags.c_contiguous)

    def test_native_output_rejects_wrong_batch_length_type_and_tail(self):
        good = np.zeros(49920,np.float32)
        for value in ([], [good,good], (good,), [good[:-1]], [np.zeros(48001,np.float32)],
                      [good.astype(np.float64)], [good.reshape(1,-1)], [np.ma.array(good)]):
            with self.assertRaises(ValueError):
                validate_outputs(value,[good],sample_count=48001,hop_length=1920)
        for bad in (float('nan'),float('inf'),16.1):
            damaged=good.copy(); damaged[-1]=bad  # Invalid padding is not silently discarded.
            with self.assertRaises(ValueError):
                validate_outputs([good],[damaged],sample_count=48001,hop_length=1920)


if __name__ == '__main__':
    unittest.main()

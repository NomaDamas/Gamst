import subprocess
import sys
import unittest

from humansense.providers.visual_microphone import recover, _filters, _bands

try:
    import numpy as np
    from scipy import signal
except ImportError:
    np = signal = None


class Discovery(unittest.TestCase):
    def test_import_is_dependency_light(self):
        p=subprocess.run([sys.executable,'-c',
            'import humansense.providers.visual_microphone as m;import sys;assert not {"numpy","scipy","torch"}&sys.modules.keys();assert "Copyright (c) 2014 MIT" in m.__doc__;assert "Permission is hereby granted" in m.__doc__'],capture_output=True)
        self.assertEqual(p.returncode,0,p.stderr)


@unittest.skipIf(np is None or signal is None,'optional NumPy/SciPy absent')
class Recovery(unittest.TestCase):
    @staticmethod
    def stimulus(count=512, width=64, height=64, frequency=125, fps=2000):
        x,y=np.meshgrid(np.arange(width),np.arange(height))
        shift=.15*np.sin(2*np.pi*frequency*np.arange(count)/fps)
        frames=np.array([128+45*np.cos(2*np.pi*(x-d)/4)+45*np.cos(2*np.pi*(y-d)/4) for d in shift]).round().astype(np.uint8)
        return frames,shift

    def test_known_subpixel_frequency_and_filter_oracle(self):
        frames,shift=self.stimulus()
        result=recover(frames,capture_fps=2000)
        m=result['metadata'];self.assertEqual(m['peak_hz'],125)
        self.assertEqual(m['nyquist_hz'],1000)
        self.assertGreater(abs(np.corrcoef(result['aligned'][20:],shift[20:])[0,1]),.999)
        # Independent filter coefficients/normalization verify waveform output.
        b,a=signal.butter(3,.05,btype='highpass')
        expected=signal.lfilter(b,a,result['aligned']);expected[:10]=expected.mean()
        expected=2*(expected-expected.min())/np.ptp(expected)-1
        np.testing.assert_allclose(result['waveform'],expected,rtol=0,atol=0)
        self.assertFalse(m['official_matlab_parity_verified'])
        self.assertFalse(m['hardware_validated']);self.assertFalse(m['intelligibility_validated'])
        self.assertTrue(m['full_window_lookahead']);self.assertEqual(m['quality'],'not_assessed')
        self.assertEqual(result['waveform'].shape,(512,))
        self.assertEqual(len(m['input_sha256']),64)

    def test_analytic_complex_band_translation(self):
        # A carrier inside the first band shifts phase by exactly -2*pi*d/4.
        x,y=np.meshgrid(np.arange(64),np.arange(64))
        filters=_filters(64,64)
        a=_bands(np.cos(2*np.pi*x/4),filters)[0]
        b=_bands(np.cos(2*np.pi*(x-.1)/4),filters)[0]
        np.testing.assert_allclose(np.angle(b/a),-2*np.pi*.1/4,atol=1e-12)
        self.assertTrue(all(mask.shape==(64,64) for mask in filters))

    def test_constant_static_and_illumination_abstention(self):
        flat=np.full((128,32,32),128,np.uint8)
        self.assertEqual(recover(flat,capture_fps=1000)['metadata']['status'],'insufficient_texture')
        frames,_=self.stimulus(count=128,width=32,height=32)
        frames[:]=frames[0]
        self.assertEqual(recover(frames,capture_fps=1000)['metadata']['status'],'insufficient_variation')
        for i in range(128):flat[i]=i
        self.assertIsNone(recover(flat,capture_fps=1000)['waveform'])

    def test_texture_loss_and_repeated_calls_no_state(self):
        frames,_=self.stimulus(count=128,width=32,height=32)
        baseline=recover(frames,capture_fps=2000)
        damaged=frames.copy();damaged[64]=128
        self.assertEqual(recover(damaged,capture_fps=2000)['metadata']['status'],'texture_lost')
        np.testing.assert_array_equal(recover(frames,capture_fps=2000)['waveform'],baseline['waveform'])
        self.assertFalse(np.shares_memory(baseline['waveform'],baseline['aligned']))

    def test_single_orientation_and_odd_dimensions(self):
        x,y=np.meshgrid(np.arange(65),np.arange(63))
        shift=.15*np.sin(2*np.pi*125*np.arange(256)/2000)
        frames=np.array([128+80*np.cos(2*np.pi*(x-d)/4) for d in shift]).round().astype(np.uint8)
        value=recover(frames,capture_fps=2000)
        self.assertEqual(value['metadata']['peak_hz'],125)
        self.assertTrue(np.isfinite(value['waveform']).all())
        self.assertEqual(value['metadata']['valid_orientations'],1)

    def test_malformed_and_oversized_reject_before_copy(self):
        frames,_=self.stimulus(count=128,width=32,height=32)
        for fps in (True,99,100001,float('nan'),float('inf'),'2000'):
            with self.assertRaises(ValueError):recover(frames,capture_fps=fps)
        for value in (frames.astype(np.float32),frames[:127],frames[:,:,:31],frames[0],
                      np.ma.array(frames),np.broadcast_to(np.uint8(0),(4096,128,128))):
            with self.assertRaises(ValueError):recover(value,capture_fps=2000)

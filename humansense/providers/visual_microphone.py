"""Bounded offline phase-based visual microphone, not ordinary-video ASR.

Single-scale/two-orientation adaptation of Davis et al. SIGGRAPH 2014's
basic released implementation and Simoncelli/Portilla complex steerable bands.
See docs/visual-microphone.md and licenses/visual-microphone.txt.
"""
import math
import hashlib

SOURCE_SHA256 = '443988c19051b8dbdc18d14e153b95ea16e91f70b5bd8e35f035ecb7b35a2b97'
ALGORITHM = 'visualmic.scfpyr1-order1.v1'
MAX_BYTES = 16 * 1024 * 1024


def _dependencies():
    try:
        import numpy as np
        from scipy import signal
    except ImportError as exc:
        raise RuntimeError('visual microphone requires existing NumPy>=1.24 and SciPy>=1.10; no automatic installation') from exc
    return np, signal


def _filters(height, width):
    """First complex band level; LUT interpolation matches original construction.

    No unused residual bands/recursive levels are allocated. FFT origin uses
    MATLAB's ceil((dims+.5)/2) convention, including odd dimensions.
    """
    np, _ = _dependencies()
    cy, cx = math.ceil((height + .5) / 2) - 1, math.ceil((width + .5) / 2) - 1
    x, y = np.meshgrid((np.arange(width) - cx) / (width / 2),
                       (np.arange(height) - cy) / (height / 2))
    radius = np.hypot(x, y)
    radius[cy, cx] = radius[cy, cx - 1]
    radius = np.log2(radius)
    angle = np.arctan2(y, x)
    samples = np.pi * np.arange(-257, 2) / 512
    raised = np.cos(samples) ** 2
    raised[0] = raised[1]; raised[-1] = raised[-2]
    radial_x = -.5 + 2 / np.pi * (samples + np.pi / 4)
    high = np.sqrt(raised)
    low0 = np.interp(radius, radial_x, np.sqrt(np.maximum(0, 1 - high ** 2)))
    band = np.interp(radius, radial_x - 1, high)
    angle_x = np.pi * np.arange(-2049, 1026) / 1024
    wrapped = (np.pi + angle_x) % (2 * np.pi) - np.pi
    angular = 2 * np.cos(angle_x) * (np.abs(wrapped) < np.pi / 2)
    return tuple(-1j * low0 * band * np.interp(angle, angle_x + shift, angular)
                 for shift in (0, np.pi / 2))


def _bands(image, filters):
    np, _ = _dependencies()
    frequency = np.fft.fftshift(np.fft.fft2(image))
    return tuple(np.fft.ifft2(np.fft.ifftshift(frequency * mask)) for mask in filters)


def recover(frames, *, capture_fps):
    """Recover one complete contiguous grayscale ROI window; no temporal state.

    Caller supplies real capture fps (not playback/container fps), fixed ROI and
    shutter/clock provenance. Returns owned local waveform arrays and metadata.
    Arrays must never be inserted directly into an LLM observation. This method
    neither verifies capture continuity nor assigns physical pressure units.
    """
    np, signal = _dependencies()
    if (type(capture_fps) not in (int, float) or not math.isfinite(capture_fps)
            or not 100 <= capture_fps <= 100000):
        raise ValueError('declared true capture_fps must be100..100000Hz')
    if (type(frames) is not np.ndarray or frames.dtype != np.uint8 or frames.ndim != 3
            or not 128 <= frames.shape[0] <= 4096
            or any(not 32 <= v <= 128 for v in frames.shape[1:]) or frames.nbytes > MAX_BYTES):
        raise ValueError('plain uint8 grayscale N,H,W required:128..4096frames,32..128axes,at most16MiB')
    frames = frames.copy()  # Bounded owned snapshot; caller must synchronize capture.
    count, height, width = frames.shape
    filters = _filters(height, width)
    reference = _bands(frames[0].astype(np.float64) / 255, filters)
    reference_phase = [np.angle(value) for value in reference]
    energy = np.array([np.abs(value).sum() for value in reference])
    valid = energy > height * width * 1e-8
    metadata = dict(algorithm=ALGORITHM, source_archive_sha256=SOURCE_SHA256,
        processing_mode='offline', input_sha256=hashlib.sha256(frames.tobytes()).hexdigest(), frames=count, height=height, width=width,
        capture_fps=float(capture_fps), sample_interval_seconds=1 / capture_fps,
        sample_span_seconds=(count - 1) / capture_fps, analysis_duration_seconds=count / capture_fps,
        nyquist_hz=capture_fps / 2, highpass_cutoff_hz=.025 * capture_fps,
        frequency_bin_hz=capture_fps / count, full_window_lookahead=True,
        waveform_unit='normalized_arbitrary', sound_pressure_calibrated=False,
        hardware_validated=False, intelligibility_validated=False,
        official_matlab_parity_verified=False, rolling_shutter_recovery=False,
        quality='not_assessed', valid_orientations=int(valid.sum()))
    if not valid.any():
        return dict(waveform=None, aligned=None, metadata=dict(metadata, status='insufficient_texture', peak_hz=None))
    traces = np.zeros((2, count), np.float64)
    for i, image in enumerate(frames):
        for j, value in enumerate(_bands(image.astype(np.float64) / 255, filters)):
            if not valid[j]:
                continue
            amplitude = np.abs(value)
            total = amplitude.sum()
            if total <= height * width * 1e-8:
                return dict(waveform=None, aligned=None, metadata=dict(metadata, status='texture_lost', peak_hz=None))
            phase = (np.pi + np.angle(value) - reference_phase[j]) % (2 * np.pi) - np.pi
            traces[j, i] = np.mean(phase * amplitude ** 2) / total
    # Strongest usable initial orientation avoids aligning against an empty band.
    anchor = int(np.argmax(energy))
    aligned = np.zeros(count, np.float64)
    shifts = []
    for j in np.flatnonzero(valid):
        correlation = np.convolve(traces[j], traces[anchor][::-1])
        shift = count - 1 - int(np.argmax(correlation))
        aligned += np.roll(traces[j], shift)
        shifts.append(shift)
    b, a = signal.butter(3, .05, btype='highpass')
    waveform = signal.lfilter(b, a, aligned)
    waveform[:10] = waveform.mean()  # Released basic example's transient treatment.
    extent = float(np.ptp(waveform))
    metadata.update(alignment_reference_orientation=anchor, circular_shifts=shifts,
                    raw_phase_signal_range=float(np.ptp(aligned)))
    # This numerical abstention is not a learned/reliable acoustic SNR threshold.
    if not np.isfinite(waveform).all() or extent <= 1e-12:
        return dict(waveform=None, aligned=None, metadata=dict(metadata, status='insufficient_variation', peak_hz=None))
    waveform = 2 * (waveform - waveform.min()) / extent - 1
    spectrum = np.abs(np.fft.rfft(signal.detrend(waveform) * np.hanning(count))) ** 2
    frequencies = np.fft.rfftfreq(count, 1 / capture_fps)
    search = (frequencies >= metadata['highpass_cutoff_hz']) & (frequencies < metadata['nyquist_hz'])
    selected = np.flatnonzero(search)
    peak = int(selected[np.argmax(spectrum[search])])
    metadata.update(status='reconstructed_unvalidated', peak_hz=float(frequencies[peak]),
                    peak_fraction=float(spectrum[peak] / spectrum[search].sum()))
    return dict(waveform=waveform.copy(), aligned=aligned.copy(), metadata=metadata)

"""Bounded numerical speech features; no calibrated voice or health claims."""
import math


def spectral_features(np, samples, rate):
    """25 ms Hann STFT, 10 ms hop, 40 mel powers, orthonormal DCT-II13.

    No padding across capture gaps. Only aggregated spectral features leave the
    provider, never a growing spectrogram. At96kHz/250ms: <=23x2049 bins.
    """
    width=round(rate*.025);hop=round(rate*.010)
    frames=np.stack([samples[n:n+width] for n in range(0,len(samples)-width+1,hop)])
    frames=frames-frames.mean(axis=1,keepdims=True)
    window=np.hanning(width);nfft=1<<(width-1).bit_length()
    power=np.abs(np.fft.rfft(frames*window,n=nfft,axis=1))**2/max(float(np.sum(window**2)),1e-12)
    frequencies=np.fft.rfftfreq(nfft,1/rate)
    mel_max=2595*math.log10(1+rate/1400)
    edges=700*(10**(np.linspace(0,mel_max,42)/2595)-1)
    filters=np.maximum(0,np.minimum((frequencies[None,:]-edges[:-2,None])/(edges[1:-1]-edges[:-2])[:,None],
        (edges[2:,None]-frequencies[None,:])/(edges[2:]-edges[1:-1])[:,None]))
    mel=power@filters.T
    logmel=np.log(np.maximum(mel,1e-12))
    dct=np.cos(np.pi/40*np.arange(13)[:,None]*(np.arange(40)+.5)[None,:])*math.sqrt(2/40)
    dct[0]/=math.sqrt(2)
    cepstra=logmel@dct.T
    norm=power/np.maximum(power.sum(axis=1,keepdims=True),1e-12)
    flux=float(np.mean(np.sum(np.diff(norm,axis=0)**2,axis=1))) if len(frames)>1 else None
    return dict(method='hann_stft_mel40_orthonormal_dct2',frame_samples=width,hop_samples=hop,
        fft_samples=nfft,frame_count=len(frames),mel_power_mean=mel.mean(axis=0).tolist(),
        mfcc_mean=cepstra.mean(axis=0).tolist(),mfcc_std=cepstra.std(axis=0).tolist(),
        spectral_flux=flux,log_power_floor=1e-12,preemphasis=False)


def cycle_features(np,samples,rate,pitch,periodicity,clipping):
    """Positive zero-crossing cycle estimates on eligible near-periodic PCM.

    This deliberately abstains for complex waveforms with extra crossings,
    insufficient cycles, unstable periods, clipping, or absent periodicity.
    """
    result=dict(eligible=False,reason='not_periodic',cycle_count=0,
        hnr_db=None,jitter_local=None,shimmer_local=None,
        method='positive_zero_crossing_cycles_autocorrelation_hnr',
        interpretation='periodic_signal_estimates_not_voice_diagnosis')
    if clipping>0:
        result['reason']='clipped';return result
    if pitch is None or periodicity<.9:return result
    centered=samples-float(np.mean(samples))
    indices=np.flatnonzero((centered[:-1]<=0)&(centered[1:]>0))
    if len(indices)<6:
        result['reason']='insufficient_cycles';return result
    crossings=indices-centered[indices]/(centered[indices+1]-centered[indices])
    periods=np.diff(crossings);expected=rate/pitch
    if np.any(periods<.75*expected) or np.any(periods>1.25*expected):
        result['reason']='ambiguous_or_unstable_cycles';return result
    amplitudes=np.array([float(np.ptp(centered[int(math.ceil(a)):int(math.ceil(b))]))
        for a,b in zip(crossings[:-1],crossings[1:])])
    if np.any(amplitudes<=1e-8):
        result['reason']='insufficient_amplitude';return result
    result.update(eligible=True,reason='eligible_periodic_signal',cycle_count=len(periods),
        hnr_db=10*math.log10(min(periodicity,1-1e-6)/(1-min(periodicity,1-1e-6))),
        jitter_local=float(np.mean(np.abs(np.diff(periods)))/np.mean(periods)),
        shimmer_local=float(np.mean(np.abs(np.diff(amplitudes)))/np.mean(amplitudes)))
    return result

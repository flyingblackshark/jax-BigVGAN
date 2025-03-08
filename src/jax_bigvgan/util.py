import numpy as np
from librosa.filters import mel as librosa_mel_fn
import jax.numpy as jnp
import jax
def dynamic_range_compression_jax(x, C=1, clip_val=1e-5):
    return jnp.log(jnp.clip(x,min=clip_val) * C)
def get_mel(y, n_mels=128,n_fft=2048,win_size=2048,hop_length=512,fmin=40,fmax=16000,clip_val=1e-5,sampling_rate=44100):

    mel_basis = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    pad_left = (win_size - hop_length) //2
    pad_right = max((win_size - hop_length + 1) //2, win_size - y.shape[-1] - pad_left)
    y = jnp.pad(y, ((0,0),(pad_left, pad_right)))
    _,_,spec = jax.scipy.signal.stft(y,nfft=n_fft,noverlap=win_size-hop_length,nperseg=win_size,boundary=None)
    spectrum_win = jnp.sin(jnp.linspace(0, jnp.pi, win_size, endpoint=False)) ** 2
    spec *= spectrum_win.sum()
    spec = jnp.sqrt(spec.real**2 + spec.imag**2 + (1e-9))
    spec = jnp.matmul(mel_basis, spec)
    spec = dynamic_range_compression_jax(spec, clip_val=clip_val)
    return spec
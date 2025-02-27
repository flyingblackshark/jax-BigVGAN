import numpy as np
from librosa.filters import mel as librosa_mel_fn
import jax.numpy as jnp
import jax
def dynamic_range_compression_jax(x, C=1, clip_val=1e-5):
    return jnp.log(jnp.clip(x,min=clip_val) * C)
def get_mel(y, keyshift=0, speed=1,n_mels=128,n_fft=2048,win_size=2048,hop_length=512,fmin=40,fmax=16000,clip_val=1e-5,sampling_rate=44100):
    factor = 2 ** (keyshift / 12)       
    n_fft_new = int(np.round(n_fft * factor))
    win_size_new = int(np.round(win_size * factor))
    hop_length_new = int(np.round(hop_length * speed))
    
    # if torch.min(y) < -1.:
    #     print('min value is ', torch.min(y))
    # if torch.max(y) > 1.:
    #     print('max value is ', torch.max(y))
    

    mel_basis = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    
    pad_left = (win_size_new - hop_length_new) //2
    pad_right = max((win_size_new - hop_length_new + 1) //2, win_size_new - y.shape[-1] - pad_left)
    # if pad_right < y.size(-1):
    #     mode = 'reflect'
    # else:
    #     mode = 'constant'
    y = jnp.pad(y, ((0,0),(pad_left, pad_right)))
    _,_,spec = jax.scipy.signal.stft(y,nfft=n_fft_new,noverlap=win_size_new-hop_length_new,nperseg=win_size_new,boundary=None)
    spectrum_win = jnp.sin(jnp.linspace(0, jnp.pi, win_size_new, endpoint=False)) ** 2
    spec *= spectrum_win.sum()
    spec = jnp.sqrt(spec.real**2 + spec.imag**2 + (1e-9))
    # spec = torch.stft(y, n_fft_new, hop_length=hop_length_new, win_length=win_size_new, window=self.hann_window[keyshift_key],
    #                     center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)                          
    # spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + (1e-9))
    if keyshift != 0:
        size = n_fft // 2 + 1
        resize = spec.size(1)
        if resize < size:
            spec = jnp.pad(spec, ((0, 0),(0, size-resize)))
        spec = spec[:, :size, :] * win_size / win_size_new   
    spec = jnp.matmul(mel_basis, spec)
    spec = dynamic_range_compression_jax(spec, clip_val=clip_val)
    return spec
class DotDict(dict):
    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

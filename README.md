# Nvidia Big-VGAN jax version 
## This version is working perfectly fine. 😀 
### Original https://github.com/NVIDIA/BigVGAN
# Example
```
from jax_bigvgan import load_model
bigvgan_model,bigvgan_params = load_model(model_type="24khz")
rng = {'params': jax.random.PRNGKey(0), 'dropout': jax.random.PRNGKey(0)}
res = jax.jit(bigvgan_model.apply)({"params":bigvgan_params},out[:,ref_len:],rngs=rng)
import soundfile as sf
sf.write("output.wav",res[0,0],samplerate=24000)
```
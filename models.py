from typing import Any
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from jax.scipy.special import i0

# LRELU_SLOPE = 0.1
# def sinc(x):
#     """Implementation of sinc, i.e., sin(pi * x) / (pi * x)."""
#     return jnp.where(
#         x == 0,
#         1.0,
#         jnp.sin(jnp.pi * x) / (jnp.pi * x)
#     )

# Define Kaiser window function in JAX
def kaiser_window(window_length, beta):
    """Generate a Kaiser window of specified length with parameter beta."""
    n = jnp.arange(0, window_length)
    # Standard Kaiser window formula as used in SciPy and PyTorch (symmetric)
    return i0(beta * jnp.sqrt(1 - (2.0 * n / (window_length - 1) - 1)**2)) / i0(beta)

# Define the filter generation function
def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    """Generate a 1D Kaiser-windowed sinc filter."""
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    # Kaiser window parameters
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * jnp.pi * delta_f + 7.95
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21)**0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0

    window = kaiser_window(kernel_size, beta)

    # Time array based on kernel_size parity
    if even:
        time = jnp.arange(-half_size, half_size) + 0.5
    else:
        time = jnp.arange(kernel_size) - half_size

    # Compute filter and normalize
    if cutoff == 0:
        filter_ = jnp.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * jnp.sinc(2 * cutoff * time)
        filter_ /= filter_.sum()  # Normalize to sum to 1

    return filter_.reshape(1, 1, kernel_size)

# Define the LowPassFilter1d module in Flax
class LowPassFilter1d(nn.Module):
    """
    A 1D low-pass filter module using a Kaiser-windowed sinc function, implemented in Flax.
    
    Args:
        cutoff (float): Cutoff frequency (0 to 0.5).
        half_width (float): Half-width of the transition band.
        stride (int): Stride of the convolution. Default is 1.
        padding (bool): Whether to pad the input. Default is True.
        padding_mode (str): Padding mode, 'replicate' or 'constant'. Default is 'replicate'.
        kernel_size (int): Size of the filter kernel. Default is 12.
    """
    cutoff: float
    half_width: float
    stride: int = 1
    padding: bool = True
    padding_mode: str = "edge"
    kernel_size: int = 12

    def setup(self):
        """Initialize the filter and padding parameters."""
        filter = kaiser_sinc_filter1d(self.cutoff, self.half_width, self.kernel_size)
        self.filter = filter[0, 0, :]  # Shape: [kernel_size]
        self.even = self.kernel_size % 2 == 0
        self.pad_left = self.kernel_size // 2 - int(self.even)
        self.pad_right = self.kernel_size // 2

    def __call__(self, x):
        """
        Apply a low-pass filter to the input using 1D convolution.

        Args:
            x (jax.numpy.ndarray): Input tensor of shape [B, C, T].

        Returns:
            jax.numpy.ndarray: Filtered output of shape [B, C, T'].
        """
        
        # Apply padding if necessary
        if self.padding:
            x = jnp.pad(
                x,
                ((0, 0), (0, 0), (self.pad_left, self.pad_right)),
                mode=self.padding_mode
            )
        B, C, T = x.shape
        # T_padded = x.shape[2]
        # Reshape to [B*C, T_padded, 1] for independent convolution on each channel
        x = x.reshape(B * C, T, 1)
        # Filter shape: [kernel_size, 1, 1]
        rhs = self.filter[:, None, None]  # Ensure self.filter is [kernel_size]
        # Perform convolution
        out = jax.lax.conv_general_dilated(
            lhs=x,                      # [B*C, T_padded, 1]
            rhs=rhs,                    # [kernel_size, 1, 1]
            window_strides=(self.stride,),
            padding="VALID",
            dimension_numbers=("NWC", "WIO", "NWC")
        )  # Output: [B*C, T', 1]
        # Extract output spatial dimension
        T_out = out.shape[1]  # T'
        # Reshape back to [B, C, T']
        out = out.reshape(B, C, T_out)
        return out
    
class UpSample1d(nn.Module):
    """
    A 1D upsampling module using a Kaiser-windowed sinc filter, implemented in Flax.
    
    Args:
        ratio (int): Upsampling ratio. Default is 2.
        kernel_size (int, optional): Size of the filter kernel. If None, it is computed based on the ratio.
    """
    ratio: int = 2
    kernel_size: int = None

    def setup(self):
        """Initialize the filter and padding parameters."""
        self.stride = self.ratio
        # Compute kernel_size if not provided
        if self.kernel_size is None:
            self.kernel_size = int(6 * self.ratio // 2) * 2
        # Padding calculations
        self.pad = self.kernel_size // self.ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        # Generate the Kaiser-windowed sinc filter
        filter = kaiser_sinc_filter1d(
            cutoff=0.5 / self.ratio,
            half_width=0.6 / self.ratio,
            kernel_size=self.kernel_size
        )
        self.filter = filter[0, 0, :]  # Shape: [kernel_size]

    def __call__(self, x):
        
        # Padding
        #T_padded = T + self.pad_left + self.pad_right
        x = jnp.pad(x, ((0, 0), (0, 0), (self.pad, self.pad)), mode="edge")

        B, C, T = x.shape

        # Reshape to [B*C, T_padded, 1]
        x = x.reshape(B * C, T, 1)
        rhs = self.filter[:, None, None]  # [kernel_size, 1, 1]
        out = jax.lax.conv_transpose(
            lhs=x,                      # [B*C, T_padded, 1]
            rhs=rhs,                    # [kernel_size, 1, 1]
            strides=(self.stride,),
            padding="VALID",
            dimension_numbers=("NWC", "WIO", "NWC")
        )  # Output: [B*C, W', 1]
        # Scale and crop
        out = self.ratio * out
        #out = out[:, self.pad_left : -self.pad_right, :]
        # Get upsampled time dimension and reshape
        out = out[:, self.pad_left : -self.pad_right]
        T_out = out.shape[1]  # W' is the second dimension
        out = out.reshape(B, C, T_out)  # [B, C, T_out]
        return out


class DownSample1d(nn.Module):
    """
    A 1D downsampling module using a low-pass filter, implemented in Flax.
    
    Args:
        ratio (int): Downsampling ratio. Default is 2.
        kernel_size (int, optional): Size of the filter kernel. If None, it is computed based on the ratio.
    """
    ratio: int = 2
    kernel_size: int = None

    def setup(self):
        """Initialize the low-pass filter."""
        # Compute kernel_size if not provided
        if self.kernel_size is None:
            self.kernel_size = int(6 * self.ratio // 2) * 2
        # Initialize the low-pass filter module
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / self.ratio,
            half_width=0.6 / self.ratio,
            stride=self.ratio,
            kernel_size=self.kernel_size
        )

    def __call__(self, x):
        """
        Apply downsampling to the input.
        
        Args:
            x: Input tensor of shape [B, C, T], where B is batch size, C is channels, T is time.
        
        Returns:
            Downsampled tensor of shape [B, C, T_out], where T_out ≈ T / ratio.
        """
        return self.lowpass(x)
    
class Activation1d(nn.Module):
    activation:nn.Module
    up_ratio: int = 2
    down_ratio: int = 2
    up_kernel_size: int = 12
    down_kernel_size: int = 12
    # x: [B,C,T]
    @nn.compact
    def __call__(self, x):
        x = UpSample1d(self.up_ratio, self.up_kernel_size)(x)
        x = self.activation(x)
        x = DownSample1d(self.down_ratio, self.down_kernel_size)(x)

        return x
    
class Snake(nn.Module):
    """
    Implementation of a sine-based periodic activation function
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter
    References:
        - This activation function is from this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snake(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """
    no_div_by_zero:float = 0.000000001
    alpha_logscale:bool = False
    # def __init__(
    #     self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False
    # ):
    #     """
    #     Initialization.
    #     INPUT:
    #         - in_features: shape of the input
    #         - alpha: trainable parameter
    #         alpha is initialized to 1 by default, higher values = higher-frequency.
    #         alpha will be trained along with the rest of your model.
    #     """
    #     super(Snake, self).__init__()
    #     self.in_features = in_features

    #     # Initialize alpha
    #     self.alpha_logscale = alpha_logscale
    #     if self.alpha_logscale:  # Log scale alphas initialized to zeros
    #         self.alpha = Parameter(torch.zeros(in_features) * alpha)
    #     else:  # Linear scale alphas initialized to ones
    #         self.alpha = Parameter(torch.ones(in_features) * alpha)

    #     self.alpha.requires_grad = alpha_trainable

    #     self.no_div_by_zero = 0.000000001
    @nn.compact
    def __call__(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        Snake ∶= x + 1/a * sin^2 (xa)
        """
        alpha = self.param('alpha', lambda rng, shape: jnp.zeros(shape), x.shape[-1])
        if self.alpha_logscale:
            alpha = jnp.exp(alpha)
        x = x + (1.0 / (alpha + self.no_div_by_zero)) * pow(jnp.sin(x * alpha), 2)

        return x
class SnakeBeta(nn.Module):
    """
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snakebeta(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """
    no_div_by_zero:float = 0.000000001
    alpha_logscale:bool = False
    # def __init__(
    #     self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False
    # ):
    #     """
    #     Initialization.
    #     INPUT:
    #         - in_features: shape of the input
    #         - alpha - trainable parameter that controls frequency
    #         - beta - trainable parameter that controls magnitude
    #         alpha is initialized to 1 by default, higher values = higher-frequency.
    #         beta is initialized to 1 by default, higher values = higher-magnitude.
    #         alpha will be trained along with the rest of your model.
    #     """
    #     super(SnakeBeta, self).__init__()
    #     self.in_features = in_features

    #     # Initialize alpha
    #     self.alpha_logscale = alpha_logscale
    #     if self.alpha_logscale:  # Log scale alphas initialized to zeros
    #         self.alpha = Parameter(torch.zeros(in_features) * alpha)
    #         self.beta = Parameter(torch.zeros(in_features) * alpha)
    #     else:  # Linear scale alphas initialized to ones
    #         self.alpha = Parameter(torch.ones(in_features) * alpha)
    #         self.beta = Parameter(torch.ones(in_features) * alpha)

    #     self.alpha.requires_grad = alpha_trainable
    #     self.beta.requires_grad = alpha_trainable

    #     self.no_div_by_zero = 0.000000001
    @nn.compact
    def __call__(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta ∶= x + 1/b * sin^2 (xa)
        """
        alpha = self.param('alpha', lambda rng, shape: jnp.zeros(shape), x.shape[-2])
        beta = self.param('beta', lambda rng, shape: jnp.zeros(shape), x.shape[-2])
        alpha = alpha[jnp.newaxis,:,jnp.newaxis]  # Line up with x to [B, C, T]
        beta = beta[jnp.newaxis,:,jnp.newaxis]
        if self.alpha_logscale:
            alpha = jnp.exp(alpha)
            beta = jnp.exp(beta)
        x = x + (1.0 / (beta + self.no_div_by_zero)) * jnp.pow(jnp.sin(x * alpha), 2)

        return x
class AMPBlock1(nn.Module):
    channels:int
    kernel_size:int=3
    dilation:tuple=(1, 3, 5)
    def setup(self):
        
        self.convs1 =[
            nn.WeightNorm(nn.Conv(self.channels,[ self.kernel_size], 1, kernel_dilation=self.dilation[0])),
            nn.WeightNorm(nn.Conv(self.channels, [self.kernel_size], 1, kernel_dilation=self.dilation[1])),
            nn.WeightNorm(nn.Conv(self.channels, [self.kernel_size], 1, kernel_dilation=self.dilation[2]))]
        self.convs2 = [
            nn.WeightNorm(nn.Conv(self.channels, [self.kernel_size], 1, kernel_dilation=1)),
            nn.WeightNorm(nn.Conv(self.channels, [self.kernel_size], 1, kernel_dilation=1)),
            nn.WeightNorm(nn.Conv(self.channels, [self.kernel_size], 1, kernel_dilation=1))
        ]
        self.num_layers = len(self.convs1) + len(self.convs2)
        self.activations = [Activation1d(activation=SnakeBeta(alpha_logscale=config.snake_logscale)) for _ in range(self.num_layers)]
        
    def __call__(self, x,train=True):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for a1,c1,a2,c2 in zip(acts1,self.convs1,acts2,self.convs2):
            xt = a1(x)
            xt = c1(xt.transpose(0,2,1)).transpose(0,2,1)
            xt = a2(xt)
            xt = c2(xt.transpose(0,2,1)).transpose(0,2,1)
            x = xt + x
        return x

class Generator(nn.Module):
    config : Any
    def setup(self):
        self.num_kernels = len(self.config.resblock_kernel_sizes)
        self.num_upsamples = len(self.config.upsample_rates)
        self.conv_pre = nn.WeightNorm(nn.Conv(features=self.config.upsample_initial_channel, kernel_size=[7], strides=[1]))
        ups = []
        for i, (u, k) in enumerate(zip(self.config.upsample_rates, self.config.upsample_kernel_sizes)):
            ups.append(
                    nn.WeightNorm(nn.ConvTranspose(
                        self.config.upsample_initial_channel // (2 ** (i + 1)),
                        (k,),
                        (u,),
                        transpose_kernel = True))
                )

        resblocks = []
        for i in range(len(ups)):
            ch = self.config.upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(self.config.resblock_kernel_sizes, self.config.resblock_dilation_sizes):
                resblocks.append(AMPBlock1(ch, k, d))

        self.conv_post =  nn.WeightNorm(nn.Conv(features=1, kernel_size=[7], strides=1 , use_bias=config.use_bias_at_final))
        self.cond = nn.Conv(self.config.upsample_initial_channel, 1)
        self.ups = ups
        self.resblocks = resblocks
        self.upp = int(np.prod(self.config.upsample_rates))
        self.activation_post = Activation1d(activation=SnakeBeta(alpha_logscale=config.snake_logscale))

    def __call__(self, x, train=False):

        x = self.conv_pre(x.transpose(0,2,1)).transpose(0,2,1)
        for i in range(self.num_upsamples):
            x = self.ups[i](x.transpose(0,2,1)).transpose(0,2,1)

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x,train=train)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x,train=train)
            x = xs / self.num_kernels

        x = self.activation_post(x)
        x = self.conv_post(x.transpose(0,2,1)).transpose(0,2,1)
        if config.use_tanh_at_final:
            x = jnp.tanh(x)
        else:
            x = jnp.clip(x, min=-1.0, max=1.0)  # Bound the output to [-1, 1]
        return x

if __name__ == "__main__":
    import flax.traverse_util
    from util import get_mel

    #import torch
    from omegaconf import OmegaConf
    #packs = torch.load("/home/fbs/jax-BigVGAN/bigvgan_generator_3msteps.pt")
    config = OmegaConf.load("./base.yaml")
    model = Generator(config)
    #wav = jnp.ones((1,44100))
    import librosa
    import soundfile as sf
    wav,sr = librosa.load("./test.wav",sr=44100)
    wav = wav[np.newaxis,:]
    mel = get_mel(wav,n_mels=config.num_mels,n_fft=config.n_fft,win_size=config.win_size,hop_length=config.hop_size,fmin=config.fmin)
    mel = np.load("/home/fbs/torch-test/test.np.npy")
    #n_frames = int(44100 // 512) + 1
    #params = model.init(jax.random.PRNGKey(0),mel)
    #flatten_param = flax.traverse_util.flatten_dict(params,sep='.')
    from convert import convert_torch_weights
    params = convert_torch_weights("/home/fbs/jax-BigVGAN/bigvgan_generator.pt")
    rng = {'params': jax.random.PRNGKey(0), 'dropout': jax.random.PRNGKey(0),'rnorms':jax.random.PRNGKey(0)}
    res = model.apply({"params":params},mel,rngs=rng)
    sf.write("output.wav",res[0,0],samplerate=44100)
    #res = model.apply(params,mel,rngs=rng)
    breakpoint()
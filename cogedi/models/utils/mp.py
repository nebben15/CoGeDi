import torch
import numpy as np
    
def normalize(x, dim=None, eps=1e-4):
    """Normalize by vector norm with a blended epsilon for stability."""
    if dim is None:
        dim = list(range(1, x.ndim))
    # Blend numerical eps with RMS to keep scale roughly preserved.
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)

def mp_silu(x):
    """Magnitude-preserving SiLU activation."""
    # Rescale SiLU so unit-variance inputs stay roughly unit-variance.
    return torch.nn.functional.silu(x) / 0.596

def mp_sum(a, b, t=0.5):
    """Variance-preserving weighted residual merge."""
    # print(a.mean(), a.std(), b.mean(), b.std())
    # Weighted residual merge with variance preservation.
    return a.lerp(b, t) / np.sqrt((1 - t) ** 2 + t ** 2)

class MPConv(torch.nn.Module):
    """Magnitude-preserving linear/conv layer with on-the-fly weight norm."""
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, *kernel))

    def forward(self, x, gain=1):
        """Apply a normalized weight transform with optional gain."""
        w = self.weight.to(torch.float32)
        w = normalize(w) # traditional weight normalization
        # Gain rescales weights while keeping overall variance constant.
        w = w * (gain / np.sqrt(w[0].numel()))
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 4
        return torch.nn.functional.conv2d(x, w, padding=(w.shape[-1]//2,))
    
class LazyMPConv(torch.nn.Module):
    """Magnitude-preserving linear/conv layer with lazy input channel inference."""
    def __init__(self, out_channels, kernel):
        super().__init__()
        self.out_channels = out_channels
        self.kernel = kernel
        self.weight = None  # created lazily

    def _initialize_weight(self, in_channels, device, dtype):
        weight = torch.randn(
            self.out_channels,
            in_channels,
            *self.kernel,
            device=device,
            dtype=torch.float32  # keep master in fp32
        )
        self.weight = torch.nn.Parameter(weight)

    def forward(self, x, gain=1):
        # Infer in_channels from input
        if self.weight is None:
            if x.ndim == 2:
                in_channels = x.shape[-1]
            elif x.ndim == 4:
                in_channels = x.shape[1]
            else:
                raise ValueError(f"Unsupported input shape {x.shape}")
            self._initialize_weight(in_channels, x.device, x.dtype)

        w = self.weight.to(torch.float32)
        w = normalize(w)
        w = w * (gain / np.sqrt(w[0].numel()))
        w = w.to(x.dtype)

        if w.ndim == 2:
            return x @ w.t()

        assert w.ndim == 4
        return torch.nn.functional.conv2d(
            x,
            w,
            padding=(w.shape[-1] // 2),
        )


class MPFourier(torch.nn.Module):
    """Random Fourier feature embedder for scalar noise levels."""
    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer('freqs', 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer('phases', 2 * np.pi * torch.rand(num_channels))

    def forward(self, x):
        """Embed a 1D tensor of noise levels into Fourier features."""
        # Random Fourier features turn scalar noise levels into a rich embedding.
        y = x.to(torch.float32)
        y = y.ger(self.freqs.to(torch.float32))
        y = y + self.phases.to(torch.float32)
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype)
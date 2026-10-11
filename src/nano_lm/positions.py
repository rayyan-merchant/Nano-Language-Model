"""Explicit sinusoidal and pairwise rotary position calculations."""
import math
import torch


def sinusoidal(length: int, width: int):
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(torch.arange(0, width, 2) * (-math.log(10000.0) / width))
    angles = positions * frequencies
    result = torch.zeros(length, width)
    result[:, 0::2] = angles.sin()
    result[:, 1::2] = angles[:, :width // 2].cos()
    return result


def apply_rope(x):
    # x: batch, heads, sequence, head_dimension. Adjacent coordinate pairs rotate.
    width = x.shape[-1]
    if width % 2:
        raise ValueError("Even head dimension required")
    positions = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)
    inverse = 10000.0 ** (-torch.arange(0, width, 2, device=x.device).float() / width)
    angles = positions[:, None] * inverse[None, :]
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

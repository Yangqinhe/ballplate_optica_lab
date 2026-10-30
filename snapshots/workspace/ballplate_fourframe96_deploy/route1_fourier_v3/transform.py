"""Differentiable Fourier integral on arbitrary uniform physical grids.

Bluestein convolution, with chirps constructed directly in FP64. No iterative
complex-power drift, no SciPy runtime, no optical/device imports.
"""
import math

import torch
from torch import nn


class FourierAxis(nn.Module):
    def __init__(self, n, m, x0, dx, f0, df, dtype=torch.complex64):
        super().__init__()
        if min(n, m) < 1 or dx <= 0 or df <= 0 or not all(math.isfinite(v) for v in (x0, dx, f0, df)):
            raise ValueError('finite uniform physical axes required')
        self.n, self.m = int(n), int(m)
        self.nfft = 1 << (n+m-2).bit_length()
        j = torch.arange(n, dtype=torch.float64)
        k = torch.arange(m, dtype=torch.float64)
        shift = torch.arange(-(n-1), m, dtype=torch.float64)
        delta = dx*df
        pre = torch.exp(-2j*math.pi*f0*(x0+j*dx)-1j*math.pi*delta*j.square())
        post = torch.exp(-2j*math.pi*k*df*x0-1j*math.pi*delta*k.square())
        kernel = torch.exp(1j*math.pi*delta*shift.square())
        self.register_buffer('pre', pre.to(dtype))
        self.register_buffer('post', post.to(dtype))
        self.register_buffer('kernel_fft', torch.fft.fft(kernel.to(dtype), n=self.nfft))

    def forward(self, values):
        if values.shape[-1] != self.n:
            raise ValueError('wrong transform input axis')
        spectrum = torch.fft.fft(values*self.pre, n=self.nfft)
        result = torch.fft.ifft(spectrum*self.kernel_fft)
        return result[..., self.n-1:self.n-1+self.m]*self.post


def transform2(values, y_axis, x_axis):
    return y_axis(x_axis(values).transpose(-1, -2)).transpose(-1, -2)


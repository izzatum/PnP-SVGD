"""Building blocks of the DRUNet architecture.

Adapted from KAIR, ``models/basicblock.py`` (https://github.com/cszn/KAIR),
MIT License, Copyright (c) 2019 Kai Zhang. Parts of KAIR derive from BasicSR
(https://github.com/XPixelGroup/BasicSR), Apache License 2.0. Only the blocks used by
:class:`~pnpsvgd.model.UNetRes` are kept, with their original module structure so
that published DRUNet weights load unchanged.
"""

from __future__ import annotations

from collections import OrderedDict

import torch.nn as nn

__all__ = [
    "sequential", "conv", "ResBlock",
    "upsample_pixelshuffle", "upsample_upconv", "upsample_convtranspose",
    "downsample_strideconv", "downsample_maxpool", "downsample_avgpool",
]


def sequential(*args: nn.Module) -> nn.Module:
    """Chain modules into one ``nn.Sequential``, flattening nested sequentials.

    A single module is returned as it is.
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            modules.extend(module.children())
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv(in_channels: int = 64, out_channels: int = 64, kernel_size: int = 3, stride: int = 1,
         padding: int = 1, bias: bool = True, mode: str = 'CBR', negative_slope: float = 0.2) -> nn.Module:
    """Build a stack of layers from a mode string.

    Each character adds one layer: ``C`` convolution, ``T`` transposed
    convolution, ``B`` batch norm, ``I`` instance norm, ``R``/``r`` ReLU
    (in place / not in place), ``L``/``l`` leaky ReLU, ``2``/``3``/``4`` pixel
    shuffle, ``U``/``u``/``v`` nearest upsampling by 2/3/4, ``M`` max pooling and
    ``A`` average pooling.
    """
    layers = []
    for t in mode:
        if t == 'C':
            layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'T':
            layers.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                             kernel_size=kernel_size, stride=stride, padding=padding,
                                             bias=bias))
        elif t == 'B':
            layers.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
        elif t == 'I':
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == 'R':
            layers.append(nn.ReLU(inplace=True))
        elif t == 'r':
            layers.append(nn.ReLU(inplace=False))
        elif t == 'L':
            layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif t == 'l':
            layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
        elif t in '234':
            layers.append(nn.PixelShuffle(upscale_factor=int(t)))
        elif t == 'U':
            layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        elif t == 'u':
            layers.append(nn.Upsample(scale_factor=3, mode='nearest'))
        elif t == 'v':
            layers.append(nn.Upsample(scale_factor=4, mode='nearest'))
        elif t == 'M':
            layers.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == 'A':
            layers.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        else:
            raise NotImplementedError(f'Undefined layer type: {t}')
    return sequential(*layers)


class ResBlock(nn.Module):
    """Residual block ``x + f(x)`` where ``f`` is built by :func:`conv`."""

    def __init__(self, in_channels: int = 64, out_channels: int = 64, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, bias: bool = True, mode: str = 'CRC',
                 negative_slope: float = 0.2) -> None:
        super().__init__()
        if in_channels != out_channels:
            raise ValueError('ResBlock needs in_channels == out_channels.')
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]
        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)

    def forward(self, x):
        return x + self.res(x)


def _check_scale_mode(mode: str, scales: str) -> None:
    if not (len(mode) < 4 and mode[0] in scales):
        raise ValueError(f"mode must start with a scale factor in {list(scales)}, "
                         f"e.g. '2' or '2R'; got {mode!r}")


def upsample_pixelshuffle(in_channels: int = 64, out_channels: int = 3, kernel_size: int = 3,
                          stride: int = 1, padding: int = 1, bias: bool = True, mode: str = '2R',
                          negative_slope: float = 0.2) -> nn.Module:
    """Convolution followed by pixel shuffle."""
    _check_scale_mode(mode, '234')
    return conv(in_channels, out_channels * (int(mode[0]) ** 2), kernel_size, stride, padding, bias,
                mode='C' + mode, negative_slope=negative_slope)


def upsample_upconv(in_channels: int = 64, out_channels: int = 3, kernel_size: int = 3,
                    stride: int = 1, padding: int = 1, bias: bool = True, mode: str = '2R',
                    negative_slope: float = 0.2) -> nn.Module:
    """Nearest-neighbour upsampling followed by a convolution."""
    _check_scale_mode(mode, '234')
    uc = {'2': 'UC', '3': 'uC', '4': 'vC'}[mode[0]]
    mode = mode.replace(mode[0], uc)
    return conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode,
                negative_slope=negative_slope)


def upsample_convtranspose(in_channels: int = 64, out_channels: int = 3, kernel_size: int = 2,
                           stride: int = 2, padding: int = 0, bias: bool = True, mode: str = '2R',
                           negative_slope: float = 0.2) -> nn.Module:
    """Transposed convolution with kernel size and stride equal to the scale factor."""
    _check_scale_mode(mode, '234')
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'T')
    return conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)


def downsample_strideconv(in_channels: int = 64, out_channels: int = 64, kernel_size: int = 2,
                          stride: int = 2, padding: int = 0, bias: bool = True, mode: str = '2R',
                          negative_slope: float = 0.2) -> nn.Module:
    """Strided convolution with kernel size and stride equal to the scale factor."""
    _check_scale_mode(mode, '234')
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'C')
    return conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)


def downsample_maxpool(in_channels: int = 64, out_channels: int = 64, kernel_size: int = 3,
                       stride: int = 1, padding: int = 0, bias: bool = True, mode: str = '2R',
                       negative_slope: float = 0.2) -> nn.Module:
    """Max pooling followed by a convolution."""
    _check_scale_mode(mode, '23')
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], 'MC')
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0], negative_slope=negative_slope)
    pool_tail = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode[1:],
                     negative_slope=negative_slope)
    return sequential(pool, pool_tail)


def downsample_avgpool(in_channels: int = 64, out_channels: int = 64, kernel_size: int = 3,
                       stride: int = 1, padding: int = 1, bias: bool = True, mode: str = '2R',
                       negative_slope: float = 0.2) -> nn.Module:
    """Average pooling followed by a convolution."""
    _check_scale_mode(mode, '23')
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], 'AC')
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0], negative_slope=negative_slope)
    pool_tail = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode[1:],
                     negative_slope=negative_slope)
    return sequential(pool, pool_tail)

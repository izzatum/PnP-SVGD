"""DRUNet: the residual U-Net denoiser used as the plug-and-play prior.

Adapted from KAIR, ``models/network_unet.py`` (https://github.com/cszn/KAIR),
MIT License, Copyright (c) 2019 Kai Zhang.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from . import basicblock as B

__all__ = ["UNetRes"]


class UNetRes(nn.Module):
    """DRUNet: a U-Net with residual blocks for Gaussian denoising.

    For plug-and-play use, the network takes two channels: the noisy image and
    a constant map holding the noise level. Layer names match the published
    DRUNet weights (``drunet_gray.pth``), so they load with ``load_state_dict``.

    Parameters
    ----------
    in_nc : int, optional
        Number of input channels. Use 2 for a grayscale image plus its noise-level
        map. Default is 1.
    out_nc : int, optional
        Number of output channels. Default is 1.
    nc : sequence of int, optional
        Channels at each of the four scales. Default is ``(64, 128, 256, 512)``.
    nb : int, optional
        Residual blocks per scale. Default is 4.
    act_mode : str, optional
        Activation inside the residual blocks, as a :func:`~pnpsvgd.basicblock.conv`
        mode character. Default is ``'R'`` (ReLU).
    downsample_mode : {'strideconv', 'avgpool', 'maxpool'}, optional
        Downsampling layer. Default is ``'strideconv'``.
    upsample_mode : {'convtranspose', 'upconv', 'pixelshuffle'}, optional
        Upsampling layer. Default is ``'convtranspose'``.

    References
    ----------
    .. [1] Zhang, K., Li, Y., Zuo, W., Zhang, L., Van Gool, L., and Timofte, R.
       (2022). Plug-and-play image restoration with deep denoiser prior. IEEE
       Transactions on Pattern Analysis and Machine Intelligence, 44(10).

    Examples
    --------
    >>> import torch
    >>> from pnpsvgd.model import UNetRes
    >>> net = UNetRes(in_nc=2, nc=(4, 8, 16, 32), nb=1)
    >>> net(torch.zeros(1, 2, 32, 32)).shape
    torch.Size([1, 1, 32, 32])
    """

    def __init__(self, in_nc: int = 1, out_nc: int = 1, nc: Sequence[int] = (64, 128, 256, 512),
                 nb: int = 4, act_mode: str = 'R', downsample_mode: str = 'strideconv',
                 upsample_mode: str = 'convtranspose') -> None:
        super().__init__()

        downsample_blocks = {'avgpool': B.downsample_avgpool, 'maxpool': B.downsample_maxpool,
                             'strideconv': B.downsample_strideconv}
        upsample_blocks = {'upconv': B.upsample_upconv, 'pixelshuffle': B.upsample_pixelshuffle,
                           'convtranspose': B.upsample_convtranspose}
        if downsample_mode not in downsample_blocks:
            raise NotImplementedError(f'downsample mode [{downsample_mode}] is not found')
        if upsample_mode not in upsample_blocks:
            raise NotImplementedError(f'upsample mode [{upsample_mode}] is not found')
        downsample_block = downsample_blocks[downsample_mode]
        upsample_block = upsample_blocks[upsample_mode]

        def res_blocks(channels: int):
            return [B.ResBlock(channels, channels, bias=False, mode='C' + act_mode + 'C') for _ in range(nb)]

        self.m_head = B.conv(in_nc, nc[0], bias=False, mode='C')

        self.m_down1 = B.sequential(*res_blocks(nc[0]), downsample_block(nc[0], nc[1], bias=False, mode='2'))
        self.m_down2 = B.sequential(*res_blocks(nc[1]), downsample_block(nc[1], nc[2], bias=False, mode='2'))
        self.m_down3 = B.sequential(*res_blocks(nc[2]), downsample_block(nc[2], nc[3], bias=False, mode='2'))

        self.m_body = B.sequential(*res_blocks(nc[3]))

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'), *res_blocks(nc[2]))
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'), *res_blocks(nc[1]))
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'), *res_blocks(nc[0]))

        self.m_tail = B.conv(nc[0], out_nc, bias=False, mode='C')

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        """Denoise ``x0`` of shape ``(n, in_nc, h, w)`` with ``h`` and ``w`` divisible by 8."""
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        return self.m_tail(x + x1)

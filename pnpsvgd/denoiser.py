"""Apply a pretrained DRUNet denoiser to seismic models.

DRUNet expects images scaled to [0, 1] plus a constant noise-level map as a
second input channel. These helpers rescale the models, pad them to a square
size that the network can process, run the network and undo the scaling.
"""

from __future__ import annotations

from math import sqrt
from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["apply_DRUNET", "apply_DRUNET_optimization"]


def _model_device_dtype(model: torch.nn.Module) -> Tuple[torch.device, torch.dtype]:
    """Device and dtype of the model parameters (for example ``cuda`` and ``float16``)."""
    param = next(model.parameters())
    return param.device, param.dtype


def apply_DRUNET(model: torch.nn.Module, img: torch.Tensor, mu: float, sigma: float,
                 shape: Sequence[int], edge_max: int = 512, verb: bool = False) -> torch.Tensor:
    r"""Denoise a batch of flattened models with DRUNet.

    Every model is scaled to [0, 1] with its own minimum and maximum, padded with
    zeros to ``edge_max x edge_max``, denoised at noise level
    :math:`\gamma = \sqrt{\sigma \mu}` and mapped back to its original range.

    Parameters
    ----------
    model : torch.nn.Module
        DRUNet with two input channels, for example
        :class:`~pnpsvgd.model.UNetRes` with ``in_nc=2``. It must be a module with
        parameters: the network runs on their device and in their precision.
    img : torch.Tensor
        Batch of flattened models of shape ``(n, shape[0] * shape[1])``.
    mu : float
        Noise-level factor.
    sigma : float
        Noise-level factor. The network sees the noise level
        :math:`\sqrt{\sigma \mu}`.
    shape : sequence of int
        Two-dimensional shape ``(nz, nx)`` of one model.
    edge_max : int, optional
        Size of the padded square image. It must be at least as large as both
        entries of ``shape`` and divisible by 8. Default is 512.
    verb : bool, optional
        Print the value range before and after scaling. Default is False.

    Returns
    -------
    torch.Tensor
        Denoised models on the CPU, with the batch dimension squeezed out when
        ``n == 1``.

    References
    ----------
    .. [1] Zhang, K., Li, Y., Zuo, W., Zhang, L., Van Gool, L., and Timofte, R.
       (2022). Plug-and-play image restoration with deep denoiser prior. IEEE
       Transactions on Pattern Analysis and Machine Intelligence, 44(10).
    """
    if verb:
        print(f'img, min{img.min()}, max{img.max()}')
    device, dtype = _model_device_dtype(model)

    batch_size, _ = img.size()
    img = img.unsqueeze(1)
    img_min = torch.min(img, -1, keepdim=True)[0]
    numerator = img - img_min
    numerator_max = torch.max(numerator, -1, keepdim=True)[0]
    scaled_img = (numerator / numerator_max).view(batch_size, 1, shape[0], shape[1])
    if verb:
        print(f'scaled_img, min{scaled_img.min()}, max{scaled_img.max()}')

    scaled_img = F.pad(scaled_img, (0, edge_max - shape[1], 0, edge_max - shape[0]),
                       mode='constant', value=0)
    gamma = sqrt(sigma * mu)
    sigma_map = torch.full((batch_size, 1, edge_max, edge_max), gamma, dtype=torch.float,
                           device=scaled_img.device)
    scaled_img = torch.cat((scaled_img, sigma_map), dim=1)

    with torch.no_grad():
        denoise_img = model(scaled_img.to(device=device, dtype=dtype))
    denoise_img = denoise_img[..., :shape[0], :shape[1]].reshape(batch_size, 1, -1)
    denoise_img = denoise_img.to(img.device) * numerator_max + img_min
    if verb:
        print(f'denoise_img, min{denoise_img.min()}, max{denoise_img.max()}')

    return denoise_img.squeeze().detach().cpu()


def apply_DRUNET_optimization(model: torch.nn.Module, img: torch.Tensor, mu: float, sigma: float,
                              edge_max: int = 512, verb: bool = False) -> np.ndarray:
    r"""Denoise a single two-dimensional model with DRUNet.

    This is the variant used as the plug-and-play step of the deterministic
    primal-dual baseline (:func:`~pnpsvgd.pnpseismic.PlugAndPlay_PrimalDual`).

    Parameters
    ----------
    model : torch.nn.Module
        DRUNet with two input channels. It runs on the device and in the
        precision of its parameters.
    img : torch.Tensor
        Model of shape ``(nz, nx)`` on the CPU.
    mu : float
        Noise-level factor.
    sigma : float
        Noise-level factor. The network sees the noise level
        :math:`\sqrt{\sigma \mu}`.
    edge_max : int, optional
        Size of the padded square image. Default is 512.
    verb : bool, optional
        Print the value range before and after scaling. Default is False.

    Returns
    -------
    numpy.ndarray
        Denoised model of shape ``(nz, nx)``, in the precision of the network.

    References
    ----------
    .. [1] Zhang, K., Li, Y., Zuo, W., Zhang, L., Van Gool, L., and Timofte, R.
       (2022). Plug-and-play image restoration with deep denoiser prior. IEEE
       Transactions on Pattern Analysis and Machine Intelligence, 44(10).
    """
    if verb:
        print(f'img, min{img.min()}, max{img.max()}')
    device, dtype = _model_device_dtype(model)

    img_min = img.min()
    numerator = img - img_min
    scaled_img = numerator / numerator.max()
    if verb:
        print(f'scaled_img, min{scaled_img.min()}, max{scaled_img.max()}')

    scaled_img = F.pad(scaled_img[None, None], (0, edge_max - img.shape[1], 0, edge_max - img.shape[0]),
                       mode='constant', value=0)
    gamma = sqrt(sigma * mu)
    sigma_map = torch.full((1, 1, edge_max, edge_max), gamma, dtype=torch.float)
    scaled_img = torch.cat((scaled_img, sigma_map), dim=1)

    with torch.no_grad():
        denoise_img = model(scaled_img.to(device=device, dtype=dtype))
    denoise_img = denoise_img[0, 0, :img.shape[0], :img.shape[1]]
    denoise_img = denoise_img * (numerator.max()) + img_min
    if verb:
        print(f'denoise_img, min{denoise_img.min()}, max{denoise_img.max()}')

    return denoise_img.cpu().detach().numpy()

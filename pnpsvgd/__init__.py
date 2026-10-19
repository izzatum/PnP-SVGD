"""PnP-SVGD: posterior sampling with plug-and-play deep priors.

Reference implementation of Izzatullah et al. (2024), Geophysics 89(2), R137,
https://doi.org/10.1190/geo2023-0035.1.
"""

from .denoiser import apply_DRUNET, apply_DRUNET_optimization
from .kernels import IMQ, RBF
from .model import UNetRes
from .operators import TorchOperator
from .pnpseismic import PlugAndPlay_PrimalDual, PrimalDual_
from .posterior import PosteriorDistribution
from .svgd import SVGD, PnPSVGD
from .utils import callback, callback0, snr

__version__ = "1.0.0"

__all__ = [
    "SVGD", "PnPSVGD", "RBF", "IMQ", "TorchOperator", "PosteriorDistribution",
    "UNetRes", "apply_DRUNET", "apply_DRUNET_optimization",
    "PrimalDual_", "PlugAndPlay_PrimalDual", "snr", "callback", "callback0",
]

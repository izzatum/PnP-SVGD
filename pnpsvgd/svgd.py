"""Stein variational gradient descent (SVGD) and its plug-and-play variant.

The kernels, the PyLops-to-PyTorch operator and the posterior live in their own
modules and are re-exported here, so ``from pnpsvgd.svgd import *`` keeps working
as in the example notebooks.
"""

from __future__ import annotations

from typing import Callable, Tuple

import torch
import torch.autograd as autograd

from .kernels import IMQ, RBF
from .operators import TorchOperator
from .posterior import PosteriorDistribution

__all__ = ["SVGD", "PnPSVGD", "RBF", "IMQ", "TorchOperator", "PosteriorDistribution"]


class SVGD:
    r"""Stein variational gradient descent.

    Moves a set of particles :math:`\{x_i\}_{i=1}^n` towards the posterior
    :math:`p` along the Stein direction

    .. math::

        \phi(x) = \frac{1}{n} \sum_{j=1}^{n} \left[ k(x_j, x)\, \nabla_{x_j} \log p(x_j)
        + \nabla_{x_j} k(x_j, x) \right].

    The first term pulls particles towards high posterior density and the
    second keeps them apart. The particle update itself is done by a PyTorch
    optimizer, which receives :math:`-\phi` as the gradient.

    Parameters
    ----------
    P : PosteriorDistribution
        Target distribution. Any object with a differentiable ``log_prob``
        method that returns a scalar works.
    K : torch.nn.Module
        Kernel, for example :class:`~pnpsvgd.kernels.RBF` or
        :class:`~pnpsvgd.kernels.IMQ`.
    optimizer : torch.optim.Optimizer
        Optimizer built on the particle tensor, for example
        ``torch.optim.Adam([X], lr=0.1)``.
    scheduler : torch.optim.lr_scheduler.LRScheduler, optional
        Learning-rate scheduler, stepped once per iteration.

    References
    ----------
    .. [1] Liu, Q., and Wang, D. (2016). Stein variational gradient descent: a
       general purpose Bayesian inference algorithm. Advances in Neural
       Information Processing Systems 29.

    Examples
    --------
    Sample a standard normal distribution in two dimensions:

    >>> import torch
    >>> from pnpsvgd.svgd import SVGD, RBF
    >>> class StdNormal:
    ...     def log_prob(self, X):
    ...         return -0.5 * (X ** 2).sum()
    >>> _ = torch.manual_seed(0)
    >>> X = torch.randn(50, 2) + 3.0
    >>> sampler = SVGD(StdNormal(), RBF(), torch.optim.Adam([X], lr=0.1))
    >>> for _ in range(200):
    ...     _ = sampler.step(X)
    >>> bool(X.mean(0).abs().max() < 0.5)
    True
    """

    def __init__(self, P, K: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 scheduler=None) -> None:
        self.P = P
        self.K = K
        self.optimizer = optimizer
        self.scheduler = scheduler

    def phi(self, X: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Compute the update direction for all particles.

        Parameters
        ----------
        X : torch.Tensor
            Particles of shape ``(n_particles, n_model)``.

        Returns
        -------
        grad : torch.Tensor
            Negative Stein direction :math:`-\\phi(X)`, to be used as the
            gradient of the particles.
        log_prob : float
            Log-posterior of the current particles.
        """
        X = X.detach().requires_grad_(True)
        log_prob = self.P.log_prob(X)
        score_func = autograd.grad(log_prob, X)[0]

        K_XX = self.K(X, X.detach())
        grad_K = -autograd.grad(K_XX.sum(), X)[0]

        phi = (K_XX.detach().matmul(score_func) + grad_K) / X.size(0)
        return -phi, log_prob.item()

    def step(self, X: torch.Tensor) -> float:
        """Run one SVGD iteration in place.

        Parameters
        ----------
        X : torch.Tensor
            Particles registered with the optimizer. They are updated in place.

        Returns
        -------
        float
            Log-posterior of the particles before the update.
        """
        self.optimizer.zero_grad()
        X.grad, log_prob = self.phi(X)
        self.optimizer.step()
        self._after_update(X)
        if self.scheduler is not None:
            self.scheduler.step()
        return log_prob

    def _after_update(self, X: torch.Tensor) -> None:
        """Hook called after the optimizer step and before the scheduler step."""


class PnPSVGD(SVGD):
    """Plug-and-play SVGD: an SVGD step followed by denoising every particle.

    Each iteration first moves the particles with :class:`SVGD` and then
    replaces every particle :math:`x_i` with :math:`D(x_i)`, the output of a
    pretrained denoiser. The denoiser acts as an implicit prior that keeps the
    particles close to the structures it was trained on.

    Parameters
    ----------
    denoiser : callable
        Function that takes a batch of particles of shape
        ``(n, n_model)`` and returns denoised particles of the same shape, for
        example a wrapper around :func:`~pnpsvgd.denoiser.apply_DRUNET`.
    P : PosteriorDistribution
        Target distribution.
    K : torch.nn.Module
        Kernel.
    optimizer : torch.optim.Optimizer
        Optimizer built on the particle tensor.
    scheduler : torch.optim.lr_scheduler.LRScheduler, optional
        Learning-rate scheduler, stepped once per iteration.
    loops : int, optional
        Denoise the particles in chunks of ``n_particles // loops`` to limit GPU
        memory, so 100 particles with ``loops=5`` go through the denoiser 20 at a
        time. When ``loops`` does not divide the number of particles, one extra
        smaller chunk takes the remainder. Fewer, larger chunks are faster when
        memory allows. By default all particles are denoised in one call.

    References
    ----------
    .. [1] Izzatullah, M., Alkhalifah, T., Romero, J., Corrales, M., Luiken, N.,
       and Ravasi, M. (2024). Posterior sampling with convolutional neural
       network-based plug-and-play regularization with applications to
       poststack seismic inversion. Geophysics, 89(2), R137.
       https://doi.org/10.1190/geo2023-0035.1
    .. [2] Venkatakrishnan, S. V., Bouman, C. A., and Wohlberg, B. (2013).
       Plug-and-play priors for model based reconstruction. IEEE Global
       Conference on Signal and Information Processing.
    .. [3] Liu, Q., and Wang, D. (2016). Stein variational gradient descent: a
       general purpose Bayesian inference algorithm. Advances in Neural
       Information Processing Systems 29.

    Examples
    --------
    >>> import torch
    >>> from pnpsvgd.svgd import PnPSVGD, RBF
    >>> class StdNormal:
    ...     def log_prob(self, X):
    ...         return -0.5 * (X ** 2).sum()
    >>> X = torch.ones(10, 4)
    >>> shrink = lambda z: 0.9 * z
    >>> sampler = PnPSVGD(shrink, StdNormal(), RBF(), torch.optim.SGD([X], lr=0.1), loops=2)
    >>> _ = sampler.step(X)
    >>> X.shape
    torch.Size([10, 4])
    """

    def __init__(self, denoiser: Callable[[torch.Tensor], torch.Tensor], P, K: torch.nn.Module,
                 optimizer: torch.optim.Optimizer, scheduler=None,
                 loops: int | None = None) -> None:
        super().__init__(P, K, optimizer, scheduler)
        self.denoiser = denoiser
        self.loops = loops

    def _after_update(self, X: torch.Tensor) -> None:
        if self.loops is None:
            X.data = self.denoiser(X.data)
            return
        chunk = X.shape[0] // self.loops
        if chunk == 0:
            raise ValueError(f"loops={self.loops} is larger than the number of particles ({X.shape[0]})")
        for i in range(0, X.shape[0], chunk):
            X.data[i:i + chunk] = self.denoiser(X.data[i:i + chunk])

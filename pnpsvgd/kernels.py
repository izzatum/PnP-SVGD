"""Positive-definite kernels used by Stein variational gradient descent."""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["RBF", "IMQ"]


def _squared_distances(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Pairwise squared Euclidean distances between the rows of ``X`` and ``Y``."""
    XX = X.matmul(X.t())
    XY = X.matmul(Y.t())
    YY = Y.matmul(Y.t())
    return -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)


def _bandwidth(dnorm2: torch.Tensor, n: int, sigma: float | None) -> float:
    """Return ``sigma``, or the median-heuristic bandwidth when ``sigma`` is None."""
    if sigma is not None:
        return sigma
    h = np.median(dnorm2.detach().cpu().numpy()) / (2 * np.log(n + 1))
    return np.sqrt(h).item()


class RBF(torch.nn.Module):
    r"""Radial basis function (Gaussian) kernel.

    .. math::

        k(x, y) = \exp\left(-\frac{\|x - y\|^2}{10^{-8} + 2\sigma^2}\right)

    Parameters
    ----------
    sigma : float, optional
        Kernel bandwidth. When None, it is set at every call by a median
        heuristic: :math:`\sigma^2 = \mathrm{med}\{\|x_i - y_j\|^2\} / (2\log(n + 1))`,
        a variant of the rule in Liu and Wang (2016).

    References
    ----------
    .. [1] Liu, Q., and Wang, D. (2016). Stein variational gradient descent: a
       general purpose Bayesian inference algorithm. Advances in Neural
       Information Processing Systems 29.

    Examples
    --------
    >>> import torch
    >>> from pnpsvgd.kernels import RBF
    >>> K = RBF(sigma=1.0)(torch.zeros(3, 2), torch.zeros(3, 2))
    >>> K.shape
    torch.Size([3, 3])
    """

    def __init__(self, sigma: float | None = None) -> None:
        super().__init__()
        self.sigma = sigma

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Evaluate the kernel matrix.

        Parameters
        ----------
        X : torch.Tensor
            Particles of shape ``(n, d)``.
        Y : torch.Tensor
            Particles of shape ``(m, d)``.

        Returns
        -------
        torch.Tensor
            Kernel matrix of shape ``(n, m)``.
        """
        dnorm2 = _squared_distances(X, Y)
        sigma = _bandwidth(dnorm2, X.size(0), self.sigma)
        gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
        return (-gamma * dnorm2).exp()


class IMQ(torch.nn.Module):
    r"""Inverse multiquadric kernel.

    .. math::

        k(x, y) = \left(1 + \frac{\|x - y\|^2}{10^{-8} + 2\sigma^2}\right)^{-1/2}

    Its heavier tails than the RBF kernel keep particles interacting when they
    are far apart.

    Parameters
    ----------
    sigma : float, optional
        Kernel bandwidth. When None, the median heuristic is used, as in
        :class:`RBF`.

    References
    ----------
    .. [1] Gorham, J., and Mackey, L. (2017). Measuring sample quality with
       kernels. Proceedings of the 34th International Conference on Machine
       Learning.

    Examples
    --------
    >>> import torch
    >>> from pnpsvgd.kernels import IMQ
    >>> K = IMQ(sigma=1.0)(torch.zeros(3, 2), torch.zeros(3, 2))
    >>> float(K[0, 0])
    1.0
    """

    def __init__(self, sigma: float | None = None) -> None:
        super().__init__()
        self.sigma = sigma

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Evaluate the kernel matrix.

        Parameters
        ----------
        X : torch.Tensor
            Particles of shape ``(n, d)``.
        Y : torch.Tensor
            Particles of shape ``(m, d)``.

        Returns
        -------
        torch.Tensor
            Kernel matrix of shape ``(n, m)``.
        """
        dnorm2 = _squared_distances(X, Y)
        sigma = _bandwidth(dnorm2, X.size(0), self.sigma)
        gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
        return (1. + gamma * dnorm2) ** (-0.5)

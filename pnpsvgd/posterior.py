"""Posterior distribution for linear inverse problems built on PyLops operators."""

from __future__ import annotations

import torch
from torch.distributions.distribution import Distribution

from .operators import TorchOperator

__all__ = ["PosteriorDistribution"]


class PosteriorDistribution(Distribution):
    r"""Posterior with a Gaussian likelihood and a Gaussian, uniform or TV prior.

    The unnormalised log-posterior of a batch of models :math:`X` is

    .. math::

        \log p(X \mid d) = \sum \log \mathcal{N}(d;\, G X, C_d)
        + \sum \log p_0(X) + \sum \log \mathrm{Laplace}(D X;\, x_{tv}, C_{tv}),

    where :math:`p_0` is a Gaussian :math:`\mathcal{N}(x_0, C_0)` or a uniform
    :math:`\mathcal{U}(x_0, C_0)` prior and the last term, the total-variation
    (TV) prior, is optional. The sums run over all particles and entries, so the
    result is a single scalar.

    Parameters
    ----------
    x0 : torch.Tensor
        Prior mean for the Gaussian prior, or lower bound for the uniform prior.
    C0 : torch.Tensor
        Prior standard deviation for the Gaussian prior, or upper bound for the
        uniform prior.
    d : torch.Tensor
        Observed data.
    Gop : TorchOperator
        Forward modelling operator :math:`G`.
    Cd : torch.Tensor
        Standard deviation of the data noise.
    uniprior : bool, optional
        Use a uniform prior instead of a Gaussian one. Default is False.
    Gtv : TorchOperator, optional
        Derivative operator :math:`D` of the TV prior. The TV prior is used only
        when this is given.
    xtv : torch.Tensor, optional
        Location of the Laplace distribution of the TV prior.
    Ctv : torch.Tensor, optional
        Scale of the Laplace distribution of the TV prior.
    verb : bool, optional
        Print the prior and likelihood terms at every evaluation. Default is False.

    References
    ----------
    .. [1] Izzatullah, M., Alkhalifah, T., Romero, J., Corrales, M., Luiken, N.,
       and Ravasi, M. (2024). Posterior sampling with convolutional neural
       network-based plug-and-play regularization with applications to
       poststack seismic inversion. Geophysics, 89(2), R137.
       https://doi.org/10.1190/geo2023-0035.1
    """

    arg_constraints = {}

    def __init__(
        self,
        x0: torch.Tensor,
        C0: torch.Tensor,
        d: torch.Tensor,
        Gop: TorchOperator,
        Cd: torch.Tensor,
        uniprior: bool = False,
        Gtv: TorchOperator | None = None,
        xtv: torch.Tensor | None = None,
        Ctv: torch.Tensor | None = None,
        verb: bool = False,
    ) -> None:
        super().__init__()
        self.d = d
        self.Gop = Gop
        self.Gtv = Gtv
        self.verb = verb

        if uniprior:
            self.prior = torch.distributions.Uniform(x0, C0)
        else:
            self.prior = torch.distributions.Normal(x0, C0)
        if self.Gtv is not None:
            self.prior1 = torch.distributions.Laplace(xtv, Ctv)
        self.likelihood = torch.distributions.Normal(d, Cd)

    def log_prob(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate the unnormalised log-posterior.

        Parameters
        ----------
        X : torch.Tensor
            Models (particles) of shape ``(n_particles, n_model)``.

        Returns
        -------
        torch.Tensor
            Scalar log-posterior summed over all particles.
        """
        d = self.Gop.apply(X)
        prior_log_prob = self.prior.log_prob(X).sum()
        if self.Gtv is not None:
            prior1_log_prob = self.prior1.log_prob(self.Gtv.apply(X)).sum()
        likelihood_log_prob = self.likelihood.log_prob(d).sum()

        log_prob = prior_log_prob + likelihood_log_prob
        if self.Gtv is not None:
            log_prob += prior1_log_prob

        if self.verb:
            if self.Gtv is None:
                print(f'prior:{prior_log_prob.item()}, likelihood:{likelihood_log_prob.item()}')
            else:
                print(f'prior:{prior_log_prob.item()}, priorTV:{prior1_log_prob.item()}, '
                      f'likelihood:{likelihood_log_prob.item()}')

        return log_prob

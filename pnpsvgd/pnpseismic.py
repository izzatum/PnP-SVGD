"""Deterministic plug-and-play primal-dual inversion, used as a baseline for PnP-SVGD."""

from __future__ import annotations

import time
from typing import Callable, Sequence

from pylops.utils.backend import get_array_module
from pyproximal import ProxOperator
from pyproximal.ProxOperator import _check_tau

__all__ = ["PrimalDual_", "PlugAndPlay_PrimalDual"]


class _Denoise(ProxOperator):
    """Proximal operator that calls a denoiser instead of a closed-form prox.

    Parameters
    ----------
    denoiser : callable
        Function ``denoiser(x, tau)`` that receives the model reshaped to ``dims``
        and the step size ``tau``, and returns the denoised model.
    dims : tuple of int
        Shape used to reshape the flat vector before calling ``denoiser``.
    """

    def __init__(self, denoiser: Callable, dims: Sequence[int]) -> None:
        super().__init__(None, False)
        self.denoiser = denoiser
        self.dims = dims

    def __call__(self, x):
        return 0.

    @_check_tau
    def prox(self, x, tau):
        x = x.reshape(self.dims)
        return self.denoiser(x, tau).ravel()


def PrimalDual_(proxf, proxg, A, x0, tau: float, mu: float, z=None, theta: float = 1.,
                niter: int = 10, gfirst: bool = True, callback: Callable | None = None,
                show: bool = False):
    r"""Primal-dual solver for :math:`\min_x f(x) + g(Ax) + z^T x`.

    Implements the Chambolle-Pock iteration as in
    ``pyproximal.optimization.primaldual.PrimalDual``, with a callback that also
    receives the dual update. That extra output is what the Hess notebook uses to
    track the image produced by the :math:`g` step.

    Parameters
    ----------
    proxf : pyproximal.ProxOperator
        Proximal operator of :math:`f`.
    proxg : pyproximal.ProxOperator
        Proximal operator of :math:`g`. Its ``proxdual`` method is used.
    A : pylops.LinearOperator
        Linear operator inside :math:`g`.
    x0 : numpy.ndarray
        Starting model.
    tau : float
        Primal step size.
    mu : float
        Dual step size. Convergence needs :math:`\tau \mu \|A\|^2 < 1`.
    z : numpy.ndarray, optional
        Linear term added to the objective.
    theta : float, optional
        Relaxation parameter, between 0 and 1. Default is 1.
    niter : int, optional
        Number of iterations. Default is 10.
    gfirst : bool, optional
        Update the dual variable (the :math:`g` step) before the primal one.
        Default is True.
    callback : callable, optional
        Called after every iteration as ``callback(xhat, ydiff)``, where ``xhat``
        is the extrapolated primal variable and ``ydiff`` holds the first
        ``xhat.size`` entries of :math:`-(y_{k+1} - y_k - \mu A \hat{x}) / \mu`.
        With :math:`A = [I; D]` this is the image produced by the :math:`g` step.
    show : bool, optional
        Print a progress log. Default is False.

    Returns
    -------
    numpy.ndarray
        Estimated model.

    References
    ----------
    .. [1] Chambolle, A., and Pock, T. (2011). A first-order primal-dual
       algorithm for convex problems with applications to imaging. Journal of
       Mathematical Imaging and Vision, 40.
    .. [2] Ravasi, M., Valtonen Örnhag, M., Luiken, N., Leblanc, O., and
       Uruñuela, E. (2024). PyProximal: scalable convex optimization in Python.
       Journal of Open Source Software, 9(95), 6326.
    """
    ncp = get_array_module(x0)
    if show:
        tstart = time.time()
        print('Primal-dual: min_x f(Ax) + x^T z + g(x)\n'
              '---------------------------------------------------------\n'
              f'Proximal operator (f): {type(proxf)}\n'
              f'Proximal operator (g): {type(proxg)}\n'
              f'Linear operator (A): {type(A)}\n'
              f"Additional vector (z): {None if z is None else 'vector'}\n"
              f'tau = {tau:10e}\tmu = {mu:10e}\ntheta = {theta:.2f}\t\tniter = {niter:d}\n')
        print('   Itn       x[0]          f           g          z^x       J = f + g + z^x')

    x = x0.copy()
    xhat = x.copy()
    y = ncp.zeros(A.shape[0], dtype=x.dtype)
    yold = y.copy()

    for iiter in range(niter):
        xold = x.copy()
        if gfirst:
            y = proxg.proxdual(yold + mu * A.matvec(xhat), mu)
            y_d = (y - yold - mu * A.matvec(xhat)) / mu * -1
            ATy = A.rmatvec(y)
            if z is not None:
                ATy += z
            x = proxf.prox(x - tau * ATy, tau)
            xhat = x + theta * (x - xold)
            yold = y.copy()
        else:
            ATy = A.rmatvec(y)
            if z is not None:
                ATy += z
            x = proxf.prox(x - tau * ATy, tau)
            xhat = x + theta * (x - xold)
            yprev = y
            y = proxg.proxdual(yprev + mu * A.matvec(xhat), mu)
            y_d = (y - yprev - mu * A.matvec(xhat)) / mu * -1

        if callback is not None:
            callback(xhat, y_d[:xhat.size])

        if show and (iiter < 10 or niter - iiter < 10 or iiter % (niter // 10) == 0):
            pf, pg = proxf(x), proxg(A.matvec(x))
            pf = 0. if isinstance(pf, bool) else pf
            pg = 0. if isinstance(pg, bool) else pg
            zx = 0. if z is None else ncp.dot(z, x)
            print(f'{iiter + 1:6g}  {x[0]:12.5e}  {pf:10.3e}  {pg:10.3e}  {zx:10.3e}'
                  f'      {pf + pg + zx:10.3e}')
    if show:
        print(f'\nTotal time (s) = {time.time() - tstart:.2f}')
        print('---------------------------------------------------------\n')
    return x


def PlugAndPlay_PrimalDual(proxf, denoiser: Callable, A, dims: Sequence[int], x0, tau: float,
                           mu: float, niter: int = 10, gfirst: bool = True,
                           callback: Callable | None = None, show: bool = False):
    """Plug-and-play primal-dual: :func:`PrimalDual_` with a denoiser as ``g``.

    Parameters
    ----------
    proxf : pyproximal.ProxOperator
        Proximal operator of the data term.
    denoiser : callable
        Function ``denoiser(x, tau)`` applied in place of the proximal operator of
        the regulariser, for example a wrapper around
        :func:`~pnpsvgd.denoiser.apply_DRUNET_optimization`.
    A : pylops.LinearOperator
        Linear operator inside the regulariser (often the identity).
    dims : tuple of int
        Shape of the model passed to ``denoiser``.
    x0 : numpy.ndarray
        Starting model.
    tau, mu : float
        Primal and dual step sizes.
    niter : int, optional
        Number of iterations. Default is 10.
    gfirst : bool, optional
        Update the dual variable first. Default is True.
    callback : callable, optional
        See :func:`PrimalDual_`.
    show : bool, optional
        Print a progress log. Default is False.

    Returns
    -------
    numpy.ndarray
        Estimated model.

    References
    ----------
    .. [1] Venkatakrishnan, S. V., Bouman, C. A., and Wohlberg, B. (2013).
       Plug-and-play priors for model based reconstruction. IEEE Global
       Conference on Signal and Information Processing.
    .. [2] Chambolle, A., and Pock, T. (2011). A first-order primal-dual
       algorithm for convex problems with applications to imaging. Journal of
       Mathematical Imaging and Vision, 40.
    """
    proxpnp = _Denoise(denoiser, dims=dims)
    return PrimalDual_(proxf, proxpnp, A, x0=x0, tau=tau, mu=mu, niter=niter,
                       callback=callback, gfirst=gfirst, show=show)

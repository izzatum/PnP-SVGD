"""Metrics and solver callbacks used in the example notebooks."""

from __future__ import annotations

from typing import List

import numpy as np

__all__ = ["snr", "callback", "callback0"]


def snr(x: np.ndarray, x_est: np.ndarray) -> float:
    r"""Signal-to-noise ratio of an estimate, in dB.

    .. math::

        \mathrm{SNR} = 10 \log_{10} \frac{\|x\|_2}{\|x - x_{est}\|_2}

    Parameters
    ----------
    x : numpy.ndarray
        Reference signal.
    x_est : numpy.ndarray
        Estimate of ``x``.

    Returns
    -------
    float
        SNR in dB.

    Examples
    --------
    >>> import numpy as np
    >>> from pnpsvgd.utils import snr
    >>> round(snr(np.ones(4), 0.9 * np.ones(4)), 6)
    10.0
    """
    return 10. * np.log10(np.linalg.norm(x) / np.linalg.norm(x - x_est))


def callback0(x: np.ndarray, x_true: np.ndarray, x_hist: List, err_hist: List) -> None:
    """Record an iterate and its relative error against ``x_true``.

    Parameters
    ----------
    x : numpy.ndarray
        Current iterate.
    x_true : numpy.ndarray
        Reference model.
    x_hist : list
        List that receives ``x``.
    err_hist : list
        List that receives the relative error ``||x - x_true|| / ||x_true||``.
    """
    x_hist.append(x)
    err_hist.append(np.linalg.norm(x - x_true) / np.linalg.norm(x_true))


def callback(x: np.ndarray, y: np.ndarray, x_true: np.ndarray, x_hist: List, y_hist: List,
             err_histx: List, err_histy: List) -> None:
    """Record primal and dual iterates of :func:`~pnpsvgd.pnpseismic.PrimalDual_`.

    Parameters
    ----------
    x : numpy.ndarray
        Primal iterate.
    y : numpy.ndarray
        Image from the dual update, as passed by ``PrimalDual_``.
    x_true : numpy.ndarray
        Reference model.
    x_hist, y_hist : list
        Lists that receive ``x`` and ``y``.
    err_histx, err_histy : list
        Lists that receive the relative errors of ``x`` and ``y`` against ``x_true``.
    """
    x_hist.append(x)
    y_hist.append(y)
    err_histx.append(np.linalg.norm(x - x_true) / np.linalg.norm(x_true))
    err_histy.append(np.linalg.norm(y - x_true) / np.linalg.norm(x_true))

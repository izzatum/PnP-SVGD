import numpy as np


def callback0(x, x_true, x_hist, err_hist):
    """Callback function for calculating error history
    """
    xhist.append(x)
    err_hist.append(np.linalg.norm(x - x_true) / np.linalg.norm(x_true))


def callback(x, y, x_true, x_hist, y_hist, err_histx, err_histy):
    """Callback function for calculating the primal-dual
    variables error history
    """
    x_hist.append(x)
    y_hist.append(y)
    err_histx.append(np.linalg.norm(x - x_true) / np.linalg.norm(x_true))
    err_histy.append(np.linalg.norm(y - x_true) / np.linalg.norm(x_true))


def snr(x, x_est):
    """Computing signal-to-noise ratio
    """
    return 10. * np.log10(np.linalg.norm(x) / np.linalg.norm(x - x_est))

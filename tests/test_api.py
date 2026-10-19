"""Public API checks: the names used by the notebooks, and fixed bugs."""

import numpy as np
import pylops
import pyproximal
import pytest
import torch


def test_notebook_star_imports_provide_every_name_they_use():
    namespace = {}
    exec("from pnpsvgd.svgd import *", namespace)
    exec("from pnpsvgd.denoiser import *", namespace)
    exec("from pnpsvgd.utils import *", namespace)
    exec("from pnpsvgd.pnpseismic import *", namespace)
    exec("from pnpsvgd.pnpseismic import _Denoise", namespace)
    exec("from pnpsvgd.model import UNetRes", namespace)
    used = {"SVGD", "PnPSVGD", "RBF", "PosteriorDistribution", "TorchOperator", "UNetRes",
            "apply_DRUNET", "apply_DRUNET_optimization", "PrimalDual_", "PlugAndPlay_PrimalDual",
            "_Denoise", "callback", "snr"}
    assert used <= set(namespace)


def test_package_exports():
    import pnpsvgd

    assert pnpsvgd.__version__ == "1.0.0"
    for name in pnpsvgd.__all__:
        assert hasattr(pnpsvgd, name)


def test_callback0_records_history():
    from pnpsvgd.utils import callback0

    x_hist, err_hist = [], []
    callback0(np.array([1.0, 1.0]), np.array([2.0, 2.0]), x_hist, err_hist)
    assert len(x_hist) == 1
    assert err_hist == [pytest.approx(0.5)]


def test_primal_dual_callback_with_dual_step_last():
    from pnpsvgd.pnpseismic import PrimalDual_

    rng = np.random.default_rng(0)
    Mop = pylops.MatrixMult(rng.standard_normal((20, 12)))
    l2 = pyproximal.L2(Op=Mop, b=Mop @ rng.standard_normal(12), niter=5)
    seen = []
    PrimalDual_(l2, pyproximal.L21(ndim=2, sigma=1e-2), pylops.Gradient(dims=(4, 3), kind="backward"),
                x0=np.zeros(12), tau=0.5, mu=0.5, niter=3, gfirst=False,
                callback=lambda x, y: seen.append((x.shape, y.shape)))
    assert seen == [((12,), (12,))] * 3


def test_pnpsvgd_rejects_too_many_loops():
    from pnpsvgd.svgd import RBF, PnPSVGD

    class StdNormal:
        def log_prob(self, X):
            return -0.5 * (X ** 2).sum()

    X = torch.ones(3, 2)
    sampler = PnPSVGD(lambda z: z, StdNormal(), RBF(), torch.optim.SGD([X], lr=0.1), loops=5)
    with pytest.raises(ValueError, match="loops=5"):
        sampler.step(X)


def test_denoiser_runs_in_model_precision_on_cpu():
    from pnpsvgd.denoiser import apply_DRUNET
    from pnpsvgd.model import UNetRes

    net = UNetRes(in_nc=2, nc=(4, 8, 16, 32), nb=1).double().eval()
    img = torch.rand(2, 16 * 12, dtype=torch.float64) + 1.0
    out = apply_DRUNET(net, img, 0.1, 0.1, (16, 12), edge_max=16)
    assert out.shape == (2, 16 * 12)
    assert out.dtype == torch.float64

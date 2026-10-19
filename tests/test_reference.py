"""Regression tests against outputs of the original implementation.

``tests/data/reference_outputs.npz`` was produced on the CPU by the code at
commit 889b256, before the package was restructured. Every public function must
reproduce those numbers, so the refactor cannot change the results in the paper.
For the denoiser wrappers, the original ``.cuda()`` and ``.half()`` calls were
replaced by CPU float32 when the reference was recorded.
"""

from pathlib import Path

import numpy as np
import pylops
import pyproximal
import pytest
import torch

from pnpsvgd.denoiser import apply_DRUNET, apply_DRUNET_optimization
from pnpsvgd.model import UNetRes
from pnpsvgd.pnpseismic import PlugAndPlay_PrimalDual, PrimalDual_
from pnpsvgd.svgd import IMQ, RBF, SVGD, PnPSVGD, PosteriorDistribution, TorchOperator
from pnpsvgd.utils import callback, snr

REF = np.load(Path(__file__).parent / "data" / "reference_outputs.npz")
EXACT = dict(rtol=1e-12, atol=1e-12)


def t64(name):
    return torch.from_numpy(REF[name])


@pytest.fixture(autouse=True)
def float64_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


@pytest.mark.parametrize("kernel, key", [(RBF(), "rbf_median"), (RBF(sigma=0.7), "rbf_sigma"),
                                         (IMQ(), "imq_median"), (IMQ(sigma=0.7), "imq_sigma")])
def test_kernels(kernel, key):
    np.testing.assert_allclose(kernel(t64("kern_X"), t64("kern_Y")).numpy(), REF[key], **EXACT)


def operators():
    Gop = pylops.Gradient(dims=(6, 5), edge=True, dtype="float64", kind="forward")
    Aop = pylops.MatrixMult(REF["top_A"], dtype="float64")
    return Gop, Aop


def test_torch_operator_batch():
    Gop, _ = operators()
    x = t64("top_xb").clone().requires_grad_(True)
    y = TorchOperator(Gop, batch=True).apply(x)
    (y * t64("top_wb")).sum().backward()
    np.testing.assert_allclose(y.detach().numpy(), REF["top_yb"], **EXACT)
    np.testing.assert_allclose(x.grad.numpy(), REF["top_gradb"], **EXACT)


def test_torch_operator_single():
    _, Aop = operators()
    x = t64("top_x1").clone().requires_grad_(True)
    y = TorchOperator(Aop).apply(x)
    (y * t64("top_w1")).sum().backward()
    np.testing.assert_allclose(y.detach().numpy(), REF["top_y1"], **EXACT)
    np.testing.assert_allclose(x.grad.numpy(), REF["top_grad1"], **EXACT)


def posteriors():
    Gop, Aop = operators()
    n = 30
    x0, d = t64("post_x0"), t64("post_d")
    A, G = TorchOperator(Aop, batch=True), TorchOperator(Gop, batch=True)
    tv = PosteriorDistribution(x0, 0.5 * torch.ones(n), d, A, 0.1 * torch.ones(9),
                               xtv=torch.zeros(Gop.shape[0]), Gtv=G,
                               Ctv=0.05 * torch.ones(Gop.shape[0]), verb=False)
    plain = PosteriorDistribution(x0, 0.5 * torch.ones(n), d, A, 0.1 * torch.ones(9))
    uni = PosteriorDistribution(x0 - 10.0, x0 + 10.0, d, A, 0.1 * torch.ones(9), uniprior=True)
    return tv, plain, uni


def test_posterior_log_prob():
    tv, plain, uni = posteriors()
    P = t64("post_P")
    np.testing.assert_allclose(tv.log_prob(P).item(), REF["post_logp_tv"], **EXACT)
    np.testing.assert_allclose(plain.log_prob(P).item(), REF["post_logp_plain"], **EXACT)
    np.testing.assert_allclose(uni.log_prob(P).item(), REF["post_logp_uni"], **EXACT)


def test_svgd_direction():
    tv, _, _ = posteriors()
    phi, lp = SVGD(tv, RBF(), None).phi(t64("post_P"))
    np.testing.assert_allclose(phi.numpy(), REF["svgd_phi"], **EXACT)
    np.testing.assert_allclose(lp, REF["svgd_logp"], **EXACT)


def run(cls, opt_name, steps, **kw):
    tv, _, _ = posteriors()
    X = t64("post_P").clone()
    opt = torch.optim.SGD([X], lr=1e-2) if opt_name == "sgd" else torch.optim.Adam([X], lr=1e-1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 20, eta_min=0)
    if cls is SVGD:
        sampler = SVGD(tv, RBF(), opt, sched)
    else:
        sampler = PnPSVGD(lambda z: 0.9 * z + 0.01, tv, RBF(), opt, scheduler=sched, **kw)
    lps = [sampler.step(X) for _ in range(steps)]
    return X.detach().numpy(), np.array(lps)


@pytest.mark.parametrize("cls, opt, kw, key", [
    (SVGD, "sgd", {}, "svgd_sgd"),
    (SVGD, "adam", {}, "svgd_adam"),
    (PnPSVGD, "adam", {}, "pnp_adam"),
    (PnPSVGD, "adam", {"loops": 2}, "pnp_adam_loops"),
])
def test_sampler_steps(cls, opt, kw, key):
    X, lps = run(cls, opt, 3, **kw)
    np.testing.assert_allclose(X, REF[key + "_X"], **EXACT)
    np.testing.assert_allclose(lps, REF[key + "_lp"], **EXACT)


def test_drunet_weight_names_and_shapes():
    sd = UNetRes(in_nc=2).state_dict()
    assert list(sd) == list(REF["drunet_keys"])
    assert [str(tuple(v.shape)) for v in sd.values()] == list(REF["drunet_shapes"])


def small_drunet():
    torch.set_default_dtype(torch.float32)
    net = UNetRes(in_nc=2, nc=(4, 8, 16, 32), nb=1)
    net.load_state_dict({k[len("small_sd__"):]: torch.from_numpy(REF[k])
                         for k in REF.files if k.startswith("small_sd__")})
    return net.eval()


def test_drunet_forward():
    net = small_drunet()
    with torch.no_grad():
        out = net(torch.from_numpy(REF["small_in"]))
    np.testing.assert_allclose(out.numpy(), REF["small_out"], rtol=1e-6, atol=1e-6)


def test_apply_drunet_batch():
    out = apply_DRUNET(small_drunet(), torch.from_numpy(REF["den_batch_in"]), 0.11, 0.01, (20, 24),
                       edge_max=32)
    np.testing.assert_allclose(out.numpy(), REF["den_batch_out"], rtol=1e-6, atol=1e-6)


def test_apply_drunet_single():
    out = apply_DRUNET_optimization(small_drunet(), torch.from_numpy(REF["den_single_in"]), 0.3, 0.5,
                                    edge_max=32)
    np.testing.assert_allclose(out, REF["den_single_out"], rtol=1e-6, atol=1e-6)


def pd_problem():
    return pylops.MatrixMult(REF["pd_M"], dtype="float64"), REF["pd_mtrue"]


def test_primal_dual_and_snr():
    Mop, mtrue = pd_problem()
    l2 = pyproximal.L2(Op=Mop, b=Mop @ mtrue, niter=10, warm=True)
    xh, yh, ex, ey = [], [], [], []
    x = PrimalDual_(l2, pyproximal.L21(ndim=2, sigma=1e-2),
                    pylops.Gradient(dims=(8, 7), edge=True, kind="backward"), x0=np.zeros(56),
                    tau=0.5, mu=0.5, niter=15, callback=lambda a, b: callback(a, b, mtrue, xh, yh, ex, ey))
    np.testing.assert_allclose(x, REF["pd_x"], **EXACT)
    np.testing.assert_allclose(ex, REF["pd_errx"], **EXACT)
    np.testing.assert_allclose(ey, REF["pd_erry"], **EXACT)
    np.testing.assert_allclose(snr(mtrue, x), REF["snr_val"], **EXACT)


def test_plug_and_play_primal_dual():
    Mop, mtrue = pd_problem()
    l2 = pyproximal.L2(Op=Mop, b=Mop @ mtrue, niter=10, warm=True)
    x = PlugAndPlay_PrimalDual(l2, lambda z, tau: 0.8 * z, pylops.Identity(56), (8, 7),
                               x0=np.zeros(56), tau=0.5, mu=0.5, niter=15)
    np.testing.assert_allclose(x, REF["pnp_pd_x"], **EXACT)

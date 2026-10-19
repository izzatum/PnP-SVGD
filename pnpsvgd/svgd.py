import numpy as np
import torch
import torch.autograd as autograd
import torch.optim as optim

from torch.distributions.distribution import Distribution
from pylops import LinearOperator
from pylops.utils import deps


class _TorchOperator(torch.autograd.Function):
    """Wrapper class for PyLops operators into Torch functions"""

    @staticmethod
    def forward(ctx, x, forw, adj):
        ctx.forw = forw
        ctx.adj = adj

        # prepare input
        # bring x to cpu and numpy
        x = x.cpu().detach().numpy()

        # apply forward operator
        y = ctx.forw(x)

        # prepare output
        # move y to torch and device
        y = torch.from_numpy(y)

        return y

    @staticmethod
    def backward(ctx, y):
        # prepare input
        y = y.cpu().detach().numpy()

        # apply adjoint operator
        x = ctx.adj(y)

        # prepare output
        x = torch.from_numpy(x)
        return x, None, None, None, None


class TorchOperator(LinearOperator):
    def __init__(self, Op, batch=False):
        if not batch:
            self.matvec = Op.matvec
            self.rmatvec = Op.rmatvec
        else:
            self.matvec = lambda x: Op.matmat(x.T).T
            self.rmatvec = lambda x: Op.rmatmat(x.T).T
        self.Top = _TorchOperator.apply
        self.shape = Op.shape
        self.dtype = Op.dtype

    def apply(self, x):
        return self.Top(x, self.matvec, self.rmatvec)


class RBF(torch.nn.Module):
    """Radial Basis Function (RBF) kernel"""
    def __init__(self, sigma=None):
        super(RBF, self).__init__()

        self.sigma = sigma

    def forward(self, X, Y):
        XX = X.matmul(X.t())
        XY = X.matmul(Y.t())
        YY = Y.matmul(Y.t())

        dnorm2 = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)

        # Apply the median
        if self.sigma is None:
            np_dnorm2 = dnorm2.detach().cpu().numpy()
            h = np.median(np_dnorm2) / (2 * np.log(X.size(0) + 1))
            sigma = np.sqrt(h).item()
        else:
            sigma = self.sigma

        gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
        K_XY = (-gamma * dnorm2).exp()

        return K_XY


class IMQ(torch.nn.Module):
    """Inverse Multiquadric Kernel (IMQ)"""
    def __init__(self, sigma=None):
        super(IMQ, self).__init__()
        self.sigma = sigma

    def forward(self, X, Y):
        XX = X.matmul(X.t())
        XY = X.matmul(Y.t())
        YY = Y.matmul(Y.t())

        dnorm2 = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)

        # Apply the median
        if self.sigma is None:
            np_dnorm2 = dnorm2.detach().cpu().numpy()
            h = np.median(np_dnorm2) / (2 * np.log(X.size(0) + 1))
            sigma = np.sqrt(h).item()
        else:
            sigma = self.sigma

        gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
        K_XY = (1. + gamma * dnorm2) ** (-0.5)

        return K_XY


class PosteriorDistribution(Distribution):
    """Posterior distribution coupled with Pylops
    """
    def __init__(self, x0, C0, d, Gop, Cd, uniprior=False, Gtv=None, xtv=None, Ctv=None, verb=False):
        """
        Posterior distribution with Gaussian likelihood and uniform/Gaussian/TV prior

        x0: initial model/particles
        C0: model prior covariance
        d: data
        Gop: forward operator
        Cd: data noise covariance
        uniprior: uniform model prior
        Gtv: Derivative operator for TV prior
        xtv: mean model for TV prior
        Ctv: TV covariance
        verb: verbosity
        """
        super(PosteriorDistribution, self).__init__()
        self.d = d
        self.Gop = Gop
        self.Gtv = Gtv
        self.verb = verb

        if not uniprior:
            self.prior = torch.distributions.Normal(x0, C0)
        else:
            self.prior = torch.distributions.Uniform(x0, C0)
        if self.Gtv is not None:
            self.prior1 = torch.distributions.Laplace(xtv, Ctv)
        self.likelihood = torch.distributions.Normal(d, Cd)

    def log_prob(self, X):
        """Compute log probabilities
        """
        d = self.Gop.apply(X)
        prior_log_prob = self.prior.log_prob(X).sum()
        if self.Gtv is not None:
            prior1_log_prob = self.prior1.log_prob(self.Gtv.apply(X)).sum()
        likelihood_log_prob = self.likelihood.log_prob(d).sum()

        # sum probabilities 
        log_prob = prior_log_prob + likelihood_log_prob
        if self.Gtv is not None:
            log_prob += prior1_log_prob

        if self.verb:
            if self.Gtv is None:
                print(f'prior:{prior_log_prob.item()}, likelihood:{likelihood_log_prob.item()}')
            else:
                print(
                    f'prior:{prior_log_prob.item()}, priorTV:{prior1_log_prob.item()}, likelihood:{likelihood_log_prob.item()}')

        return log_prob


class SVGD:
    def __init__(self, P, K, optimizer, scheduler):
        """
        Simulate SVGD for n number of particles

        P: Posterior Distributions
        K: Kernel (RBF or IMQ)
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler
        """
        self.P = P
        self.K = K
        self.optimizer = optimizer
        self.scheduler = scheduler

    def phi(self, X):
        """
        Compute the Stein Gradient. The terms inside the square bracket above.

        X: n number of models (particles)
        """

        X = X.detach().requires_grad_(True)
        log_prob = self.P.log_prob(X)
        score_func = autograd.grad(log_prob, X)[0]

        K_XX = self.K(X, X.detach())
        grad_K = -autograd.grad(K_XX.sum(), X)[0]

        phi = (K_XX.detach().matmul(score_func) + grad_K) / X.size(0)

        return -phi, log_prob.item()

    def step(self, X):
        self.optimizer.zero_grad()
        X.grad, log_prob = self.phi(X)
        self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()

        return log_prob


class PnPSVGD:
    def __init__(self, Denoiser, P, K, optimizer, scheduler=None, loops=None):
        """
        Simulate SVGD for n number of particles
        
        Denoiser: PyTorch denoiser model in lambda
        P: Posterior Distributions
        K: Kernel (RBF or IMQ)
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler
        loops: Number of batches to separate particle into for denoising. This should be tuned to avoid the CNN-based denoiser 
        running into GPU memories issues (the smaller the faster as denoising leverages GPU vectorization)
        """
        self.denoiser = Denoiser
        self.P = P
        self.K = K
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loops = loops

    def phi(self, X):
        """
        Compute the Stein Gradient. The terms inside the square bracket above.

        X: n number of models (particles)
        """

        X = X.detach().requires_grad_(True)
        log_prob = self.P.log_prob(X)
        score_func = autograd.grad(log_prob, X)[0]

        K_XX = self.K(X, X.detach())
        grad_K = -autograd.grad(K_XX.sum(), X)[0]

        phi = (K_XX.detach().matmul(score_func) + grad_K) / X.size(0)

        return -phi, log_prob.item()

    def step(self, X):
        self.optimizer.zero_grad()
        X.grad, log_prob = self.phi(X)
        self.optimizer.step()

        if self.loops is not None:
            for i in range(0, X.shape[0], X.shape[0] // self.loops):
                X.data[i:i + X.shape[0] // self.loops] = self.denoiser(X.data[i:i + X.shape[0] // self.loops])
        else:
            X.data = self.denoiser(X.data)

        if self.scheduler:
            self.scheduler.step()

        return log_prob


"""Bridge between PyLops linear operators and PyTorch autograd."""

from __future__ import annotations

from typing import Callable

import torch

__all__ = ["TorchOperator"]


class _TorchOperatorFunction(torch.autograd.Function):
    """Autograd function: forward applies the operator, backward its adjoint.

    Data move to NumPy on the CPU for the PyLops call and come back as CPU tensors.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, forw: Callable, adj: Callable) -> torch.Tensor:
        ctx.adj = adj
        y = forw(x.cpu().detach().numpy())
        return torch.from_numpy(y)

    @staticmethod
    def backward(ctx, y: torch.Tensor):
        x = ctx.adj(y.cpu().detach().numpy())
        return torch.from_numpy(x), None, None


class TorchOperator:
    """Wrap a PyLops operator so that it can be used inside a PyTorch graph.

    The forward pass applies the operator and the backward pass applies its
    adjoint, so gradients of any loss built on top of the operator are exact.
    PyLops 1.x ships no equivalent wrapper, which is why this class exists.

    Parameters
    ----------
    Op : pylops.LinearOperator
        Operator to wrap.
    batch : bool, optional
        If True, inputs have shape ``(n_samples, Op.shape[1])`` and the operator
        is applied to every row through ``Op.matmat``. If False (default), inputs
        are single vectors of length ``Op.shape[1]``.

    Attributes
    ----------
    shape : tuple
        Shape of the wrapped operator.
    dtype : str or numpy.dtype
        Data type of the wrapped operator.

    Notes
    -----
    The operator runs on the CPU with NumPy. Inputs must be CPU tensors and
    outputs are CPU tensors.

    References
    ----------
    .. [1] Ravasi, M., and Vasconcelos, I. (2020). PyLops: a linear-operator
       Python library for scalable algebra and optimization. SoftwareX, 11,
       100361.

    Examples
    --------
    >>> import numpy as np
    >>> import pylops
    >>> import torch
    >>> from pnpsvgd.operators import TorchOperator
    >>> Op = TorchOperator(pylops.Diagonal(np.array([1.0, 2.0])))
    >>> x = torch.ones(2, dtype=torch.float64, requires_grad=True)
    >>> Op.apply(x).sum().backward()
    >>> x.grad
    tensor([1., 2.], dtype=torch.float64)
    """

    def __init__(self, Op, batch: bool = False) -> None:
        if batch:
            self.matvec = lambda x: Op.matmat(x.T).T
            self.rmatvec = lambda x: Op.rmatmat(x.T).T
        else:
            self.matvec = Op.matvec
            self.rmatvec = Op.rmatvec
        self.shape = Op.shape
        self.dtype = Op.dtype

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the operator to ``x``.

        Parameters
        ----------
        x : torch.Tensor
            Input vector, or a batch of row vectors when ``batch=True``.

        Returns
        -------
        torch.Tensor
            Operator output on the CPU, connected to the autograd graph of ``x``.
        """
        return _TorchOperatorFunction.apply(x, self.matvec, self.rmatvec)

    __call__ = apply

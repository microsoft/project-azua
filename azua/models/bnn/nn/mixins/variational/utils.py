import functools
import math
import operator

import torch
import torch.nn.functional as F


EPS = 1e-4


def prod(iterable):
    return functools.reduce(operator.mul, iterable, 1)


def inverse_softplus(x):
    if torch.is_tensor(x):
        return x.expm1().log()
    else:
        return math.log(math.expm1(x))


def vec_to_chol(x, out=None):
    """Transforms a batch of (d+1)*d/2-dimensional vector to a batch of d x d dimensional lower triangular matrices
    with positive diagonals."""

    # calculate d using quadratic formula from x.shape[-1]
    d_float = 0.5 * (math.sqrt(1 + 8 * x.shape[-1]) - 1)
    d = int(d_float)
    if d != d_float:
        raise ValueError("Trailing dimension of input must be of size d+1 choose 2.")

    if out is None:
        out = x.new_zeros(x.shape[:-1] + (d, d))
    # set diagonal to positive elements
    out[..., torch.arange(d), torch.arange(d)] = F.softplus(x[..., :d])
    # set off-diagonal elements
    tril_row_indices, tril_col_indices = torch.tril_indices(d, d, offset=-1)
    out[..., tril_row_indices, tril_col_indices] = x[..., d:]
    return out

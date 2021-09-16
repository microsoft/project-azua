import math

import torch
import torch.distributions as dist
from torch.distributions import constraints


def kron(t1: torch.Tensor, t2: torch.Tensor):
    r"""Calculates the Kronecker product between batches of matrices, i.e. the elementwise product between every
    element of t1 and the entire matrix t2. Note that in contrast to  the numpy implementation, the present one treats
    any leading dimensions before the first two as batch dimensions and calculates the batch of kronecker products.
    Numpy calculates the product along all dimensions, i.e. for a, b: m x n x k, whereas np.kron(a, b) returns an
    m^2 x n^2 x k^2 array."""

    # a x b -> a x 1 x b x 1
    t1 = t1.unsqueeze(-1).unsqueeze(-3)
    # c x d -> 1 x c x 1 x d
    t2 = t2.unsqueeze(-2).unsqueeze(-4)
    # a x c x b x d
    t_stacked = t1 * t2
    # ac x bd
    return t_stacked.flatten(-4, -3).flatten(-2, -1)


class _RealMatrix(constraints.Constraint):
    def check(self, value: torch.Tensor):
        return value.eq(value).all(-1).all(-1)


real_matrix = _RealMatrix()


def _batch_kf_mahalanobis(bLu: torch.Tensor, bLv: torch.Tensor, bX: torch.Tensor):
    r"""Calculates the squared Mahalanobis distance :math:`vec(\mathbf{x})^\top\mathbf{M}^{-1}vec(\mathbf{x})`
    for a Kronecker factored :math:`\mathbf{M} = \mathbf{U} \otimes \mathbf{V}` with the lower Cholesky factors
    :math:`\mathbf{U} = \mathbf{L}_u\mathbf{L}_u^\top` and :math:`\mathbf{V} = \mathbf{L}_v\mathbf{L}_v^\top`
    being provided as inputs. This implementation defines :math:`vec(\mathbf{X})` as the column vector of the
    concatenated rows of :math:`\mathbf{X}`, which means that the matrix-vector product for a Kronecker factored
    matrix can be calculated as
    :math:`\mathbf{U}\otimes\mathbf{V}vec(\mathbf{X}) = vec(\mathbf{U}\mathbf{X}\mathbf{V}^top`. This defintion
    of the vec operation is equivalent to using `.flatten` with the usual C/row-major order, whereas the definition
    e.g. on Wikipedia (see MatrixNormal for link) of concatenating the columns uses Fortran/column-major order."""
    x1 = bX.transpose(-2, -1).triangular_solve(bLv, upper=False).solution
    x2 = x1.transpose(-2, -1).triangular_solve(bLu, upper=False).solution
    return x2.pow(2).flatten(-2).sum(-1)


class MatrixNormal(dist.Distribution):
    r"""Basic implementation of a matrix normal distribution, which is equivalent to a multivariate normal with
    a Kronecker factored covariance matrix. Makes use of Kronecker product identities to support efficient sampling
    and calculation of log probabilities, entropy and KL divergences.

    See https://en.wikipedia.org/wiki/Matrix_normal_distribution for an overview.

    Besides the mean matrix, it expects the square roots of the row and
    column covariance matrices. Precision matrices are not currently supported, but could be incorporated in a
    similar manner to pytorch's multivariate normal distribution."""

    arg_constraints = {
        "loc": real_matrix,
        "row_scale_tril": constraints.lower_cholesky,
        "col_scale_tril": constraints.lower_cholesky,
    }
    support = constraints.real  # type: ignore
    has_rsample = True

    def __init__(self, loc, row_scale_tril, col_scale_tril, validate_args=None):
        if loc.dim() < 2:
            raise ValueError("loc must be at least two-dimensional")
        rows, cols = loc.shape[-2:]
        if row_scale_tril.dim() < 2 or row_scale_tril.size(-1) != rows or row_scale_tril.size(-2) != rows:
            raise ValueError(
                "row_scale_tril must be at-least two dimensional with the final two dimensions matching "
                "the penultimate dimension of loc"
            )
        if col_scale_tril.dim() < 2 or col_scale_tril.size(-1) != cols or col_scale_tril.size(-2) != cols:
            raise ValueError(
                "col_scale_tril must be at-least two dimensional with the final two dimensions matching "
                "the last dimension of loc"
            )

        batch_shape, event_shape = loc.shape[:-2], loc.shape[-2:]
        super(MatrixNormal, self).__init__(batch_shape, event_shape, validate_args=validate_args)

        self.loc = loc
        self.row_scale_tril = row_scale_tril
        self.col_scale_tril = col_scale_tril

    @property
    def num_rows(self):
        return self._event_shape[-2]

    @property
    def num_cols(self):
        return self._event_shape[-1]

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return (
            self.row_scale_tril.pow(2)
            .sum(-1)
            .unsqueeze(-1)
            .mul(self.col_scale_tril.pow(2).sum(-1))
            .expand(self._batch_shape + self._event_shape)
        )

    def _half_log_det(self):
        """Calculates the determinant of the covariance times 0.5"""
        row_half_log_det = self.row_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        col_half_log_det = self.col_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        return self.num_cols * row_half_log_det + self.num_rows * col_half_log_det

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = torch.randn(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + self.row_scale_tril.matmul(eps).matmul(self.col_scale_tril.transpose(-2, -1))

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        diff = value - self.loc
        M = _batch_kf_mahalanobis(self.row_scale_tril, self.col_scale_tril, diff)
        return -0.5 * (self.num_rows * self.num_cols * math.log(2 * math.pi) + M) - self._half_log_det()

    def entropy(self):
        H = 0.5 * self.num_rows * self.num_cols * (1.0 + math.log(2 * math.pi)) + self._half_log_det()
        if len(self._batch_shape) == 0:
            return H
        else:
            return H.expand(self._batch_shape)

    def to_multivariatenormal(self):
        """Converts this to a 'flat' multivariate normal. This is mainly for testing and to support KL divergences to
        multivariate normals, but typically this will be highly inefficient both in terms of memory and computation."""
        return dist.MultivariateNormal(self.loc.flatten(-2), scale_tril=kron(self.row_scale_tril, self.col_scale_tril),)

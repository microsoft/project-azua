import numpy as np
import pytest
import torch
import torch.distributions as dist
from torch.distributions.multivariate_normal import _batch_mahalanobis

import azua.models.bnn.distributions.matrix_normal as mn


def _rand_psd(d, batch_shape=None):
    if batch_shape is None:
        batch_shape = tuple()
    elif isinstance(batch_shape, int):
        batch_shape = (batch_shape,)
    a = torch.randn(*batch_shape, d, d + 1)
    return a @ a.transpose(-2, -1) + 1e-2 * torch.eye(d)


def test_kron():
    a = torch.ones(2, 2)
    b = torch.arange(6).reshape(3, 2).float()

    assert torch.allclose(mn.kron(a, b), b.repeat(2, 2))


@pytest.mark.parametrize("a_rows,a_cols,b_rows,b_cols", [(2, 3, 4, 5), (10, 12, 3, 6), (11, 3, 8, 19)])
def test_kron_numpy(a_rows, a_cols, b_rows, b_cols):
    a = torch.randn(a_rows, a_cols)
    b = torch.randn(b_rows, b_cols)
    mn_result = mn.kron(a, b)
    np_result = torch.from_numpy(np.kron(a.numpy(), b.numpy()))

    assert torch.allclose(mn_result, np_result)


def test_kron_batch():
    a = torch.randn(2, 2, 2)
    b = torch.randn(2, 3, 3)
    batched_kron = mn.kron(a, b)
    iter_kron = torch.stack([mn.kron(aa, bb) for aa, bb in zip(a, b)])
    assert torch.allclose(batched_kron, iter_kron)


@pytest.mark.parametrize(
    "x_batch,u_batch,v_batch",
    [(None, None, None), ((2,), 2, 2), ((2,), None, None), ((3, 2), 2, 2)],
)
def test_batch_kf_mahalanobis(x_batch, u_batch, v_batch):
    rows, cols = 3, 4
    if x_batch is None:
        x_batch = tuple()
    bX = torch.randn(*x_batch, rows, cols)
    bLU = _rand_psd(rows, u_batch).cholesky()
    bLV = _rand_psd(cols, v_batch).cholesky()
    bL = mn.kron(bLU, bLV)

    kf_result = mn._batch_kf_mahalanobis(bLU, bLV, bX)
    t_result = _batch_mahalanobis(bL, bX.flatten(-2))

    assert torch.allclose(kf_result, t_result)


def test_single_sample_shape():
    matnorm = mn.MatrixNormal(torch.zeros(2, 3), torch.eye(2), torch.eye(3))
    x = matnorm.rsample()
    assert x.shape == (2, 3)


def test_batch_sample_shape():
    matnorm = mn.MatrixNormal(torch.zeros(5, 6), torch.eye(5), torch.eye(6))
    x = matnorm.rsample((10,))
    assert x.shape == (10, 5, 6)


def test_to_multivariate_normal():
    rows, cols = 3, 4
    loc = torch.randn(rows, cols)
    row_cov = _rand_psd(rows)
    col_cov = _rand_psd(cols)

    matnorm = mn.MatrixNormal(loc, row_cov.cholesky(), col_cov.cholesky())
    multinorm = dist.MultivariateNormal(loc.flatten(), mn.kron(row_cov, col_cov))

    assert torch.allclose(matnorm.to_multivariatenormal().loc, multinorm.loc)
    assert torch.allclose(matnorm.to_multivariatenormal().scale_tril, multinorm.scale_tril, atol=1e-4)


def test_batch_to_multivariate_normal():
    batch, rows, cols = 2, 3, 4
    loc = torch.randn(batch, rows, cols)
    row_cov = _rand_psd(rows, batch)
    col_cov = _rand_psd(cols, batch)

    matnorm = mn.MatrixNormal(loc, row_cov.cholesky(), col_cov.cholesky())
    multinorm = dist.MultivariateNormal(loc.flatten(-2), mn.kron(row_cov, col_cov))

    assert torch.allclose(matnorm.to_multivariatenormal().loc, multinorm.loc)
    assert torch.allclose(matnorm.to_multivariatenormal().scale_tril, multinorm.scale_tril, atol=1e-4)


def test_mean():
    rows, cols = 3, 4
    loc = torch.randn(3, 4)
    row_cov = _rand_psd(rows)
    col_cov = _rand_psd(cols)

    matnorm = mn.MatrixNormal(loc, row_cov.cholesky(), col_cov.cholesky())
    multinorm = matnorm.to_multivariatenormal()

    assert torch.allclose(matnorm.mean.flatten(-2), multinorm.mean)


def test_variance():
    rows, cols = 3, 4
    loc = torch.randn(3, 4)
    row_cov = _rand_psd(rows)
    col_cov = _rand_psd(cols)

    matnorm = mn.MatrixNormal(loc, row_cov.cholesky(), col_cov.cholesky())
    multinorm = matnorm.to_multivariatenormal()

    assert torch.allclose(matnorm.variance.flatten(-2), multinorm.variance)


def test_entropy():
    rows, cols = 3, 4
    loc = torch.randn(3, 4)
    row_cov = _rand_psd(rows)
    col_cov = _rand_psd(cols)

    matnorm = mn.MatrixNormal(loc, row_cov.cholesky(), col_cov.cholesky())
    multinorm = matnorm.to_multivariatenormal()

    assert torch.isclose(matnorm.entropy(), multinorm.entropy())


def test_log_prob():
    rows, cols = 3, 4
    loc = torch.randn(rows, cols)
    row_cov = _rand_psd(rows)
    col_cov = _rand_psd(cols)

    matnorm = mn.MatrixNormal(loc, row_cov.cholesky(), col_cov.cholesky())
    # using row major order for vec/flatten operation as implemented in pytorch means that the covariance of the
    # multivariate normal equivalent to the matrix normal has covariance U \otimes V rather than V \otimes U
    # as on Wikipedia (where theye use column major order)
    multinorm = matnorm.to_multivariatenormal()

    x = matnorm.sample()

    assert torch.isclose(matnorm.log_prob(x), multinorm.log_prob(x.flatten()))


def test_sample_cov():
    torch.manual_seed(42)
    rows, cols = 3, 4
    sample_size = int(1e6)

    loc = torch.randn(rows, cols)
    row_cov = _rand_psd(rows)
    col_cov = _rand_psd(cols)

    matnorm = mn.MatrixNormal(loc, row_cov.cholesky(), col_cov.cholesky())
    true_cov = mn.kron(row_cov, col_cov)
    multinorm = matnorm.to_multivariatenormal()

    matnorm_sample = matnorm.sample((sample_size,)).flatten(-2)
    multinorm_sample = multinorm.sample((sample_size,))

    matnorm_cov = torch.from_numpy(np.cov(matnorm_sample.t().numpy())).float()
    multinorm_cov = torch.from_numpy(np.cov(multinorm_sample.t().numpy())).float()

    # we're testing if the empirical covariance of samples from the matrix normal is similarly close to the true
    # covariance as samples from the equivalent multivariate normal. The threshold is relatively high at 0.01, but
    # increasing the sample size -- which slows down the test noticeably -- would enable a smaller threshold,
    # e.g. 0.001 for a 10 times larger sample
    assert torch.isclose(
        matnorm_cov.sub(true_cov).abs().mean(),
        multinorm_cov.sub(true_cov).abs().mean(),
        atol=1e-2,
    )


def test_1d_loc_raises():
    with pytest.raises(ValueError):
        mn.MatrixNormal(torch.randn(3), _rand_psd(3), _rand_psd(3))


def test_1d_row_cov_raises():
    with pytest.raises(ValueError):
        mn.MatrixNormal(torch.randn(3, 4), torch.rand(3), _rand_psd(4))


def test_1d_col_cov_raises():
    with pytest.raises(ValueError):
        mn.MatrixNormal(torch.randn(3, 4), _rand_psd(3), torch.rand(4))


def test_row_cov_mismatch_raises():
    with pytest.raises(ValueError):
        mn.MatrixNormal(torch.randn(3, 4), _rand_psd(2), _rand_psd(4))


def test_col_cov_mismatch_raises():
    with pytest.raises(ValueError):
        mn.MatrixNormal(torch.randn(3, 4), _rand_psd(3), _rand_psd(2))

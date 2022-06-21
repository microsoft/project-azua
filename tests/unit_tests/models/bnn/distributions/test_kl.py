import torch
import torch.distributions as dist

import azua.models.bnn.distributions.matrix_normal as mn


def _rand_psd(d, batch_shape=None):
    if batch_shape is None:
        batch_shape = tuple()
    elif isinstance(batch_shape, int):
        batch_shape = (batch_shape,)
    a = torch.randn(*batch_shape, d, d + 1)
    return a @ a.transpose(-2, -1) + 1e-2 * torch.eye(d)


def test_matmat_kl():
    rows, cols = 3, 4
    loc_p = torch.randn(rows, cols)
    row_cov_p = _rand_psd(rows)
    col_cov_p = _rand_psd(cols)
    p = mn.MatrixNormal(loc_p, row_cov_p.cholesky(), col_cov_p.cholesky())

    loc_q = torch.randn(rows, cols)
    row_cov_q = _rand_psd(rows)
    col_cov_q = _rand_psd(cols)
    q = mn.MatrixNormal(loc_q, row_cov_q.cholesky(), col_cov_q.cholesky())

    kl1 = dist.kl_divergence(p, q)
    kl2 = dist.kl_divergence(p.to_multivariatenormal(), q.to_multivariatenormal())
    assert torch.isclose(kl1, kl2)


def test_matdiag_kl():
    rows, cols = 3, 2
    loc_p = torch.randn(rows, cols)
    row_cov_p = _rand_psd(rows)
    col_cov_p = _rand_psd(cols)
    p = mn.MatrixNormal(loc_p, row_cov_p.cholesky(), col_cov_p.cholesky())
    p_multi = p.to_multivariatenormal()

    loc_q = torch.randn(rows, cols)
    scale_q = torch.rand(rows, cols)
    q = dist.Normal(loc_q, scale_q)
    q_multi = dist.MultivariateNormal(loc_q.flatten(), scale_tril=scale_q.flatten().diag_embed())

    kl1 = dist.kl_divergence(p, q)
    kl2 = dist.kl_divergence(p_multi, q_multi)
    assert torch.isclose(kl1, kl2)

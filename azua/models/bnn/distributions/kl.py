import torch.distributions as dist

from .matrix_normal import MatrixNormal, _batch_kf_mahalanobis


# below are functions for calculating KL divergences with matrix normals. They are registered with pytorch's
# distribution module, through the dist.register_kl decorator so that calling dist.kl_divergence will dispatch
# the call to these functions. The KLs that use an efficient implementation are between two matrix normals and
# between a matrix normal and a diagonal Normal, otherwise the distributions are converted to multivariate normals
# which might be prohibitive in terms of memory usage
@dist.register_kl(MatrixNormal, MatrixNormal)
def _kl_matrixnormal_matrixnormal(p, q):
    if p.event_shape != q.event_shape:
        raise NotImplementedError("Cannot calculate Kl-divergence for matrix normals of different shape.")
    half_log_det_diff = q._half_log_det() - p._half_log_det()
    d = p.event_shape[0] * p.event_shape[1]
    row_trace_term = p.row_scale_tril.triangular_solve(q.row_scale_tril, upper=False)[0].pow(2).sum((-2, -1))
    col_trace_term = p.col_scale_tril.triangular_solve(q.col_scale_tril, upper=False)[0].pow(2).sum((-2, -1))
    trace_term = row_trace_term * col_trace_term
    mahalanobis_term = _batch_kf_mahalanobis(q.row_scale_tril, q.col_scale_tril, q.loc - p.loc)
    return half_log_det_diff + 0.5 * (trace_term + mahalanobis_term - d)


@dist.register_kl(MatrixNormal, dist.Normal)
def _kl_matrixnormal_normal(p, q):
    if p.event_shape != q.batch_shape[-2:]:
        raise ValueError(
            "Cannot calculate KL-divergence if trailing batch dimensions of the factorized normal do not"
            "match the event shape of the matrix normal"
        )
    p_halflogdet = p._half_log_det()
    q_halflogdet = q.scale.log().sum()
    diff_halflogdet = q_halflogdet - p_halflogdet

    d = p.event_shape[0] * p.event_shape[1]

    row_cov_diag = p.row_scale_tril.pow(2).sum(-1).unsqueeze(-1)
    col_cov_diag = p.col_scale_tril.pow(2).sum(-1).unsqueeze(-2)
    trace_term = (q.scale.pow(-2) * row_cov_diag * col_cov_diag).sum((-2, -1))

    delta = q.loc - p.loc
    mahalanobis_term = delta.pow(2).div(q.scale.pow(2)).sum((-2, -1))

    return diff_halflogdet + 0.5 * (trace_term - d + mahalanobis_term)


@dist.register_kl(dist.Normal, MatrixNormal)
def _kl_normal_matrixnormal(p, q):
    return dist.kl_divergence(
        dist.MultivariateNormal(p.loc.flatten(-2), scale_tril=p.scale.diag_embed()), q.to_multivariatenormal(),
    )


@dist.register_kl(MatrixNormal, dist.MultivariateNormal)
def _kl_matrixnormal_multivariatenormal(p, q):
    return dist.kl_divergence(p.to_multivariatenormal(), q)


@dist.register_kl(dist.MultivariateNormal, MatrixNormal)
def _kl_multivariatenormal_matrixnormal(p, q):
    return dist.kl_divergence(p, q.to_multivariatenormal())

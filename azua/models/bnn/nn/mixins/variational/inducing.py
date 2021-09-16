from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist


from .base import VariationalMixin
from .utils import EPS, prod, inverse_softplus, vec_to_chol
from ....distributions import MatrixNormal


__all__ = ["InducingDeterministicMixin", "InducingMixin"]


def _jittered_cholesky(m):
    j = EPS * m.detach().diagonal().mean() * torch.eye(m.shape[-1], device=m.device)
    return m.add(j).cholesky()


class _InducingBase(VariationalMixin):
    def __init__(
        self,
        *args,
        inducing_rows=None,
        inducing_cols=None,
        prior_sd=1.0,
        init_lamda=1e-4,
        learn_lamda=True,
        max_lamda=None,
        q_inducing="diagonal",
        whitened_u=True,
        max_sd_u=None,
        sqrt_width_scaling=False,
        cache_cholesky=False,
        ensemble_size=8,
        **kwargs,
    ):
        """Base class for inducing weight mixins. Implements all parameter-related logic as well as methods for 
        sampling parameters and converting them to the correct shape, the forward pass and calculating KL divergences.
        Subclasses only need to implement the `conditional_mean_and_noise` method to calculate the mean and noise
        as in the reparameterization of a Gaussian for the conditional distribution on the weights.
        The Mixin class is expected to be inherited from in a class that also inherits from a torch.nn.Module that has
        .weight and .bias parameter attributes, such as Linear and Conv layers.

        Args:
            inducing_rows (int): number of inducing rows, i.e. first dimension of U
            inducing_cols (int): number of inducing columns , i.e. second dimension of U
            prior_sd (float): standard deviation of the prior on W.
            init_lamda (float): initial value for the lamda parameter that down-scale the conditional covariance of the
                inference distribution on W (typo is intentional since lambda is a keyword in python)
            learn_lamda (bool): whether to learn lamda or keep it constant and the initial value
            max_lamda (float): maximum value of lamda
            q_inducing (str): 'diagonal', 'matrix', 'full' or 'ensemble'. Determines the structure of the covariance
                that is used for the Gaussian that is the variational posterior on U if not 'ensemble', mixture of
                Delta distributions otherwise.
            whitened_u (bool): whether to whiten U, i.e. make the marginal prior a standard normal
            max_sd_u (float): maximum value of the scale of the variational posterior on U if q_inducing is "diagonal"
            sqrt_width_scaling (bool): whether or not to divide prior_sd by the square root of the number of inputs.
                Setting this flag to true and the prior_sd to 1 corresponds to the 'neal' prior scale. 
        """
        super().__init__(*args, **kwargs)
        self.has_bias = self.bias is not None
        self.whitened_u = whitened_u
        self._weight_shape = self.weight.shape

        # _u caches the per-layer u sample -- this will be needed for the between-layer correlations
        self._u = None

        self._caching_mode = False
        self.reset_cache()

        self._i = 0

        del self.weight
        if self.has_bias:
            del self.bias

        self._d_out = self._weight_shape[0]
        self._d_in = prod(self._weight_shape[1:]) + int(self.has_bias)

        # if no number of inducing rows or columns is given, use the square root of the corresponding dimension
        if inducing_rows is None:
            inducing_rows = int(self._d_out ** 0.5)
        if inducing_cols is None:
            inducing_cols = int(self._d_in ** 0.5)

        self.inducing_rows = inducing_rows
        self.inducing_cols = inducing_cols

        if sqrt_width_scaling:
            prior_sd /= self._d_in ** 0.5

        self.prior_sd = prior_sd
        self.max_sd_u = max_sd_u

        # set attributes with the size of the augmented space, i.e. total_cols is the number of columns
        # in weight space plus the number of inducing columns
        self.total_cols = self._d_in + self.inducing_cols
        self.total_rows = self._d_out + self.inducing_rows

        self.z_row = nn.Parameter(torch.randn(inducing_rows, self._d_out) * inducing_rows ** -0.5)
        self.z_col = nn.Parameter(torch.randn(inducing_cols, self._d_in) * inducing_cols ** -0.5)

        d = inducing_rows * inducing_cols
        if q_inducing == "full":
            # use a multivariate Gaussian with full covariance for inference on U, parameterized by the
            # Cholesky decomposition of the covariance
            self.inducing_mean = nn.Parameter(torch.randn(d))
            self._inducing_scale_tril = nn.Parameter(
                torch.cat([torch.full((d,), inverse_softplus(1e-3)), torch.zeros((d * (d - 1)) // 2),])
            )
            self.register_parameter("_inducing_row_scale_tril", None)
            self.register_parameter("_inducing_col_scale_tril", None)
            self.register_parameter("_inducing_sd", None)
        elif q_inducing == "matrix":
            # use a matrix Gaussian with full row and column covariance for inference on U, both parameterized
            # by the corresponding Cholesky decomposition
            self.inducing_mean = nn.Parameter(torch.randn(inducing_rows, inducing_cols))
            self._inducing_row_scale_tril = nn.Parameter(
                torch.cat(
                    [
                        torch.full((inducing_rows,), inverse_softplus(1e-3)),
                        torch.zeros((inducing_rows * (inducing_rows - 1)) // 2),
                    ]
                )
            )
            self._inducing_col_scale_tril = nn.Parameter(
                torch.cat(
                    [
                        torch.full((inducing_cols,), inverse_softplus(1e-3)),
                        torch.zeros((inducing_cols * (inducing_cols - 1)) // 2),
                    ]
                )
            )
            self.register_parameter("_inducing_scale_tril", None)
            self.register_parameter("_inducing_sd", None)
        elif q_inducing == "diagonal":
            self.inducing_mean = nn.Parameter(torch.randn(inducing_rows, inducing_cols))
            self._inducing_sd = nn.Parameter(torch.full((inducing_rows, inducing_cols), inverse_softplus(1e-3)))
            self.register_parameter("_inducing_scale_tril", None)
            self.register_parameter("_inducing_row_scale_tril", None)
            self.register_parameter("_inducing_col_scale_tril", None)
        elif q_inducing == "ensemble":
            tmp = torch.randn(inducing_rows, inducing_cols)
            self.inducing_mean = nn.Parameter(tmp + torch.randn(ensemble_size, inducing_rows, inducing_cols) * 0.1)
            self.register_parameter("_inducing_scale_tril", None)
            self.register_parameter("_inducing_row_scale_tril", None)
            self.register_parameter("_inducing_col_scale_tril", None)
            self.register_parameter("_inducing_sd", None)
        else:
            raise ValueError("q_inducing must be one of 'full', 'matrix', 'diagonal', 'ensemble'.")
        self.q_inducing = q_inducing

        self.max_lamda = max_lamda
        if learn_lamda:
            # unconstrained parameterization of the lambda parameter which rescales the covariance of the
            # conditional Gaussian on W given U. Positivity is ensured through softplus function
            self._lamda = nn.Parameter(torch.tensor(inverse_softplus(init_lamda)))
        else:
            self.register_buffer("_lamda", torch.tensor(inverse_softplus(init_lamda)))

        self._d_row = nn.Parameter(torch.full((inducing_rows,), inverse_softplus(1e-3)))
        self._d_col = nn.Parameter(torch.full((inducing_cols,), inverse_softplus(1e-3)))

        self.weight, self.bias = self.sample_shaped_parameters()
        self._caching_mode = cache_cholesky

    def reset_cache(self):
        self._L_r_cached = None
        self._L_c_cached = None
        self._prior_inducing_row_scale_tril_cached = None
        self._prior_inducing_col_scale_tril_cached = None
        self._row_transform_cached = None
        self._col_transform_cached = None

    def extra_repr(self):
        """Helper function for displaying and printing nn.Module objects."""
        s = super().extra_repr()
        s += f", inducing_cols={self.inducing_cols}, inducing_rows={self.inducing_rows}"
        return s

    def init_from_deterministic_params(self, param_dict):
        raise NotImplementedError

    def forward(self, x):
        self.weight, self.bias = self.sample_shaped_parameters()
        return super().forward(x)

    def sample_shaped_parameters(self):
        parameters = self.sample_parameters()
        if self.has_bias:
            w = parameters[:, :-1]
            b = parameters[:, -1]
        else:
            w = parameters
            b = None
        return w.view(self._weight_shape), b

    def sample_parameters(self):
        # self._u will be set from outside the forward pass if we're using global inducing weights
        # if that is the case, we use the sample that has been written into the instance attribute,
        # otherwise sample it here independently of other layers
        if self._u is None:
            self._u = self.sample_u()
        u = self._u
        self._u = None
        row_chol = self.prior_inducing_row_scale_tril
        col_chol = self.prior_inducing_col_scale_tril
        if self.whitened_u:
            u = row_chol @ u @ col_chol.t()
        return self.sample_conditional_parameters(u, row_chol, col_chol)

    def sample_u(self):
        if self.q_inducing == "ensemble":
            i = self._i
            self._i = (self._i + 1) % len(self.inducing_mean)
            u = self.inducing_mean[i]
        else:
            u = self.inducing_dist.rsample()
        return u.view(self.inducing_rows, self.inducing_cols)

    def sample_conditional_parameters(self, u, row_chol, col_chol):
        # mean and noise are the conditional mean and additive noise s.t.
        # W = mean + noise follows a Gaussian with covariance equal to the covariance of noise
        mean, noise = self.conditional_mean_and_noise(u, row_chol, col_chol)
        rescaled_noise = self.lamda * noise
        # self.prior_sd is the standard deviation of the prior on W set in the init method
        return self.prior_sd * (mean + rescaled_noise)

    def kl_divergence(self):
        # TODO potentially add log prob of ensemble u positions to kl

        # kl divergence is the sum of the kl divergence in U space and the expected KL in W space.
        # The latter can be evaluated analytically for our cases, since everything is Gaussian
        if self.inducing_dist is None:
            inducing_kl = 0.0
        else:
            inducing_kl = dist.kl_divergence(self.inducing_dist, self.inducing_prior_dist).sum()
        conditional_kl = self.conditional_kl_divergence()
        return inducing_kl + conditional_kl

    def conditional_kl_divergence(self):
        """Computes KL[q(W|U) || p(W|U)]. q and p are assumed to be Gaussians that share the same mean
        and only differ in covariance through a rescaling by self.lambda^2."""
        return self._d_in * self._d_out * (0.5 * self.lamda ** 2 - self.lamda.log() - 0.5)

    def compute_row_transform(self, row_chol):
        # Z_r^T (Z_r Z_r^T + D_r^2)^-1
        if self._row_transform_cached is None:
            row_transform = self.z_row.cholesky_solve(row_chol).t()
        else:
            row_transform = self._row_transform_cached

        if self.caching_mode:
            self._row_transform_cached = row_transform
        return row_transform

    def compute_col_transform(self, col_chol):
        # (Z_c Z_c^T + D_c^2)^-1 Z_c
        if self._col_transform_cached is None:
            col_transform = self.z_col.cholesky_solve(col_chol)
        else:
            col_transform = self._col_transform_cached

        if self.caching_mode:
            self._col_transform_cached = col_transform
        return col_transform

    @abstractmethod
    def conditional_mean_and_noise(self, u, row_chol, col_chol):
        pass

    @property
    def caching_mode(self):
        return self._caching_mode

    @caching_mode.setter
    def caching_mode(self, mode):
        self._caching_mode = mode
        if mode is False:
            self.reset_cache()

    # below are properties that transform the unconstrained pytorch parameters into constrained space
    @property
    def d_row(self):
        return F.softplus(self._d_row).diag_embed()

    @property
    def d_col(self):
        return F.softplus(self._d_col).diag_embed()

    @property
    def lamda(self):
        return F.softplus(self._lamda).clamp(0.0, self.max_lamda) if self._lamda is not None else None

    @property
    def prior_inducing_row_cov(self):
        return self.z_row @ self.z_row.t() + self.d_row.pow(2)

    @property
    def prior_inducing_row_scale_tril(self):
        if self._prior_inducing_row_scale_tril_cached is not None:
            prior_inducing_row_scale_tril = self._prior_inducing_row_scale_tril_cached
        else:
            prior_inducing_row_scale_tril = _jittered_cholesky(self.prior_inducing_row_cov)

        if self.caching_mode:
            self._prior_inducing_row_scale_tril_cached = prior_inducing_row_scale_tril
        return prior_inducing_row_scale_tril

    @property
    def prior_inducing_col_cov(self):
        return self.z_col @ self.z_col.t() + self.d_col.pow(2)

    @property
    def prior_inducing_col_scale_tril(self):
        if self._prior_inducing_col_scale_tril_cached is not None:
            prior_inducing_col_scale_tril = self._prior_inducing_col_scale_tril_cached
        else:
            prior_inducing_col_scale_tril = _jittered_cholesky(self.prior_inducing_col_cov)

        if self.caching_mode:
            self._prior_inducing_col_scale_tril_cached = prior_inducing_col_scale_tril
        return prior_inducing_col_scale_tril

    @property
    def inducing_scale_tril(self):
        return vec_to_chol(self._inducing_scale_tril) if self._inducing_scale_tril is not None else None

    @property
    def inducing_row_scale_tril(self):
        return vec_to_chol(self._inducing_row_scale_tril) if self._inducing_row_scale_tril is not None else None

    @property
    def inducing_col_scale_tril(self):
        return vec_to_chol(self._inducing_col_scale_tril) if self._inducing_col_scale_tril is not None else None

    @property
    def inducing_sd(self):
        return F.softplus(self._inducing_sd).clamp(0.0, self.max_sd_u) if self._inducing_sd is not None else None

    @property
    def inducing_prior_dist(self):
        if self.inducing_mean is None:
            # If the inducing_mean has been set to None, i.e. we're using global inducing weights, the marginal prior
            # distribution calculated below is no longer correct (as we're conditioning on some global inducing
            # weight), so return None. We could also make the forward hook for the global inducing weight assign
            # the marginal distribution on the per-layer inducing weights of each layer conditioned on the sample
            # for the global inducing weights to an _inducing_prior_dist attribute and return that here, however
            # I can't think of an actual use case for this
            return None

        loc = self.z_row.new_zeros(self.inducing_rows, self.inducing_cols)
        if self.whitened_u:
            # after whitening the prior on u it is a standard Gaussian N(0, 1), below parameterising q
            # in different ways for computational efficiency
            if self.q_inducing == "full":
                cov = torch.eye(self.inducing_rows * self.inducing_cols, device=self.device)
                return dist.MultivariateNormal(loc.flatten(), cov)
            elif self.q_inducing == "matrix":
                row_scale_tril = torch.eye(self.inducing_rows, device=self.device)
                col_scale_tril = torch.eye(self.inducing_cols, device=self.device)
                return MatrixNormal(loc, row_scale_tril, col_scale_tril)
            else:  # self.q_inducing == "diagonal"
                scale = self.z_row.new_ones(self.inducing_rows, self.inducing_cols)
                return dist.Normal(loc, scale)
        else:
            return MatrixNormal(loc, self.prior_inducing_row_scale_tril, self.prior_inducing_col_scale_tril,)

    @property
    def inducing_dist(self):
        # TODO potentially return mixture of deltas for q_inducing == "ensemble"
        if self.inducing_mean is None or self.q_inducing == "ensemble":
            return None

        if self.q_inducing == "full":
            return dist.MultivariateNormal(self.inducing_mean, scale_tril=self.inducing_scale_tril)
        elif self.q_inducing == "matrix":
            return MatrixNormal(self.inducing_mean, self.inducing_row_scale_tril, self.inducing_col_scale_tril,)
        else:  # self.q_inducing == "diagonal"
            return dist.Normal(self.inducing_mean, self.inducing_sd)

    @property
    def device(self):
        return self.z_row.device


class InducingDeterministicMixin(_InducingBase):
    """Inducing model where we only sample u and deterministiclly transform it into the weights."""

    def conditional_mean_and_noise(self, u, row_chol, col_chol):
        # Z_r^T (Z_r Z_r^T + D_r^2)^-1
        row_transform = self.compute_row_transform(row_chol)
        # (Z_c Z_c^T + D_c^2)^-1 Z_c
        col_transform = self.compute_col_transform(col_chol)
        M_w = row_transform @ u @ col_transform
        return M_w, torch.zeros_like(M_w)


class InducingMixin(_InducingBase):
    """Inducing mixin which uses Matheron's rule to sample from the conditional multivariate normal on W, i.e.
    sample from the joint and linearly transform the samples to get the correct mean and covariance. This is necessary
    since the covariance of W is a difference of two Kronecker products, i.e. has no simple structure."""

    def conditional_mean_and_noise(self, u, row_chol, col_chol):
        # Z_r^T (Z_r Z_r^T + D_r^2)^-1  -- row_chol is the Cholesky of Z_r Z_r^T + D_r^2
        row_transform = self.compute_row_transform(row_chol)
        # (Z_c Z_c^T + D_c^2)^-1 Z_c -- col_chol is the Cholesky of Z_c Z_c^T + D_c^2
        col_transform = self.compute_col_transform(col_chol)

        M_w = row_transform @ u @ col_transform

        # Below is the implementation of Matheron's rule where we transform a sample from the joint
        # p(W, U) such that the resulting M_w + noise_term follows the conditional Normal p(W|U)
        #
        # Sampling from the joint p(W, U) cannot be done directly in an efficient manner since it is
        # a multivariate normal with a covariance that has Khatri-Rao product structure (Kronecker factored blocks).
        # Instead we augment the joint variable to:
        # W   U_c
        # U_r U
        # such that the joint variable has a matrix normal distribution
        # We then only compute the W and U samples. The latter can be structured into a sum of four blocks
        # of noise terms. The first term is a transformation of the noise needed for W, however two of the
        # remaining three terms are projections from the corresponding dimension (rows or columns) of W into
        # the smaller dimension of U by Z. Hence the noise can also be seen as coming from a lower dimensional
        # matrix normal distribution with covariance ZZ^T. Hence we calculate the Cholesky of Z and directly
        # sample in U space to reduce noise
        e1 = self.z_row.new_empty(self._d_out, self._d_in).normal_()
        e2, e3, e4 = self.z_row.new_empty(3, self.inducing_rows, self.inducing_cols).normal_()

        w_bar = e1

        if self._L_r_cached is None:
            # either not in caching mode or Cholesky has not been computed yet
            L_r = _jittered_cholesky(self.z_row.mm(self.z_row.t()))
        else:
            L_r = self._L_r_cached

        if self._L_c_cached is None:
            # either not in caching mode or Cholesky has not been computed yet
            L_c = _jittered_cholesky(self.z_col.mm(self.z_col.t()))
        else:
            L_c = self._L_c_cached

        if self.caching_mode:
            self._L_r_cached = L_r
            self._L_c_cached = L_c

        t1 = self.z_row @ e1 @ self.z_col.t()
        t2 = L_r @ e2 @ self.d_col
        t3 = self.d_row @ e3 @ L_c.t()
        t4 = self.d_row @ e4 @ self.d_col
        u_bar = t1 + t2 + t3 + t4

        noise_term = w_bar - row_transform @ u_bar @ col_transform

        return M_w, noise_term


def _iter_inducing_modules(module):
    yield from filter(lambda m: isinstance(m, _InducingBase), module.modules())


def register_global_inducing_weights_(module, inducing_rows, inducing_cols, cat_dim=0, **inducing_kwargs):
    """Function for adding global inducing weights to an nn.Module that contains inducing modules to add correlation
    between their inducing weights. For this we need to concatenate/stack the per-layer inducing weights along either
    the rows or columns. cat_dim indicates which dimension we concatenate (0 for rows, 1 for columns) so the other
    dimension needs to be the same for all inducing layers of module.

    Args:
        module (nn.Module): A pytorch module that contains at least two submodules that inherit from _InducingBase.
            The variational parameters of those modules will be removed and sampling the inducing weights taken
            care of through a forward hook that samples their respective inducing weights jointly. Note that an
            extra nn.Module is added to module, so this will show up when iterating over the .modules(). The forward
            pass of this module is implemented as the identity function to be compatible e.g. with nn.Sequential,
            which does such an iteration in the forward pass
        inducing_rows (int): number of global inducing rows
        inducing_cols (int): number of inducing columns
        cat_dim (int; 0 or 1): dimension of the per-layer inducing weights to concatenate for creating the joint
            object. 0 for concatenating the rows (hence the number of inducing columns must be the same across all
            inducing layers), 1 for concatenating columns (hence number of rows must be the same)
        inducing_kwargs: inducing parameters for the global InduxingMixin layer

    Returns:
        The pre forward hook registered on module that samples the inducing weights.
    """

    class _GlobalInducingModule(InducingMixin, nn.Linear):
        """Dummy class to use the sampling logic from the inducing mixin for sampling global inducing weights. The
        in_features and out_features attributes correspond to the total number inducing columns and rows in the network.
        Internal to the function to avoid having it registered as a subclass of VariationalMixin for bayesianize."""

        def __init__(self, *args, **kwargs):
            kwargs["bias"] = False
            super().__init__(*args, **kwargs)

        def forward(self, input):
            # implement identity function so that this works inside nn.Sequential, which calls the forward
            # method of all module that are registered within it
            return input

    if cat_dim not in [0, 1]:
        raise ValueError("Must concatenate either the rows (cat_dim=0) or columns(cat_dim=1) of the inducing weights")

    if len(list(_iter_inducing_modules(module))) < 2:
        raise ValueError(
            "'module' must contain at least two inducing layers as the point of this function is to add "
            "correlation between such layers."
        )

    non_cat_size = None
    cat_size = 0
    # check that all inducing layers dimensions match for concatenation and that they are whitened, and add up
    # the sizes along the concatenation dimension. Note that rather than iterating over all inducing modules we
    # could also have the function accept an iterator as an argument, e.g. to exclude input or output layers
    # which might have differently shaped inducing weights than the inner modules
    for m in _iter_inducing_modules(module):
        m_non_cat_size = m.inducing_rows if cat_dim == 1 else m.inducing_cols
        if non_cat_size is None:
            non_cat_size = m_non_cat_size
        elif m_non_cat_size != non_cat_size:
            raise ValueError("Size of the inducing dimension which is not concatenated must match across all layers.")

        if not m.whitened_u:
            raise ValueError("All inducing weight layers must use whitened inducing weights.")

        cat_size += m.inducing_rows if cat_dim == 0 else m.inducing_cols

        # delete the variational parameters over u by setting them to None -- this allows for the unconstrained
        # properties to still be None rather than raising an AttributeError
        m.inducing_mean = None
        m._inducing_row_scale_tril = None
        m._inducing_col_scale_tril = None
        m._inducing_scale_tril = None
        m._inducing_sd = None

    if cat_dim == 0:
        num_cols = non_cat_size
        num_rows = cat_size
    else:
        num_cols = cat_size
        num_rows = non_cat_size

    # set the global inducing module as an attribute on the original module so that the parameters are registered
    module._global_inducing_module = _GlobalInducingModule(
        num_cols, num_rows, inducing_rows=inducing_rows, inducing_cols=inducing_cols, **inducing_kwargs,
    )

    # hook function that is called before the forward pass of module is executed. Jointly samples the inducing weights
    # of all layers and assigns the corresponding slice to the ._u attribute of each inducing module, such that the
    # per-layer weights are conditioned on those
    def inducing_sampling_hook(m, input):
        # draw the joint sample of inducing weights for all layers stacked into one big matrix. Note that
        # the 'weight matrix' of the global inducing module corresponds to the joint inducing weights of the layers.
        # The inducing weights of the module are the 'global' inducing weights
        inducing_weights = m._global_inducing_module.sample_parameters()

        offset = 0
        for im in _iter_inducing_modules(m):
            # skip the global inducing module
            if im is m._global_inducing_module:
                continue

            if cat_dim == 0:
                delta = im.inducing_rows
                im._u = inducing_weights[offset : offset + delta]
            else:
                delta = im.inducing_cols
                im._u = inducing_weights[:, offset : offset + delta]
            offset += delta

    return module.register_forward_pre_hook(inducing_sampling_hook)


class InducingDeterministicConditionalCholeskyMixin(_InducingBase):
    """Hybrid model which has the mean of the marginalized inducing model (i.e. the mean of the conditional
    distribution is a function of only U, but not U_r or U_c) and the covariance is that of the fully
    conditional model, i.e. the matrix normal that corresponds to p(W|U_r, U_c, U)."""

    def conditional_mean_and_noise(self, u, row_chol, col_chol):
        # Z_r^T (Z_r Z_r^T + D_r^2)^-1
        row_transform = self.compute_row_transform(row_chol)
        # (Z_c Z_c^T + D_c^2)^-1 Z_c
        col_transform = self.compute_col_transform(col_chol)

        M_w = row_transform @ u @ col_transform

        row_cov = (1 + EPS) * torch.eye(self._d_out, device=self.device) - row_transform @ self.z_row
        L_r = row_cov.cholesky()
        col_cov = (1 + EPS) * torch.eye(self._d_in, device=self.device) - self.z_col.t() @ col_transform
        L_c = col_cov.cholesky()

        noise_term = L_r @ torch.randn_like(M_w) @ L_c.t()

        return M_w, noise_term


class InducingMarginalizedCholeskyMixin(_InducingBase):
    """Marginalized inducing model that explicitly calculates the Cholesky of the multivariate normal conditional
    distribution over the weights. This is intractable for even moderately sized layers, the main purpose of this
    class is to test the sampling of the Matheron variant."""

    def conditional_mean_and_noise(self, u, row_chol, col_chol):
        # Z_r^T (Z_r Z_r^T + D_r^2)^-1
        row_transform = self.compute_row_transform(row_chol)
        # (Z_c Z_c^T + D_c^2)^-1 Z_c
        col_transform = self.compute_col_transform(col_chol)

        M_w = row_transform @ u @ col_transform

        row_cov = row_transform @ self.z_row
        col_cov = self.z_col.t() @ col_transform
        weight_cov = (1 + EPS) * torch.eye(self._d_out * self._d_in, device=self.device) - kron(row_cov, col_cov)
        L_w = weight_cov.cholesky()
        noise_term = L_w.mv(torch.randn_like(M_w).flatten()).view_as(M_w)

        return M_w, noise_term

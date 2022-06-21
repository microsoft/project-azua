import math
import operator
from functools import reduce
from typing import Union, cast

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector

from .base import VariationalMixin

__all__ = ["FCGMixin"]


def _prod(iterable):
    return reduce(operator.mul, iterable, 1)


def _normal_sample(mean, scale_tril):
    return mean + scale_tril @ torch.randn_like(mean)


# TODO: reuse more code between this and FFGMixin class
class FCGMixin(VariationalMixin):
    """Variational module that places a multivariate Gaussian with full covariance jointly
    over .weight and .bias attributes. The forward pass always explicitly samples the weights."""

    def __init__(
        self,
        *args,
        prior_mean: float = 0.0,
        prior_weight_sd: Union[float, str] = 1.0,
        prior_bias_sd: float = 1.0,
        init_sd: float = 1e-4,
        nonlinearity_scale: float = 1.0,
        **kwargs,
    ):

        # TODO: don't ignore below mypy error
        # None of the superclasses have initializer which takes args and kwards
        # However, I believe that this code works, since:
        # -we never use this class directly
        # -when we inherit from this class, we also inherit either from nn.Linear or nn.ConvNd.
        # Both of these classes have initializers which take args/kwargs
        # This is really theory, which requires to be checked.
        # I don't know how to stop ignoring below mypy error
        # Similar problem is in ffg class
        super().__init__(*args, **kwargs)  # type: ignore
        self.has_bias = self.bias is not None
        self.weight_shape = self.weight.shape
        self.bias_shape = self.bias.shape if self.has_bias else None

        current_parameters = parameters_to_vector(self.parameters())
        num_params = current_parameters.numel()

        self.mean = nn.Parameter(current_parameters.data.detach().clone())
        _init_sd = math.log(math.expm1(init_sd))
        self._scale_diag = nn.Parameter(torch.full((num_params,), _init_sd))
        self._scale_tril = nn.Parameter(torch.zeros(num_params, num_params))

        prior_weight_sd_float: float
        if prior_weight_sd == "neal":
            input_dim = _prod(self.weight.shape[1:])
            prior_weight_sd_float = input_dim ** -0.5
        else:
            prior_weight_sd_float = cast(float, prior_weight_sd)

        prior_weight_sd_float *= nonlinearity_scale

        prior_weight_sd_tensor = torch.full((self.weight.flatten().shape), prior_weight_sd_float)
        if self.has_bias:
            prior_bias_sd_tensor = torch.full((self.bias.flatten().shape), prior_bias_sd)
            prior_sd_diag = torch.cat((prior_weight_sd_tensor, prior_bias_sd_tensor))
        else:
            prior_sd_diag = prior_weight_sd_tensor

        del self._parameters["weight"]
        if self.has_bias:
            del self._parameters["bias"]
        self.assign_params(self.mean.data)

        self.register_buffer("prior_mean", torch.full((num_params,), prior_mean))
        self.register_buffer("prior_scale_tril", prior_sd_diag.diag_embed())

    def extra_repr(self):
        s = super().extra_repr()
        m = self.prior_mean.data[0]
        if torch.allclose(m, self.prior_mean):
            s += f", prior mean={m.item():.2f}"
        sd = self.prior_scale_tril[0, 0]
        if torch.allclose(sd, self.prior_scale_tril) and torch.allclose(self.prior_scale_tril.tril(diagonal=-1), 0):
            s += f", prior sd={sd.item():.2f}"
        return s

    def init_from_deterministic_params(self, param_dict):
        weight = param_dict["weight"]
        bias = param_dict.get("bias")
        with torch.no_grad():
            mean = weight.flatten()
            if bias is not None:
                mean = torch.cat([mean, bias.flatten()])
            self.mean.data.copy_(mean)

    @property
    def parameter_tensors(self):
        if self.has_bias:
            return self.weight, self.bias
        return (self.weight,)

    @property
    def scale_tril(self):
        return F.softplus(self._scale_diag).diagflat() + torch.tril(self._scale_tril, diagonal=-1)

    @property
    def parameter_distribution(self):
        return dist.MultivariateNormal(self.mean, scale_tril=self.scale_tril)

    @property
    def prior_distribution(self):
        return dist.MultivariateNormal(self.prior_mean, scale_tril=self.prior_scale_tril)

    def assign_params(self, parameters: torch.Tensor):
        if self.has_bias:
            num_bias_params = _prod(self.bias_shape)
            self.weight = parameters[:-num_bias_params].view(self.weight_shape)
            self.bias = parameters[-num_bias_params:].view(self.bias_shape)
        else:
            self.weight = parameters.view(self.weight_shape)

    def forward(self, x: torch.Tensor):
        parameter_sample = _normal_sample(self.mean, self.scale_tril)
        self.assign_params(parameter_sample)
        return super().forward(x)

    def kl_divergence(self):
        return dist.kl_divergence(self.parameter_distribution, self.prior_distribution)

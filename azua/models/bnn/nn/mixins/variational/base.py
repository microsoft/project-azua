from abc import abstractmethod

from typing import Any, Optional, Union

from ..base import BayesianMixin

import torch


class VariationalMixin(BayesianMixin):

    # TODO: Add type annotation for bias
    # We use VariationalMixin to override either linear or convolution layers
    # These layer have different type of bias: Parameter for lineary layer, Optional[Tensor] for convolution layer
    # As override happens through multiple inheritance, we can't currently specify type of bias
    # One solution would be to make the whole class taking generic parameter which would be type of bias
    # (I assume that this is possible in python)
    bias: Any  # Union[torch.nn.parameter.Parameter, Optional[torch.Tensor]]
    weight: torch.Tensor

    def parameter_loss(self):
        return self.kl_divergence()

    @abstractmethod
    def kl_divergence(self):
        raise NotImplementedError

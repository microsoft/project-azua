from abc import ABC, abstractmethod

import torch.nn as nn


class BayesianMixin(ABC, nn.Module):
    @abstractmethod
    def parameter_loss(self):
        """Calculates generic parameter-dependent loss. For a probabilistic module with some prior over the parameters,
        e.g. for MAP inference or MCMC sampling, this would be the negative log prior, for Variational inference the
        KL divergence between approximate posterior and prior."""
        raise NotImplementedError

    @abstractmethod
    def init_from_deterministic_params(self, param_dict):
        """Initializes from the parameters of a deterministic network. For a variational module, this might mean
        setting the mean of the approximate posterior to those parameters, whereas a MAP/MCMC module would simply
        copy the parameter values."""
        raise NotImplementedError

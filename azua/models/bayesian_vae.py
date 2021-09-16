from typing import Tuple

import torch

from .vae import VAE
from .bnn import bayesianize_


class BayesianVAE(VAE):
    def __init__(self, *args, inference_config=None, dataset_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._bayes_init(inference_config, dataset_size)

    def _bayes_init(self, inference_config, dataset_size):
        if inference_config is None:
            raise ValueError("inference_config must not be None for Bayesian VAE")
        if dataset_size is None:
            raise ValueError("dataset_size must not be None for Bayesian VAE")

        if "bnn_kl_beta" in inference_config:
            self._bnn_kl_beta = inference_config["bnn_kl_beta"]  # the kl coefficent for the BNN weights
            inference_config.pop("bnn_kl_beta")
        else:
            self._bnn_kl_beta = 1.0

        bayesianize_(self._decoder, **inference_config)
        # Ensure Bayesianized decoder is on same device as the rest of the model
        self._decoder.to(self._device)
        self.dataset_size = dataset_size

    def _loss(self, *input_tensors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, kl_vae, nll = super()._loss(*input_tensors)
        kl_bnn = self.scaled_decoder_kl(len(input_tensors[0]))
        kl = kl_vae + kl_bnn
        # note the usage of a different kl coefficent for the BNN KL
        loss = self._bnn_kl_beta * kl_bnn + self._beta * kl_vae + nll

        return loss, kl, nll

    def scaled_decoder_kl(self, batch_size):
        kl = torch.tensor(0.0, device=self._device)
        for module in self._decoder.modules():
            if hasattr(module, "parameter_loss"):
                kl = kl + module.parameter_loss()
        # note in training the total_loss will be devided by batch_size (as the nll is computed by summing in mini-batch)
        # so the correct scale of the BNN KL here is batch_size / dataset_size
        return batch_size * kl / self.dataset_size

    @classmethod
    def name(cls) -> str:
        return "bayesian_vae"

    @classmethod
    def bayesianize_(cls, vae, inference_config, dataset_size):
        vae.__class__ = cls
        vae._bayes_init(inference_config, dataset_size)
        return vae

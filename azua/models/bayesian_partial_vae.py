from typing import Tuple

import torch

from .partial_vae import PartialVAE
from .bayesian_vae import BayesianVAE
from ..models.torch_model import TorchModel


class BayesianPartialVAE(PartialVAE, TorchModel):
    # TODO why the double inheritance?
    def __init__(self, *args, **kwargs):
        # assertion and retrival for inference config and dataset size
        assert "inference_config" in kwargs.keys()
        inference_config = kwargs["inference_config"]
        kwargs.pop("inference_config")
        assert "dataset_size" in kwargs.keys()
        dataset_size = kwargs["dataset_size"]
        kwargs.pop("dataset_size")

        if "bnn_kl_beta" in inference_config:
            self._bnn_kl_beta = inference_config["bnn_kl_beta"]  # the kl coefficent for the BNN weights
            inference_config.pop("bnn_kl_beta")
        else:
            self._bnn_kl_beta = 1.0

        super(BayesianPartialVAE, self).__init__(*args, **kwargs)
        BayesianVAE.bayesianize_(self._vae, inference_config, dataset_size)
        if len(inference_config.keys()) > 0:
            print("constructing the decoder of PVAE with the following inference config:")
            print(inference_config)
            print(self._vae._decoder)

    @classmethod
    def name(cls) -> str:
        return "bayesian_pvae"

    def _loss(
        self, x: torch.Tensor, input_mask: torch.Tensor, scoring_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, kl_vae, nll = super()._loss(x, input_mask, scoring_mask)
        kl_bnn = self._vae.scaled_decoder_kl(len(x))  # type: ignore
        kl = kl_vae + kl_bnn
        loss = self._beta * kl_vae + self._bnn_kl_beta * kl_bnn + nll

        return loss, kl, nll

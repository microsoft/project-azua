# This is required in python 3 to allow return types of the same class.
from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from ..datasets.variables import Variables
from ..models.torch_training_types import LossConfig, VAELossResults
from ..utils.training_objectives import kl_divergence, negative_log_likelihood
from .decoder import Decoder
from .encoder import Encoder
from .vae import VAE


# TODO: Reuse more code from base VAE class
class PredictiveVAE(VAE):
    """
    Predictive variational autoencoder.

    It is basically a NN predictor (takes x and latent code as input and predicts y) coupled with an encoder (takes y and x, and predicts latent code z)
    """

    def __init__(
        self,
        model_id: str,
        variables: Variables,
        save_dir: str,
        device: torch.device,
        feature_dim: int,
        encoder_layers: List[int],
        latent_dim: int,  # Encoder options
        decoder_layers: List[int],
        decoder_variances: float,  # Decoder options
        categorical_likelihood_coefficient: float,
        kl_coefficient: float,
        variance_autotune: bool = False,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        metadata_filepath: Optional[str] = None,
    ) -> None:
        """
        Args:
            model_id (str): Unique model ID for referencing this model instance.
            variables (Variables): Information about variables/features used
                by this model.
            save_dir (str): Location to save any information about this model, including training data.
                be created if it doesn't exist.
            device (`torch.device`): Device to load model to.

            input_dim (int): Dimension of input data.
            output_dim (int): Dimension of data output.
            feature_dim: feature dimension for predicting y.

            encoder_layers (list of int): Sizes of internal hidden layers. i.e. [a, b] is three linear layers with shapes (input_dim, a), (a, b), (b, output_dim)
            latent_dim (int): Dimension of output latent space.

            decoder_layers (list of int): Sizes of internal hidden layers. [a, b] is three linear layers with shapes (input_dim, a), (a, b), (b, output_dim)
            decoder_variances (float): Output variance to use.

            categorical_likelihood_coefficient (float): coefficient for balancing likelihoods
            kl_coefficient (float): coefficient for KL terms
            variance_autotune (bool): automatically tune variances or not
        """
        if input_dim is None:
            input_dim = variables.num_processed_cols

        if output_dim is None:
            output_dim = variables.num_processed_cols

        # Initialise both abstract classes.
        VAE.__init__(
            self,
            model_id=model_id,
            variables=variables,
            save_dir=save_dir,
            device=device,
            input_dim=input_dim,
            output_dim=output_dim,
            encoder_layers=encoder_layers,
            latent_dim=latent_dim,
            decoder_layers=decoder_layers,
            decoder_variances=decoder_variances,
            categorical_likelihood_coefficient=categorical_likelihood_coefficient,
            kl_coefficient=kl_coefficient,
            variance_autotune=variance_autotune,
        )

        self._encoder = Encoder(feature_dim, encoder_layers, latent_dim, device)
        self._decoder = Decoder(
            latent_dim + feature_dim,
            output_dim,
            variables,
            decoder_layers,
            decoder_variances,
            device,
            variance_autotune=variance_autotune,
        )

        self.__output_dim = output_dim

    # CLASS METHODS #
    @classmethod
    def name(cls) -> str:
        return "vae_predictive"

    # IMPLEMENTATION OF ABSTRACT METHODS #

    def loss(self, loss_config: LossConfig, input_tensors: Tuple[torch.Tensor, ...]) -> VAELossResults:
        x_and_y = input_tensors[0]
        feature = x_and_y[:, 0 : -self.__output_dim]
        y = x_and_y[:, -self.__output_dim :]
        total_loss, kl, total_nll = self._loss(feature, y)
        # set mask, as loss is calculcated over all ys
        return VAELossResults(loss=total_loss, kl=kl, nll=total_nll, mask_sum=torch.tensor(y.numel()))

    def _loss(self, feature: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # type: ignore[override]
        """
        Calculate the loss for data with mask.
        loss = kl_divergence(N(0, 1), N(μ(x), σ(x))) + nll(y, y(x)')
        i.e., although this loss function takes both x and y as input, the nll loss is only calculated on the y dimension only.
        This makes it different from the loss function of the VAE base class.
        where
            y is the target data in training set
            y' is the prediction
            x is the input feature
            μ is a latent variable representing the mean of the distribution
            σ is a latent variable representing the standard deviation of the distribution

        Args:
            y: data with shape (batch_size, input_dim)
            feature: input features with shape (batch_size, feature_dim) for predicting y.

        Returns:
            loss: total loss for all batches.
            kl_divergence: for tracking training. kld summed over all batches.
            negative log likelihood: for tracking training. nll summed over all batches.
        """
        # pass through encoder and decoder
        (dec_mean, dec_logvar), _, encoder_output = self.reconstruct(feature, y)

        # compute loss
        # KL divergence between approximate posterior q and prior p
        kl = kl_divergence(encoder_output).sum()
        # nll is only calcualted on y dimension
        nll = negative_log_likelihood(y, dec_mean, dec_logvar, self.variables, self._alpha, mask=None)

        loss = self._beta * kl + nll
        return loss, kl, nll

    def decode(self, data: torch.Tensor, *input_tensors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run decoding part of the VAE.

        Args:
            data: Input tensor with shape (batch_size, latent_dim)

        Returns:
            mean, logvar: Output of shape (batch_size, output_dim)
        """

        # TODO: account for case when count in recostruct is different than 1
        # we need to tile input_tensors appropriate amount of times then
        return self._decoder(torch.cat((data, input_tensors[0]), 1))

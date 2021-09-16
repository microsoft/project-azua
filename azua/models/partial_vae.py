# This is required in python 3 to allow return types of the same class.
from __future__ import annotations

import os

import numpy as np
import torch

from ..models.models_factory import create_set_encoder

from ..models.pvae_base_model import PVAEBaseModel
from ..models.vae import VAE

from ..utils.training_objectives import (
    kl_divergence,
    negative_log_likelihood,
    gaussian_negative_log_likelihood,
    get_input_and_scoring_processed_masks,
)

from ..datasets.variables import Variables
from ..datasets.data_processor import DataProcessor
from typing import List, Optional, Tuple
from ..models.torch_training import train_model
from ..models.torch_training_types import LossConfig, VAELossResults


class PartialVAE(PVAEBaseModel):
    """
    Subclass of `models.pvae_base_model.PVAEBaseModel` representing a Partial VAE.
    """

    def __init__(
        self,
        model_id: str,
        variables: Variables,
        save_dir: str,
        device: torch.device,
        embedding_dim: int,
        set_embedding_dim: int,  # Set encoder options
        set_embedding_multiply_weights: bool,
        encoding_function: str,
        metadata_filepath: str,
        encoder_layers: List[int],
        latent_dim: int,  # Encoder options
        decoder_layers: List[int],
        decoder_variances: float,  # Decoder options
        categorical_likelihood_coefficient: float,
        kl_coefficient: float,
        set_encoder_type: str = "default",
        variance_autotune: bool = False,
        use_importance_sampling: bool = False,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        non_linearity: Optional[str] = "ReLU",
        activation_for_continuous: Optional[str] = "Sigmoid",
        squash_input: Optional[bool] = True,
        init_method: Optional[str] = "default",
        **extra_set_encoder_kwargs,
    ) -> None:
        """
        Args:
            model_id (str): Unique model ID for referencing this model instance.
            variables (Variables): Information about variables/features used
                by this model.
            save_dir (str): Location to save any information about this model, including training data.
                be created if it doesn't exist.
            device (`torch.device`): Device to load model to.

            input_dim (int): Dimension of input data to embedding model.
            output_dim (int): Dimension of data output from PVAE.
            embedding_dim (int): Dimension of embedding for each input.
            set_embedding_dim (int): Dimension of output set embedding.

            set_embedding_multiply_weights (bool): Whether or not to take the product of x with embedding weights when feeding through.
            encoding_function (str): Function to use to summarise set input.
            metadata_filepath (str): Path to NumPy array containing feature meta.
            encoder_layers (list of int): Sizes of internal hidden layers. i.e. [a, b] is three linear layers with shapes (input_dim, a), (a, b), (b, output_dim)
            latent_dim (int): Dimension of output latent space.

            decoder_layers (list of int): Sizes of internal hidden layers. [a, b] is three linear layers with shapes (input_dim, a), (a, b), (b, output_dim)
            decoder_variances (float): Output variance to use.

            categorical_likelihood_coefficient (float): coefficient for balancing likelihoods
            kl_coefficient (float): coefficient for KL terms
            set_encoder_type (str): Type of set encoder to use. Can currently be either "default" or "sparse".
            variance_autotune (bool): automatically tune variances or not
            use_importance_sampling (bool): use importance weighted ELBO or not
            non_linearity (str): Non linear activation function used between Linear layers. Defaults to ReLU.
            activation_for_continuous (str): activation function for continuous variable outputs. Defaults to Sigmoid.
            squash_input (bool): squash VAE input or not
            init_method (str): initialization method
            extra_set_encoder_kwargs (dict): extra kwargs to pass to set encoder constructor
            TODO: instead, pass model_config as single object and pass that object on to set encoder constructor
            TODO are input_dim and variables.num_processed_cols always the same? If so, we could drop the input_dim argument
        """

        super().__init__(model_id, variables, save_dir, device)
        if input_dim is None:
            input_dim = variables.num_processed_cols
        if output_dim is None:
            output_dim = variables.num_processed_non_aux_cols

        if metadata_filepath:
            assert os.path.exists(metadata_filepath)
            metadata = np.load(metadata_filepath)
            metadata = torch.from_numpy(metadata).float()
            metadata = metadata.to(device)
        else:
            metadata = None
        self._alpha = categorical_likelihood_coefficient
        self._beta = kl_coefficient
        self._set_encoder = create_set_encoder(
            set_encoder_type,
            {
                "input_dim": input_dim,
                "embedding_dim": embedding_dim,
                "set_embedding_dim": set_embedding_dim,
                "device": device,
                "multiply_weights": set_embedding_multiply_weights,
                "encoding_function": encoding_function,
                "metadata": metadata,
                **extra_set_encoder_kwargs,
            },
        )

        self.data_processor = DataProcessor(variables, squash_input=squash_input)

        self._vae = VAE(
            model_id=model_id,
            variables=variables,
            save_dir=save_dir,
            device=device,
            input_dim=set_embedding_dim,
            output_dim=output_dim,
            encoder_layers=encoder_layers,
            latent_dim=latent_dim,  # Encoder options
            decoder_layers=decoder_layers,
            decoder_variances=decoder_variances,  # Decoder options
            categorical_likelihood_coefficient=categorical_likelihood_coefficient,
            kl_coefficient=kl_coefficient,
            variance_autotune=variance_autotune,
            non_linearity=non_linearity,
            activation_for_continuous=activation_for_continuous,
            init_method=init_method,
        )
        self._use_importance_sampling = use_importance_sampling
        if use_importance_sampling:
            self.importance_count = 20
        self.input_dim = input_dim
        self.latent_dim = latent_dim

    # CLASS METHODS
    @classmethod
    def name(cls) -> str:
        return "pvae"

    def save_onnx(self, save_dir: str) -> None:
        self._vae.save_onnx(save_dir)
        self._set_encoder.save_onnx(save_dir)

    def _train(self, dataset, train_output_dir, report_progress_callback, **train_config_dict):
        train_model(
            self,
            dataset=dataset,
            train_output_dir=train_output_dir,
            report_progress_callback=report_progress_callback,
            **train_config_dict,
        )

    def encode(self, data: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run encoding part of the PVAE.

        Args:
            data: Input tensor with shape (batch_size, input_dim).
            mask: Mask indicting observed variables with shape (batch_size, input_dim). 1 is observed, 0 is un-observed.

        Returns:
            mean, logvar: Latent space samples of shape (batch_size, latent_dim).
        """
        embedded = self._set_encoder(data, mask)
        mean, logvar = self._vae.encode(embedded)

        return mean, logvar

    def _loss(self, x: torch.Tensor, input_mask: torch.Tensor, scoring_mask: torch.Tensor):
        if self._use_importance_sampling:
            return self._loss_IWAE(x, input_mask, scoring_mask)
        else:
            return self._loss_ELBO(x, input_mask, scoring_mask)

    def loss(self, loss_config: LossConfig, input_tensors: Tuple[torch.Tensor, ...]) -> VAELossResults:
        """
        Calculate tensor loss on a single batch of data.

        TODO categorical likelihood and KL coefficients, as well as choice of IWAE vs. ELBO, should live in loss_config (part of training config), 
        not model_config.

        Args:
            loss_config (LossConfig): Parameters that specify how the loss is calculated.
            input_tensors: Single batch of data and mask, given as (data, mask). The mask indicates 
                which data is present. 1 is observed, 0 is unobserved.

        """

        x, mask = input_tensors
        assert loss_config.max_p_train_dropout is not None
        assert loss_config.score_imputation is not None
        assert loss_config.score_reconstruction is not None
        assert loss_config.score_reconstruction or loss_config.score_imputation

        input_mask, scoring_mask = get_input_and_scoring_processed_masks(
            self.data_processor,
            mask,
            max_p_train_dropout=loss_config.max_p_train_dropout,
            score_imputation=loss_config.score_imputation,
            score_reconstruction=loss_config.score_reconstruction,
        )

        total_loss, kl, total_nll = self._loss(x, input_mask=input_mask, scoring_mask=scoring_mask)
        return VAELossResults(loss=total_loss, kl=kl, nll=total_nll, mask_sum=scoring_mask.sum())

    def reconstruct(
        self, data: torch.Tensor, mask: Optional[torch.Tensor], sample: bool = True, count: int = 1, **kwargs
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Reconstruct data by passing them through the VAE.

        Args:
            data: Input data with shape (batch_size, input_dim).
            mask: If not None, mask indicating observed variables with shape (batch_size, input_dim). 1 is observed,
                  0 is un-observed. If None, everything is observed.
            sample: If True, samples the latent variables, otherwise uses the mean.
            count: Number of samples to reconstruct.

        Returns:
            (decoder_mean, decoder_logvar): Reconstucted variables, output from the decoder. Both are shape (count, batch_size, output_dim). Count dim is removed if 1.
            samples: Latent variable used to create reconstruction (input to the decoder). Shape (count, batch_size, latent_dim). Count dim is removed if 1.
            (encoder_mean, encoder_logvar): Output of the encoder. Both are shape (batch_size, latent_dim)
        """
        assert data.shape[1] == self.input_dim
        if mask is None:
            mask = torch.ones_like(data)
        assert data.shape == mask.shape
        embedded = self._set_encoder(data, mask)
        return self._vae.reconstruct(embedded, sample=sample, count=count)

    # Internal helper functions for model #

    def _loss_ELBO(
        self, x: torch.Tensor, input_mask: torch.Tensor, scoring_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate the ELBO loss for data with mask.
        loss = kl_divergence(N(0, 1), N(μ, σ)) + nll(x, x')
        where
            x' is the reconstructed data
            μ is a latent variable representing the mean of the distribution
            σ is a latent variable representing the standard deviation of the distribution

        Args:in
            x: data with shape (batch_size, input_dim)
            input_mask: mask indicating observed variables with shape (batch_size, input_dim). 1 is observed, 0 is un-observed.
            scoring_mask: mask indicating which variables should be included in NLL calculation.

        Returns:
            loss: total loss for all batches.
            kl_divergence: for tracking training. kld summed over all batches.
            negative log likelihood: for tracking training. nll summed over all batches.
        """
        # pass through encoder and decoder

        (dec_mean, dec_logvar), _, encoder_output = self.reconstruct(x, input_mask)
        # compute loss
        # KL divergence between approximate posterior q and prior p
        kl = kl_divergence(encoder_output).sum()
        nll = negative_log_likelihood(x, dec_mean, dec_logvar, self.variables, self._alpha, scoring_mask)

        loss = self._beta * kl + nll
        return loss, kl, nll

    def _loss_IWAE(
        self, x: torch.Tensor, input_mask: torch.Tensor, scoring_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate the IWAE loss for data with mask.
        Please refer to https://arxiv.org/abs/1509.00519 for more details
        loss = logsumexp( log p(x, x'|z_k) + log p(z_k) - log q(z_k) ) + const.
        where
            x' is the reconstructed data
            z_k is a posterior sample from latent space
            p is the prior distribution (standard gaussian)
            q is the PNP encoder distribution

        Args:in
            x: data with shape (batch_size, input_dim)
            input_mask: mask indicating observed variables with shape (batch_size, input_dim). 1 is observed, 0 is un-observed.
            scoring_mask: mask indicating which variables should be included in the NLL calculation.
            count: number of importance samples
        Returns:
            loss: total loss for all batches.
            kl_divergence: for tracking training. kld summed over all batches. Not used in training, for monitoring training only.
            negative log likelihood: for tracking training. nll summed over all batches.
        """
        # pass through encoder and decoder
        (dec_mean, dec_logvar), latent_samples, (enc_mean, enc_logvar) = self.reconstruct(
            x, input_mask, count=self.importance_count
        )

        batch_size, feature_count = x.shape[0], self.variables.num_processed_non_aux_cols
        dec_mean = dec_mean.reshape(-1, feature_count)
        dec_logvar = dec_logvar.reshape(-1, feature_count)
        x = x.unsqueeze(0).repeat((self.importance_count, 1, 1)).reshape(-1, x.shape[1])
        scoring_mask = (
            scoring_mask.unsqueeze(0).repeat((self.importance_count, 1, 1)).reshape(-1, scoring_mask.shape[1])
        )

        nll = negative_log_likelihood(x, dec_mean, dec_logvar, self.variables, self._alpha, scoring_mask, sum_type=None)
        nll = nll.sum(1).reshape(-1, batch_size)
        importance_weights = gaussian_negative_log_likelihood(
            latent_samples, torch.zeros_like(enc_mean), torch.zeros_like(enc_logvar), mask=None, sum_type=None
        ).sum(2) - gaussian_negative_log_likelihood(latent_samples, enc_mean, enc_logvar, mask=None, sum_type=None).sum(
            2
        )

        loss = (torch.logsumexp(nll + importance_weights, 0) - np.log(self.importance_count)).sum()
        kl = kl_divergence((enc_mean, enc_logvar)).sum()
        return loss, kl, nll.sum()

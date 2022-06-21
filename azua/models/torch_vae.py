from abc import abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributions as tdist

from ..datasets.dataset import Dataset, SparseDataset
from ..datasets.variables import Variables
from ..models.imodel import IModelWithReconstruction
from ..models.torch_model import TorchModel
from ..models.torch_training import train_model
from ..models.torch_training_types import LossConfig, VAELossResults
from ..utils.training_objectives import kl_divergence, negative_log_likelihood
from .models_utils import assert_elements_have_equal_first_dimension, assert_elements_have_equal_shape


class TorchVAE(TorchModel, IModelWithReconstruction):
    def __init__(
        self,
        model_id: str,
        variables: Variables,
        save_dir: str,
        device: torch.device,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        categorical_likelihood_coefficient: float,
        kl_coefficient: float,
    ) -> None:
        """
        Args:
            model_id (str): Unique model ID for referencing this model instance.
            variables (Variables): Information about variables/features used
                by this model.
            save_dir (str): Location to save any information about this model, including training data.
                be created if it doesn't exist.
            device (`torch.device`): Device to load model to.
            encoder (`torch.nn.Module`): torch encoder to be used in encode()
            decoder (`torch.nn.Module`): torch decoder to be used in decode()
            categorical_likelihood_coefficient (float): coefficient for balancing likelihoods
            kl_coefficient (float): coefficient for KL terms

        """

        super().__init__(model_id, variables, save_dir, device)
        self._encoder = encoder

        self._decoder = decoder

        self._alpha = categorical_likelihood_coefficient
        self._beta = kl_coefficient

    # IMPLEMENTATION OF ABSTRACT METHODS #
    def save_onnx(self, save_dir: str) -> None:
        # I am not sure to what extent this is used
        # TODO: Investigate to what extent this is being/planned to be used
        # If this is used, add new interface for save_onnx (and fix annotation)
        # If this is not used, remove the method
        self._decoder.save_onnx(save_dir)  # type: ignore
        self._encoder.save_onnx(save_dir)  # type: ignore

    def validate_loss_config(self, loss_config: LossConfig) -> None:
        # At the moment, we don't use loss_config (thus ignore it):
        # -We don't apply max_p_dropout_train
        # -Mask is not used as input mask to the model (i.e. )
        # -Mask is used as scoring mask
        pass

    def loss(self, loss_config: LossConfig, input_tensors: Tuple[torch.Tensor, ...]) -> VAELossResults:
        total_loss, kl, total_nll = self._loss(*input_tensors)
        mask_sum = input_tensors[1].sum() if len(input_tensors) > 1 else torch.tensor(input_tensors[0].numel())
        return VAELossResults(loss=total_loss, kl=kl, nll=total_nll, mask_sum=mask_sum)

    def _train(
        self,  # type: ignore[override]
        dataset: Union[Dataset, SparseDataset],
        train_output_dir: str,
        report_progress_callback: Optional[Callable[[str, int, int], None]],
        learning_rate: float,
        batch_size: int,
        iterations: int,
        epochs: int,
    ) -> Dict[str, List[float]]:
        return train_model(
            model=self,
            dataset=dataset,
            train_output_dir=train_output_dir,
            report_progress_callback=report_progress_callback,
            learning_rate=learning_rate,
            batch_size=batch_size,
            iterations=iterations,
            epochs=epochs,
            rewind_to_best_epoch=False,
        )

    def _loss(self, *input_tensors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate the loss for data
        loss = kl_divergence(N(0, 1), N(μ, σ)) + nll(x, x')
        where
            x is the first element of input tensors
            x' is the reconstructed first element of input tensors
            μ is a latent variable representing the mean of the distribution
            σ is a latent variable representing the standard deviation of the distribution

        Args:
            input_tensors: input tensors. each with shape (batch_size, input_dim)

        Returns:
            loss: total loss for all batches.
            kl_divergence: for tracking training. kld summed over all batches.
            negative log likelihood: for tracking training. nll summed over all batches.
        """
        assert_elements_have_equal_shape(*input_tensors)

        # TODO dedup with PVAE's _loss_ELBO
        # pass through encoder and decoder
        (dec_mean, dec_logvar), _, encoder_output = self.reconstruct(*input_tensors)

        # compute loss
        # KL divergence between approximate posterior q and prior p
        kl = kl_divergence(encoder_output).sum()
        # Assume that second tensor, if present, is a mask
        mask = input_tensors[1] if len(input_tensors) > 1 else None
        nll = negative_log_likelihood(input_tensors[0], dec_mean, dec_logvar, self.variables, self._alpha, mask=mask)

        loss = self._beta * kl + nll
        return loss, kl, nll

    @abstractmethod
    def encode(self, *input_tensors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run encoding part of the VAE.

        Args:
            input_tensors: Input tensors. Each with shape (batch_size, input_dim).

        Returns:
            mean, logvar: Latent space samples of shape (batch_size, latent_dim).
        """
        raise NotImplementedError()

    @abstractmethod
    def decode(self, data: torch.Tensor, *input_tensors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run decoding part of the VAE.

        Args:
            data: Input tensor with shape (batch_size, latent_dim).
            input_tensors: Input tensors. Each with shape (batch_size, input_dim).

        Returns:
            mean, logvar: Output of shape (batch_size, output_dim)
        """
        raise NotImplementedError()

    # TODO: Fix IModelForReconstruction to use *input_tensors, and remove below type_ignore
    def reconstruct(  # type: ignore
        self, *input_tensors: torch.Tensor, sample: bool = True, count: int = 1
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Tuple[torch.Tensor, torch.Tensor],]:
        """
        Reconstruct a fully observed version of x by passing through the VAE.

        Args:
            input_tensors: Input tensors. Each with shape (batch_size, input_dim).
            sample: Boolean. Defaults to True. If True, samples the latent variables, otherwise uses the mean.
            count: Int. Defaults to 1. Number of samples to reconstruct.

        Returns:
            (decoder_mean, decoder_logvar): Reconstucted variables, output from the decoder. Both are shape (count, batch_size, output_dim). Count dim is removed if 1.
            samples: Latent variable used to create reconstruction (input to the decoder). Shape (count, batch_size, latent_dim). Count dim is removed if 1.
            (encoder_mean, encoder_logvar): Output of the encoder. Both are shape (batch_size, latent_dim)

        """
        assert_elements_have_equal_first_dimension(*input_tensors)

        # Run through the encoder.
        encoder_mean, encoder_logvar = self.encode(*input_tensors)  # Each with shape (batch_size, latent_dim)

        # Clamp encoder_logvar (better numerical stability)
        encoder_logvar = torch.clamp(encoder_logvar, -20, 20)
        encoder_stddev = torch.sqrt(torch.exp(encoder_logvar))

        # Sample from latent space
        if sample:
            gaussian = tdist.Normal(encoder_mean, encoder_stddev)
            samples = gaussian.rsample((count,)).to(self._device)  # Shape (count, batch_size, latent_dim)
        else:
            samples = encoder_mean.repeat((count, 1, 1))  # Shape (count, batch_size, latent_dim)

        # Reshape samples so they're shaped correctly for running through decoder.
        batch_size, latent_dim = encoder_mean.shape
        samples_batched = samples.reshape(count * batch_size, latent_dim)

        # Run through decoder
        # Pass around input_tensors to enable more variations of VAE e.g. PredictiveVAE
        decoder_mean, decoder_logvar = self.decode(samples_batched, *input_tensors)

        if count != 1:
            # Reshape back into (sample_count, batch_size, output_dim)
            _, output_dim = decoder_mean.shape
            samples_batched = samples_batched.reshape(count, batch_size, latent_dim)
            decoder_mean = decoder_mean.reshape(count, batch_size, output_dim)
            decoder_logvar = decoder_logvar.reshape(count, batch_size, output_dim)

        return (
            (decoder_mean, decoder_logvar),
            samples_batched,
            (encoder_mean, encoder_logvar),
        )

    def run_train(self, *args, **kwargs):
        raise NotImplementedError

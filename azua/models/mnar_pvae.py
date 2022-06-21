import warnings
from typing import Optional, Tuple

import numpy as np
import torch
import torch.distributions as tdist
from torch.nn import Identity

from ..datasets.variables import Variable, Variables
from ..models.torch_model import TorchModel
from ..utils.training_objectives import gaussian_negative_log_likelihood, kl_divergence, negative_log_likelihood
from .decoder import Decoder
from .encoder import Encoder
from .mask_net import MaskNet
from .partial_vae import PartialVAE


class MNARPartialVAE(PartialVAE, TorchModel):
    """
        Implements MNAR Partial VAE.
        Basic introduction: see https://microsofteur-my.sharepoint.com/personal/chezha_microsoft_com/_layouts/OneNote.aspx?id=%2Fpersonal%2Fchezha_microsoft_com%2FDocuments%2FNotebooks%2FMinDataAI_Master&wd=target%28New%20Section%20Group%2FMNAR%20code%20design.one%7C18B31B8B-1108-4985-9FF4-E903B5AA90A4%2F%29
    onenote:https://microsofteur-my.sharepoint.com/personal/chezha_microsoft_com/Documents/Notebooks/MinDataAI_Master/New%20Section%20Group/MNAR%20code%20design.one#section-id={18B31B8B-1108-4985-9FF4-E903B5AA90A4}&end
    """

    def __init__(self, *args, mask_net_config, prior_net_config, **kwargs) -> None:
        """
        Args:
            mask_net_config: configs for mask net
            prior_net_config: configs for prior net
            **kwargs: extra parameters
        """
        super().__init__(*args, **kwargs)
        # create missing model
        self._mask_net_config = mask_net_config
        self._prior_net_config = prior_net_config
        self._mask_net, self._mask_variables = self._create_mask_nn(self.variables, self._device)
        self._mask_net_coefficient = mask_net_config["mask_net_coefficient"]
        # create prior net
        self._prior_net_input_list = self.variables.proc_always_observed_list
        self._prior_net_input_dim = self._prior_net_input_list.count(True)

        if prior_net_config["use_prior_net_to_train"]:
            if not any(self._prior_net_input_list):
                warnings.warn("no fully observed columns, switch to degenerate prior")
                if not self._prior_net_config["degenerate_prior"] == "gaussian":
                    self._prior_net = Encoder(
                        input_dim=self.input_dim,
                        hidden_dims=prior_net_config["encoder_layers"],
                        latent_dim=self.latent_dim,
                        device=self._device,
                        non_linearity=Identity,
                    )

                    self._prior_net.train(False)
            else:
                self._prior_net = Encoder(
                    input_dim=self._prior_net_input_dim,
                    hidden_dims=prior_net_config["encoder_layers"],
                    latent_dim=self.latent_dim,
                    device=self._device,
                    non_linearity=Identity,
                )
                self._prior_net.train(False)

    @classmethod
    def name(cls) -> str:
        return "mnar_pvae"

    @staticmethod
    def _create_mask_variables(variables):
        mask_all_variables = []
        for x_variable in variables:
            mask_all_variable = Variable(
                name=f"R_{x_variable.name}",
                query=False,
                always_observed=True,
                type="binary",
                lower=0,
                upper=1,
                overwrite_processed_dim=x_variable.processed_dim,
            )
            mask_all_variables.append(mask_all_variable)

        mask_variables = mask_all_variables[0 : variables.num_unprocessed_non_aux_cols]
        mask_aux_variables = mask_all_variables[variables.num_unprocessed_non_aux_cols :]

        return Variables(mask_variables, mask_aux_variables)

    def _create_mask_nn(self, variables, device: torch.device):
        """
        create a neural net that models the missing mechanism
        Args:
            variables (`model_interface.variables.Variables`): Information about input variables x.
            device (str or int): Name of Torch device to create the model on. Valid options are 'cpu', 'gpu',
            or a device ID (e.g. 0 or 1 on a two-GPU machine).
        Returns:
            mask_net: a Decoder model representing the missing mechanism
            mask_variables: information about the mask variables
        """
        #  if latent connection is true, our mask net model will be specified by the connections Z->X->mask and Z->mask
        #  otherwise, it is just the Z->X->mask model
        if self._mask_net_config["latent_connection"]:
            mask_net_input_dim = variables.num_processed_non_aux_cols + self.latent_dim
        else:
            mask_net_input_dim = variables.num_processed_non_aux_cols
        output_dim = variables.num_processed_cols
        mask_variables = self._create_mask_variables(variables)
        if self._mask_net_config["latent_connection"]:
            return (
                Decoder(
                    mask_net_input_dim,
                    output_dim,
                    mask_variables,
                    hidden_dims=self._mask_net_config["decoder_layers"],
                    variance=0,
                    device=device,
                ),
                mask_variables,
            )
        else:
            return MaskNet(mask_net_input_dim, device=device), mask_variables

    def _loss_ELBO(
        self, x: torch.Tensor, input_mask: torch.Tensor, scoring_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate the ELBO loss for data with mask.
        loss = kl_divergence(N(μ_prior, σ_prior), N(μ_posterior, σ_posterior)) + nll(x, x') + nll(mask, mask'|x)
        where
            x' is the reconstructed data
            μ is a latent variable representing the mean of the distribution
            σ is a latent variable representing the standard deviation of the distribution

        Args:in
            x: data with shape (batch_size, input_dim)
            mask: mask indicting observed variables with shape (batch_size, input_dim). 1 is observed, 0 is un-observed.

        Returns:
            loss: total loss for all batches.
            kl_divergence: for tracking training. kld summed over all batches.
            negative log likelihood: for tracking training. nll summed over all batches.
        """
        # pass through encoder and decoder
        (dec_mean_x, dec_logvar_x), posterior_samples, (enc_mean, enc_logvar) = self._generate_from_inference_net(
            x, input_mask
        )

        use_prior_net = self._prior_net_config["use_prior_net_to_train"] and (
            any(self._prior_net_input_list) or self._prior_net_config["degenerate_prior"] != "gaussian"
        )
        if use_prior_net:
            _, _, (prior_mean_x, prior_logvar_x) = self._generate_from_prior_net(x, input_mask)
        else:
            prior_mean_x = torch.zeros_like(enc_mean)
            prior_logvar_x = torch.zeros_like(enc_mean)

        # compute loss
        # KL divergence between approximate posterior q and prior p
        if self._prior_net_config["use_prior_net_to_train"]:
            kl = kl_divergence((enc_mean, enc_logvar), (prior_mean_x, prior_logvar_x)).sum()
        else:
            kl = kl_divergence((enc_mean, enc_logvar)).sum()

        mixed_x = (
            dec_mean_x * (1 - input_mask[:, 0 : self._mask_variables.num_processed_non_aux_cols])
            + x[:, 0 : self._mask_variables.num_processed_non_aux_cols]
            * input_mask[:, 0 : self._mask_variables.num_processed_non_aux_cols]
        )
        if self._mask_net_config["latent_connection"]:
            mask_net_input = torch.cat([mixed_x, posterior_samples.reshape([mixed_x.shape[0], -1])], 1)
            (dec_mean_mask, _) = self._mask_net(mask_net_input)
        else:
            mask_net_input = mixed_x
            dec_mean_mask = self._mask_net(mask_net_input)
        nll_mask = negative_log_likelihood(
            scoring_mask[:, 0 : self._mask_variables.num_processed_non_aux_cols],
            dec_mean_mask,
            torch.zeros_like(dec_mean_mask),
            self._mask_variables,
            self._alpha,
            torch.ones_like(scoring_mask),
        )
        nll_x = negative_log_likelihood(x, dec_mean_x, dec_logvar_x, self.variables, self._alpha, scoring_mask)
        nll = nll_x + self._mask_net_coefficient * nll_mask

        loss = self._beta * kl + nll
        return loss, kl, nll

    def _loss_IWAE(
        self, x: torch.Tensor, input_mask: torch.Tensor, scoring_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate the IWAE loss for data with mask.
        Please refer to https://arxiv.org/abs/1509.00519 for more details
        loss = logsumexp( log p(x, x'|z_k) log p(mask, mask'|x_imputed) + log p(z_k) - log q(z_k) ) + const.
        where
            x' is the reconstructed data
            z_k is a posterior sample from latent space
            p is the prior distribution (standard gaussian)
            q is the PNP encoder distribution

        Args:in
            x: data with shape (batch_size, input_dim)
            mask: mask indicting observed variables with shape (batch_size, input_dim). 1 is observed, 0 is un-observed.
            count: number of importance samples
        Returns:
            loss: total loss for all batches.
            kl_divergence: for tracking training. kld summed over all batches. Not used in training,
            for monitoring training only.
            negative log likelihood: for tracking training. nll summed over all batches.
        """
        # pass through encoder and decoder
        (dec_mean_x, dec_logvar_x), posterior_samples, (enc_mean, enc_logvar) = self._generate_from_inference_net(
            x, input_mask, count=self.importance_count
        )

        use_prior_net = self._prior_net_config["use_prior_net_to_train"] and (
            any(self._prior_net_input_list) or self._prior_net_config["degenerate_prior"] != "gaussian"
        )
        if use_prior_net:
            _, _, (prior_mean_x, prior_logvar_x) = self._generate_from_prior_net(
                x, input_mask, count=self.importance_count
            )
        else:
            prior_mean_x = torch.zeros_like(enc_mean)
            prior_logvar_x = torch.zeros_like(enc_mean)

        batch_size, feature_count = x.shape
        dec_mean_x = dec_mean_x.reshape(-1, feature_count)
        dec_logvar_x = dec_logvar_x.reshape(-1, feature_count)
        x = x.unsqueeze(0).repeat((self.importance_count, 1, 1)).reshape(-1, feature_count)
        scoring_mask = scoring_mask.unsqueeze(0).repeat((self.importance_count, 1, 1)).reshape(-1, feature_count)
        input_mask = input_mask.unsqueeze(0).repeat((self.importance_count, 1, 1)).reshape(-1, feature_count)
        nll_x = negative_log_likelihood(
            x, dec_mean_x, dec_logvar_x, self.variables, self._alpha, scoring_mask, sum_type=None
        )

        mixed_x = (
            dec_mean_x * (1 - input_mask[:, 0 : self._mask_variables.num_processed_non_aux_cols])
            + x * input_mask[:, 0 : self._mask_variables.num_processed_non_aux_cols]
        )
        if self._mask_net_config["latent_connection"]:
            mask_net_input = torch.cat([mixed_x, posterior_samples.reshape([mixed_x.shape[0], -1])], 1)
            (dec_mean_mask, _) = self._mask_net(mask_net_input)
        else:
            mask_net_input = mixed_x
            dec_mean_mask = self._mask_net(mask_net_input)
        nll_mask = negative_log_likelihood(
            scoring_mask[:, 0 : self._mask_variables.num_processed_non_aux_cols],
            dec_mean_mask,
            torch.zeros_like(dec_logvar_x),
            self._mask_variables,
            1,
            torch.ones_like(scoring_mask),
            sum_type=None,
        )
        nll = torch.cat([nll_x, self._mask_net_coefficient * nll_mask], 1)
        nll = nll.sum(1).reshape(-1, batch_size)
        if self._prior_net_config["use_prior_net_to_train"]:
            importance_weights = gaussian_negative_log_likelihood(
                posterior_samples, prior_mean_x, prior_logvar_x, mask=None, sum_type=None
            ).sum(2) - gaussian_negative_log_likelihood(
                posterior_samples, enc_mean, enc_logvar, mask=None, sum_type=None
            ).sum(
                2
            )
        else:
            importance_weights = gaussian_negative_log_likelihood(
                posterior_samples, torch.zeros_like(enc_mean), torch.zeros_like(enc_logvar), mask=None, sum_type=None
            ).sum(2) - gaussian_negative_log_likelihood(
                posterior_samples, enc_mean, enc_logvar, mask=None, sum_type=None
            ).sum(
                2
            )
        loss = -(torch.logsumexp(-nll - importance_weights, 0) - np.log(self.importance_count)).sum()
        if self._prior_net_config["use_prior_net_to_train"]:
            kl = kl_divergence((enc_mean, enc_logvar), (prior_mean_x, prior_logvar_x)).sum()
        else:
            kl = kl_divergence((enc_mean, enc_logvar)).sum()
        return loss, kl, nll.sum()

    def _generate_from_prior_net(
        self, data: torch.Tensor, mask: torch.Tensor, sample: bool = True, count: int = 1
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Tuple[torch.Tensor, torch.Tensor],]:
        """
        reconstrutc samples from prior net, rather than from encoder.

        Args:
            data: partially observed input data with shape (batch_size, input_dim).
            sample: Boolean. Defaults to True. If True, samples the latent variables, otherwise uses the mean.
            count: Int. Defaults to 1. Number of samples to reconstruct.

        Returns:
            (decoder_mean, decoder_logvar): Reconstucted variables, output from the decoder.
                                            Both are shape (count, batch_size, output_dim). Count dim is removed if 1.
            samples: Latent variable used to create reconstruction (input to the decoder).
                    Shape (count, batch_size, latent_dim). Count dim is removed if 1.
            (encoder_mean, encoder_logvar): Output of the encoder. Both are shape (batch_size, latent_dim)

        """
        # Run through the encoder.
        if not any(self._prior_net_input_list):
            if not self._prior_net_config["degenerate_prior"] == "gaussian":
                prior_net_input = mask  # this is only used when there are no aux variables
                encoder_mean, encoder_logvar = self._prior_net(
                    prior_net_input
                )  # Each with shape (batch_size, latent_dim)
            else:
                encoder_mean = torch.zeros((data.shape[0], self.latent_dim))
                encoder_logvar = torch.zeros((data.shape[0], self.latent_dim))
        else:
            prior_net_input = data[:, self._prior_net_input_list]
            encoder_mean, encoder_logvar = self._prior_net(prior_net_input)  # Each with shape (batch_size, latent_dim)

        # Clamp encoder_logvar (better numerical stability)
        encoder_logvar = torch.clamp(encoder_logvar, -10, 10)
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
        decoder_mean, decoder_logvar = self._vae._decoder(samples_batched)

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

    def _generate_from_inference_net(
        self,
        data: torch.Tensor,
        mask: Optional[torch.Tensor],
        sample: bool = True,
        count: int = 1,
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
            (decoder_mean, decoder_logvar): Reconstucted variables, output from the decoder.
                                            Both are shape (count, batch_size, output_dim). Count dim is removed if 1.
            samples: Latent variable used to create reconstruction (input to the decoder).
                    Shape (count, batch_size, latent_dim). Count dim is removed if 1.
            (encoder_mean, encoder_logvar): Output of the encoder. Both are shape (batch_size, latent_dim)
        """
        assert data.shape[1] == self.input_dim
        if mask is None:
            mask = torch.ones_like(data)
        assert data.shape == mask.shape
        embedded = self._set_encoder(data, mask)
        return self._vae.reconstruct(embedded, sample=sample, count=count)

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
            (decoder_mean, decoder_logvar): Reconstucted variables, output from the decoder.
                                            Both are shape (count, batch_size, output_dim). Count dim is removed if 1.
            samples: Latent variable used to create reconstruction (input to the decoder).
                    Shape (count, batch_size, latent_dim). Count dim is removed if 1.
            (encoder_mean, encoder_logvar): Output of the encoder. Both are shape (batch_size, latent_dim)
        """
        assert data.shape[1] == self.input_dim
        if mask is None:
            mask = torch.ones_like(data)
        assert data.shape == mask.shape
        if self._prior_net_config["use_prior_net_to_impute"] and self._prior_net_config["use_prior_net_to_train"]:
            return self._generate_from_prior_net(data, mask, sample=sample, count=count)
        else:
            if self._prior_net_config["use_prior_net_to_impute"]:
                warnings.warn("Prior Net not used in training phase, switch to impute using inference net")
            return self._generate_from_inference_net(data, mask, sample=sample, count=count)

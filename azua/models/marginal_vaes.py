from ..utils.exceptions import ONNXNotImplemented

from ..models.torch_vae import TorchVAE

from ..models.vae import VAE
from ..datasets.variables import Variables
import os
from typing import Any, Dict, Tuple

import torch


class MarginalVAEs(TorchVAE):
    def __init__(
        self, model_id: str, variables: Variables, save_dir: str, device: torch.device, **model_config_dict,
    ):
        # TODO: Once we extract part of this class logic (encode(), decode() and _marginal_vaes) into
        # StackedEncoder and StackedDecoder, we could just create instances of these classes
        # and pass them into parent's constructor. In the meantime, we artifically pass None for encoder and decoder
        categorical_likelihood_coefficient = model_config_dict["categorical_likelihood_coefficient"]
        kl_coefficient = model_config_dict["kl_coefficient"]
        super().__init__(model_id, variables, save_dir, device, None, None, categorical_likelihood_coefficient, kl_coefficient)  # type: ignore

        self._marginal_vaes = self._create_marginal_vaes(model_config_dict)

        # TODO: remove the need for it
        self.vae_latent_dim = model_config_dict["latent_dim"]
        # Total processed dim
        self._output_dim = sum([var.processed_dim for var in self.variables])

    def _create_marginal_vaes(self, vae_config: Dict[str, Any]) -> torch.nn.ModuleList:
        # Returns a nn.ModuleList with one VAE per variable

        return torch.nn.ModuleList(
            [self._create_one_marginal_vae(v, idx, vae_config, self.save_dir) for idx, v in enumerate(self.variables)]
        )

    def _create_one_marginal_vae(self, variable, idx, vae_config, all_vaes_dir):
        vae_model_id = f"marginal_{idx}"
        vae_variables = Variables([variable])
        vae_dir = os.path.join(all_vaes_dir, vae_model_id)
        os.makedirs(vae_dir, exist_ok=True)

        return VAE.create(
            model_id=vae_model_id,
            save_dir=vae_dir,
            variables=vae_variables,
            model_config_dict=vae_config,
            device=self._device,
        )

    # CLASS METHODS #
    @classmethod
    def name(cls) -> str:
        return "marginal_vaes"

    def encode(self, *input_tensors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encoding part of Marginal VAEs

        Args:
            input_tensors: Input tensors. Each with shape (batch_size, total_processed_dim)

        Returns:
            mean, logvar: Latent space samples of shape (batch_size, variable_count * latent_dim).
        """
        data = input_tensors[0]
        output_tensor_size = (data.shape[0], len(self.variables) * self.vae_latent_dim)
        all_encoder_mean = torch.zeros(output_tensor_size, device=self._device)
        all_encoder_logvar = torch.zeros(output_tensor_size, device=self._device)
        for i, variable in enumerate(self.variables):  # TODO: parallelize
            var_data = self.variables.get_var_cols_from_data(i, data)  # Shape(batch_size, processed_dim)
            # Encode data into latent space.
            var_latent_slice = slice(self.vae_latent_dim * i, self.vae_latent_dim * i + self.vae_latent_dim)
            all_encoder_mean[:, var_latent_slice], all_encoder_logvar[:, var_latent_slice] = self._encode_single_var(
                i, var_data
            )  # Each with shape (batch_size, latent_dim)

        # Allow running with mask and without
        # TODO: refactor VAEM, so not allowing running without mask anymore
        if len(input_tensors) > 1:
            # Nullify rows which are masked
            mask = input_tensors[1]
            for idx in range(len(self.variables)):
                var_mask = self.variables.get_var_cols_from_data(idx, mask)

                rows_not_to_keep = torch.where(var_mask[:, 0] == 0)[0]
                var_cols = slice(self.vae_latent_dim * idx, self.vae_latent_dim * idx + self.vae_latent_dim)
                all_encoder_mean[rows_not_to_keep, var_cols] = 0
                all_encoder_logvar[rows_not_to_keep, var_cols] = 0
        return all_encoder_mean, all_encoder_logvar

    def _encode_single_var(self, var_idx: int, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        vae = self._marginal_vaes[var_idx]
        return vae.encode(data)

    def decode(self, data: torch.Tensor, *input_tensors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decoding part of Marginal VAEs

        Args:
            data: Input tensor with shape (batch_size, variables_count * latent_dim).

        Returns:
            mean, logvar: Output of shape (batch_size, total_processed_dim)
        """
        batch_size = data.shape[0]
        data = data.reshape(batch_size, len(self.variables), -1).permute(1, 0, 2)

        recon_x_means = torch.zeros((batch_size, self._output_dim), device=self._device)
        recon_x_logvars = torch.zeros((batch_size, self._output_dim), device=self._device)
        i = 0
        for x_var_id, var_z in enumerate(data):  # TODO: parallelize
            # decode data into observation space.
            recon_x_mean, recon_x_logvar = self._decode_single_var(x_var_id, var_z)
            num_cols = recon_x_mean.shape[1]

            # Add results to tensors.
            recon_x_means[:, i : i + num_cols] = recon_x_mean
            recon_x_logvars[:, i : i + num_cols] = recon_x_logvar
            i += num_cols
        return recon_x_means, recon_x_logvars

    def _decode_single_var(self, var_id: int, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        vae = self._marginal_vaes[var_id]
        return vae.decode(data)

    # TODO: remove once, we start using StackedEncoder and StackedDecoder which support save_onnx() method
    def save_onnx(self, save_dir: str) -> None:
        raise ONNXNotImplemented

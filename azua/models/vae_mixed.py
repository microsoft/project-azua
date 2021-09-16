# This is required in python 3 to allow return types of the same class.
from __future__ import annotations

import logging
from ..models.dependency_network_creator import DependencyNetworkCreator
from ..models.marginal_vaes import MarginalVAEs
import os

import torch
import torch.distributions as tdist
import numpy as np
from scipy.sparse import csr_matrix

from ..datasets.variables import Variables
from ..datasets.dataset import Dataset, SparseDataset
from ..models.pvae_base_model import PVAEBaseModel
from ..models.torch_training import train_model
from ..utils.io_utils import save_json
from ..utils.data_mask_utils import to_tensors
from typing import Dict, Optional, Tuple, Callable, Any, Union, cast


class VAEMixed(PVAEBaseModel):
    """
    Instance of `models.pvae_base_model.PVAEBaseModel` representing a VAEM.
    """

    _marginal_network_save_dir = "marginal_vaes"
    _dependency_network_save_dir = "dependency_network"
    _var_id_to_model_id_file = "model_ids.json"

    _marginal_vaes: MarginalVAEs

    def __init__(
        self, model_id: str, variables: Variables, save_dir: str, device: torch.device, **model_config_dict
    ) -> None:
        """
        Args:
            model_id (str): Unique model ID for referencing this model instance.
            variables (Variables): Information about variables/features used
                by this model.
            save_dir (str): Location to save any information about this model, including training data.
                be created if it doesn't exist.
            device (`torch.device`): Device to use
            model_config_dict: model config
        """
        super().__init__(model_id, variables, save_dir, device)

        vae_config, dep_network_config = self._split_configs(model_config_dict)
        all_vaes_dir = os.path.join(self.save_dir, self._marginal_network_save_dir)
        self._marginal_vaes = self._create_marginal_vaes(
            variables=variables, all_vaes_dir=all_vaes_dir, device=device, **vae_config
        )
        marginal_latent_dim = vae_config["latent_dim"]
        dependency_save_dir = os.path.join(save_dir, self._dependency_network_save_dir)
        self._dependency_network = DependencyNetworkCreator.create(
            variables, dependency_save_dir, self._device, marginal_latent_dim, dep_network_config
        )

        self.input_dim = variables.num_processed_cols
        self.output_dim = variables.num_processed_cols

    # Having a separate method, so the logic can be overriden
    # TODO: rethink the design here: write it in the way so we don't need to do it
    def _create_marginal_vaes(self, variables: Variables, all_vaes_dir: str, device: torch.device, **vae_config):
        return MarginalVAEs(
            model_id=self._marginal_network_save_dir,
            variables=variables,
            save_dir=all_vaes_dir,
            device=device,
            **vae_config,
        )

    # CLASS METHODS
    @classmethod
    def name(cls) -> str:
        return "vaem"

    def _train(  # type: ignore[override]
        self,
        dataset: Union[Dataset, SparseDataset],
        train_output_dir: str,
        report_progress_callback: Optional[Callable[[str, int, int], None]],
        dep_learning_rate: float,
        dep_batch_size: int,
        dep_iterations: int,
        dep_epochs: int,
        marginal_learning_rate: float,
        marginal_batch_size: int,
        marginal_iterations: int,
        marginal_epochs: int,
        max_p_train_dropout: float,
        score_reconstruction: bool,
        score_imputation: bool,
        rewind_to_best_epoch: bool,
        lr_warmup_epochs: int = 0,
    ):
        """
        Train the model using the given data.

        Args:
            dataset: Dataset object containing data and mask in processed form.
            train_output_dir (str): Path to save any training information to, including tensorboard summary files.
            report_progress_callback: Function to report model progress for API.        # TODO: needs to be used.
            dep_learning_rate: Learning rate for dependency network
            dep_batch_size: Minibatch size for dependency network
            dep_iterations: Number of minibatch samples with replacement per epoch for training dependency network
            dep_epochs: Number of epochs for training dependency network
            marginal_learning_rate: Learning rate for marginal networks
            marginal_batch_size: Minibatch size for marginal networks
            marginal_iterations: Number of minibatch samples with replacement per epoch for training marginal networks
            marginal_epochs: Number of epochs for training marginal networks
            max_p_train_dropout: max proportion of inputs to mask during training
            score_reconstruction: flag indicating whether to score reconstructed values in NLL
            score_imputation: flag indicating whether to score imputed values in NLL
            lr_warmup_epochs: number of epochs for learning rate warm-up for the dependency network
        Returns:
            results_dict (dictionary): Train loss, KL divergence, and NLL for each epoch as a dictionary.
        """
        logger = logging.getLogger()

        logger.info("Training marginal networks.")
        train_data, train_mask = dataset.train_data_and_mask
        marginal_vaes_dir = os.path.join(train_output_dir, self._marginal_network_save_dir)
        os.makedirs(marginal_vaes_dir, exist_ok=True)
        self._marginal_vaes._train(
            dataset,
            train_output_dir=marginal_vaes_dir,
            report_progress_callback=report_progress_callback,
            learning_rate=marginal_learning_rate,
            batch_size=marginal_batch_size,
            iterations=marginal_iterations,
            epochs=marginal_epochs,
        )

        logger.info("Training dependency network.")
        # Train by running data through encoder in marginals.
        # Place dependency network inputs on CPU, they will be moved to GPU during training.
        tensor_train_data, tensor_train_mask = to_tensors(train_data, train_mask, device=torch.device("cpu"))
        dep_network_input_data, dep_network_input_mask = self._get_dependency_network_input(
            tensor_train_data, tensor_train_mask
        )

        dep_network_val_data_array: Optional[np.ndarray]
        dep_network_val_mask_array: Optional[np.ndarray]
        if dataset.has_val_data:
            val_data, val_mask = dataset.val_data_and_mask
            val_data = cast(Union[csr_matrix, np.ndarray], val_data)
            val_mask = cast(Union[csr_matrix, np.ndarray], val_mask)
            dep_network_val_data, dep_network_val_mask = self._get_dependency_network_input(
                *to_tensors(val_data, val_mask, device=torch.device("cpu"))
            )
            dep_network_val_data_array = dep_network_val_data.detach().numpy()
            dep_network_val_mask_array = dep_network_val_mask.detach().numpy()
        else:
            dep_network_val_data_array, dep_network_val_mask_array = None, None

        # Then use that as data to train dep network.
        # TODO dedup with predictive_vae_mixed.py
        dep_dir = os.path.join(train_output_dir, self._dependency_network_save_dir)
        os.makedirs(dep_dir, exist_ok=True)

        self._set_dep_net_variables_min_max(dep_network_input_data, dep_network_input_mask)
        self._squash_dep_net_input(dep_network_input_data)
        if dataset.has_val_data:
            self._squash_dep_net_input(dep_network_val_data_array)  # type: ignore

        dataset_for_dependency_network = Dataset(
            train_data=dep_network_input_data.detach().numpy(),
            train_mask=dep_network_input_mask.detach().numpy(),
            val_data=dep_network_val_data_array,
            val_mask=dep_network_val_mask_array,
            test_data=None,
            test_mask=None,
            variables=None,
        )

        dep_training_results = train_model(
            model=self._dependency_network,
            dataset=dataset_for_dependency_network,
            train_output_dir=dep_dir,
            report_progress_callback=report_progress_callback,
            learning_rate=dep_learning_rate,
            batch_size=dep_batch_size,
            iterations=dep_iterations,
            epochs=dep_epochs,
            max_p_train_dropout=max_p_train_dropout,
            lr_warmup_epochs=lr_warmup_epochs,
            rewind_to_best_epoch=rewind_to_best_epoch,
            score_reconstruction=score_reconstruction,
            score_imputation=score_imputation,
        )
        self.save()

        # Save dependency network training results as top-level training results.
        # TODO do we really want to do this?
        if dep_training_results is not None:
            train_results_save_path = os.path.join(self.save_dir, "training_results_dict.json")
            save_json(dep_training_results, train_results_save_path)

    def encode(self, data: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run encoding part of the PVAE.

        Args:
            data: Input tensor with shape (batch_size, input_dim).
            mask: Mask indicating observed variables with shape (batch_size, input_dim). 1 is observed, 0 is un-observed.

        Returns:
            mean, logvar: Latent space samples of shape (batch_size, latent_dim).
        """

        dep_network_input_data, dep_network_input_mask = self._get_dependency_network_input(data, mask)
        self._squash_dep_net_input(dep_network_input_data)
        mean, logvar = self._dependency_network.encode(dep_network_input_data, dep_network_input_mask)

        return mean, logvar

    @torch.no_grad()
    def _get_dependency_network_input(
        self, data: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dep_network_mask_z = []

        data_device = data.device

        all_encoder_mean, all_encoder_logvar = self._marginal_vaes.encode(
            data.to(self._marginal_vaes.get_device())
        )  # Each with shape (batch_size, variables_count * latent_dim)
        gaussian = tdist.Normal(all_encoder_mean, (all_encoder_logvar * 0.5).exp())
        samples = gaussian.rsample().to(data_device)  # Shape (batch_size, variables_count * latent_dim)

        # Create mask for dependency network's z
        for idx in range(len(self.variables)):
            var_mask = self.variables.get_var_cols_from_data(idx, mask)  # Shape(batch_size, processed_dim)
            var_mask = torch.unsqueeze(var_mask[:, 0], dim=1)  # Shape (batch_size, 1)

            repeated_var_mask = var_mask.expand(-1, self._marginal_vaes.vae_latent_dim)  # Shape(batch_size, latent_dim)
            dep_network_mask_z.append(repeated_var_mask)

        # Create 2D tensor from output marginals.
        dep_network_data_z_tensor = samples  # Shape (batch_size, variable_count * latent_dim)
        dep_network_mask_z_tensor = torch.cat(
            dep_network_mask_z, dim=1
        )  # Shape (batch_size, variable_count * latent_dim)

        return dep_network_data_z_tensor, dep_network_mask_z_tensor

    # Figure out min/max for data, and overwrite variables
    def _set_dep_net_variables_min_max(self, data, mask):
        # TODO: Write it correctly (account for idx being a list..)
        for idx, variable in enumerate(self._dependency_network.variables):
            var_data = data[:, idx]
            var_mask = mask[:, idx]
            var_min = min(var_data[np.where(var_mask == 1)]).item()
            var_max = max(var_data[np.where(var_mask == 1)]).item()
            variable.lower = var_min
            variable.upper = var_max

    def _squash_dep_net_input(self, dep_network_input_data: torch.Tensor):
        # squash zs into (0, 1)
        def squash(vals, lower, upper):
            return (vals - lower) / (upper - lower)

        # TODO: Write it correctly (account for idx being a list..)
        # TODO: reuse DataProcessor for it?
        for idx, variable in enumerate(self._dependency_network.variables):
            dep_network_input_data[:, idx] = squash(dep_network_input_data[:, idx], variable.lower, variable.upper)

    def _get_marginal_networks_output(
        self, z: torch.Tensor, *input_tensors: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Map from tensor z containing batch of values of marginal latent variables z_i to decoded means and logvars for 
        the corresponding features x_i.
        """

        # unsquash zs from (0, 1)
        def unsquash(vals, lower, upper):
            return (vals * (upper - lower)) + lower

        # TODO: Write it correctly (account for idx being a list..)
        # TODO: reuse DataProcessor for it?
        for idx, variable in enumerate(self._dependency_network.variables):
            z[:, idx] = unsquash(z[:, idx], variable.lower, variable.upper)

        return self._marginal_vaes.decode(z, *input_tensors)  # Each of shape (batch_size, total_processed_dim)

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
            samples: Latent variable used to create reconstruction (input to the decoder). Shape (count, batch_size, 
                latent_dim). Count dim is removed if 1.
            (encoder_mean, encoder_logvar): Output of the encoder. Both are shape (batch_size, latent_dim)
        """
        assert data.shape[1] == self.input_dim
        batch_size = data.shape[0]
        if mask is None:
            mask = torch.ones_like(data)

        dep_network_input_data, dep_network_input_mask = self._get_dependency_network_input(data, mask)

        self._squash_dep_net_input(dep_network_input_data)
        (recon_z_mean, recon_z_logvar), samples, (h_mean, h_logvar) = self._dependency_network.reconstruct(
            dep_network_input_data, dep_network_input_mask, sample=sample, count=count, **kwargs
        )
        z_dim = recon_z_mean.shape[-1]

        recon_z_mean = recon_z_mean.reshape(count * batch_size, z_dim)
        recon_z_logvar = recon_z_logvar.reshape(count * batch_size, z_dim)

        # HACK: currently have to pass data and mask to _get_marginal_networks_output() for predictive VAEM, but the
        # VAEMMixed implementation of this method doesn't use them.
        if sample:
            gaussian = tdist.Normal(recon_z_mean, torch.sqrt(torch.exp(recon_z_logvar)))
            z_samples = gaussian.rsample().to(self._device)
            recon_x_mean, recon_x_logvar = self._get_marginal_networks_output(z_samples, data, mask)
        else:
            recon_x_mean, recon_x_logvar = self._get_marginal_networks_output(recon_z_mean, data, mask)

        if count != 1:
            # Reshape back into (sample_count, batch_size, output_dim)
            recon_x_mean = recon_x_mean.reshape(count, batch_size, self.output_dim)
            recon_x_logvar = recon_x_logvar.reshape(count, batch_size, self.output_dim)
        return (recon_x_mean, recon_x_logvar), samples, (h_mean, h_logvar)

    @staticmethod
    def _split_configs(config_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        dep_config = {}
        vae_config = {}
        dep_str = "dep"
        vae_str = "marginal"
        for key, val in config_dict.items():
            key_split = key.split("_", 1)
            if len(key_split) != 2:
                key_type = None
            else:
                key_type, key_name = key_split

            if key_type == dep_str:
                dep_config[key_name] = val
            elif key_type == vae_str:
                vae_config[key_name] = val
            else:
                # Put in both.
                dep_config[key] = val
                vae_config[key] = val

        return vae_config, dep_config

import copy
import logging
from ..utils.exceptions import ONNXNotImplemented
from ..datasets.dataset import Dataset, SparseDataset
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from ..models.predictive_vae import PredictiveVAE
from ..models.marginal_vaes import MarginalVAEs
from ..models.torch_vae import TorchVAE
import torch
from ..datasets.variables import Variable, Variables


# The model combines MarginalVAES and PredictiveVAE in one model:
# -All except last variables are modelled by MarginalVAEs model
# -The last variable is a target, and is modelled with Predictive VAE
class MarginalVAEsWithPredictiveVAE(TorchVAE):

    # TODO: rethink where to put it
    _marginal_network_save_dir = "marginal_vaes"

    def __init__(
        self, model_id: str, variables: Variables, save_dir: str, device: torch.device, **model_config_dict,
    ):
        # TODO: Rethink what to pass as encoder &decoder, as we currently artifically pass None for them
        categorical_likelihood_coefficient = model_config_dict["categorical_likelihood_coefficient"]
        kl_coefficient = model_config_dict["kl_coefficient"]
        super().__init__(model_id, variables, save_dir, device, None, None, categorical_likelihood_coefficient, kl_coefficient)  # type: ignore

        # TODO: remove the need for it
        self.vae_latent_dim = model_config_dict["latent_dim"]

        variables_without_y = Variables([var for var in variables if var.query])
        self._marginal_vaes = MarginalVAEs(
            model_id=self._marginal_network_save_dir,
            variables=variables_without_y,
            save_dir=save_dir,
            device=device,
            **model_config_dict,
        )

        target_variables = [var for var in variables if not var.query]
        assert len(target_variables) == 1, "MarginalVAEsWithPredictiveVAE requires exactly one target variable"
        y_variable = target_variables[0]
        self._y_variable = y_variable
        self._y_dim = y_variable.processed_dim
        self._y_vae = self._create_predictive_vae(
            variable=y_variable, all_vaes_dir=save_dir, vae_config=model_config_dict
        )

    def _create_predictive_vae(
        self, variable: Variable, all_vaes_dir: str, vae_config: Dict[str, Any]
    ) -> PredictiveVAE:
        vae_config = copy.deepcopy(vae_config)
        vae_config["feature_dim"] = sum([var.processed_dim for var in self.variables]) - variable.processed_dim
        vae_variables = Variables([variable])
        vae_model_id = f"marginal_{variable.name}"
        vae_dir = os.path.join(all_vaes_dir, vae_model_id)
        os.makedirs(vae_dir, exist_ok=True)
        return PredictiveVAE.create(vae_model_id, vae_dir, vae_variables, vae_config, self._device)

    @classmethod
    def name(cls) -> str:
        return "marginal_vaes_with_predictive_vae"

    def encode(self, *input_tensors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        data = input_tensors[0]
        data_without_y = data[:, : -self._y_dim]
        non_y_encoder_mean, non_y_encoder_logvar = self._marginal_vaes.encode(
            data_without_y.to(self._marginal_vaes.get_device())
        )  # Each with shape (batch_size, (feature_count-1) * latent_dim)
        y_encoder_mean, y_encoder_logvar = self._y_vae.encode(
            data_without_y.to(self._device)
        )  # Each with shape (batch_size, latent_dim)
        all_encoder_mean = torch.cat([non_y_encoder_mean, y_encoder_mean], dim=1)
        all_encoder_logvar = torch.cat([non_y_encoder_logvar, y_encoder_logvar,], dim=1)
        return (all_encoder_mean, all_encoder_logvar)

    def decode(self, data: torch.Tensor, *input_tensors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = data
        x = input_tensors[0]
        mask = input_tensors[1]

        # Decode non-target variables
        recon_x_means_without_y, recon_x_logvars_without_y = self._marginal_vaes.decode(
            z[:, : -self._marginal_vaes.vae_latent_dim]
        )  # Each of shape (batch_size, total_processed_dim-latent_dim)

        # Decode target variable
        # Get data to use for this variable only.
        y_var_data = z[:, -self._marginal_vaes.vae_latent_dim :]

        # decode data into observation space.
        # slice masks on input dimensions
        x_without_y = x[:, : -self._y_dim]
        mask_without_y = mask[:, : -self._y_dim]
        # since z may contain multiple samples, we hereby tile x to match the shape of z
        sample_count = z.shape[0] / x_without_y.shape[0]
        x_without_y = x_without_y.repeat(int(sample_count), 1)
        mask_without_y = mask_without_y.repeat(int(sample_count), 1)
        # insert observed data
        masked_x = x_without_y * mask_without_y
        masked_imputation = recon_x_means_without_y * (1 - mask_without_y)
        imputation_x = masked_imputation + masked_x
        # decode y
        recon_x_mean_for_y, recon_x_logvar_for_y = self._y_vae.decode(
            y_var_data, imputation_x
        )  # Each of shape (batch_size, latent_dim)

        # Combine decoded means&logvars for non-target and target variables
        recon_x_means = torch.cat([recon_x_means_without_y, recon_x_mean_for_y], dim=1)
        recon_x_logvars = torch.cat([recon_x_logvars_without_y, recon_x_logvar_for_y], dim=1)
        return recon_x_means, recon_x_logvars  # Each of shape (batch_size, total_processed_dim)

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
        logger = logging.getLogger()
        logger.info("Training marginal VAEs.")
        data, mask = dataset.train_data_and_mask
        val_data, val_mask = dataset.val_data_and_mask
        # TODO: account for SparseDataset below
        # TODO: add validation data below
        dataset_without_ys = Dataset(data[:, : -self._y_dim], mask[:, : -self._y_dim])
        marginal_vaes_results = self._marginal_vaes._train(
            dataset_without_ys,
            train_output_dir=train_output_dir,
            report_progress_callback=report_progress_callback,
            learning_rate=learning_rate,
            batch_size=batch_size,
            iterations=iterations,
            epochs=epochs,
        )
        var_dir = os.path.join(train_output_dir, "var_id_%s" % self._y_variable.name)
        os.makedirs(var_dir, exist_ok=True)
        logger.info("Training Predictive VAE for variable %s" % self._y_variable.name)
        predictive_vae_results = self._y_vae._train(
            dataset, var_dir, report_progress_callback, learning_rate, batch_size, iterations, epochs,
        )

        # Combine both training results, and propagate back
        marginal_vaes_results = {f"marginal_vae.{key}": val for key, val in marginal_vaes_results.items()}
        predictive_vae_results = {f"predictive_vae.{key}": val for key, val in predictive_vae_results.items()}
        return {**marginal_vaes_results, **predictive_vae_results}

    # TODO: rethink what to do with this method
    def save_onnx(self, save_dir: str) -> None:
        raise ONNXNotImplemented

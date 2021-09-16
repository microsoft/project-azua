# This is required in python 3 to allow return types of the same class.
from __future__ import annotations

import logging
from ..models.marginal_vaes_with_predictive_vae import MarginalVAEsWithPredictiveVAE
import os

import torch
import numpy as np

from ..datasets.variables import Variables
from ..datasets.dataset import Dataset, SparseDataset
from ..models.vae_mixed import VAEMixed
from ..utils.data_mask_utils import to_tensors
from ..utils.training_objectives import get_input_and_scoring_masks
from typing import Optional, Callable, Union


class PredictiveVAEMixed(VAEMixed):
    """
    Predictive VAEM.

    A predictive VAEM is just a VAEM, except that the marginal VAE for the target dimension (y) is replaced by a predictive VAE.

    Also the training procedure is different.

    Note - this is not an instance of torch.nn.Module, but contains instances of that.
    """

    def __init__(self, model_id: str, variables: Variables, save_dir: str, device: torch.device, **model_config_dict):
        # Currently, assumes the only last variable is a target
        assert all(var.query for var in variables[:-1]), "All non-last variables should be querable"
        assert variables[-1].query is False, "Last variable is a target"

        super().__init__(model_id, variables, save_dir, device, **model_config_dict)

        # We may remove it, once we move finetune training over to child class (MarginaLVAEsWith..)
        self._y_variable = variables[-1]

    # CLASS METHODS
    @classmethod
    def name(cls) -> str:
        return "vaem_predictive"

    def _create_marginal_vaes(self, variables: Variables, all_vaes_dir: str, device: torch.device, **vae_config):
        return MarginalVAEsWithPredictiveVAE(
            model_id=self._marginal_network_save_dir,
            variables=variables,
            save_dir=all_vaes_dir,
            device=device,
            **vae_config,
        )

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
        rewind_to_best_epoch: bool,
        score_reconstruction: bool,
        score_imputation: bool,
        lr_warmup_epochs: int = 0,
    ):
        """
        Train the model using the given data.

        Details of the training procedures can be found in the OneNote page:

        https://microsofteur-my.sharepoint.com/personal/chezha_microsoft_com/_layouts/OneNote.aspx?id=%2Fpersonal%2Fchezha_microsoft_com%2FDocuments%2FNotebooks%2FMinDataAI_Master&wd=target%28Engineering.one%7C0186B790-A439-4F98-A829-382BF4D93B7D%2FVAEM%20%28with%20predictors%5C%29%7CC632CF50-A2E8-4804-92CD-11DFA2E8283D%2F%29


        Args:
            data (numpy array of shape (batch_size, feature_count)): Data to be used to train the model,
                in processed form.
            mask (numpy array of shape (batch_size, feature_count)): Corresponding mask, where observed
                values are 1 and unobserved values are 0.
            train_output_dir (str): Path to save any training information to, including tensorboard summary files.
            report_progress_callback: Function to report model progress for API.        # TODO: needs to be used.
            lr_warmup_epochs: number of epochs for learning rate warmup for the dependency network
        Returns:
            results_dict (dictionary): Train loss, KL divergence, and NLL for each epoch as a dictionary.
        """

        # Stage 1 and 2 of training (training marginal networks and dependency network)
        # are done in the parent class (VAEM)
        super()._train(
            dataset=dataset,
            train_output_dir=train_output_dir,
            report_progress_callback=report_progress_callback,
            dep_learning_rate=dep_learning_rate,
            dep_batch_size=dep_batch_size,
            dep_iterations=dep_iterations,
            dep_epochs=dep_epochs,
            marginal_learning_rate=marginal_learning_rate,
            marginal_batch_size=marginal_batch_size,
            marginal_iterations=marginal_iterations,
            marginal_epochs=marginal_epochs,
            max_p_train_dropout=max_p_train_dropout,
            score_reconstruction=score_reconstruction,
            score_imputation=score_imputation,
            rewind_to_best_epoch=rewind_to_best_epoch,
            lr_warmup_epochs=lr_warmup_epochs,
        )

        logger = logging.getLogger()
        data, mask = dataset.train_data_and_mask

        # Stage 3 Finetune predictive VAE using generated (imputed) data to boost the AL performance of the first few steps
        logger.info("Retraining predictive VAE.")

        # manually create missingness (at the same time, forcing the y dimension to be unobserved)

        # TODO update
        y_dim = self._y_variable.processed_dim
        mask_without_y = np.copy(mask)
        mask_without_y[:, -y_dim:] = 0
        input_mask, _ = get_input_and_scoring_masks(
            torch.as_tensor(mask_without_y, device=self._device, dtype=torch.float),
            max_p_train_dropout=max_p_train_dropout,
            score_imputation=True,
            score_reconstruction=True,
        )

        # ask the pretrained VAEM to impute data under such missingness

        data_tensor = to_tensors(data, device=self._device)[0]
        # TODO batching

        imputation_with_y = self.impute_processed_batch(
            data_tensor, input_mask, sample_count=1, vamp_prior_data=None, preserve_data=True
        )  # shape (1, batch_size, output_dim)
        # add the real data of y dimension to the imputed data, which gives the tensor imputation_with_y
        imputation_with_y[0, :, -y_dim:] = data_tensor[:, -y_dim:]
        imputation_with_y_array = imputation_with_y.detach().cpu().numpy()

        # finally, use the tensor imputation_with_y to finetune the predictive VAE for the y dimension
        # TODO: move over to MarginalVAEWIthPredictiveVAE.fine_tune_training
        ignored_mask = np.ones_like(imputation_with_y_array[0])  # mask is ignored
        imputation_with_y_dataset = Dataset(imputation_with_y_array[0], ignored_mask)
        var_dir = os.path.join(train_output_dir, "var_id_%s_finetune" % self._y_variable.name)
        self._marginal_vaes._y_vae._train(  # type: ignore
            imputation_with_y_dataset,
            var_dir,
            report_progress_callback,
            marginal_learning_rate,
            marginal_batch_size,
            marginal_iterations,
            marginal_epochs,
        )

        # # Stage 4 Retrain dependency network (Optional) (will be detailed on OneNote Page)
        # logger.info("Retraining dependency network.")
        # # Train by running data through encoder in marginals.
        # dep_network_input_data, dep_network_input_mask = self._get_dependency_network_input(data, mask)
        # # Then use that as data to train dep network.
        # dep_dir = os.path.join(train_output_dir, self._dependency_network_save_dir)
        # os.makedirs(dep_dir, exist_ok=True)
        # results_dict = self._dependency_network._train(
        #     dep_network_input_data, dep_network_input_mask, dep_dir, report_progress_callback,
        #     dep_learning_rate, dep_batch_size, dep_iterations, dep_epochs, max_p_train_dropout)
        self.save()

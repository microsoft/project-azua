# This is required in python 3 to allow return types of the same class.
from __future__ import annotations

import logging
import os
from tqdm import trange, tqdm  # type: ignore
from typing import Dict, List, Optional, Tuple, Union, Callable, Any

import numpy as np
import torch
from torch.nn import ReLU
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, BatchSampler, RandomSampler, SequentialSampler, Sampler
from scipy.sparse import issparse

from ..models.torch_model import TorchModel
from ..models.model import Model
from ..utils.torch_utils import (
    get_torch_device,
    generate_fully_connected,
)
from ..utils.io_utils import save_json
from ..datasets.variables import Variables
from ..datasets.dataset import Dataset
from ..experiment.azua_context import AzuaContext
from dependency_injector.wiring import Provide, inject


class DeepMatrixFactorization(TorchModel):
    """
    Deep Matrix Factorization (IJCAI 2017)
    https://www.ijcai.org/Proceedings/2017/0447.pdf
    Hong-Jian Xue, Xin-Yu Dai, Jianbing Zhang, Shujian Huang, Jiajun Chen
    """

    __model_config_path = "model_config.json"
    __model_type_path = "model_type.txt"
    __variables_path = "variables.json"

    __model_file = "model.pt"

    def __init__(
        self,
        model_id: str,
        variables: Variables,
        save_dir: str,
        device: torch.device,
        input_dim_user_i: int,
        input_dim_item_j: int,
        layers_user_i: List[int],
        layers_item_j: List[int],
        output_dim: int,
    ) -> None:
        """
        Args:
            model_id (str): Unique model ID for referencing this model instance.
            variables (Variables): Information about variables/features used
                by this model.
            save_dir (str): Location to save any information about this model, including training data.
                be created if it doesn't exist.
            device (str or int): Name of Torch device to create the model on. Valid options are 'cpu', 'gpu', or a device ID
                (e.g. 0 or 1 on a two-GPU machine).
            input_dim_user_i, input_dim_item_j (int): column and row sizes of the input matrix.
            layers_user_i, layers_item_j (list): hidden layer sizes for the two FNNs that respectively take y_i, y_j as inputs.
            output_dim (int): dimensionality of the latent representations p_i and q_j. Note that dim(p_i) == dim(q_j).
        """
        self._device = device

        # TODO: infer automatically during dataloading
        self.input_dim_user_i = input_dim_user_i
        self.input_dim_item_j = input_dim_item_j
        Model.__init__(self, model_id, variables, save_dir)
        torch.nn.Module.__init__(self)

        self._user_i_fnn = generate_fully_connected(
            input_dim_user_i, output_dim, layers_user_i, ReLU, activation=None, device=device,
        )

        self._item_j_fnn = generate_fully_connected(
            input_dim_item_j, output_dim, layers_item_j, ReLU, activation=None, device=device,
        )

        self._cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    # CLASS METHODS #
    @classmethod
    def name(cls) -> str:
        return "deep_matrix_factorization"

    def forward(self, input_user_i: torch.Tensor, input_item_j: torch.Tensor, mu: float = 1e-6) -> torch.Tensor:
        """
        Forward pass of deep matrix factorization.

        Args:
            input_user_i (torch tensor of shape (batch_size, feature_count)): Data to be used for the 
                forward pass. Corresponds to the rows of the input matrix.

            input_item_j (torch tensor of shape (batch_size, item_count)): Data to be used for the 
                forward pass. Corresponds to the columns of the input matrix.

            mu: A small value added to make the cosine similarity non-negative.
        """
        user_i_representation = self._user_i_fnn(input_user_i)
        item_j_representation = self._item_j_fnn(input_item_j)

        cos = self._cosine_similarity(user_i_representation, item_j_representation)

        # Make sure that the return value is non-negative.
        cos[cos < mu] = mu
        return cos

    def run_train(
        self,
        dataset: Dataset,
        train_config_dict: Dict[str, Any] = {},
        report_progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> None:
        """
        Train the model.
        Training results will be saved.

        Args:
            dataset: Dataset object with data and masks in unprocessed form.
            train_config_dict (dictionary): Any other parameters needed by a specific concrete class. Of
                the form {arg_name: arg_value}. e.g. {"learning_rate": 1e-3, "epochs": 100}
            report_progress_callback: Function to report model progress for API.
        """
        train_output_dir = self._create_train_output_dir_and_save_config(train_config_dict)
        self.train_output_dir = train_output_dir

        train_config_save_path = os.path.join(train_output_dir, "train_config.json")
        save_json(train_config_dict, train_config_save_path)

        # Run the training.
        train_results = self._train(dataset, train_output_dir, report_progress_callback, **train_config_dict,)

        # Save train results.
        if train_results is not None:
            train_results_save_path = os.path.join(self.save_dir, "training_results_dict.json")
            save_json(train_results, train_results_save_path)

        # Reload best saved model into this class.
        self = self.load(self.model_id, self.save_dir, self._device)  # type: ignore

    @inject
    def _train(
        self,  # type: ignore[override]
        dataset: Dataset,
        train_output_dir: str,
        report_progress_callback: Optional[Callable[[str, int, int], None]],
        learning_rate: float,
        batch_size: int,
        iterations: int,
        epochs: int,
        loss_function: str = "BCE",
        missing_fill_val: float = 0.5,
        azua_context: AzuaContext = Provide[AzuaContext],
    ) -> Dict[str, List[float]]:
        """
        Train the model using the given data.

        Args:
            dataset: Dataset object with data and masks in unprocessed form.
            train_output_dir (str): Path to save any training information to, including tensorboard summary files.
            report_progress_callback: Function to report model progress for API.
            learning_rate (float): Learning rate for Adam optimiser.
            batch_size (int): Size of minibatches to use.
            iterations (int): Iterations to train for. -1 is all iterations per epoch.
            epochs (int): Number of epochs to train for.
            loss_function (str): Loss function for training. Currently, only MSE and BCE are supported.
            missing_fill_val (float): TODO In current binary train_data, missing values and negative labels are both set to be 0.
                Add argument missing_fill_val, which is in default set to be 0.5 to differentiate this.
        Returns:
            train_results (dictionary): Train loss for each epoch as a dictionary.
        """
        writer = SummaryWriter(os.path.join(train_output_dir, "summary"), flush_secs=1)
        logger = logging.getLogger()
        metrics_logger = azua_context.metrics_logger()

        # Put PyTorch into train mode.
        self.train()

        assert loss_function in ["BCE", "MSE"], NotImplementedError(
            "Only BCE and MSE are supported as loss functions for deep matrix factorization. "
        )

        # Process the train data.
        data, mask = self.data_processor.process_data_and_masks(*dataset.train_data_and_mask)
        data_test, mask_test = self.data_processor.process_data_and_masks(*dataset.test_data_and_mask)
        data_val, mask_val = self.data_processor.process_data_and_masks(*dataset.val_data_and_mask)

        if issparse(data):
            data_, mask_ = data.todense(), mask.todense()
            data_test_, mask_test_ = data_test.todense(), mask_test.todense()
            data_val_, mask_val_ = data_val.todense(), mask_val.todense()

        # TODO: reuse the logic for resizing elementwise data splits in SparseCSVDatasetLoader
        # TODO: Do unit test for this
        def change_dim_data_and_mask(data_, mask_, idxs):
            data = np.zeros((self.input_dim_item_j, self.input_dim_user_i))
            mask = np.zeros((self.input_dim_item_j, self.input_dim_user_i))
            data[dataset.data_split[idxs]] = data_
            mask[dataset.data_split[idxs]] = mask_
            return data, mask

        data, mask = change_dim_data_and_mask(data_, mask_, "train_idxs")

        data_test, mask_test = change_dim_data_and_mask(data_test_, mask_test_, "test_idxs")
        data_val, mask_val = change_dim_data_and_mask(data_val_, mask_val_, "val_idxs")

        results_dict: Dict[str, List] = {"training_loss": [], "training_acc": []}

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        best_val_acc = np.nan
        is_quiet = logger.level > logging.INFO
        for epoch in trange(epochs, desc="Epochs", disable=is_quiet):
            training_loss_full = 0.0
            mask_sum = 0.0
            correct_sum = 0.0

            dataloader = self._create_index_dataloader_for_dmf(
                mask, batch_size=batch_size, iterations=iterations, sample_randomly=True,
            )

            for indices_i, indices_j in tqdm(dataloader, desc="Batches", disable=is_quiet):
                batch_size_ = len(indices_i)
                mask_sum += batch_size_

                total_loss, y, output = self._get_loss(data, indices_i, indices_j, loss_function, True)

                optimizer.zero_grad()

                y_hat = np.zeros(y.shape)
                y_hat[np.where(output > 0.5)] = 1.0
                y_hat[np.where(output < 0.5)] = 0.0
                current_sum = sum(y_hat == y.detach().cpu().numpy())
                correct_sum += current_sum
                loss = total_loss / batch_size_
                loss.backward()
                optimizer.step()

                training_loss_full += total_loss.item()

            # average loss over most recent epoch
            training_loss_avg = training_loss_full / mask_sum
            training_acc = correct_sum / mask_sum

            dataloader_val = self._create_index_dataloader_for_dmf(
                mask_val, batch_size=len(mask_val), iterations=-1, sample_randomly=False,
            )

            loss_val, y_val, output_val = self._get_loss(data_val, *next(iter(dataloader_val)), loss_function, False)  # type: ignore
            if output_val.is_cuda:
                output_val = output_val.cpu().data.numpy()
            else:
                output_val = output_val.data.numpy()
            y_hat_val = np.zeros(y_val.shape)
            y_hat_val[np.where(output_val > 0.5)] = 1.0
            y_hat_val[np.where(output_val < 0.5)] = 0.0
            val_acc = sum(y_hat_val == y_val.detach().cpu().numpy()) / len(y_hat_val)

            # TODO: replace this with train_model() in torch_training.py
            if np.isnan(best_val_acc) or val_acc > best_val_acc:
                best_val_acc = val_acc
                best_training_acc = training_acc
                best_epoch = epoch

                dataloader_test = self._create_index_dataloader_for_dmf(
                    mask_test, batch_size=len(mask_test), iterations=-1, sample_randomly=False,
                )

                loss_test, y_test, output_test = self._get_loss(
                    data_test, *next(iter(dataloader_test)), loss_function, False  # type: ignore
                )

                y_hat_test = np.zeros(y_test.shape)
                y_hat_test[np.where(output_test > 0.5)] = 1.0
                y_hat_test[np.where(output_test < 0.5)] = 0.0
                test_acc = sum(y_hat_test == y_test.detach().cpu().numpy()) / len(y_hat_test)

                # Save model.
                self.save()

            # Save useful quantities.
            writer.add_scalar("train/loss-train", training_loss_avg, epoch)
            writer.add_scalar("train/train-acc", training_acc, epoch)
            results_dict["training_loss"].append(training_loss_avg)
            results_dict["training_acc"].append(training_acc)
            if report_progress_callback:
                report_progress_callback(self.model_id, epoch + 1, epochs)

            if np.isnan(training_loss_full):
                logger.info("Training loss is NaN. Exiting early.")
                break

        metrics_logger.log_dict({"test_data.all.Accuracy": test_acc})
        logger.info(
            "Best model found at epoch %d, with val_acc %.4f, test_acc %.4f, training_acc %.4f"
            % (best_epoch, best_val_acc, test_acc, best_training_acc)
        )
        writer.close()

        return results_dict

    def _get_loss(
        self, data: np.ndarray, indices_i: np.ndarray, indices_j: np.ndarray, loss_function: str, is_train: bool
    ):
        """
        Compute loss based on minibatch dataset.

        Args:
            data: Dense input data array.
            indices_i: Indices of the rowwise input for DMF.
            indices_j: Indices of the columnwise input for DMF.
            loss_function: Type of lossfunction chosen from (MSE, BCE). 
            is_train: boolean that tells whether current computation of loss is for training for not (inference).
        """
        with torch.set_grad_enabled(is_train):
            if not is_train:
                self.eval()

            input_user_i = torch.as_tensor(data[indices_i, :], dtype=torch.float, device=self._device)
            input_item_j = torch.as_tensor(data[:, indices_j].T, dtype=torch.float, device=self._device)
            y = torch.as_tensor(data[indices_i, indices_j], dtype=torch.float, device=self._device).flatten()

            output = self(input_user_i, input_item_j)
            if loss_function == "BCE":
                total_loss = F.binary_cross_entropy(output, y, reduction="mean")
            else:
                total_loss = F.mse_loss(output, y, reduction="mean")

        return total_loss, y, output

    # Iteratively find non-empty indices using masks and return in return it as an insance of Dataloader.
    def _create_index_dataloader_for_dmf(
        self, mask: np.ndarray, batch_size: int, iterations: int = -1, sample_randomly: bool = True
    ) -> DataLoader:
        """
        Create dataset loader specifically for DMF.
        
        Args:
            mask (numpy array of shape (batch_size, feature_count)): Corresponding mask, where observed
                values are 1 and unobserved values are 0.
            bash_size: Batch size for training.
            iteration: Number of interations for a single epoch. When set to -1, every datapoints are used
                for each epoch.
            sample_randomly: Boolean that decides whether to sample datapoints randomly or not.
                If set to False, when sampling is done sequentially. 
        """

        def to_tensors(
            data_i: np.ndarray, data_j: np.ndarray, *, device: Optional[Union[str, int, torch.device]] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            device = self._device if device is None else get_torch_device(device)
            return (
                torch.as_tensor(data_i, dtype=torch.int),
                torch.as_tensor(data_j, dtype=torch.int),
            )

        dataset = TensorDataset(*to_tensors(*np.where(mask)))

        row_count = len(dataset)
        max_iterations = np.ceil(row_count / batch_size)
        if iterations > max_iterations:
            iterations = -1

        if sample_randomly:
            if iterations == -1:
                sampler: Sampler = RandomSampler(dataset)
            else:
                sampler = RandomSampler(dataset, replacement=True, num_samples=iterations * batch_size)
        else:
            sampler = SequentialSampler(dataset)

        batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
        index_dataloader = DataLoader(dataset, batch_sampler=batch_sampler)
        return index_dataloader

    def impute(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        impute_config_dict: Optional[Dict[str, int]] = None,
        vamp_prior_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        average: bool = True,
        batch_size: int = 100000,
    ) -> np.ndarray:
        """
        Fill in unobserved variables using a trained model.
        Data should be provided in unprocessed form, and will be processed before running, and
        will be de-processed before returning (i.e. variables will be in their normal, rather than
        squashed, ranges).

        Args:
            data (numpy array of shape (batch_size, feature_count)): Data to be used to train the model,
                in unprocessed form.
            mask (numpy array of shape (batch_size, feature_count)): Corresponding mask, where observed
                values are 1 and unobserved values are 0.
            impute_config_dict (dictionary): Any other parameters needed by a specific concrete class. Of
                the form {arg_name: arg_value}. e.g. {"sample_count": 10}
            vamp_prior_data (Tuple of (data, mask)): Data to be used to fill variables if using the vamp
                prior method. This defaults to None, in which case the vamp prior method will not be used.
            batch_size: Batch size for loading the data. 

        Returns:
            imputed (numpy array of shape (batch_size, feature_count)): Input data with missing values filled in.
        """
        self.eval()
        return_array = np.zeros(mask.shape)

        index_dataloader = self._create_index_dataloader_for_dmf(
            mask, batch_size=batch_size, iterations=-1, sample_randomly=True
        )
        for indices_i, indices_j in tqdm(index_dataloader, desc="Batches", disable=True):
            input_user_i = torch.as_tensor(data[indices_i, :], dtype=torch.float, device=self._device)
            input_item_j = torch.as_tensor(data[:, indices_j].T, dtype=torch.float, device=self._device)
            with torch.no_grad():
                output = self(input_user_i, input_item_j)
            return_array[indices_i, indices_j] = output.numpy()

        return return_array

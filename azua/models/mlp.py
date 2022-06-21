# This is required in python 3 to allow return types of the same class.
from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from torch.nn import ReLU
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange  # type: ignore

from ..datasets.dataset import Dataset
from ..datasets.variables import Variables
from ..models.torch_model import TorchModel
from ..utils.io_utils import save_json
from ..utils.torch_utils import CrossEntropyLossWithConvert, create_dataloader, generate_fully_connected


class MLP(TorchModel):
    def __init__(
        self,
        model_id: str,
        variables: Variables,
        save_dir: str,
        device: torch.device,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
    ) -> None:
        # TODO ensure input_dim, output_dim are compatible with variables
        super().__init__(model_id, variables, save_dir, device)

        # This defaults to a ReLU network
        self.__network = MLPNetwork(input_dim, hidden_dims, output_dim, activation=None, device=device)

        # Setup our target variable type (only supports target variable of 1 length)
        self._set_network_targets()

    def _set_network_targets(self):
        # checks the target_id, sets up the appropriate activation and loss functions
        target_var_idxs = self.variables.target_var_idxs
        assert len(target_var_idxs) == 1, "Must have one target variable"
        self._target_type = self.variables[target_var_idxs[0]].type
        (
            self._loss_function,
            self._activation_function,
        ) = self._get_loss_and_activation_function()

    def _get_loss_and_activation_function(self):
        if self._target_type == "categorical":
            return CrossEntropyLossWithConvert(), torch.nn.Softmax(dim=1)
        elif self._target_type == "binary":
            return torch.nn.BCEWithLogitsLoss(), torch.nn.Sigmoid()
        elif self._target_type == "continuous":
            return torch.nn.MSELoss(), torch.nn.Identity()
        else:
            raise ValueError(
                "Invalid feature type. Expected continuous, categorical or binary, got %s" % self._target_type
            )

    @classmethod
    def name(cls) -> str:
        return "mlp"

    def save_onnx(self, save_dir):
        self.__network.save_onnx(save_dir)

    def run_train(
        self,
        dataset: Dataset,
        train_config_dict: Dict[str, Any] = {},
        report_progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> None:

        train_output_dir = self._create_train_output_dir_and_save_config(train_config_dict)

        # Run the training.
        # TODO use validation data
        data, _ = dataset.train_data_and_mask
        train_results = self._train(data, train_output_dir, report_progress_callback, **train_config_dict)

        # Save train results.
        if train_results is not None:
            train_results_save_path = os.path.join(self.save_dir, "training_results_dict.json")
            save_json(train_results, train_results_save_path)

        # Reload best saved model into this class.
        self = self.load(self.model_id, self.save_dir, self._device)

    def _train(
        self,
        data: np.ndarray,
        train_output_dir: str,
        report_progress_callback: Optional[Callable[[str, int, int], None]],
        learning_rate: float,
        batch_size: int,
        epochs: int,
    ):

        self.train()
        writer = SummaryWriter(os.path.join(train_output_dir, "summary"), flush_secs=1)
        logger = logging.getLogger()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        dataloader = create_dataloader(data, batch_size=batch_size)
        results_dict: Dict[str, List] = {metric: [] for metric in ["training_loss"]}

        best_train_loss = np.nan

        for epoch in trange(epochs, desc="Epochs"):
            training_loss = 0.0

            for data in tqdm(dataloader, desc="Batches"):
                data = data[0]
                data = data.to(self._device)
                x, y = data[:, :-1], data[:, -1]
                batch_size, _ = x.shape

                loss = self._loss(x, y)

                # Needed for pruning gradients in PackNet for instance
                self._optimize_step(optimizer, loss)

                training_loss += loss.item()

            # average loss over most recent epoch
            training_loss_avg = training_loss / float(x.numel())

            if np.isnan(best_train_loss) or training_loss_avg < best_train_loss:
                best_train_loss = training_loss_avg
                best_epoch = epoch
                results_dict["training_loss"].append(training_loss_avg)
                # Save model.
                model_path = os.path.join(self.save_dir, self._model_file)
                torch.save(self.state_dict(), model_path)

            # Save useful quantities.
            writer.add_scalar("train/loss-train", training_loss_avg, epoch)

        logger.info("Best model found at epoch %d, with train_loss %.4f" % (best_epoch, best_train_loss))
        writer.close()
        self.eval()

        return results_dict

    def _optimize_step(self, optimizer, loss):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def _loss(self, x: torch.Tensor, y: torch.Tensor):
        # Uses numerically stable losses, which is why we don't invoke forward
        y_pred = self.__network(x)
        return self._loss_function(y_pred, y)

    def predict(self, x_array: np.ndarray):
        # Prediction using numpy arrays in and out
        # Outputs actual classes for non-continuous variables
        with torch.no_grad():
            x: torch.Tensor = torch.FloatTensor(x).to(self._device)
            x = self(x)
            if self._target_type == "categorical":
                x = x.argmax(1)
            elif self._target_type == "binary":
                x = x.round()
        return x.cpu().detach().numpy()

    def forward(self, x: torch.Tensor):
        output = self.__network(x)
        return self._activation_function(output)


class MLPNetwork(torch.nn.Module):
    """
    Generic Feed Forward MLP network
    """

    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        device,
        non_linearity=ReLU,
        activation=None,
    ):
        """
        Args:
            input_dim: Dimension of observed features.
            hidden_dims: List of int. Sizes of internal hidden layers. i.e. [a, b] is three linear layers with shapes (input_dim, a), (a, b), (b, output_dim)
            output_dims: Dimension of output dimension
            non_linearity: Non linear activation function used between Linear layers. Defaults to ReLU.
            activation: Final layer activation function
            device: torch device to use.
        """
        super().__init__()
        self.__input_dim = input_dim

        self._forward_sequence = generate_fully_connected(
            input_dim, output_dim, hidden_dims, non_linearity, activation, device
        )

        self._device = device

    def forward(self, x):
        """
        Args:
            x: Input tensor with shape (batch_size, input_dim).

        Returns:
            output: Output tensor with shape (batch_size, output_dim)
        """
        output = self._forward_sequence(x)
        return output

    def save_onnx(self, save_dir):
        dummy_input = torch.rand(1, self.__input_dim, dtype=torch.float, device=self._device)

        path = os.path.join(save_dir, "mlp.onnx")
        torch.onnx.export(self, dummy_input, path)

    def replace_network(self, new_network):
        self._forward_sequence = new_network

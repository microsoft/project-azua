# This is required in python 3 to allow return types of the same class.
from __future__ import annotations

import os
from typing import Any, Dict, Optional, List, Callable, Tuple, Union
from time import gmtime, strftime

from tqdm import trange, tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from ..models.model import Model
from ..datasets.variables import Variables
from ..datasets.dataset import Dataset
from ..utils.torch_utils import get_torch_device, set_random_seeds
from ..utils.io_utils import read_json_as, save_json, save_txt
from ..utils.data_transform import transform_from_configs, TransformableFirstTensorDataset

from .bnn import bayesianize_
from .bnn.nn.nets import make_network
from .bnn.nn.mixins.base import BayesianMixin


# TODO: inherit from TorchModel,so we can use code for basic operations (save/load/create)
class BNN(Model):

    model_file = "model.pt"
    onnx_file = "net.onnx"
    model_config_file = "model_config.json"
    model_type_file = "model_type.txt"
    variables_file = "variables.json"

    def __init__(
        self,
        model_id: str,
        variables: Variables,
        save_dir: str,
        net: nn.Module,
        ensemble_size: int,
        device: torch.device,
        input_shape: Union[int, Tuple[int, ...]],
    ) -> None:
        super().__init__(model_id, variables, save_dir)

        self.net = net.to(device)
        self.ensemble_size = ensemble_size
        self.device = device
        self.input_shape = (input_shape,) if isinstance(input_shape, int) else input_shape

        target_var_idxs = self.variables.target_var_idxs
        if len(target_var_idxs) > 1:
            raise ValueError("Got variables with more than on target.")
        # TODO this likely isn't the case anymore?
        # id enumeration seems to start from 1, subtracting 1 to match 0 based indexing
        self.target_col = target_var_idxs[0] - 1
        self.target_type = self.variables[self.target_col + 1].type

    @classmethod
    def name(cls) -> str:
        return "bnn"

    @classmethod
    def create(
        cls,
        model_id: str,
        save_dir: str,
        variables: Variables,
        model_config_dict: Dict[str, Any],
        device: Union[str, int],
    ) -> BNN:
        model = cls._create(model_id, variables, save_dir, device=device, **model_config_dict)

        # Save all the model information.
        # Save model config to save dir.
        model_config_save_path = os.path.join(save_dir, cls.model_config_file)
        save_json(model_config_dict, model_config_save_path)

        # Save variables file.
        variables_path = os.path.join(save_dir, cls.variables_file)
        variables.save(variables_path)

        # Save model type.
        model_type_path = os.path.join(save_dir, cls.model_type_file)
        save_txt(cls.name(), model_type_path)

        # Save the model that has just been created.
        model.save()
        return model

    @classmethod
    def _create(
        cls,
        model_id: str,
        variables: Variables,
        save_dir: str,
        device: Union[str, int],
        network_config: dict,
        seed: int,
        ensemble_size: int,
        input_shape: Union[int, Tuple[int, ...]],
        inference_config: dict = None,
        pretrained_path: Optional[str] = None,
    ):
        # GPU and set random seed
        set_random_seeds(seed)
        torch_device = get_torch_device(device)

        net = make_network(**network_config)
        reference_sd = torch.load(pretrained_path, map_location="cpu") if pretrained_path is not None else None
        if inference_config is not None:
            bayesianize_(net, reference_state_dict=reference_sd, **inference_config)
        return cls(model_id, variables, save_dir, net, ensemble_size, device=torch_device, input_shape=input_shape,)

    @classmethod
    def load(cls, model_id: str, save_dir: str, device: Union[str, int]) -> BNN:
        # Load variables.
        variables = Variables.create_from_json(os.path.join(save_dir, cls.variables_file))

        # Load model_config_fict
        model_config_path = os.path.join(save_dir, cls.model_config_file)
        model_config_dict = read_json_as(model_config_path, dict)

        return cls._load(model_id, variables, save_dir, device, model_config_dict)

    @classmethod
    def _load(
        cls,
        model_id: str,
        variables: Variables,
        save_dir: str,
        device: Union[str, int],
        model_config_dict: Dict[str, Any],
    ) -> BNN:
        model = cls.create(model_id, save_dir, variables, model_config_dict, device)
        model_path = os.path.join(save_dir, cls.model_file)
        model.net.load_state_dict(torch.load(model_path))
        return model

    def save(self) -> None:
        # Save variables
        self.variables.save(os.path.join(self.save_dir, self.variables_file))

        # Save model
        model_path = os.path.join(self.save_dir, self.model_file)
        torch.save(self.net.state_dict(), model_path)

        # disabling onnx saving since linalg ops like cholesky that are used in inducing weights are not supported
        # onnx_path = os.path.join(self.save_dir, self.onnx_file)
        # with maintain_random_state():
        #     dummy_input = torch.randn(1, *self.input_shape)
        #     torch.onnx.export(self.net.to("cpu"), dummy_input, onnx_path)
        # self.net.to(self.device)

    def run_train(
        self,
        dataset: Dataset,
        train_config_dict: Dict[str, Any] = {},
        report_progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> None:
        # Create directory and save config to that folder.
        # TODO use TorchModel method
        starttime = strftime("%Y-%m-%d_%H%M%S", gmtime())
        train_output_dir = os.path.join(self.save_dir, "train_%s" % starttime)
        os.makedirs(train_output_dir, exist_ok=True)

        train_config_save_path = os.path.join(train_output_dir, "train_config.json")
        save_json(train_config_dict, train_config_save_path)
        # Run the training.
        # TODO use validation data
        data, _ = dataset.train_data_and_mask
        train_results = self._train(data, train_output_dir, report_progress_callback, **train_config_dict)

        # Save train results.
        if train_results is not None:
            train_results_save_path = os.path.join(self.save_dir, "training_results_dict.json")
            save_json(train_results, train_results_save_path)

    def _train(
        self,
        data: np.ndarray,
        train_output_dir: str,
        report_progress_callback: Optional[Callable[[str, int, int], None]] = None,
        lr: float = 0.01,
        batch_size: int = 100,
        epochs: int = 100,
        ml_epochs: int = 0,
        annealing_epochs: int = 0,
        max_kl_factor: float = 1.0,
        num_workers: int = 0,
        transforms: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, List]:
        self.net.train()
        writer = SummaryWriter(os.path.join(train_output_dir, "summary"), flush_secs=1)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        inputs, targets = self._split_inputs_and_targets_and_reshape(data, self.target_col)
        transform = transform_from_configs(transforms) if transforms is not None else None
        dataset = TransformableFirstTensorDataset(
            torch.from_numpy(inputs).float(), torch.from_numpy(targets).long(), transform=transform,
        )
        dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers)
        results_dict: Dict[str, List] = {metric: [] for metric in ["data_loss", "param_loss", "loss"]}

        kl_factor = 0.0 if ml_epochs > 0 or annealing_epochs > 0 else 1.0
        annealing_rate = max_kl_factor * annealing_epochs ** -1 if annealing_epochs > 0 else 1.0
        for epoch in trange(epochs, desc="Epochs"):
            training_loss = 0.0
            n = 0

            for x, y in tqdm(dataloader, desc="Batches", leave=False):
                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                data_loss = self._data_loss(x, y) * len(data)
                param_loss = sum(
                    (module.parameter_loss() for module in self.net.modules() if isinstance(module, BayesianMixin)),
                    start=torch.tensor(0.0, device=self.device),
                )
                loss = data_loss + kl_factor * param_loss

                loss.backward()
                optimizer.step()

                results_dict["data_loss"].append(data_loss.item())
                results_dict["param_loss"].append(param_loss.item())
                results_dict["loss"].append(loss.item())

                training_loss += x.shape[0] * loss.item()
                n += x.shape[0]

            # Save useful quantities.
            writer.add_scalar("train/loss-train", training_loss / n, epoch)

            if epoch >= ml_epochs:
                kl_factor = min(max_kl_factor, kl_factor + annealing_rate)

        writer.close()

        return results_dict

    def _split_inputs_and_targets_and_reshape(self, data: np.ndarray, target_col: int) -> Tuple[np.ndarray, np.ndarray]:
        inputs, targets = Model._split_inputs_and_targets(data, target_col)
        return inputs.reshape(-1, *self.input_shape), targets

    def predict(self, x: np.ndarray):
        self.net.eval()
        x_tensor = torch.from_numpy(x).float().to(self.device)
        predictions = self.net(x_tensor)
        if self.target_type == "categorical":
            output = predictions.softmax(-1)
        elif self.target_type == "binary":
            output = predictions.sigmoid()
        elif self.target_type == "continuous":
            output = predictions
        else:
            self._raise_target_type_error()
        return output.detach().cpu().numpy()

    def _data_loss(self, x: torch.Tensor, y: torch.Tensor):
        yhat = self.net(x)
        if self.target_type == "categorical":
            return F.cross_entropy(yhat, y.long())
        elif self.target_type == "binary":
            return F.binary_cross_entropy_with_logits(yhat, y.long())
        elif self.target_type == "continuous":
            return F.mse_loss(yhat, y)
        else:
            self._raise_target_type_error()

    def impute(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        impute_config_dict: Dict[str, int] = None,
        vamp_prior_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        average: bool = True,
    ) -> np.ndarray:
        if ((mask == 0).sum(1) > 1).any():
            raise ValueError("Only target variables are allowed to be missing.")
        if (mask[:, self.target_col] == 0).sum() != len(mask):
            raise ValueError("All target values must be missing.")

        inputs, _ = self._split_inputs_and_targets_and_reshape(data, self.target_col)
        predictions = np.stack([self.predict(inputs) for _ in range(self.ensemble_size)]).mean(0)
        if self.target_type == "categorical":
            outputs = predictions.argmax(-1)
        elif self.target_type == "binary":
            outputs = (predictions > 0.5)[..., 0]
        elif self.target_type == "continuous":
            outputs = predictions[..., 0]
        else:
            self._raise_target_type_error()

        data[:, self.target_col] = np.cast[data.dtype](outputs)
        return data

    def _raise_target_type_error(self):
        raise RuntimeError(
            f"Unrecognized target type: '{self.target_type}'. " f"Expected one of (categorical, binary, continuous)."
        )

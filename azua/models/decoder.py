import warnings
from typing import Type

import numpy as np
import torch
from torch.jit import TracerWarning
from torch.nn import ReLU, Sigmoid, Softmax

from ..datasets.variables import Variables
from ..models.torch_model import ONNXNotImplemented
from ..utils.torch_utils import generate_fully_connected

warnings.filterwarnings("ignore", category=TracerWarning)


class Decoder(torch.nn.Module):
    """
    Fully connected decoder.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        variables,
        hidden_dims,
        variance: float,
        device,
        variance_autotune=False,
        non_linearity=ReLU,
        activation_for_continuous=Sigmoid,
        init_method="default",
    ):
        """
        Args:
            input_dim: int representing input dimension, e.g. the dimension of the latent space.
            output_dim: int representing output dimenion.
            variables: Variables used in the model.
            hidden_dims: List of int. Sizes of internal hidden layers. [a, b] is three linear layers with shapes (input_dim, a), (a, b), (b, output_dim)
            variance: Output variance to use.
            device: torch device to use.
            variance_autotune: automatically tune variance or not.
            non_linearity: Non linear activation function used between Linear layers. Defaults to ReLU.
            activation_for_continuous: activation function for continuous variable outputs. Defaults to Sigmoid.
            init_method (str): initialization method
        """
        super().__init__()

        self._device = device

        self.__input_dim = input_dim
        self.__variables = variables

        self.__logvar = np.log(variance)
        self.__variance_autotune = variance_autotune

        self._forward_sequence = generate_fully_connected(
            input_dim, output_dim, hidden_dims, non_linearity, activation=None, device=device, init_method=init_method
        )

        self.__trainable_logvar = Logvar(output_dim, self.__logvar, device)

        self._device = device

        self._featurewise_activation = FeaturewiseActivation(
            activation_for_continuous=activation_for_continuous, variables=variables
        )

    def forward(self, latent_sample):
        """
        Forward pass of decoder.

        Args:
            latent_sample: input to the decoder, sample from latent space with shape (batch_size, dim_in)

        Returns:
            means: shape (batch_size, output_dimension)
            log_variances: shape (batch_size, output_dimension)
        """
        mean = self._forward_sequence(latent_sample)  # Shape (batch_size, output_dim)

        mean = self._featurewise_activation(mean)
        if not self.__variance_autotune:
            logvar = torch.full_like(mean, fill_value=self.__logvar, device=self._device)
        else:
            logvar = self.__trainable_logvar(mean)

        return mean, logvar

    def save_onnx(self, save_dir):
        raise ONNXNotImplemented
        # dummy_input = (
        #     torch.rand(1, self.__input_dim, device=self._device)
        # )

        # path = os.path.join(save_dir, "decoder.onnx")
        # torch.onnx.export(self, dummy_input, path, opset_version=11)


class Logvar(torch.nn.Module):
    def __init__(self, output_dim, logvar_init, device):
        """
        This module creates learnable variances.

        Args:
            output_dim: dimensionality of output
            logvar_init: intitialization value of logvar
        """
        super(Logvar, self).__init__()
        self._device = device
        self.logvar = torch.nn.Parameter(torch.full((1, output_dim), logvar_init))

    def forward(self, x):
        logvar_expanded = torch.full_like(x, fill_value=0, device=self._device) + self.logvar.to(self._device)
        return logvar_expanded


class FeaturewiseActivation(torch.nn.Module):
    """
    Passes outputs for each feature through a different activation function depending if the feature is
    continuous, binary, or categorical.
    """

    def __init__(self, activation_for_continuous: Type, variables: Variables):
        super().__init__()
        self._variables = variables
        self._activation_for_continuous = activation_for_continuous()

    def _get_activation_func_for_variable(self, variable_type):
        if variable_type in ("continuous", "text"):
            return self._activation_for_continuous
        elif variable_type == "binary":
            return Sigmoid()
        elif variable_type.startswith("categorical"):
            return Softmax(dim=1)
        else:
            raise ValueError("Variable type must be one of 'continuous', 'text', 'binary' or 'categorical'.")

    def forward(self, data):
        """
        Runs data through activation functions based on type of feature.

        Args:
            data: Tensor with shape (count, batch_size, output_dim) or (batch_size, output_dim).
        Returns:
            data: Tensor with shape (count, batch_size, output_dim) or (batch_size, output_dim).
        """
        if data.ndim > 3 or (data.shape[-1] != self._variables.num_processed_non_aux_cols):
            raise ValueError

        all_activated_data = torch.zeros_like(data)

        for var_type, idxs in self._variables.processed_non_aux_cols_by_type.items():
            # For non categorical, we need to subset all instances at once
            for idx in idxs:
                activation = self._get_activation_func_for_variable(var_type)
                activated_data = activation(data[..., idx])
                all_activated_data[..., idx] = activated_data

        return all_activated_data

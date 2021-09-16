import os
from typing import List

import torch
from torch.nn import ReLU

from ..utils.torch_utils import generate_fully_connected


class Encoder(torch.nn.Module):
    """
    PointNet encoder
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        device: torch.device,
        non_linearity=ReLU,
        init_method="default",
    ):
        """
        Args:
            input_dim: Dimension of observed features.
            hidden_dims: List of int. Sizes of internal hidden layers. i.e. [a, b] is three linear layers with shapes (input_dim, a), (a, b), (b, output_dim)
            latent_dim: Dimension of output latent space.
            device: torch device to use.
            non_linearity: Non linear activation function used between Linear layers. Defaults to ReLU.
            init_method (str): initialization method
        """
        super().__init__()
        self._device = device
        self.__input_dim = input_dim

        self._forward_sequence = generate_fully_connected(
            input_dim,
            2 * latent_dim,
            hidden_dims,
            non_linearity,
            activation=None,
            device=device,
            init_method=init_method,
        )

        self._device = device

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input tensor with shape (batch_size, input_dim).

        Returns:
            encoded: Encoded output tensor with shape (batch_size, 2*latent_dim)
        """  # Run masked values through model.
        output = self._forward_sequence(x)  # Shape (batch_size, 2 * latent_dim)
        output = output.chunk(2, dim=1)
        return output

    def save_onnx(self, save_dir):
        dummy_input = torch.rand(1, self.__input_dim, dtype=torch.float, device=self._device)

        path = os.path.join(save_dir, "encoder.onnx")
        torch.onnx.export(self, dummy_input, path)

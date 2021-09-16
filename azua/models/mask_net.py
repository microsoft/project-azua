import os
import torch
from torch.nn import Sigmoid, Softplus


class MaskNet(torch.nn.Module):
    """
    This implements the Mask net that is used in notmiwae's implementation for self-masking mechanism
    """

    def __init__(self, input_dim: int, device):
        """
        Args:
            input_dim: Dimension of observed features.
            device: torch device to use.
        """
        super().__init__()
        self._device = device
        self.__input_dim = input_dim

        self.W = torch.nn.Parameter(torch.zeros([1, input_dim], device=device), requires_grad=True)
        self.b = torch.nn.Parameter(torch.zeros([1, input_dim], device=device), requires_grad=True)
        self._device = device

    def forward(self, x):
        """
        Args:
            x: Input tensor with shape (batch_size, input_dim).

        Returns:
            encoded: Encoded output tensor with shape (batch_size, input_dim)
        """  # Run masked values through model.
        output = Sigmoid()(-Softplus()(self.W) * (x - self.b))
        return output

    def save_onnx(self, save_dir):
        dummy_input = torch.rand(1, self.__input_dim, dtype=torch.float, device=self._device)

        path = os.path.join(save_dir, "mask_net.onnx")
        torch.onnx.export(self, dummy_input, path)

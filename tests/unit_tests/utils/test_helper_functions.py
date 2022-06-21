import numpy as np
import torch

from azua.utils.helper_functions import to_tensors
from azua.utils.torch_utils import get_torch_device

cpu_torch_device = get_torch_device("cpu")


def test_to_tensors_one_input():
    a = np.ones((3, 4))

    # _to_tensors() returns a tuple so we need to explicitly unpack a single value with a comma
    (tensor_a,) = to_tensors(a, device=cpu_torch_device)
    assert isinstance(tensor_a, torch.Tensor)

    assert a.shape == tensor_a.shape
    assert tensor_a.dtype == torch.float


def test_to_tensors_two_inputs():
    a = np.ones((3, 4))
    b = np.ones((3, 4))

    tensor_a, tensor_b = to_tensors(a, b, device=cpu_torch_device)
    assert isinstance(tensor_a, torch.Tensor)
    assert isinstance(tensor_b, torch.Tensor)

    assert a.shape == tensor_a.shape
    assert b.shape == tensor_b.shape

    assert tensor_a.dtype == torch.float
    assert tensor_b.dtype == torch.float


def test_to_tensors_specify_dtype():
    a = np.ones((3, 4))
    b = np.ones((3, 4))

    tensor_a, tensor_b = to_tensors(a, b, device=cpu_torch_device, dtype=torch.int)
    assert isinstance(tensor_a, torch.Tensor)
    assert isinstance(tensor_b, torch.Tensor)

    assert a.shape == tensor_a.shape
    assert b.shape == tensor_b.shape

    assert tensor_a.dtype == torch.int
    assert tensor_b.dtype == torch.int

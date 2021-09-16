import random
from typing import List, Optional, Type, Union, Tuple

import numpy as np
from scipy.sparse import issparse, csr_matrix
import torch
from torch.nn import Linear, Module, Sequential, Dropout
from torch.utils.data import (
    Dataset,
    DataLoader,
    TensorDataset,
    BatchSampler,
    RandomSampler,
    SequentialSampler,
    Sampler,
)
from ..utils.data_mask_utils import to_tensors


def set_random_seeds(seed):
    """
    Set random seeds for Torch, Numpy and Python, as well as Torch reproducibility settings.
    """
    if isinstance(seed, list) and len(seed) == 1:
        seed = seed[0]

    # PyTorch settings to ensure reproducibility - see https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_torch_device(device_id: Union[int, str, torch.device] = 0) -> torch.device:
    """
    Get a torch device: 
        - If CUDA is available will return a CUDA device (optionally can specify it to be the 
          `device_id`th CUDA device), otherwise "cpu".
        - If 'gpu' is specified, default to the first GPU ('cuda:0')
        - Can request a CPU by providing a device id of 'cpu' or -1.
    
    Args:
        device_id (int, str, or torch.device): The ID of a CUDA device if more than one available on the system. 
        Defaults to 0, which means GPU if it's available.
    
    Returns:
        :obj:`torch.device`: The available device.
    """
    # If input is already a Torch device, then return it as-is.
    if isinstance(device_id, torch.device):
        return device_id
    elif device_id == -1 or device_id == "cpu":
        return torch.device("cpu")
    elif torch.torch.cuda.is_available():
        if device_id == "gpu":
            device_id = 0
        return torch.device("cuda:{}".format(device_id))
    else:
        return torch.device("cpu")


def generate_fully_connected(
    input_dim: int,
    output_dim: int,
    hidden_dims: List[int],
    non_linearity: Optional[Type[Module]],
    activation: Optional[Type[Module]],
    device: torch.device,
    p_dropout: float = 0.0,
    init_method: str = "default",
) -> Module:
    """
    Generate a fully connected network.

    Args:
        input_dim: Int. Size of input to network.
        output_dim: Int. Size of output of network.
        hidden_dims: List of int. Sizes of internal hidden layers. i.e. [a, b] is three linear layers with shapes (input_dim, a), (a, b), (b, output_dim)
        non_linearity: Non linear activation function used between Linear layers.
        activation: Final layer activation to use.
        device: torch device to load weights to.
        p_dropout: Float. Dropout probability at the hidden layers.
        init_method: initialization method

    Returns:
        Sequential object containing the desired network.
    """
    layers: List[Module] = []

    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(Linear(prev_dim, hidden_dim).to(device))
        if non_linearity is not None:
            layers.append(non_linearity())
        if p_dropout != 0:
            layers.append(Dropout(p_dropout))
        prev_dim = hidden_dim

    layers.append(Linear(prev_dim, output_dim).to(device))

    if activation is not None:
        layers.append(activation())

    fcnn = Sequential(*layers)
    if init_method != "default":
        fcnn.apply((lambda x: alternative_initialization(x, init_method=init_method)))
    return fcnn


def alternative_initialization(module: Module, init_method: str) -> None:
    if isinstance(module, torch.nn.Linear):
        if init_method == "xavier_uniform":
            torch.nn.init.xavier_uniform_(module.weight)
        elif init_method == "xavier_normal":
            torch.nn.init.xavier_normal_(module.weight)
        elif init_method == "uniform":
            torch.nn.init.uniform_(module.weight)
        elif init_method == "normal":
            torch.nn.init.normal_(module.weight)
        else:
            return
        torch.nn.init.zeros_(module.bias)


class CrossEntropyLossWithConvert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = torch.nn.CrossEntropyLoss()

    def forward(self, prediction, actual):
        return self._loss(prediction, actual.long())


def create_dataloader(
    *arrays: Union[np.ndarray, csr_matrix],
    batch_size: int,
    iterations: int = -1,
    sample_randomly: bool = True,
    dtype: torch.dtype = torch.float,
    device: torch.device = torch.device("cpu"),
) -> DataLoader:
    """
    Device specifies the device on which the TensorDataset is created. This should be CPU in most cases, as we 
    typically do not wish to store the whole dataset on the GPU.
    """
    assert len(arrays) > 0
    dataset: Dataset
    if issparse(arrays[0]):
        assert all([issparse(arr) for arr in arrays])
        # TODO: To fix type error need to cast arrays from Tuple[Union[ndarray, csr_matrix]] to Tuple[csr_matrix],
        # but MyPy doesn't seem to detect it when I do this.
        dataset = SparseTensorDataset(*arrays, dtype=dtype, device=device)  # type: ignore
    else:
        assert all([not issparse(arr) for arr in arrays])
        dataset = TensorDataset(*to_tensors(*arrays, dtype=dtype, device=device))

    row_count = arrays[0].shape[0]
    max_iterations = np.ceil(row_count / batch_size)
    if iterations > max_iterations:
        iterations = -1

    if sample_randomly:
        if iterations == -1:
            # mypy throws an error when using a pytorch Dataset for the pytorch RandomSampler. This seems to be an issue in pytorch typing.
            sampler: Sampler = RandomSampler(dataset)  # type: ignore
        else:
            sampler = RandomSampler(dataset, replacement=True, num_samples=iterations * batch_size)  # type: ignore
    else:
        sampler = SequentialSampler(dataset)

    batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, pin_memory=True)
    return dataloader


class SparseTensorDataset(Dataset):
    """
    Custom dataset class which takes in a sparse matrix (assumed to be efficiently indexable row-wise, ie csr) and
    returns dense tensors containing requested rows. Ensures that the large matrices are kept sparse at all times,
    and only converted to dense matrices one minibatch at a time.
    """

    def __init__(
        self, *matrices: Tuple[csr_matrix, ...], dtype: torch.dtype = torch.float, device: torch.device,
    ):
        self._matrices = matrices
        self._dtype = dtype
        self._device = device

    def __getitem__(self, idx):
        data_rows = tuple(
            torch.as_tensor(matrix[idx, :].toarray().squeeze(axis=0), dtype=self._dtype, device=self._device,)
            for matrix in self._matrices
        )
        return data_rows

    def __len__(self):
        return self._matrices[0].shape[0]

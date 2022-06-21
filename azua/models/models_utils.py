# Utils for models
from typing import overload

import numpy as np
import torch


@overload
def assert_elements_have_equal_shape(*elements: np.ndarray) -> None:
    ...


@overload
def assert_elements_have_equal_shape(*elements: torch.Tensor) -> None:
    ...


def assert_elements_have_equal_shape(*elements) -> None:
    assert all(elements[0].shape == x.shape for x in elements), "Elements need to be of the same shape"


@overload
def assert_elements_have_equal_first_dimension(*elements: np.ndarray) -> None:
    ...


@overload
def assert_elements_have_equal_first_dimension(*elements: torch.Tensor) -> None:
    ...


def assert_elements_have_equal_first_dimension(*elements) -> None:
    assert all(elements[0].shape[0] == x.shape[0] for x in elements), "Elements need to have equal first dimension"

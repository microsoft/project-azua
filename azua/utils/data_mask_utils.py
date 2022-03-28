from typing import overload, List, Tuple, Optional
from itertools import product

import numpy as np
import pandas as pd
import torch

from ..datasets.variables import Variables


@overload
def to_tensors(
    array1: np.ndarray, array2: np.ndarray, device: torch.device, dtype: torch.dtype = torch.float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ...


@overload
def to_tensors(*arrays: np.ndarray, device: torch.device, dtype: torch.dtype = torch.float) -> Tuple[torch.Tensor, ...]:
    ...


def to_tensors(*arrays, device, dtype=torch.float):
    return tuple(torch.as_tensor(array, dtype=dtype, device=device) for array in arrays)


@overload
def sample_inducing_points(data: np.ndarray, mask: np.ndarray, row_count: int) -> Tuple[np.ndarray, np.ndarray]:
    ...


@overload
def sample_inducing_points(data: torch.Tensor, mask: torch.Tensor, row_count: int) -> Tuple[torch.Tensor, torch.Tensor]:
    ...


def sample_inducing_points(data, mask, row_count):

    # Randomly select inducing points to use to impute data.
    random_inducing_points_location = np.random.choice(data.shape[0], size=row_count, replace=True)
    inducing_data = data[random_inducing_points_location, :]
    inducing_mask = mask[random_inducing_points_location, :]

    return inducing_data, inducing_mask


def add_to_mask(variables: Variables, mask: torch.Tensor, idxs: List[int]) -> torch.Tensor:
    # Add observations to whole columns of a processed input mask for the variables with idxs given in `idxs`.
    cols_set = set()
    for var_idx in idxs:
        cols = variables.processed_cols[var_idx]
        cols_set.update(cols)
    cols = list(cols_set)

    new_mask = mask.clone()
    new_mask[:, idxs] = 1
    return new_mask


def add_to_data(variables: Variables, data: torch.Tensor, new_vals: torch.Tensor, idxs: List[int]) -> torch.Tensor:
    # Update columns of processed data `data` with values from `new_vals` for the variables with idxs given in `idxs`.
    cols_set = set()
    for var_idx in idxs:
        cols = variables.processed_cols[var_idx]
        cols_set.update(cols)
    cols = list(cols_set)

    new_data = data.clone()
    new_data[:, cols] = new_vals[:, cols]
    return new_data


def restore_preserved_values(
    variables: Variables,
    data: torch.Tensor,  # shape (batch_size, input_dim)
    imputations: torch.Tensor,  # shape (num_samples, batch_size, input_dim)
    mask: torch.Tensor,  # shape (batch_size, input_dim)
) -> torch.Tensor:  # shape (num_samples, batch_size, input_dim)
    """
    Replace values in imputations with data where mask is True

    """
    assert data.dim() == 2

    assert data.shape == mask.shape
    assert isinstance(data, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    masked_data = data * mask

    # Remove metadata dims to get shape (1, batch_size, output_dim)
    if variables.has_auxiliary:
        variables = variables.subset(list(range(0, variables.num_unprocessed_non_aux_cols)))
        output_var_idxs = torch.arange(variables.num_processed_cols, device=mask.device)
        output_mask = torch.index_select(mask, dim=1, index=output_var_idxs)
        masked_imputations = imputations * (1 - output_mask)

        masked_data = torch.index_select(masked_data, dim=1, index=output_var_idxs)

    else:
        masked_imputations = imputations * (1 - mask)

    # pytorch broadcasts by matching up trailing dimensions
    # so it is OK that masked_imputations.ndim==3 and masked_data.dim==2
    return masked_imputations + masked_data


def matrix_to_list(matrix: np.ndarray, row_ids: List[int], col_ids: List[int]) -> pd.DataFrame:
    """
    Transform a dense matrix of values to a DataFrame listing all of the matrix values, with columns 
    (row_id, col_id, value).
    """

    num_rows, num_cols = matrix.shape

    assert len(row_ids) == num_rows
    assert len(col_ids) == num_cols
    data = []
    for i, (row_idx, col_idx) in enumerate(product(range(num_rows), range(num_cols))):
        data.append((row_ids[row_idx], col_ids[col_idx], matrix[row_idx, col_idx]))

    return pd.DataFrame(data, columns=["row_id", "col_id", "value"])


def argsort_rows_exclude_nan(array: np.ndarray, ascending: bool = True, max_qs_per_row: Optional[int] = None):
    """
    Returns indices of the sorted values of each row of the input array, ignoring any NaN values.
    Args:
        array: 2D NumPy array to be argsorted.
        ascending: Whether each row of the array should be sorted in ascending or descending order.
        max_qs_per_row: Maximum number of indices to include in each row of the output array. If None, all available
            indices will be included.
    """
    if ascending:
        argsorted_rows = np.argsort(array, axis=1)
    else:
        argsorted_rows = np.argsort(-array, axis=1)
    out: List[List[int]] = [[] for _ in range(array.shape[0])]
    for i, row in enumerate(argsorted_rows):
        for feature_num, j in enumerate(row):
            if np.isnan(array[i][j]) or ((max_qs_per_row is not None) and (feature_num > max_qs_per_row - 1)):
                # Assume all elements in row are NaN after encountering first NaN, as they are always placed at the end
                # of sorting.
                continue
            else:
                out[i].append(j)
    return out

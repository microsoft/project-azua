import numpy as np
import pytest
import torch

import azua.utils.training_objectives as train
from azua.datasets.variables import Variable, Variables
from azua.utils.torch_utils import set_random_seeds


@pytest.fixture(scope="function")
def variables():
    return Variables(
        [
            Variable("binary_input", True, "binary", 1.0, 3.0),
            Variable("numeric_input", True, "continuous", 3, 13),
            Variable("binary_input_2", True, "binary", 1.0, 3.0),
            Variable("categorical_input", True, "categorical", 0.0, 2.0),
            Variable("numeric_target", False, "continuous", 2, 300),
        ]
    )


@pytest.mark.parametrize("sum_type", ["all", "cols", None])
@pytest.mark.parametrize("mask", [None, torch.arange(21).reshape((3, 7)) % 2])  # Mask alternating 0 and 1
@pytest.mark.parametrize("alpha", [1.0, 0.5])
def test_negative_log_likelihood(variables, sum_type, mask, alpha):
    dec_mean = torch.Tensor(
        [
            [0.6, 3.5, 0.4, 0.2, 0.3, 0.5, 26.5],
            [0.3, 11.8, 0.8, 0.1, 0.7, 0.2, 263.7],
            [0.7, 7.2, 0.2, 0.6, 0.1, 0.3, 139.3],
        ]
    )
    dec_logvar = torch.Tensor(
        [
            [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, np.log(2.0)],
            [0.7, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.2, 0.5, 0.0, 0.0, 0.0, 0.0, -1.5],
        ]
    )
    data = torch.Tensor(
        [
            [0.0, 5.6, 1.0, 0.0, 0.0, 1.0, 21.5],
            [1.0, 10.2, 0.0, 0.0, 1.0, 0.0, 266.7],
            [1.0, 7.9, 0.0, 0.0, 0.0, 1.0, 141.8],
        ]
    )

    nll = train.negative_log_likelihood(
        data, dec_mean, dec_logvar, variables, alpha=alpha, mask=mask, sum_type=sum_type
    )

    if mask is None:
        mask = torch.ones((3, 7))

    if sum_type is None:
        expected_nll = torch.zeros((3, 5))
        expected_nll[:, [1, 4]] = train.gaussian_negative_log_likelihood(
            data[:, [1, 6]],
            dec_mean[:, [1, 6]],
            dec_logvar[:, [1, 6]],
            mask[:, [1, 6]],
            sum_type=None,
        )
        expected_nll[:, [0, 2]] = train.bernoulli_negative_log_likelihood(
            data[:, [0, 2]], dec_mean[:, [0, 2]], mask[:, [0, 2]], sum_type=None
        )
        expected_nll[:, 3] = alpha * train.categorical_negative_log_likelihood(
            data[:, 3:6], dec_mean[:, 3:6], mask[:, 3:6], sum_type=None
        )
    elif sum_type == "cols":
        expected_nll = torch.zeros((1, 5))
        expected_nll[:, [1, 4]] = train.gaussian_negative_log_likelihood(
            data[:, [1, 6]],
            dec_mean[:, [1, 6]],
            dec_logvar[:, [1, 6]],
            mask[:, [1, 6]],
            sum_type="cols",
        )
        expected_nll[:, [0, 2]] = train.bernoulli_negative_log_likelihood(
            data[:, [0, 2]], dec_mean[:, [0, 2]], mask[:, [0, 2]], sum_type="cols"
        )
        expected_nll[:, 3] = alpha * train.categorical_negative_log_likelihood(
            data[:, 3:6], dec_mean[:, 3:6], mask[:, 3:6], sum_type="cols"
        )
    else:
        expected_nll = 0
        expected_nll += train.gaussian_negative_log_likelihood(
            data[:, [1, 6]],
            dec_mean[:, [1, 6]],
            dec_logvar[:, [1, 6]],
            mask[:, [1, 6]],
            sum_type="all",
        )
        expected_nll += train.bernoulli_negative_log_likelihood(
            data[:, [0, 2]], dec_mean[:, [0, 2]], mask[:, [0, 2]], sum_type="all"
        )
        expected_nll += alpha * train.categorical_negative_log_likelihood(
            data[:, 3:6], dec_mean[:, 3:6], mask[:, 3:6], sum_type="all"
        )

    assert torch.abs(nll - expected_nll).sum() < 1e-3


@pytest.mark.parametrize("max_p_train_dropout", [0.0, 1.0, 0.3])
def test_get_input_and_scoring_masks(max_p_train_dropout):
    set_random_seeds(0)
    data_density = 0.7
    # On average we drop 0.5 * max_p_train_dropout of the data
    expected_input_density = torch.tensor(data_density * (1 - 0.5 * max_p_train_dropout))
    data_mask = torch.bernoulli(data_density * torch.ones((10000, 200))).to(torch.float)

    # Score imputation only
    input_mask, scoring_mask = train.get_input_and_scoring_masks(
        mask=data_mask, max_p_train_dropout=max_p_train_dropout, score_imputation=True, score_reconstruction=False
    )

    assert torch.equal(input_mask + scoring_mask, data_mask)
    assert torch.allclose(input_mask.mean(), expected_input_density, atol=0.01)

    # Score reconstruction only
    input_mask, scoring_mask = train.get_input_and_scoring_masks(
        mask=data_mask, max_p_train_dropout=max_p_train_dropout, score_reconstruction=True, score_imputation=False
    )
    assert torch.allclose(input_mask.mean(), expected_input_density, atol=0.01)
    assert torch.equal(scoring_mask, input_mask)

    # Score imputation and reconstruction
    input_mask, scoring_mask = train.get_input_and_scoring_masks(
        mask=data_mask, max_p_train_dropout=max_p_train_dropout, score_reconstruction=True, score_imputation=True
    )
    assert torch.allclose(input_mask.mean(), expected_input_density, atol=0.01)
    assert torch.equal(scoring_mask, data_mask)

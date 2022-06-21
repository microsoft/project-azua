import numpy as np
import pytest

from azua.datasets.variables import Variable, Variables
from azua.utils.imputation_statistics_utils import ImputationStatistics as ImputeStats


@pytest.fixture(scope="function")
def variables():
    return Variables(
        [
            Variable("binary_input", True, "binary", 0.0, 1.0),
            Variable("numeric_input", True, "continuous", 3, 13),
            Variable("categorical_target", False, "categorical", 1, 5),
        ]
    )


def test_impute_stats_utils(variables):
    batch_size = 2
    data = np.array(
        [
            [0.0, 4.6, 3.0],
            [0.0, 7.9, 2.0],
            [1.0, 3.3, 5.0],
            [1.0, 9.7, 1.0],
            [0.0, 10.2, 2.0],
        ]
    )
    data = np.tile(data[:, np.newaxis], (1, batch_size, 1))  # repeat the data

    predictions_stats = ImputeStats.get_statistics(data, variables)

    assert isinstance(predictions_stats, dict)
    assert len(predictions_stats.items()) == variables.__len__()
    for var_id in predictions_stats.keys():
        variable = variables.__getitem__(var_id)
        assert predictions_stats[var_id]["type"] == variable.type

        # test for categorical variables
        if variable.type == "categorical":
            n_class = predictions_stats[var_id]["n_class"]
            # first test shapes and min/max values
            assert n_class == (variable.upper - variable.lower + 1)
            assert predictions_stats[var_id]["marginal_prob"].shape == (
                batch_size,
                n_class,
            )
            assert predictions_stats[var_id]["majority_vote"].shape == (batch_size,)
            assert np.min(predictions_stats[var_id]["majority_vote"]) >= variable.lower
            assert np.max(predictions_stats[var_id]["majority_vote"]) <= variable.upper
            assert predictions_stats[var_id]["majority_prob"].shape == (batch_size,)
            assert predictions_stats[var_id]["entropy"].shape == (batch_size,)
            assert np.min(predictions_stats[var_id]["entropy"]) >= 0.0
            assert np.max(predictions_stats[var_id]["entropy"]) <= np.log(n_class) + 1e-5

            # then test the computation, also taking numerical error into consideration
            assert predictions_stats[var_id]["majority_vote"][0] == 2.0
            assert np.abs(predictions_stats[var_id]["majority_prob"][0] - 0.4) < 1e-5
            target_entropy = -0.2 * np.log(0.2) * 3 - 0.4 * np.log(0.4)
            assert np.abs(predictions_stats[var_id]["entropy"][0] - target_entropy) < 1e-5

        # test for binary variables
        if variable.type == "binary":
            # first test shapes and min/max values
            assert predictions_stats[var_id]["n_class"] == 2
            assert predictions_stats[var_id]["majority_vote"].shape == (batch_size,)
            assert np.min(predictions_stats[var_id]["majority_vote"]) >= 0.0
            assert np.max(predictions_stats[var_id]["majority_vote"]) <= 1.0
            assert predictions_stats[var_id]["majority_prob"].shape == (batch_size,)
            assert predictions_stats[var_id]["entropy"].shape == (batch_size,)
            assert np.min(predictions_stats[var_id]["entropy"]) >= 0.0
            assert np.max(predictions_stats[var_id]["entropy"]) <= np.log(2) + 1e-5

            # then test the computation, also taking numerical error into consideration
            assert predictions_stats[var_id]["majority_vote"][0] == 0.0
            assert predictions_stats[var_id]["majority_prob"][0] == 0.6
            target_entropy = -0.4 * np.log(0.4) - 0.6 * np.log(0.6)
            assert np.abs(predictions_stats[var_id]["entropy"][0] - target_entropy) < 1e-5

        # test for continuous variables
        if variable.type == "continuous":
            # first test shapes
            assert predictions_stats[var_id]["min_val"].shape == (batch_size,)
            assert predictions_stats[var_id]["max_val"].shape == (batch_size,)
            assert predictions_stats[var_id]["mean"].shape == (batch_size,)
            assert predictions_stats[var_id]["quartile_1"].shape == (batch_size,)
            assert predictions_stats[var_id]["median"].shape == (batch_size,)
            assert predictions_stats[var_id]["quartile_3"].shape == (batch_size,)

            # then test the computation, also taking numerical error into consideration
            assert predictions_stats[var_id]["variable_lower"] == variable.lower
            assert predictions_stats[var_id]["variable_upper"] == variable.upper
            assert predictions_stats[var_id]["min_val"][0] == 3.3
            assert predictions_stats[var_id]["max_val"][0] == 10.2
            assert np.abs(predictions_stats[var_id]["mean"][0] - 7.14) < 1e-5
            assert np.abs(predictions_stats[var_id]["quartile_1"][0] - 4.6) < 1e-5
            assert np.abs(predictions_stats[var_id]["median"][0] - 7.9) < 1e-5
            assert np.abs(predictions_stats[var_id]["quartile_3"][0] - 9.7) < 1e-5

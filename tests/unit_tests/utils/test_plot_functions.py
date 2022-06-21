import os

import numpy as np
import pytest

from azua.datasets.variables import Variable, Variables
from azua.utils.imputation_statistics_utils import ImputationStatistics as ImputeStats
from azua.utils.plot_functions import violin_plot_imputations


@pytest.fixture(scope="function")
def variables():
    return Variables(
        [
            Variable("binary_input", True, "binary", 0.0, 1.0),
            Variable("numeric_input1", True, "continuous", 3, 13),
            Variable("numeric_input2", True, "continuous", 1, 20),
            Variable("numeric_input3", True, "continuous", -5, 5),
            Variable("categorical_target", False, "categorical", 1, 5),
        ]
    )


def test_violin_plot_imputations(variables, tmpdir):
    batch_size = 2
    data = np.array(
        [
            [0.0, 4.6, 1.5, -2.0, 3.0],
            [0.0, 7.9, 10.7, 4.3, 2.0],
            [1.0, 3.3, 17.9, 0.5, 5.0],
            [1.0, 9.7, 6.8, -3.7, 1.0],
            [0.0, 10.2, 12.4, 1.0, 2.0],
        ]
    )
    data = np.tile(data[:, np.newaxis], (1, batch_size, 1))  # repeat the data
    mask = np.zeros((batch_size, data.shape[0]))

    stats = ImputeStats.get_statistics(data, variables)
    user_id = 1
    save_path = tmpdir
    plot_name = "unit_test_violin_plot"
    user_stats, _ = violin_plot_imputations(
        data,
        stats,
        mask,
        variables,
        user_id,
        save_path=save_path,
        plot_name=plot_name,
        normalized=False,
    )

    # test the collected user stats
    assert len(user_stats["variable_names"]) == 3
    for i in range(3):
        name = "numeric_input%d" % (i + 1)
        assert name in user_stats["variable_names"]
    assert np.abs(user_stats["median"] - np.array([7.9, 10.7, 0.5])).mean() < 1e-5
    assert np.abs(user_stats["quartile_1"] - np.array([4.6, 6.8, -2.0])).mean() < 1e-5
    assert np.abs(user_stats["quartile_3"] - np.array([9.7, 12.4, 1.0])).mean() < 1e-5
    assert np.abs(user_stats["lower_fence"] - np.array([3.3, 1.5, -3.7])).mean() < 1e-5
    assert np.abs(user_stats["upper_fence"] - np.array([10.2, 17.9, 4.3])).mean() < 1e-5
    # test the plot
    assert os.path.isfile(os.path.join(save_path, plot_name + ".png"))

    plot_name = "unit_test_violin_plot_normalized"
    user_stats, _ = violin_plot_imputations(
        data,
        stats,
        mask,
        variables,
        user_id,
        save_path=save_path,
        plot_name=plot_name,
        normalized=True,
    )

    # test the collected user stats
    assert len(user_stats["variable_names"]) == 3
    for i in range(3):
        name = "numeric_input%d" % (i + 1)
        assert name in user_stats["variable_names"]
    interval = np.array([10.0, 19.0, 10.0])
    lower = np.array([3.0, 1.0, -5.0])
    assert np.abs(user_stats["median"] - (np.array([7.9, 10.7, 0.5]) - lower) / interval).mean() < 1e-5
    assert np.abs(user_stats["quartile_1"] - (np.array([4.6, 6.8, -2.0]) - lower) / interval).mean() < 1e-5
    assert np.abs(user_stats["quartile_3"] - (np.array([9.7, 12.4, 1.0]) - lower) / interval).mean() < 1e-5
    assert np.abs(user_stats["lower_fence"] - (np.array([3.3, 1.5, -3.7]) - lower) / interval).mean() < 1e-5
    assert np.abs(user_stats["upper_fence"] - (np.array([10.2, 17.9, 4.3]) - lower) / interval).mean() < 1e-5
    # test the plot
    assert os.path.isfile(os.path.join(save_path, plot_name + ".png"))

import os

import numpy as np
import pytest

from azua.objectives.eddi import EDDIObjective
from azua.objectives.eddi_rowwise import EDDIRowwiseObjective
from azua.objectives.rand import RandomObjective
from azua.objectives.sing import SINGObjective
from azua.utils.active_learning import compute_rmse_curves, save_info_gain_normalizer, select_feature
from azua.datasets.variables import Variable, Variables
from .mock_model_for_objective import MockModelForObjective


@pytest.fixture(scope="function")
def variables():
    return Variables(
        [
            Variable("categorical_input", True, "categorical", 1.0, 3.0),
            Variable("numeric_input", True, "continuous", 3, 13),
            Variable("numeric_target", False, "continuous", 2, 300),
        ]
    )


@pytest.fixture(scope="function")
def variables_without_target():
    return Variables(
        [
            Variable("numeric_input1", True, "continuous", 1.0, 3.0),
            Variable("numeric_input2", True, "continuous", 3, 13),
            Variable("numeric_input3", True, "continuous", 2, 300),
        ]
    )


@pytest.fixture(scope="function")
def variables_with_groups():
    return Variables(
        [
            Variable("categorical_input", True, "categorical", 1.0, 3.0, group_name="group_1"),
            Variable("numeric_input", True, "continuous", 3, 13, group_name="group_1"),
            Variable("numeric_input_2", True, "continuous", 3, 13, group_name="group_2"),
            Variable("binary_input", True, "binary", 0, 1, group_name="group_2"),
            Variable("numeric_target", False, "continuous", 2, 300),
        ]
    )


@pytest.fixture(scope="function")
def variables_with_groups_and_query_targets():
    return Variables(
        [
            Variable("categorical_input", True, "categorical", 1.0, 3.0, group_name="group_1"),
            Variable("numeric_input", True, "continuous", 3, 13, group_name="group_1"),
            Variable("numeric_query_target", True, "continuous", 3, 13, group_name="group_2", target=True),
            Variable("binary_query_target", True, "binary", 0, 1, group_name="group_2", target=True),
            Variable("categorical_query_target", True, "categorical", 2, 4, target=True),
            Variable("numeric_target", False, "continuous", 2, 300),
        ]
    )


def test_select_feature_random(tmpdir, variables):
    model = MockModelForObjective("dummy_id", variables, tmpdir)
    objective = RandomObjective(model)
    data = np.array([[1, 4.2, 2.2], [2, 9.2, 200.1], [3, 11.2, 100.0]])
    mask = np.ones_like(data, dtype=bool)  # all values exist in underlying data (can be queried)
    obs_mask = np.zeros_like(data, dtype=bool)  # all values unobserved to begin with

    next_qs, info_gains = select_feature(data, mask, obs_mask, objective, variables.group_idxs)
    assert len(next_qs) == 3
    for row_choice in next_qs:
        assert len(row_choice) == 1  # recommending single next best question
        assert min(row_choice) >= 0
        assert max(row_choice) <= 1  # 2 queriable variables
    assert info_gains is None


def test_select_feature_groups_random(tmpdir, variables_with_groups):
    variables = variables_with_groups
    model = MockModelForObjective("dummy_id", variables, tmpdir)
    objective = RandomObjective(model)
    data = np.array([[1, 4.2, 4.2, 1, 2.2], [2, 9.2, 9.2, 0, 200.1], [3, 11.2, 11.2, 1, 100.0]])
    mask = np.ones_like(data, dtype=bool)  # all values exist in underlying data (can be queried)
    obs_mask = np.zeros_like(data, dtype=bool)  # all values unobserved to begin with

    next_qs, info_gains = select_feature(data, mask, obs_mask, objective, variables.group_idxs)
    assert len(next_qs) == 3
    for row_choice in next_qs:
        assert len(row_choice) == 1
        assert min(row_choice) >= 0
        assert max(row_choice) <= 1  # 2 queriable groups
    assert info_gains is None


def test_select_feature_groups_and_query_targets_random(tmpdir, variables_with_groups_and_query_targets):
    variables = variables_with_groups_and_query_targets
    model = MockModelForObjective("dummy_id", variables, tmpdir)
    objective = RandomObjective(model)
    data = np.array([[1, 4.2, 4.2, 1, 2, 2.2], [2, 9.2, 9.2, 0, 3, 200.1], [3, 11.2, 11.2, 1, 4, 100.0]])
    mask = np.ones_like(data, dtype=bool)  # all values exist in underlying data (can be queried)
    obs_mask = np.zeros_like(data, dtype=bool)  # all values unobserved to begin with

    next_qs, info_gains = select_feature(data, mask, obs_mask, objective, variables.group_idxs)
    assert len(next_qs) == 3
    for row_choice in next_qs:
        assert len(row_choice) == 1
        assert min(row_choice) >= 0
        assert max(row_choice) <= 2  # 3 queriable groups
    assert info_gains is None


def test_select_feature_sing(tmpdir, variables):
    model = MockModelForObjective("dummy_id", variables, tmpdir)
    objective = SINGObjective(model, sample_count=3)
    data = np.array([[1, 4.2, 2.2], [2, 9.2, 200.1], [3, 11.2, 100.0]])
    mask = np.ones_like(data, dtype=bool)  # all values exist in underlying data (can be queried)

    obs_mask = np.zeros_like(data, dtype=bool)  # all values unobserved to begin with

    next_qs, info_gains = select_feature(data, mask, obs_mask, objective, variables.group_idxs)
    assert len(next_qs) == 3
    for row_choice in next_qs:
        assert len(row_choice) == 1
        assert min(row_choice) >= 0
        assert max(row_choice) <= 1  # 2 queriable variables
    assert len(info_gains) == 2  # 2 queriable variables


def test_select_feature_groups_sing(tmpdir, variables_with_groups):
    variables = variables_with_groups
    model = MockModelForObjective("dummy_id", variables, tmpdir)
    objective = SINGObjective(model, sample_count=3)
    data = np.array([[1, 4.2, 4.2, 1, 2.2], [2, 9.2, 9.2, 0, 200.1], [3, 11.2, 11.2, 1, 100.0]])
    mask = np.ones_like(data, dtype=bool)  # all values exist in underlying data (can be queried)
    obs_mask = np.zeros_like(data, dtype=bool)  # all values unobserved to begin with

    next_qs, info_gains = select_feature(data, mask, obs_mask, objective, variables.group_idxs)
    assert len(next_qs) == 3
    for row_choice in next_qs:
        assert len(row_choice) == 1
        assert min(row_choice) >= 0
        assert max(row_choice) <= 1  # 2 queriable groups
    assert len(info_gains) == 2  # 2 queriable groups


def test_select_feature_groups_and_query_targets_sing(tmpdir, variables_with_groups_and_query_targets):
    variables = variables_with_groups_and_query_targets
    model = MockModelForObjective("dummy_id", variables, tmpdir)
    objective = SINGObjective(model, sample_count=3)
    data = np.array([[1, 4.2, 4.2, 1, 2, 2.2], [2, 9.2, 9.2, 0, 3, 200.1], [3, 11.2, 11.2, 1, 4, 100.0]])
    mask = np.ones_like(data, dtype=bool)  # all values exist in underlying data (can be queried)
    obs_mask = np.zeros_like(data, dtype=bool)  # all values unobserved to begin with

    next_qs, info_gains = select_feature(data, mask, obs_mask, objective, variables.group_idxs)
    assert len(next_qs) == 3
    for row_choice in next_qs:
        assert len(row_choice) == 1
        assert min(row_choice) >= 0
        assert max(row_choice) <= 2
    assert len(info_gains) == 3  # 3 queriable groups


@pytest.mark.parametrize("test_objective", [EDDIObjective, EDDIRowwiseObjective])
def test_select_feature_eddi(tmpdir, variables, test_objective):
    model = MockModelForObjective("dummy_id", variables, tmpdir)
    objective = test_objective(model, sample_count=3)
    data = np.array([[1, 4.2, 2.2], [2, 9.2, 200.1], [3, 11.2, 100.0]])
    mask = np.ones_like(data, dtype=bool)  # all values exist in underlying data (can be queried)
    obs_mask = np.zeros_like(data, dtype=bool)  # all values unobserved to begin with

    next_qs, info_gains = select_feature(data, mask, obs_mask, objective, variables.group_idxs)
    assert len(next_qs) == 3
    for row_choice in next_qs:
        assert len(row_choice) == 1
        assert min(row_choice) >= 0
        assert max(row_choice) <= 1  # 2 queriable variables
    assert len(info_gains) == 3  # 3 data rows
    assert len(info_gains[0]) == 2  # 2 queriable variables


@pytest.mark.parametrize("test_objective", [EDDIObjective, EDDIRowwiseObjective])
def test_select_feature_eddi_masked(tmpdir, variables, test_objective):
    model = MockModelForObjective("dummy_id", variables, tmpdir)
    objective = test_objective(model, sample_count=3)
    data = np.array([[1, 4.2, 2.2], [2, 9.2, 200.1], [3, 11.2, 100.0]])
    mask = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]])  # all values exist in underlying data (can be queried)
    obs_mask = np.zeros_like(data)  # all values unobserved to begin with

    next_qs, info_gains = select_feature(data, mask, obs_mask, objective, variables.group_idxs)

    assert len(next_qs) == 3
    for row_choice in next_qs:
        if row_choice:
            assert min(row_choice) >= 0
            assert max(row_choice) <= 1  # 2 queriable variables
    assert len(next_qs[0]) == 1
    assert len(next_qs[1]) == 1
    assert len(next_qs[2]) == 0

    assert len(info_gains) == 3  # 3 data rows
    assert len(info_gains[0]) == 2  # 2 queriable variables on row 0
    assert len(info_gains[1]) == 1  # 1 queriable variable on row 1
    assert len(info_gains[2]) == 0  # 0 queriable variables on row 2


@pytest.mark.parametrize("test_objective", [EDDIObjective, EDDIRowwiseObjective])
def test_select_feature_eddi_with_both_masks(tmpdir, variables, test_objective):
    model = MockModelForObjective("dummy_id", variables, tmpdir)
    objective = test_objective(model, sample_count=3)
    data = np.array([[1, 4.2, 2.2], [2, 9.2, 200.1], [3, 11.2, 100.0]])
    mask = np.array([[1, 1, 0], [1, 1, 0], [1, 1, 0]])
    obs_mask = np.array([[0, 1, 0], [1, 0, 0], [1, 1, 0]])

    next_qs, _ = select_feature(data, mask, obs_mask, objective, variables.group_idxs)
    assert next_qs == [[0], [1], []]


@pytest.mark.parametrize("test_objective", [EDDIObjective, EDDIRowwiseObjective])
def test_select_feature_groups_eddi(tmpdir, variables_with_groups, test_objective):
    variables = variables_with_groups
    model = MockModelForObjective("dummy_id", variables, tmpdir)
    objective = test_objective(model, sample_count=3)
    data = np.array([[1, 4.2, 4.2, 1, 2.2], [2, 9.2, 9.2, 0, 200.1], [3, 11.2, 11.2, 1, 100.0]])
    mask = np.ones_like(data, dtype=bool)  # all values exist in underlying data (can be queried)
    obs_mask = np.zeros_like(data, dtype=bool)  # all values unobserved to begin with

    next_qs, info_gains = select_feature(data, mask, obs_mask, objective, variables.group_idxs)
    assert len(next_qs) == 3
    for row_choice in next_qs:
        assert len(row_choice) == 1
        assert min(row_choice) >= 0
        assert max(row_choice) <= 1  # 2 queriable groups
    assert len(info_gains) == 3  # 3 data rows
    assert len(info_gains[0]) == 2  # 2 queriable groups


@pytest.mark.parametrize("test_objective", [EDDIObjective, EDDIRowwiseObjective])
def test_select_feature_groups_eddi_with_both_masks(tmpdir, variables_with_groups, test_objective):
    variables = variables_with_groups
    model = MockModelForObjective("dummy_id", variables, tmpdir)
    objective = test_objective(model, sample_count=3)
    data = np.array(
        [
            [1, 4.2, 4.2, 1, 2.2],
            [2, 9.2, 9.2, 0, 200.1],
            [3, 11.2, 11.2, 1, 100.0],
            [4, 11.2, 11.2, 1, 100.0],
            [5, 11.2, 11.2, 1, 100.0],
            [6, 11.2, 11.2, 1, 100.0],
        ]
    )
    mask = np.array(
        [[1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0]]
    )
    obs_mask = np.array(
        [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    )

    next_qs, _ = select_feature(data, mask, obs_mask, objective, variables.group_idxs)
    assert next_qs == [[1], [], [], [0], [], [1]]


@pytest.mark.parametrize("test_objective", [EDDIObjective, EDDIRowwiseObjective])
def test_select_feature_groups_and_query_targets_eddi(tmpdir, variables_with_groups_and_query_targets, test_objective):
    # Test behaviour when variables include query groups and queriable targets.
    variables = variables_with_groups_and_query_targets
    model = MockModelForObjective("dummy_id", variables, tmpdir)
    objective = test_objective(model, sample_count=3)
    data = np.array([[1, 4.2, 4.2, 1, 2, 2.2], [2, 9.2, 9.2, 0, 3, 200.1], [3, 11.2, 11.2, 1, 4, 100.0]])
    mask = np.ones_like(data, dtype=bool)  # all values exist in underlying data (can be queried)
    obs_mask = np.zeros_like(data, dtype=bool)  # all values unobserved to begin with

    next_qs, info_gains = select_feature(data, mask, obs_mask, objective, variables.group_idxs)
    assert len(next_qs) == 3
    for row_choice in next_qs:
        assert len(row_choice) == 1
        assert min(row_choice) >= 0
        assert max(row_choice) <= 2
    assert len(info_gains) == 3  # 3 data rows
    assert len(info_gains[0]) == 3  # 3 queriable groups


@pytest.mark.parametrize("test_objective", [EDDIObjective, EDDIRowwiseObjective])
def test_select_multiple_features_eddi(tmpdir, variables, test_objective):
    # Test that multiple next queries are returned per row if question_count > 1
    model = MockModelForObjective("dummy_id", variables, tmpdir)
    objective = test_objective(model, sample_count=3)
    data = np.array([[1, 4.2, 2.2], [2, 9.2, 200.1], [3, 11.2, 100.0]])
    mask = np.ones_like(data)  # all values exist in underlying data (can be queried)
    obs_mask = np.zeros_like(data)  # all values unobserved to begin with

    next_qs, info_gains = select_feature(data, mask, obs_mask, objective, variables.group_idxs, question_count=2)
    assert len(next_qs) == 3
    for row_choices in next_qs:
        assert len(row_choices) == 2
        assert min(row_choices) >= 0
        assert max(row_choices) <= 1  # 2 queriable variables
    assert len(info_gains) == 3  # 3 data rows
    assert len(info_gains[0]) == 2  # 2 queriable variables


@pytest.mark.parametrize("test_objective", [EDDIObjective, EDDIRowwiseObjective])
def test_select_too_many_features_eddi(tmpdir, variables, test_objective):
    # Test that as many queries as possible are returned for each row if question_count is greater than the maximum
    # number of query groups in the row
    model = MockModelForObjective("dummy_id", variables, tmpdir)
    objective = test_objective(model, sample_count=3)
    data = np.array([[1, 4.2, 2.2], [2, 9.2, 200.1], [3, 11.2, 100.0]])
    mask = np.ones_like(data)  # all values exist in underlying data (can be queried)
    obs_mask = np.zeros_like(data)  # all values unobserved to begin with

    next_qs, info_gains = select_feature(data, mask, obs_mask, objective, variables.group_idxs, question_count=5)
    assert len(next_qs) == 3
    for row_choices in next_qs:
        assert len(row_choices) == 2
        assert min(row_choices) >= 0
        assert max(row_choices) <= 1  # 2 queriable variables
    assert len(info_gains) == 3  # 3 data rows
    assert len(info_gains[0]) == 2  # 2 queriable variables


def test_compute_rmse_curves(variables):
    imputed_values = np.array([[[1, 4.0, 3.0]]])
    test_data = np.array([[1, 5.0, 5.0]])
    test_mask = np.ones_like(test_data, dtype=bool)
    rmse_curves = compute_rmse_curves({"eedi": imputed_values}, test_data, test_mask, variables, False)
    assert rmse_curves["numeric_target"]["eedi"][0] == 2.0
    assert rmse_curves["all"]["eedi"][0] == 2.0
    assert len(rmse_curves) == 2


def test_compute_rmse_curves_no_target_variable(variables_without_target):
    imputed_values = np.array([[[1.0, 4.0, 3.0]]])
    test_data = np.array([[2.0, 5.0, 5.0]])
    test_mask = np.ones_like(test_data, dtype=bool)
    rmse_curves = compute_rmse_curves({"eedi": imputed_values}, test_data, test_mask, variables_without_target, False)
    assert rmse_curves["all"]["eedi"][0] == pytest.approx(1.4142135623730951, 0.1)  # sqrt(2)
    assert len(rmse_curves) == 1


def test_compute_rmse_curves_no_target_variable_with_mask(variables_without_target):
    imputed_values = np.array([[[1.0, 4.0, 3.0]]])
    test_data = np.array([[2.0, 5.0, 5.0]])
    test_mask = np.array([[0, 1, 1]], dtype=bool)
    rmse_curves = compute_rmse_curves({"eedi": imputed_values}, test_data, test_mask, variables_without_target, False)
    assert rmse_curves["all"]["eedi"][0] == pytest.approx(1.58113883008, 0.1)  # sqrt(2.5)
    assert len(rmse_curves) == 1

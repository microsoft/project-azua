from azua.datasets.cifar10_dataset_loader import CIFAR10DatasetLoader


def test_create_variables_no_targets():
    variables = CIFAR10DatasetLoader._create_variables(use_targets=False)
    assert len(variables) == 3 * 32 * 32

    assert variables[0].type == "continuous"
    assert variables[-1].type == "continuous"

    assert variables[0].query is True
    assert variables[-1].query is True

    assert variables[0].lower == 0.0
    assert variables[0].upper == 1.0

    assert variables[-1].lower == 0.0
    assert variables[-1].upper == 1.0


def test_create_variables_with_targets():
    variables = CIFAR10DatasetLoader._create_variables(use_targets=True)
    assert len(variables) == 3 * 32 * 32 + 1

    assert variables[0].type == "continuous"
    assert variables[-1].type == "categorical"

    assert variables[0].query is True
    assert variables[-1].query is False

    assert variables[0].lower == 0.0
    assert variables[0].upper == 1.0

    assert variables[-1].lower == 0
    assert variables[-1].upper == 9

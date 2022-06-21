import pytest
import torch
from torch.nn import Sigmoid

from azua.models.decoder import FeaturewiseActivation
from azua.datasets.variables import Variable, Variables


def get_variables():
    return Variables(
        [
            Variable(name="continuous", query=True, type="continuous", lower=0.0, upper=3.0),
            Variable(name="categorical", query=True, type="categorical", lower=0, upper=5),
            Variable(name="binary", query=False, type="binary"),
        ]
    )


def get_featurewise_activation(variables):
    return FeaturewiseActivation(activation_for_continuous=Sigmoid, variables=variables)


def test_featurewise_activation():
    variables = get_variables()
    model = FeaturewiseActivation(variables=variables, activation_for_continuous=Sigmoid)
    inputs = torch.zeros(4, 8)
    inputs[:, -1] = 0
    inputs[:, 0] = 100000
    outputs = model(inputs)
    assert outputs.shape == inputs.shape
    assert torch.allclose(
        outputs[:, 0], torch.tensor(1.0)
    )  # Sigmoid should map big values to ~1 for continuous variable
    assert torch.allclose(outputs[:, 7], torch.tensor(0.5))  # Sigmoid should map log-odds 0 to 0.5 for binary
    assert torch.allclose(
        outputs[:, 1:7], torch.tensor(1.0 / 6.0)
    )  # Softmax should map all-zeros to uniform distribution for categorical


@pytest.mark.parametrize("data", [torch.zeros(1, 1), torch.zeros(3, 4, 5), torch.zeros(17)])
def test_featurewise_activation_raises_if_wrong_shape(data):
    variables = get_variables()
    model = FeaturewiseActivation(variables=variables, activation_for_continuous=Sigmoid)
    with pytest.raises(ValueError):
        model(data)

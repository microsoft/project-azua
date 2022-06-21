import pytest
import torch

from azua.models.marginal_vaes import MarginalVAEs
from azua.datasets.variables import Variable, Variables


@pytest.fixture(scope="function")
def marginal_vaes(variables, tmpdir_factory):
    return MarginalVAEs(
        model_id="test_model",
        variables=variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        device=torch.device("cpu"),
        **{
            "encoder_layers": [50, 100],
            "latent_dim": 2,
            "decoder_layers": [50, 100],
            "decoder_variances": 0.02,
            "categorical_likelihood_coefficient": 1,
            "kl_coefficient": 0,
            "metadata_filepath": None,
        }
    )


@pytest.fixture(scope="function")
def variables():
    return Variables(
        [
            Variable("categorical_input1", True, "categorical", 0.0, 11.0),
            Variable("categorical_input2", True, "categorical", 0, 3),
            Variable("binary_target", False, "binary", 0, 1),
        ]
    )


def test_marginal_vaes_ignores_unobserved_x(marginal_vaes: MarginalVAEs):
    batch_size = 10
    unobserved_rows = [0, 1, 2]
    unobserved_cols = slice(12, 16)

    x = torch.ones((batch_size, marginal_vaes._output_dim))
    mask = torch.ones_like(x)
    mask[unobserved_rows, unobserved_cols] = 0
    output1 = marginal_vaes.encode(x, mask)

    x[unobserved_rows, unobserved_cols] = 100 * x[unobserved_rows, unobserved_cols]
    output2 = marginal_vaes.encode(x, mask)

    assert torch.allclose(output1[0], output2[0])
    assert torch.allclose(output1[1], output2[1])

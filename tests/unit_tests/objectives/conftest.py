import pytest
import torch

from azua.models.partial_vae import PartialVAE
from azua.datasets.variables import Variable, Variables

input_dim = 4
output_dim = 4
set_embedding_dim = 4
embedding_dim = 2
latent_dim = 17


@pytest.fixture(scope="function")
def pvae(variables, tmpdir_factory):
    return PartialVAE(
        model_id="test_model",
        variables=variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        device=torch.device("cpu"),
        input_dim=input_dim,
        output_dim=output_dim,
        latent_dim=latent_dim,
        embedding_dim=embedding_dim,
        set_embedding_dim=set_embedding_dim,
        set_embedding_multiply_weights=True,
        encoding_function="sum",
        metadata_filepath=None,
        encoder_layers=[5],
        decoder_layers=[6],
        decoder_variances=0.2,
        categorical_likelihood_coefficient=1.0,
        kl_coefficient=1.0,
    )


@pytest.fixture(scope="function")
def variables():
    return Variables(
        [
            Variable("binary_input1", True, "binary", 1.0, 3.0),
            Variable("numeric_input", True, "continuous", 3, 13),
            Variable("binary_input2", True, "binary", 1.0, 3.0),
            Variable("numeric_target", False, "continuous", 2, 300),
        ]
    )

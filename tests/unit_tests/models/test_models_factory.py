import pytest

from azua.datasets.variables import Variable, Variables
from azua.models.models_factory import create_set_encoder, create_model
from azua.models.point_net import PointNet, SparsePointNet
from azua.models.transformer_set_encoder import TransformerSetEncoder


@pytest.mark.parametrize(
    "set_encoder_type, kwargs, expected_return_type",
    [
        (
            "default",
            {
                "input_dim": 3,
                "embedding_dim": 2,
                "set_embedding_dim": 19,
                "metadata": None,
                "device": "cpu",
                "multiply_weights": True,
            },
            PointNet,
        ),
        (
            "sparse",
            {
                "input_dim": 3,
                "embedding_dim": 2,
                "set_embedding_dim": 19,
                "metadata": None,
                "device": "cpu",
                "multiply_weights": True,
            },
            SparsePointNet,
        ),
        (
            "transformer",
            {
                "input_dim": 3,
                "embedding_dim": 2,
                "set_embedding_dim": 19,
                "metadata": None,
                "device": "cpu",
                "multiply_weights": True,
                "num_heads": 2,
                "num_blocks": 3,
                "num_seed_vectors": 4,
            },
            TransformerSetEncoder,
        ),
    ],
)
def test_create_set_encoder(set_encoder_type, kwargs, expected_return_type):
    set_encoder = create_set_encoder(set_encoder_type, kwargs)
    assert isinstance(set_encoder, expected_return_type)


def test_create_model(tmpdir):

    variables = Variables([Variable("test", True, "continuous", lower=0, upper=1)])

    model_config_dict = {
        "embedding_dim": 10,
        "set_embedding_dim": 20,
        "set_embedding_multiply_weights": True,
        "latent_dim": 10,
        "encoder_layers": [500, 200],
        "decoder_layers": [50, 100],
        "encoding_function": "sum",
        "decoder_variances": 0.02,
        "random_seed": [0],
        "categorical_likelihood_coefficient": 1,
        "kl_coefficient": 1,
        "variance_autotune": False,
        "metadata_filepath": None,
    }

    pvae = create_model(
        model_name="pvae", models_dir=tmpdir, variables=variables, device="cpu", model_config_dict=model_config_dict
    )

    assert pvae.name() == "pvae"

import pytest
import torch

from azua.models.transformer_pvae import (
    Transformer,
    TransformerDecoder,
    TransformerEncoder,
    TransformerPVAE,
    TransformerVAE,
)
from azua.datasets.variables import Variables


@pytest.mark.parametrize(
    "use_isab, num_inducing_points",
    [(False, None), (True, 32)],
)
def test_transformer(use_isab, num_inducing_points):
    transformer_embedding_dim = 64
    module = Transformer(
        transformer_embedding_dim=transformer_embedding_dim,
        num_heads=4,
        num_blocks=4,
        use_isab=use_isab,
        num_inducing_points=num_inducing_points,
        multihead_init_type="kaiming",
        use_layer_norm=True,
        elementwise_transform_type="single",
    )

    batch_size = 128
    input_dim = 16
    data = torch.ones((batch_size, input_dim, transformer_embedding_dim))

    output = module(data=data)
    assert output.shape == (batch_size, input_dim, transformer_embedding_dim)


def test_transformer_encoder():
    transformer_embedding_dim = 64
    module = TransformerEncoder(
        transformer_embedding_dim=transformer_embedding_dim,
        num_heads=4,
        num_blocks=4,
        use_isab=False,
        num_inducing_points=None,
        multihead_init_type="kaiming",
        use_layer_norm=True,
        elementwise_transform_type="single",
    )

    batch_size = 128
    input_dim = 16
    data = torch.ones((batch_size, input_dim * transformer_embedding_dim))

    mean, logvar = module(data=data)
    assert mean.shape == (batch_size, input_dim * transformer_embedding_dim)
    assert logvar.shape == (batch_size, input_dim * transformer_embedding_dim)


def test_transformer_decoder():
    input_dim = 16
    transformer_embedding_dim = 64
    module = TransformerDecoder(
        transformer_embedding_dim=transformer_embedding_dim,
        num_heads=4,
        num_blocks=4,
        use_isab=False,
        num_inducing_points=None,
        multihead_init_type="kaiming",
        use_layer_norm=True,
        elementwise_transform_type="single",
    )

    batch_size = 128
    data = torch.ones((batch_size, input_dim * transformer_embedding_dim))

    mean, logvar = module(data=data)
    assert mean.shape == (batch_size, input_dim)
    assert logvar.shape == (batch_size, input_dim)


def test_transformer_vae(tmpdir_factory):
    input_dim = 16
    transformer_embedding_dim = 64
    variables_dict = {
        "variables": [
            {"id": i, "query": True, "type": "binary", "name": f"Column {i}", "lower": 0, "upper": 1}
            for i in range(input_dim)
        ],
        "metadata_variables": [],
        "used_cols": list(range(input_dim)),
    }
    variables = Variables.create_from_dict(variables_dict)

    model = TransformerVAE(
        model_id="model_id",
        variables=variables,
        save_dir=tmpdir_factory.mktemp("model_dir"),
        device="cpu",
        categorical_likelihood_coefficient=1.0,
        kl_coefficient=1.0,
        transformer_embedding_dim=transformer_embedding_dim,
        num_heads=4,
        num_blocks=4,
        use_isab=False,
        num_inducing_points=None,
        multihead_init_type="kaiming",
        use_layer_norm=True,
        elementwise_transform_type="single",
    )

    batch_size = 128
    data = torch.ones((batch_size, input_dim * transformer_embedding_dim))

    (decoder_mean, decoder_logvar), samples, (encoder_mean, encoder_logvar) = model.reconstruct(
        data, sample=True, count=1
    )
    assert decoder_mean.shape == (batch_size, input_dim)
    assert decoder_logvar.shape == (batch_size, input_dim)
    assert samples.shape == (batch_size, input_dim * transformer_embedding_dim)
    assert encoder_mean.shape == (batch_size, input_dim * transformer_embedding_dim)
    assert encoder_logvar.shape == (batch_size, input_dim * transformer_embedding_dim)

    count = 10
    (decoder_mean, decoder_logvar), samples, (encoder_mean, encoder_logvar) = model.reconstruct(
        data, sample=True, count=count
    )
    assert decoder_mean.shape == (count, batch_size, input_dim)
    assert decoder_logvar.shape == (count, batch_size, input_dim)
    assert samples.shape == (count, batch_size, input_dim * transformer_embedding_dim)
    assert encoder_mean.shape == (batch_size, input_dim * transformer_embedding_dim)
    assert encoder_logvar.shape == (batch_size, input_dim * transformer_embedding_dim)


def test_transformer_pvae(tmpdir_factory):
    input_dim = 16
    transformer_embedding_dim = 64
    variables_dict = {
        "variables": [
            {"id": i, "query": True, "type": "binary", "name": f"Column {i}", "lower": 0, "upper": 1}
            for i in range(input_dim)
        ],
        "metadata_variables": [],
        "used_cols": list(range(input_dim)),
    }
    variables = Variables.create_from_dict(variables_dict)

    model = TransformerPVAE(
        model_id="model_id",
        variables=variables,
        save_dir=tmpdir_factory.mktemp("model_dir"),
        device="cpu",
        categorical_likelihood_coefficient=1.0,
        kl_coefficient=1.0,
        embedding_dim=8,
        transformer_embedding_dim=transformer_embedding_dim,
        num_heads=4,
        num_blocks=4,
        use_isab=False,
        num_inducing_points=None,
        multihead_init_type="kaiming",
        use_layer_norm=True,
        elementwise_transform_type="single",
    )

    batch_size = 128
    data = torch.ones((batch_size, input_dim))
    mask = torch.ones((batch_size, input_dim))

    (decoder_mean, decoder_logvar), samples, (encoder_mean, encoder_logvar) = model.reconstruct(
        data, mask, sample=True, count=1
    )
    assert decoder_mean.shape == (batch_size, input_dim)
    assert decoder_logvar.shape == (batch_size, input_dim)
    assert samples.shape == (batch_size, input_dim * transformer_embedding_dim)
    assert encoder_mean.shape == (batch_size, input_dim * transformer_embedding_dim)
    assert encoder_logvar.shape == (batch_size, input_dim * transformer_embedding_dim)

    count = 10
    (decoder_mean, decoder_logvar), samples, (encoder_mean, encoder_logvar) = model.reconstruct(
        data, mask, sample=True, count=count
    )
    assert decoder_mean.shape == (count, batch_size, input_dim)
    assert decoder_logvar.shape == (count, batch_size, input_dim)
    assert samples.shape == (count, batch_size, input_dim * transformer_embedding_dim)
    assert encoder_mean.shape == (batch_size, input_dim * transformer_embedding_dim)
    assert encoder_logvar.shape == (batch_size, input_dim * transformer_embedding_dim)

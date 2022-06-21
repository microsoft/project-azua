import numpy as np
import pytest
import torch

from azua.models.vae_mixed import VAEMixed
from azua.datasets.variables import Variable, Variables


@pytest.fixture(scope="function")
def model_config():
    model_config = {
        "random_seed": 0,
        "marginal_encoder_layers": [1],
        "marginal_latent_dim": 4,
        "marginal_decoder_layers": [1],
        "marginal_decoder_variances": 1.0,
        "marginal_categorical_likelihood_coefficient": 0,
        "marginal_kl_coefficient": 1,
        "dep_embedding_dim": 1,
        "dep_set_embedding_dim": 1,
        "dep_set_embedding_multiply_weights": True,
        "dep_encoding_function": "sum",
        "dep_encoder_layers": [1],
        "dep_latent_dim": 5,
        "dep_decoder_layers": [1],
        "dep_decoder_variances": 1.0,
        "dep_categorical_likelihood_coefficient": 1,
        "dep_kl_coefficient": 1,
        "dep_variance_autotune": False,
        "dep_metadata_filepath": None,
    }
    return model_config


def test_split_configs():
    model_hypers = {
        "random_seed": [0],
        "marginal_encoder_layers": [50, 100],
        "dep_encoder_layers": [500, 200],
    }

    train_hypers = {
        "dep_epochs": 1000,
        "marginal_epochs": 1001,
        "max_p_train_dropout": 0.99,
    }

    marginal_model_hypers = {
        "random_seed": [0],
        "encoder_layers": [50, 100],
    }

    dep_model_hypers = {
        "random_seed": [0],
        "encoder_layers": [500, 200],
    }

    marginal_train_hypers = {
        "epochs": 1001,
        "max_p_train_dropout": 0.99,
    }

    dep_train_hypers = {
        "epochs": 1000,
        "max_p_train_dropout": 0.99,
    }

    assert VAEMixed._split_configs(model_hypers) == (marginal_model_hypers, dep_model_hypers)
    assert VAEMixed._split_configs(train_hypers) == (marginal_train_hypers, dep_train_hypers)


@pytest.mark.parametrize("batch_size, average", [(1, True), (1, False), (3, True), (3, False)])
def test_impute_one_sample(batch_size, average, model_config, tmpdir_factory):
    variables = Variables(
        [
            Variable("binary_input", True, "binary", 0, 1),
            Variable("continuous_target", False, "continuous", 0.0, 10.0),
        ]
    )
    input_dim = 2
    assert input_dim == variables.num_processed_cols
    output_dim = input_dim

    model = VAEMixed.create(
        model_id="model_id",
        variables=variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        model_config_dict=model_config,
        device="cpu",
    )

    data = np.ones((batch_size, input_dim))
    mask = np.ones((batch_size, input_dim))

    impute_config_dict = {"sample_count": 1, "batch_size": 100}
    imputed = model.impute(data=data, mask=mask, impute_config_dict=impute_config_dict, average=average)

    assert isinstance(imputed, np.ndarray)
    if average:
        assert imputed.shape == (batch_size, output_dim)
    else:
        assert imputed.shape == (1, batch_size, output_dim)


@pytest.mark.parametrize("batch_size, average", [(1, True), (1, False), (3, True), (3, False)])
def test_impute_multiple_samples(batch_size, average, model_config, tmpdir_factory):
    variables = Variables(
        [
            Variable("binary_input", True, "binary", 0, 1),
            Variable("continuous_target", False, "continuous", 0.0, 10.0),
        ]
    )
    input_dim = 2
    assert input_dim == variables.num_processed_cols
    output_dim = input_dim

    model = VAEMixed.create(
        model_id="model_id",
        variables=variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        model_config_dict=model_config,
        device="cpu",
    )

    data = np.ones((batch_size, input_dim))
    mask = np.ones((batch_size, input_dim))

    impute_config_dict = {"sample_count": 2, "batch_size": 100}
    imputed = model.impute(data=data, mask=mask, impute_config_dict=impute_config_dict, average=average)

    assert isinstance(imputed, np.ndarray)
    if average:
        assert imputed.shape == (batch_size, output_dim)
    else:
        assert imputed.shape == (2, batch_size, output_dim)


@pytest.mark.parametrize("average", [True, False])
def test_impute_vamp_prior(average, model_config, tmpdir_factory):
    variables = Variables(
        [
            Variable("binary_input", True, "binary", 0, 1),
            Variable("continuous_target", False, "continuous", 0.0, 10.0),
        ]
    )
    input_dim = 2
    batch_size = 3
    assert input_dim == variables.num_processed_cols
    output_dim = input_dim

    model = VAEMixed.create(
        model_id="model_id",
        variables=variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        model_config_dict=model_config,
        device="cpu",
    )

    data = np.ones((batch_size, input_dim))
    mask = np.ones((batch_size, input_dim))
    # Create unobserved row
    mask[0, :] = 0
    vamp_prior_data = (data[0:2, :], mask[0:2, :])

    impute_config_dict = {"sample_count": 2, "batch_size": 100}
    imputed = model.impute(
        data=data,
        mask=mask,
        impute_config_dict=impute_config_dict,
        average=average,
        vamp_prior_data=vamp_prior_data,
    )

    assert isinstance(imputed, np.ndarray)
    if average:
        assert imputed.shape == (batch_size, output_dim)
    else:
        assert imputed.shape == (2, batch_size, output_dim)


@pytest.mark.parametrize("batch_size, sample_count", [(1, 1), (1, 2), (3, 1), (3, 2)])
@pytest.mark.parametrize("preserve_data", [True, False])
def test_impute_processed_batch(batch_size, sample_count, preserve_data, model_config, tmpdir_factory):
    variables = Variables(
        [
            Variable("binary_input", True, "binary", 0, 1),
            Variable("continuous_target", False, "continuous", 0.0, 10.0),
        ]
    )
    input_dim = 2
    assert input_dim == variables.num_processed_cols
    output_dim = input_dim

    model = VAEMixed.create(
        model_id="model_id",
        variables=variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        model_config_dict=model_config,
        device="cpu",
    )

    data = torch.ones((batch_size, input_dim), dtype=torch.float)
    mask = torch.ones((batch_size, input_dim), dtype=torch.float)

    sample_count = 1
    imputed = model.impute_processed_batch(
        data=data,
        mask=mask,
        sample_count=sample_count,
        vamp_prior_data=None,
        preserve_data=preserve_data,
    )

    assert isinstance(imputed, torch.Tensor)
    assert imputed.shape == (sample_count, batch_size, output_dim)


@pytest.mark.parametrize("batch_size, sample", [(1, True), (1, False), (3, True), (3, False)])
def test_reconstruct_one_sample(batch_size, sample, model_config, tmpdir_factory):
    variables = Variables(
        [
            Variable("binary_input", True, "binary", 0, 1),
            Variable("continuous_target", False, "continuous", 0.0, 10.0),
        ]
    )
    input_dim = 2
    latent_dim = 5
    assert input_dim == variables.num_processed_cols
    output_dim = input_dim

    model = VAEMixed.create(
        model_id="model_id",
        variables=variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        model_config_dict=model_config,
        device="cpu",
    )

    data = torch.ones((batch_size, input_dim), dtype=torch.float)
    mask = torch.ones((batch_size, input_dim), dtype=torch.float)

    count = 1
    (
        (decoder_mean, decoder_logvar),
        samples,
        (encoder_mean, encoder_logvar),
    ) = model.reconstruct(data=data, mask=mask, sample=sample, count=count)

    assert isinstance(decoder_mean, torch.Tensor)
    assert decoder_mean.shape == (batch_size, output_dim)
    assert isinstance(decoder_logvar, torch.Tensor)
    assert decoder_logvar.shape == (batch_size, output_dim)

    assert isinstance(samples, torch.Tensor)
    assert samples.shape == (batch_size, latent_dim)

    assert isinstance(encoder_mean, torch.Tensor)
    assert encoder_mean.shape == (batch_size, latent_dim)
    assert isinstance(encoder_logvar, torch.Tensor)
    assert encoder_logvar.shape == (batch_size, latent_dim)


@pytest.mark.parametrize("batch_size, sample", [(1, True), (1, False), (3, True), (3, False)])
def test_reconstruct_multiple_samples(batch_size, sample, model_config, tmpdir_factory):
    variables = Variables(
        [
            Variable("binary_input", True, "binary", 0, 1),
            Variable("continuous_target", False, "continuous", 0.0, 10.0),
        ]
    )
    input_dim = 2
    latent_dim = 5
    assert input_dim == variables.num_processed_cols
    output_dim = input_dim

    model = VAEMixed.create(
        model_id="model_id",
        variables=variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        model_config_dict=model_config,
        device="cpu",
    )

    data = torch.ones((batch_size, input_dim), dtype=torch.float)
    mask = torch.ones((batch_size, input_dim), dtype=torch.float)

    count = 4
    (
        (decoder_mean, decoder_logvar),
        samples,
        (encoder_mean, encoder_logvar),
    ) = model.reconstruct(data=data, mask=mask, sample=sample, count=count)

    assert isinstance(decoder_mean, torch.Tensor)
    assert decoder_mean.shape == (count, batch_size, output_dim)
    assert isinstance(decoder_logvar, torch.Tensor)
    assert decoder_logvar.shape == (count, batch_size, output_dim)

    assert isinstance(samples, torch.Tensor)
    assert samples.shape == (count, batch_size, latent_dim)

    assert isinstance(encoder_mean, torch.Tensor)
    assert encoder_mean.shape == (batch_size, latent_dim)
    assert isinstance(encoder_logvar, torch.Tensor)
    assert encoder_logvar.shape == (batch_size, latent_dim)


def test_reconstruct_no_mask(model_config, tmpdir_factory):
    variables = Variables(
        [
            Variable("binary_input", True, "binary", 0, 1),
            Variable("continuous_target", False, "continuous", 0.0, 10.0),
        ]
    )
    input_dim = 2
    latent_dim = 5
    assert input_dim == variables.num_processed_cols
    output_dim = input_dim

    model = VAEMixed.create(
        model_id="model_id",
        variables=variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        model_config_dict=model_config,
        device="cpu",
    )

    batch_size = 3
    data = torch.ones((batch_size, input_dim), dtype=torch.float)
    mask = None

    count = 4
    (
        (decoder_mean, decoder_logvar),
        samples,
        (encoder_mean, encoder_logvar),
    ) = model.reconstruct(data=data, mask=mask, sample=True, count=count)

    assert isinstance(decoder_mean, torch.Tensor)
    assert decoder_mean.shape == (count, batch_size, output_dim)
    assert isinstance(decoder_logvar, torch.Tensor)
    assert decoder_logvar.shape == (count, batch_size, output_dim)

    assert isinstance(samples, torch.Tensor)
    assert samples.shape == (count, batch_size, latent_dim)

    assert isinstance(encoder_mean, torch.Tensor)
    assert encoder_mean.shape == (batch_size, latent_dim)
    assert isinstance(encoder_logvar, torch.Tensor)
    assert encoder_logvar.shape == (batch_size, latent_dim)

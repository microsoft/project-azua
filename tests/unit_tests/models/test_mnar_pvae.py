import glob
import os

import numpy as np
import pytest
import torch

from azua.models.mnar_pvae import MNARPartialVAE
from azua.datasets.variables import Variable, Variables
from azua.experiment.azua_context import AzuaContext
from azua.run_experiment import run_experiment
from azua.utils.io_utils import read_json_as, save_json

input_dim = 3
output_dim = 3
set_embedding_dim = 4
embedding_dim = 2
latent_dim = 17


@pytest.fixture(scope="function")
def mnar_pvae(variables, tmpdir_factory):
    return MNARPartialVAE(
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
        mask_net_config={"decoder_layers": [10], "mask_net_coefficient": 1.0, "latent_connection": True},
        prior_net_config={"use_prior_net_to_train": True, "encoder_layers": [], "use_prior_net_to_impute": True},
    )


@pytest.fixture(scope="function")
def model_config():
    model_config = {
        "embedding_dim": 20,
        "set_embedding_dim": 20,
        "set_embedding_multiply_weights": True,
        "latent_dim": 2,
        "encoder_layers": [10],
        "decoder_layers": [10],
        "non_linearity": "Tanh",
        "activation_for_continuous": "Identity",
        "init_method": "xavier_uniform",
        "encoding_function": "sum",
        "metadata_filepath": None,
        "decoder_variances": 0.01,
        "random_seed": [1],
        "categorical_likelihood_coefficient": 1.0,
        "kl_coefficient": 1.0,
        "variance_autotune": False,
        "use_importance_sampling": True,
        "squash_input": False,
        "mask_net_config": {"decoder_layers": [10], "mask_net_coefficient": 1.0, "latent_connection": True},
        "prior_net_config": {
            "use_prior_net_to_train": True,
            "encoder_layers": [],
            "use_prior_net_to_impute": True,
            "degenerate_prior": "mask",
        },
    }
    return model_config


@pytest.fixture(scope="function")
def variables():
    return Variables(
        [
            Variable("binary_input", True, "binary", 1.0, 3.0),
            Variable("numeric_input", True, "continuous", 3, 13),
            Variable("numeric_target", False, "continuous", 2, 300),
        ]
    )


def good_data():
    # Should pass validations with variables fixture defined in this file
    return np.array([[0, 3, 40], [1, 4, 16]], dtype=float)


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("average", [True, False])
@pytest.mark.parametrize("auxiliary_var", [True, False])
def test_impute_one_sample(batch_size, average, auxiliary_var, model_config, tmpdir_factory):

    input_dim = 2
    latent_dim = 5

    if auxiliary_var:
        variables = Variables(
            [Variable("continuous_target", False, "continuous", 0.0, 10.0)],
            [Variable("binary_input", True, "binary", 0, 1)],
        )
        aux_dim = 1
    else:
        variables = Variables(
            [
                Variable("binary_input", True, "binary", 0, 1),
                Variable("continuous_target", False, "continuous", 0.0, 10.0),
            ]
        )
        aux_dim = 0

    output_dim = variables.num_processed_cols - aux_dim
    assert input_dim == variables.num_unprocessed_cols
    assert aux_dim == variables.num_processed_cols - variables.num_processed_non_aux_cols

    model_config["latent_dim"] = latent_dim
    model_config["input_dim"] = input_dim
    model_config["output_dim"] = output_dim

    model = MNARPartialVAE.create(
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


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("average", [True, False])
@pytest.mark.parametrize("auxiliary_var", [True, False])
def test_impute_multiple_samples(batch_size, average, auxiliary_var, model_config, tmpdir_factory):
    input_dim = 2
    latent_dim = 5

    if auxiliary_var:
        variables = Variables(
            [Variable("continuous_target", False, "continuous", 0.0, 10.0)],
            [Variable("binary_input", True, "binary", 0, 1)],
        )
        aux_dim = 1

    else:
        variables = Variables(
            [
                Variable("binary_input", True, "binary", 0, 1),
                Variable("continuous_target", False, "continuous", 0.0, 10.0),
            ]
        )
        aux_dim = 0

    output_dim = variables.num_processed_cols - aux_dim
    assert input_dim == variables.num_unprocessed_cols
    assert aux_dim == variables.num_processed_cols - variables.num_processed_non_aux_cols

    model_config["latent_dim"] = latent_dim
    model_config["input_dim"] = input_dim
    model_config["output_dim"] = output_dim

    model = MNARPartialVAE.create(
        model_id="model_id",
        variables=variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        model_config_dict=model_config,
        device="cpu",
    )

    data = np.ones((batch_size, input_dim))
    mask = np.ones((batch_size, input_dim))

    impute_config_dict = {"sample_count": 4, "batch_size": 100}
    imputed = model.impute(data=data, mask=mask, impute_config_dict=impute_config_dict, average=average)

    assert isinstance(imputed, np.ndarray)
    if average:
        assert imputed.shape == (batch_size, output_dim)
    else:
        assert imputed.shape == (4, batch_size, output_dim)


@pytest.mark.parametrize("average, auxiliary_var", [(True, False), (False, False), (True, True), (False, True)])
def test_impute_vamp_prior(average, auxiliary_var, model_config, tmpdir_factory):
    input_dim = 2
    latent_dim = 5
    batch_size = 3

    # when auxiliary_var is true, we create an aux col (in this case, the binary_input)
    if auxiliary_var:
        variables = Variables(
            [Variable("continuous_target", False, "continuous", 0.0, 10.0)],
            [Variable("binary_input", True, "binary", 0, 1)],
        )
        aux_dim = 1

    else:
        variables = Variables(
            [
                Variable("binary_input", True, "binary", 0, 1),
                Variable("continuous_target", False, "continuous", 0.0, 10.0),
            ]
        )
        aux_dim = 0

    output_dim = variables.num_processed_cols - aux_dim
    assert input_dim == variables.num_unprocessed_cols
    assert aux_dim == variables.num_processed_cols - variables.num_processed_non_aux_cols

    model_config["latent_dim"] = latent_dim
    model_config["input_dim"] = input_dim
    model_config["output_dim"] = output_dim

    model = MNARPartialVAE.create(
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


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("sample", [True, False])
@pytest.mark.parametrize("auxiliary_var", [True, False])
def test_reconstruct_one_sample(batch_size, sample, auxiliary_var, model_config, tmpdir_factory):
    input_dim = 2
    latent_dim = 5

    if auxiliary_var:
        variables = Variables(
            [Variable("continuous_target", False, "continuous", 0.0, 10.0)],
            [Variable("binary_input", True, "binary", 0, 1)],
        )
        aux_dim = 1

    else:
        variables = Variables(
            [
                Variable("binary_input", True, "binary", 0, 1),
                Variable("continuous_target", False, "continuous", 0.0, 10.0),
            ]
        )
        aux_dim = 0

    output_dim = variables.num_processed_cols - aux_dim
    assert input_dim == variables.num_unprocessed_cols
    assert aux_dim == variables.num_processed_cols - variables.num_processed_non_aux_cols

    model_config["latent_dim"] = latent_dim
    model_config["input_dim"] = input_dim
    model_config["output_dim"] = output_dim

    model = MNARPartialVAE.create(
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


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("sample", [True, False])
@pytest.mark.parametrize("auxiliary_var", [True, False])
def test_reconstruct_multiple_samples(batch_size, sample, auxiliary_var, model_config, tmpdir_factory):
    input_dim = 2
    latent_dim = 5

    if auxiliary_var:
        variables = Variables(
            [Variable("continuous_target", False, "continuous", 0.0, 10.0)],
            [Variable("binary_input", True, "binary", 0, 1)],
        )
        aux_dim = 1

    else:
        variables = Variables(
            [
                Variable("binary_input", True, "binary", 0, 1),
                Variable("continuous_target", False, "continuous", 0.0, 10.0),
            ]
        )
        aux_dim = 0

    output_dim = variables.num_processed_cols - aux_dim
    assert input_dim == variables.num_unprocessed_cols
    assert aux_dim == variables.num_processed_cols - variables.num_processed_non_aux_cols

    model_config["latent_dim"] = latent_dim
    model_config["input_dim"] = input_dim
    model_config["output_dim"] = output_dim

    model = MNARPartialVAE.create(
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


@pytest.mark.parametrize("auxiliary_var", [True, False])
def test_reconstruct_no_mask(auxiliary_var, model_config, tmpdir_factory):
    input_dim = 2
    latent_dim = 5
    batch_size = 3

    if auxiliary_var:
        variables = Variables(
            [Variable("continuous_target", False, "continuous", 0.0, 10.0)],
            [Variable("binary_input", True, "binary", 0, 1)],
        )
        aux_dim = 1

    else:
        variables = Variables(
            [
                Variable("binary_input", True, "binary", 0, 1),
                Variable("continuous_target", False, "continuous", 0.0, 10.0),
            ]
        )
        aux_dim = 0

    output_dim = variables.num_processed_cols - aux_dim
    assert input_dim == variables.num_unprocessed_cols
    assert aux_dim == variables.num_processed_cols - variables.num_processed_non_aux_cols

    model_config["latent_dim"] = latent_dim
    model_config["input_dim"] = input_dim
    model_config["output_dim"] = output_dim

    model = MNARPartialVAE.create(
        model_id="model_id",
        variables=variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        model_config_dict=model_config,
        device="cpu",
    )

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


@pytest.mark.parametrize("batch_size", [1, 13])
@pytest.mark.parametrize("num_importance_samples", [1, 7])
@pytest.mark.parametrize("evaluate_imputation", [True, False])
@pytest.mark.parametrize("auxiliary_var", [True, False])
def test_get_log_importance_weights(
    batch_size, num_importance_samples, evaluate_imputation, auxiliary_var, model_config, tmpdir_factory
):
    input_dim = 2
    latent_dim = 5

    if auxiliary_var:
        variables = Variables(
            [Variable("continuous_target", False, "continuous", 0.0, 10.0)],
            [Variable("binary_input", True, "binary", 0, 1)],
        )
        aux_dim = 1
    else:
        variables = Variables(
            [
                Variable("binary_input", True, "binary", 0, 1),
                Variable("continuous_target", False, "continuous", 0.0, 10.0),
            ]
        )
        aux_dim = 0

    output_dim = variables.num_processed_cols - aux_dim
    assert input_dim == variables.num_unprocessed_cols
    assert aux_dim == variables.num_processed_cols - variables.num_processed_non_aux_cols

    model_config["latent_dim"] = latent_dim
    model_config["input_dim"] = input_dim
    model_config["output_dim"] = output_dim

    model = MNARPartialVAE.create(
        model_id="model_id",
        variables=variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        model_config_dict=model_config,
        device="cpu",
    )

    data = torch.ones((batch_size, input_dim), dtype=torch.float)
    observed_mask = torch.ones((batch_size, input_dim), dtype=torch.float)
    target_mask = torch.zeros((batch_size, input_dim), dtype=torch.float)
    observed_mask[0, 0] = 0
    target_mask[0, 0] = 1

    log_importance_weights = model._get_log_importance_weights(
        data=data,
        observed_mask=observed_mask,
        target_mask=target_mask,
        evaluate_imputation=evaluate_imputation,
        num_importance_samples=num_importance_samples,
    )
    assert isinstance(log_importance_weights, torch.Tensor)
    assert log_importance_weights.shape == (num_importance_samples, batch_size)


@pytest.mark.parametrize("observed_mask, target_mask", [(None, None), (torch.ones((13, 2), dtype=torch.float), None)])
def test_get_log_importance_weights_no_mask_raises_error(observed_mask, target_mask, model_config, tmpdir_factory):
    variables = Variables(
        [
            Variable("binary_input", True, "binary", 0, 1),
            Variable("continuous_target", False, "continuous", 0.0, 10.0),
        ]
    )
    input_dim = 2
    assert input_dim == variables.num_unprocessed_cols
    output_dim = variables.num_processed_cols
    model_config["input_dim"] = input_dim
    model_config["output_dim"] = output_dim

    model = MNARPartialVAE.create(
        model_id="model_id",
        variables=variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        model_config_dict=model_config,
        device="cpu",
    )

    data = torch.ones((13, input_dim), dtype=torch.float)

    with pytest.raises(AssertionError):
        model._get_log_importance_weights(
            data=data,
            observed_mask=observed_mask,
            target_mask=target_mask,
            evaluate_imputation=False,
            num_importance_samples=7,
        )


@pytest.mark.parametrize("num_rows", [1, 13])
@pytest.mark.parametrize("batch_size", [1, 5])
@pytest.mark.parametrize("num_importance_samples", [1, 7])
@pytest.mark.parametrize("evaluate_imputation", [True, False])
@pytest.mark.parametrize("auxiliary_var", [True, False])
def test_get_marginal_log_likelihood(
    num_rows, batch_size, num_importance_samples, evaluate_imputation, auxiliary_var, model_config, tmpdir_factory
):
    input_dim = 2
    latent_dim = 5

    if auxiliary_var:
        variables = Variables(
            [Variable("continuous_target", False, "continuous", 0.0, 10.0)],
            [Variable("binary_input", True, "binary", 0, 1)],
        )
        aux_dim = 1
    else:
        variables = Variables(
            [
                Variable("binary_input", True, "binary", 0, 1),
                Variable("continuous_target", False, "continuous", 0.0, 10.0),
            ]
        )
        aux_dim = 0

    output_dim = variables.num_processed_cols - aux_dim
    assert input_dim == variables.num_unprocessed_cols
    assert aux_dim == variables.num_processed_cols - variables.num_processed_non_aux_cols

    model_config["latent_dim"] = latent_dim
    model_config["input_dim"] = input_dim
    model_config["output_dim"] = output_dim

    model = MNARPartialVAE.create(
        model_id="model_id",
        variables=variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        model_config_dict=model_config,
        device="cpu",
    )

    data = np.ones((num_rows, input_dim), dtype=float)
    observed_mask = np.ones((num_rows, input_dim), dtype=bool)
    target_mask = np.zeros((num_rows, input_dim), dtype=bool)
    observed_mask[0, 0] = 0
    target_mask[0, 0] = 1

    mll = model.get_marginal_log_likelihood(
        impute_config={"batch_size": batch_size},
        data=data,
        observed_mask=observed_mask,
        target_mask=target_mask,
        evaluate_imputation=evaluate_imputation,
        num_importance_samples=num_importance_samples,
    )
    assert isinstance(mll, float)


@pytest.mark.parametrize("auxiliary_var", [True, False])
def test_get_marginal_log_likelihood_categorical_variables(model_config, auxiliary_var, tmpdir_factory):
    num_rows = 13
    num_importance_samples = 7
    if auxiliary_var:
        variables = Variables(
            [Variable("continuous_target", False, "continuous", 0.0, 10.0)],
            [Variable("categorical_input", True, "categorical", 0, 2)],
        )
        aux_dim = 3

    else:
        variables = Variables(
            [
                Variable("categorical_input", True, "categorical", 0, 2),
                Variable("continuous_target", False, "continuous", 0.0, 10.0),
            ]
        )
        aux_dim = 0

    input_dim = 2
    output_dim = variables.num_processed_cols - aux_dim
    assert variables.num_processed_cols == 4
    assert variables.num_unprocessed_cols == input_dim
    assert aux_dim == variables.num_processed_cols - variables.num_processed_non_aux_cols
    model_config["input_dim"] = variables.num_processed_cols
    model_config["output_dim"] = output_dim

    model = MNARPartialVAE.create(
        model_id="model_id",
        variables=variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        model_config_dict=model_config,
        device="cpu",
    )

    data = np.ones((num_rows, input_dim), dtype=float)
    observed_mask = np.ones((num_rows, input_dim), dtype=bool)
    target_mask = np.zeros((num_rows, input_dim), dtype=bool)
    observed_mask[0, 0] = 0
    target_mask[0, 0] = 1

    mll = model.get_marginal_log_likelihood(
        impute_config={"batch_size": 1},
        data=data,
        observed_mask=observed_mask,
        target_mask=target_mask,
        evaluate_imputation=True,
        num_importance_samples=num_importance_samples,
    )
    assert isinstance(mll, float)


@pytest.mark.parametrize("num_rows", [1, 13])
@pytest.mark.parametrize("num_importance_samples", [1, 7])
@pytest.mark.parametrize("auxiliary_var", [True, False])
def test_get_marginal_log_likelihood_no_mask(
    num_rows, num_importance_samples, auxiliary_var, model_config, tmpdir_factory
):
    input_dim = 2

    if auxiliary_var:
        variables = Variables(
            [Variable("continuous_target", False, "continuous", 0.0, 10.0)],
            [Variable("binary_input", True, "binary", 0, 1)],
        )
        aux_dim = 1
    else:
        variables = Variables(
            [
                Variable("binary_input", True, "binary", 0, 1),
                Variable("continuous_target", False, "continuous", 0.0, 10.0),
            ]
        )
        aux_dim = 0

    output_dim = variables.num_processed_cols - aux_dim
    assert input_dim == variables.num_unprocessed_cols
    assert aux_dim == variables.num_processed_cols - variables.num_processed_non_aux_cols

    model_config["input_dim"] = input_dim
    model_config["output_dim"] = output_dim

    model = MNARPartialVAE.create(
        model_id="model_id",
        variables=variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        model_config_dict=model_config,
        device="cpu",
    )

    data = np.ones((num_rows, input_dim), dtype=float)

    mll = model.get_marginal_log_likelihood(
        impute_config={"batch_size": 1},
        data=data,
        observed_mask=None,
        target_mask=None,
        evaluate_imputation=False,
        num_importance_samples=num_importance_samples,
    )
    assert isinstance(mll, float)


@pytest.mark.parametrize("num_rows, num_importance_samples", [(1, 1), (13, 1), (1, 7), (13, 7)])
def test_get_marginal_log_likelihood_no_drop_mask_raises_error(
    num_rows, num_importance_samples, model_config, tmpdir_factory
):
    variables = Variables(
        [
            Variable("binary_input", True, "binary", 0, 1),
            Variable("continuous_target", False, "continuous", 0.0, 10.0),
        ]
    )
    input_dim = 2
    assert input_dim == variables.num_unprocessed_cols
    output_dim = variables.num_processed_cols
    model_config["input_dim"] = input_dim
    model_config["output_dim"] = output_dim

    model = MNARPartialVAE.create(
        model_id="model_id",
        variables=variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        model_config_dict=model_config,
        device="cpu",
    )

    data = np.ones((num_rows, input_dim), dtype=float)

    with pytest.raises(AssertionError):
        model.get_marginal_log_likelihood(
            impute_config={"batch_size": 1},
            data=data,
            observed_mask=None,
            target_mask=None,
            evaluate_imputation=True,
            num_importance_samples=num_importance_samples,
        )
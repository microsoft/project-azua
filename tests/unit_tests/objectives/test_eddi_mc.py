import numpy as np
import pytest
import torch
import torch.distributions as tdist

from azua.objectives.eddi_mc import EDDIMCObjective, EDDIObjective
from azua.utils.torch_utils import set_random_seeds
from azua.datasets.variables import Variable, Variables
from azua.models.partial_vae import PartialVAE

input_dim = 4
output_dim = 4
set_embedding_dim = 4
embedding_dim = 2
latent_dim = 17


def good_data():
    dec_mean = np.array([[1.0, 3.5, 2.0, 26.5], [2.0, 11.8, 1.0, 263.7], [2.0, 7.2, 1.0, 139.3]])
    dec_logvar = np.array([[0.5, 0.0, 0.0, np.log(2.0)], [0.7, -1.0, 0.0, 1.0], [0.2, 0.5, 0.0, -1.5]])
    return dec_mean, dec_logvar


def gaussian_ll(x, mean, logvar):
    return tdist.Normal(mean, np.exp(0.5 * logvar)).log_prob(x)


def compute_entropy(dec_mean, dec_logvar):
    entropy = 0.0
    sample_count = dec_mean.shape[0]
    for j in range(sample_count):
        x = dec_mean[j]
        ll = []
        for k in range(sample_count):
            if j != k:
                ll.append(gaussian_ll(x, dec_mean[k], dec_logvar[k]))
        ll = np.log(np.clip(np.exp(np.asarray(ll, dtype="f")).mean(), 1e-10, 1e10))
        entropy -= ll / sample_count
    return entropy


def test_entropy_mc_approx(pvae):
    dec_mean, dec_logvar = good_data()
    dec_mean_tensor = torch.tensor(dec_mean).unsqueeze(dim=1)  # shape (3, 1, 4)
    dec_logvar_tensor = torch.tensor(dec_logvar).unsqueeze(dim=1)  # shape (3, 1, 4)
    imputed = dec_mean_tensor

    sample_count = 10
    eddi_mc_objective = EDDIMCObjective(pvae, sample_count, use_vamp_prior=False)
    entropy = eddi_mc_objective._entropy_mc_approx(imputed, dec_mean_tensor, dec_logvar_tensor, summed=False)
    # test the shapes and values
    assert entropy.shape == (1, 4)
    entropy_np = []
    for i in [1, 3]:
        entropy_np.append(compute_entropy(dec_mean[:, i], dec_logvar[:, i]))
    entropy_np = np.asarray(entropy_np, dtype="f")

    assert np.abs(entropy.detach().numpy()[0][[1, 3]] - entropy_np).sum() < 1e-3


@pytest.fixture(scope="function")
def variables_no_target():
    return Variables(
        [
            Variable("binary_input1", True, "binary", 1.0, 3.0),
            Variable("numeric_input", True, "continuous", 3, 13),
            Variable("binary_input2", True, "binary", 1.0, 3.0),
            Variable("numeric_input", True, "continuous", 2, 300),
        ]
    )


@pytest.fixture(scope="function")
def pvae_no_target(variables_no_target, tmpdir_factory):
    return PartialVAE(
        model_id="test_model",
        variables=variables_no_target,
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


def test_info_gain_no_target(pvae_no_target):
    data = torch.from_numpy(good_data()[0]).to(torch.float)
    data_mask = torch.ones(data.shape)
    obs_mask = torch.zeros(data.shape)

    sample_count = 100
    eddi_mc_objective = EDDIMCObjective(pvae_no_target, sample_count, use_vamp_prior=False)
    eddi_objective = EDDIObjective(pvae_no_target, sample_count, use_vamp_prior=False)
    set_random_seeds(0)
    reward_mc = eddi_mc_objective._information_gain(data, data_mask, obs_mask, as_array=True, vamp_prior_data=None)
    set_random_seeds(0)
    reward = eddi_objective._information_gain(data, data_mask, obs_mask, as_array=True, vamp_prior_data=None)

    assert np.abs(reward_mc - reward).sum() < 1e-3

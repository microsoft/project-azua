import numpy as np
import pytest
import torch

from azua.objectives.variance import VarianceObjective
from azua.models.transformer_imputer import TransformerImputer


@pytest.fixture(scope="function")
def transformer_imputer(variables, tmpdir_factory):
    return TransformerImputer(
        model_id="test_model",
        variables=variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        device=torch.device("cpu"),
        embedding_dim=3,
        num_heads=1,
        num_blocks=1,
        metadata_filepath=None,
        num_inducing_points=None,
        use_layer_norm=False,
        elementwise_transform_type="single",
        transformer_embedding_dim=2,
        use_isab=False,
        share_readout_weights=True,
        multihead_init_type="xavier",
    )


def test_information_gain_shapes(transformer_imputer):
    input_dim = transformer_imputer.variables.num_processed_cols

    batch_size = 13
    objective = VarianceObjective(model=transformer_imputer, sample_count=3)

    data = torch.zeros(batch_size, input_dim)
    data_mask = torch.ones_like(data)
    obs_mask = torch.zeros_like(data)

    info_gain = objective._information_gain(data=data, obs_mask=obs_mask, data_mask=data_mask, as_array=True)
    assert info_gain.shape == (batch_size, input_dim - 1)


def test_dimension_mangling(transformer_imputer):
    """
    The _information_gain method contains calls to 'repeat_interleave', 'permute', 'reshape'.
    This is a quick check that we haven't got batch and sample dimensions mixed up,
    by seeing that if you insert NaNs in data, you get NaNs in information gain in the expected places.
    """
    input_dim = transformer_imputer.variables.num_processed_cols
    batch_size = 13
    objective = VarianceObjective(model=transformer_imputer, sample_count=3)

    # There is one NaN in each row in the batch. All info gains should be nan
    data = torch.zeros(batch_size, input_dim)
    data_mask = torch.ones_like(data)
    obs_mask = torch.zeros_like(data)
    data[:, 0] = np.nan
    info_gain = objective._information_gain(data=data, obs_mask=obs_mask, data_mask=data_mask, as_array=True)
    assert info_gain.shape == (batch_size, input_dim - 1)  # input_dim - 1 since target dim is not queriable
    assert np.all(np.isnan(info_gain))

    # First row in the batch is NaNs. First row only of info gains should be nan.
    data = torch.zeros(batch_size, input_dim)
    data[0, :] = np.nan
    info_gain = objective._information_gain(data=data, obs_mask=obs_mask, data_mask=data_mask, as_array=True)
    assert info_gain.shape == (batch_size, input_dim - 1)
    assert np.all(np.isnan(info_gain[0, :]))
    assert np.all(np.logical_not(np.isnan(info_gain[1:, :])))

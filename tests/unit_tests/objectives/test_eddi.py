import numpy as np
import torch

from azua.objectives.eddi import EDDIObjective


def test_information_gain_shapes(pvae):
    input_dim = pvae.variables.num_processed_cols

    batch_size = 13
    objective = EDDIObjective(model=pvae, sample_count=3, use_vamp_prior=False)

    data = torch.zeros(batch_size, input_dim)
    data_mask = torch.ones_like(data)
    obs_mask = torch.zeros_like(data)

    info_gain = objective._information_gain(data=data, obs_mask=obs_mask, data_mask=data_mask, as_array=True)
    assert info_gain.shape == (batch_size, input_dim - 1)


def test_dimension_mangling(pvae):
    """
    The _information_gain method contains calls to 'repeat_interleave', 'permute', 'reshape'.
    This is a quick check that we haven't got batch and sample dimensions mixed up,
    by seeing that if you insert NaNs in data, you get NaNs in information gain in the expected places.
    """
    input_dim = pvae.variables.num_processed_cols
    batch_size = 13
    objective = EDDIObjective(model=pvae, sample_count=3, use_vamp_prior=False)

    # There is one NaN in each row in the batch. All info gains should be nan
    data = torch.zeros(batch_size, input_dim)
    data_mask = torch.ones_like(data)
    obs_mask = torch.zeros_like(data)
    data[:, 0] = np.nan
    info_gain = objective._information_gain(data=data, obs_mask=obs_mask, data_mask=data_mask, as_array=True)
    assert info_gain.shape == (batch_size, input_dim - 1)  # input_dim - 1 since target dim is not queriable
    assert np.all(np.isnan(info_gain))

    # First row in the batch is NaNs. First row of info gains should be nan, and no others
    data = torch.zeros(batch_size, input_dim)
    data[0, :] = np.nan
    info_gain = objective._information_gain(data=data, obs_mask=obs_mask, data_mask=data_mask, as_array=True)
    assert info_gain.shape == (batch_size, input_dim - 1)
    assert np.all(np.isnan(info_gain[0, :]))
    assert np.all(np.logical_not(np.isnan(info_gain[1:, :])))

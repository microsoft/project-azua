import numpy as np
import pytest
import torch

from azua.models.transformer_imputer import ParallelLinear, TransformerImputer
from azua.datasets.variables import Variables


def get_variables():
    variables_dict = {
        "variables": [
            {"id": 0, "query": True, "type": "binary", "name": "Column 0", "lower": 0, "upper": 1},
            {"id": 1, "query": True, "type": "binary", "name": "Column 1", "lower": 0, "upper": 1},
            {"id": 2, "query": True, "type": "binary", "name": "Column 2", "lower": 0, "upper": 1},
        ],
        "metadata_variables": [],
        "used_cols": [0, 1, 2],
    }
    return Variables.create_from_dict(variables_dict)


def get_basic_config():
    return {
        "embedding_dim": 3,
        "num_heads": 1,
        "num_blocks": 2,
        "metadata_filepath": None,
        "num_inducing_points": 3,
        "multihead_init_type": "xavier",
        "use_layer_norm": False,
        "elementwise_transform_type": "double",
        "use_isab": False,
        "transformer_embedding_dim": 10,
        "share_readout_weights": True,
    }


def get_model(save_dir, config_updates):
    config = get_basic_config()
    config.update(config_updates)
    return TransformerImputer(
        model_id="blah", save_dir=save_dir, variables=get_variables(), device=torch.device("cpu"), **config
    )


@pytest.mark.parametrize("use_isab,share_readout_weights", [(True, True), (False, True), (False, False), (True, False)])
def test_impute_processed_batch_shape(tmpdir, use_isab, share_readout_weights):
    # Check that 'impute_processed_batch' returns values of expected shape.
    model = get_model(
        save_dir=tmpdir, config_updates={"use_isab": use_isab, "share_readout_weights": share_readout_weights}
    )
    batch_size = 4
    sample_count = 11
    num_cols = 3
    data = torch.zeros((batch_size, num_cols))
    mask = torch.ones_like(data)
    output = model.impute_processed_batch(data, mask, sample_count=sample_count)
    assert output.shape == (sample_count, batch_size, num_cols)


@pytest.mark.parametrize("use_isab", [True, False])
def test_impute_shape(tmpdir, use_isab):
    # Check that 'impute' returns values of expected shape.
    model = get_model(save_dir=tmpdir, config_updates={"use_isab": use_isab})
    num_rows = 4
    sample_count = 11
    num_cols = 3
    data = np.zeros((num_rows, num_cols))
    mask = np.ones_like(data, dtype=bool)
    output = model.impute(data, mask, impute_config_dict={"batch_size": 2, "sample_count": sample_count})
    assert output.shape == (num_rows, num_cols)
    output = model.impute(data, mask, impute_config_dict={"batch_size": 3, "sample_count": sample_count}, average=False)
    assert output.shape == (sample_count, num_rows, num_cols)


def test_masked_data_is_ignored(tmpdir):
    # Check that modifying unmasked values changes output, but modifying
    # masked values does not.
    model = get_model(save_dir=tmpdir, config_updates={})
    batch_size = 4
    num_cols = 3
    data = torch.zeros((batch_size, num_cols))
    mask = torch.zeros_like(data)
    mask[0, :] = 1
    output1 = model(data, mask)
    output2 = model(data + 200 * (1 - mask), mask)
    output3 = model(data + mask, mask)

    assert torch.allclose(output1, output2)
    assert not torch.allclose(output1, output3)


@pytest.mark.parametrize(
    "config_updates,expected_num_params",
    [({"share_readout_weights": False}, 1458), ({"share_readout_weights": True}, 1414)],
)
def test_num_parameters(tmpdir, config_updates, expected_num_params):
    # Check the model has the expected number of trainable parameters.  Bigger when readout weights are not shared.
    model = get_model(save_dir=tmpdir, config_updates=config_updates)
    num_trainable_parameters = sum(p.numel() for p in model.parameters())
    assert num_trainable_parameters == expected_num_params


def test_num_parameters_parallel():
    num_parallel = 13
    input_dim = 3
    output_dim = 23
    x = ParallelLinear(num_parallel, input_dim, output_dim)
    c = torch.nn.Linear(input_dim, output_dim)
    num_params_parallel = sum([p.numel() for p in x.parameters()])
    num_params_one = sum([p.numel() for p in c.parameters()])
    assert num_params_parallel == num_params_one * num_parallel

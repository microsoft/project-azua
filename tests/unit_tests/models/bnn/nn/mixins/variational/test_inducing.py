import copy
import itertools

import pytest
import torch
import torch.nn as nn

from azua.models.bnn import bayesianize_, register_global_inducing_weights_
from azua.models.bnn.nn import (
    InducingConv2d,
    InducingDeterministicConv2d,
    InducingDeterministicLinear,
    InducingLinear,
)


@pytest.mark.parametrize(
    "q_inducing,whitened,max_lamda,max_sd_u,bias,layer_type,sqrt_width_scaling",
    itertools.product(
        ("diagonal", "matrix", "full"),
        (False, True),
        (None, 0.3),
        (None, 0.3),
        (False, True),
        ("linear", "conv"),
        (False, True),
    ),
)
def test_forward_shape(q_inducing, whitened, max_lamda, max_sd_u, bias, layer_type, sqrt_width_scaling):
    inducing_rows = 4
    inducing_cols = 2
    batch_size = 5
    inducing_kwargs = dict(
        inducing_rows=inducing_rows,
        inducing_cols=inducing_cols,
        q_inducing=q_inducing,
        whitened_u=whitened,
        max_lamda=max_lamda,
        max_sd_u=max_sd_u,
        init_lamda=1,
        bias=bias,
        sqrt_width_scaling=sqrt_width_scaling,
    )

    if layer_type == "linear":
        in_features = 3
        out_features = 5
        layer = InducingLinear(in_features, out_features, **inducing_kwargs)

        x = torch.randn(batch_size, in_features)
        expected_shape = (batch_size, out_features)
    elif layer_type == "conv":
        in_channels = 3
        out_channels = 6
        kernel_size = 3
        padding = 1
        layer = InducingConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            **inducing_kwargs,
        )

        h, w = 7, 7
        x = torch.randn(batch_size, in_channels, h, w)
        expected_shape = (batch_size, out_channels, h, w)
    else:
        raise ValueError(f"Invalid layer_type: {layer_type}")

    assert layer(x).shape == expected_shape


@pytest.mark.parametrize("inference", ["inducing", "inducingdeterministic"])
def test_bayesianize_compatible(inference):
    net = nn.Sequential(nn.Conv2d(3, 8, 3), nn.Conv2d(8, 8, 3), nn.Linear(32, 16), nn.Linear(16, 8))
    bnn = copy.deepcopy(net)
    bayesianize_(bnn, inference)

    for m, bm in zip(net.modules(), bnn.modules()):
        if m is net:
            continue

        if inference == "inducing":
            if isinstance(m, nn.Linear):
                assert isinstance(bm, InducingLinear)
            elif isinstance(m, nn.Conv2d):
                assert isinstance(bm, InducingConv2d)
            else:  # unreachable
                assert False
        else:
            if isinstance(m, nn.Linear):
                assert isinstance(bm, InducingDeterministicLinear)
            elif isinstance(m, nn.Conv2d):
                assert isinstance(bm, InducingDeterministicConv2d)
            else:  # unreachable
                assert False


def test_global_inducing_cat_rows():
    net = nn.Sequential(
        InducingLinear(5, 5, inducing_rows=3, inducing_cols=3, whitened_u=True),
        nn.ReLU(),
        InducingLinear(5, 4, inducing_rows=3, inducing_cols=3, whitened_u=True),
    )
    register_global_inducing_weights_(net, inducing_rows=3, inducing_cols=3, cat_dim=0)

    # net[3] is the helper model that contains the parameters for the inducing variables
    # in_features is the total number of inducing columns, out_features the number of inducing rows
    assert net[3].in_features == 3
    assert net[3].out_features == 6


def test_global_inducing_cat_cols():
    net = nn.Sequential(
        InducingLinear(5, 5, inducing_rows=3, inducing_cols=3, whitened_u=True),
        nn.ReLU(),
        InducingLinear(5, 4, inducing_rows=3, inducing_cols=3, whitened_u=True),
    )
    register_global_inducing_weights_(net, inducing_rows=3, inducing_cols=3, cat_dim=1)

    # net[3] is the helper model that contains the parameters for the inducing variables
    # in_features is the total number of inducing columns, out_features the number of inducing rows
    assert net._global_inducing_module.in_features == 6
    assert net._global_inducing_module.out_features == 3


def test_global_inducing_shape():
    net = nn.Sequential(
        InducingLinear(5, 5, inducing_rows=5, inducing_cols=5, whitened_u=True),
        nn.ReLU(),
        InducingLinear(5, 4, inducing_rows=5, inducing_cols=2, whitened_u=True),
    )
    register_global_inducing_weights_(net, inducing_rows=3, inducing_cols=3, cat_dim=1)
    x = torch.randn(2, 5)
    assert net(x).shape == (2, 4)


def test_global_inducing_dim_mismatch_raises():
    net = nn.Sequential(
        InducingLinear(5, 5, inducing_rows=5, inducing_cols=5, whitened_u=True),
        nn.ReLU(),
        InducingLinear(5, 4, inducing_rows=5, inducing_cols=2, whitened_u=True),
    )

    with pytest.raises(ValueError):
        register_global_inducing_weights_(net, inducing_rows=3, inducing_cols=3)


def test_global_inducing_non_whitened_raises():
    net = nn.Sequential(
        InducingLinear(5, 5, inducing_rows=5, inducing_cols=5, whitened_u=False),
        nn.ReLU(),
        InducingLinear(5, 4, inducing_rows=5, inducing_cols=2, whitened_u=False),
    )

    with pytest.raises(ValueError):
        register_global_inducing_weights_(net, inducing_rows=3, inducing_cols=3, cat_dim=1)


def test_global_inducing_raises_on_no_inducing_layers():
    net = nn.Sequential(nn.Linear(5, 5), nn.ReLU(), nn.Linear(5, 4))

    with pytest.raises(ValueError):
        register_global_inducing_weights_(net, inducing_rows=3, inducing_cols=3, cat_dim=1)


def test_global_inducing_raises_on_single_inducing_layer():
    net = nn.Sequential(
        InducingLinear(5, 5, inducing_rows=3, inducing_cols=3),
        nn.ReLU(),
        nn.Linear(5, 4),
    )

    with pytest.raises(ValueError):
        register_global_inducing_weights_(net, inducing_rows=3, inducing_cols=3, cat_dim=1)


def test_global_inducing_deletes_layerwise_variational_parameters():
    net = nn.Sequential(
        InducingLinear(5, 5, inducing_rows=5, inducing_cols=5, whitened_u=True),
        nn.ReLU(),
        InducingLinear(5, 4, inducing_rows=5, inducing_cols=2, whitened_u=True),
    )
    register_global_inducing_weights_(net, inducing_rows=3, inducing_cols=3, cat_dim=1)

    for m in (net[0], net[2]):
        param_names = [n for n, _ in m.named_parameters()]
        assert "inducing_mean" not in param_names
        assert "inducing_sd" not in param_names
        assert "inducing_scale_tril" not in param_names
        assert "inducing_col_scale_tril" not in param_names
        assert "inducing_row_scale_tril" not in param_names

        assert "z_row" in param_names
        assert "z_col" in param_names
        assert "_d_row" in param_names
        assert "_d_col" in param_names

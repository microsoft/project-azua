import pytest
import torch

import azua.models.bnn as bnn


@pytest.mark.parametrize("local_reparam", [True, False])
def test_sampling(local_reparam):
    """Tests that the ffg layer samples from the correct distribution."""
    torch.manual_seed(24)

    layer = bnn.nn.FFGLinear(2, 3, bias=False, init_sd=0.1, local_reparameterization=local_reparam)
    x = torch.randn(1, 2)

    mu = x.mm(layer.weight_mean.t())
    sd = x.pow(2).mm(layer.weight_sd.pow(2).t()).sqrt()

    a = torch.stack([layer(x) for _ in range(1000)])
    assert torch.allclose(mu, a.mean(0), atol=1e-2)
    assert torch.allclose(sd, a.std(0), atol=1e-2)


def test_init_from_deterministic_params():
    layer = bnn.nn.FFGLinear(5, 3)
    weight = torch.randn(3, 5)
    bias = torch.randn(3)
    layer.init_from_deterministic_params({"weight": weight, "bias": bias})
    assert torch.allclose(weight, layer.weight_mean)
    assert torch.allclose(bias, layer.bias_mean)


def test_init_from_deterministic_params_no_bias():
    layer = bnn.nn.FFGLinear(5, 3, bias=False)
    weight = torch.randn(3, 5)
    layer.init_from_deterministic_params({"weight": weight})
    assert torch.allclose(weight, layer.weight_mean)

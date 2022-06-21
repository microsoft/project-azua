import torch

import azua.models.bnn as bnn


def test_init_from_deterministic_params():
    layer = bnn.nn.FCGLinear(5, 3)
    weight = torch.randn(3, 5)
    bias = torch.randn(3)
    layer.init_from_deterministic_params({"weight": weight, "bias": bias})
    assert torch.allclose(torch.cat([weight.flatten(), bias]), layer.mean)


def test_init_from_deterministic_params_no_bias():
    layer = bnn.nn.FCGLinear(5, 3, bias=False)
    weight = torch.randn(3, 5)
    layer.init_from_deterministic_params({"weight": weight})
    assert torch.allclose(weight.flatten(), layer.mean)


def test_sampling():
    """Tests that the ffg layer samples from the correct distribution."""
    torch.manual_seed(24)

    layer = bnn.nn.FCGLinear(3, 1, bias=False, init_sd=0.1)
    x = torch.randn(1, 3)

    # for w ~ N(\mu, \Sigma), x^T w ~ N(x^T \mu, x^T \Sigma x)
    mu = x.mv(layer.mean).squeeze()
    sd = x.mm(layer.scale_tril).pow(2).sum().sqrt()

    a = torch.stack([layer(x).squeeze() for _ in range(1000)])
    assert torch.isclose(mu, a.mean(), atol=1e-2)
    assert torch.isclose(sd, a.std(), atol=1e-2)

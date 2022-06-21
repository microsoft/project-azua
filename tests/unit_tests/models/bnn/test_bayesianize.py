import copy

import pytest
import torch
import torch.nn as nn

import azua.models.bnn as bnn
from azua.models.bnn.bayesianize import bayesian_from_template, bayesianize_


def assert_layers_equal(l1, l2):
    if isinstance(l1, nn.Linear):
        assert_linear_equal(l1, l2)
    elif isinstance(l1, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        assert_conv_equal(l1, l2)
    else:
        raise ValueError("Unrecognized torch layer class:", l1.__class__.__name__)


def assert_linear_equal(l1, l2):
    assert l1.in_features == l2.in_features
    assert l1.out_features == l2.out_features
    assert l1.weight.shape == l2.weight.shape
    assert (l1.bias is not None) == (l2.bias is not None)
    if l1.bias is not None:
        assert l1.bias.shape == l2.bias.shape


def assert_conv_equal(l1, l2):
    assert all(
        getattr(l1, a) == getattr(l2, a)
        for a in [
            "in_channels",
            "out_channels",
            "kernel_size",
            "stride",
            "padding",
            "dilation",
            "groups",
        ]
    )
    assert l1.weight.shape == l2.weight.shape
    assert (l1.bias is not None) == (l2.bias is not None)
    if l1.bias is not None:
        assert l1.bias.shape == l2.bias.shape


@pytest.mark.parametrize(
    "torch_layer,inference,target_class",
    [
        (nn.Linear(5, 3, True), "ffg", bnn.nn.FFGLinear),
        (nn.Linear(5, 3, False), "ffg", bnn.nn.FFGLinear),
        (nn.Linear(5, 3, True), "fcg", bnn.nn.FCGLinear),
        (nn.Linear(5, 3, False), "fcg", bnn.nn.FCGLinear),
        (nn.Conv2d(16, 32, 3, bias=True), "ffg", bnn.nn.FFGConv2d),
        (nn.Conv2d(16, 32, 3, bias=False), "ffg", bnn.nn.FFGConv2d),
        (nn.Conv2d(16, 32, 3, stride=2, padding=1), "ffg", bnn.nn.FFGConv2d),
        (nn.Conv2d(16, 32, 3, bias=True), "fcg", bnn.nn.FCGConv2d),
        (nn.Conv2d(16, 32, 3, bias=False), "fcg", bnn.nn.FCGConv2d),
    ],
)
def test_template(torch_layer, inference, target_class):
    bayesian_layer = bayesian_from_template(torch_layer, inference)
    assert isinstance(bayesian_layer, target_class)
    assert_layers_equal(torch_layer, bayesian_layer)


def test_template_raises_on_unkown_inference():
    layer = nn.Linear(3, 2)
    with pytest.raises(KeyError):
        bayesian_from_template(layer, "abcdefghi")


@pytest.mark.parametrize("inference,target_class", [("ffg", bnn.nn.FFGLinear), ("fcg", bnn.nn.FCGLinear)])
def test_bayesianize_all(inference, target_class):
    net = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    bnet = copy.deepcopy(net)
    bayesianize_(bnet, inference)

    assert isinstance(bnet[0], target_class)
    assert_layers_equal(net[0], bnet[0])
    assert isinstance(bnet[2], target_class)
    assert_layers_equal(net[2], bnet[2])
    assert isinstance(bnet[1], nn.ReLU)


def test_bayesianize_resnet():
    net = bnn.nn.nets.make_network("resnet18", kernel_size=3, out_features=10)
    bnet = copy.deepcopy(net)
    bayesianize_(bnet, "ffg")
    assert len(list(net.modules())) == len(list(bnet.modules()))
    for module, bmodule in zip(net.modules(), bnet.modules()):
        if isinstance(module, nn.Linear):
            assert isinstance(bmodule, bnn.nn.FFGLinear)
            assert_linear_equal(module, bmodule)
        elif isinstance(module, nn.Conv2d):
            assert isinstance(bmodule, bnn.nn.FFGConv2d)
            assert_conv_equal(module, bmodule)
        elif not list(module.modules()):
            # check for "elementary" modules like batchnorm, nonlinearities etc that the
            # class hasn't been changed. Checking for equality would be better, but isn't
            # really supported by pytorch, e.g. l == copy.deepcopy(l) will return False
            # for a BatchNorm layer
            assert module.__class__ == bmodule.__class__
        else:
            # skip modules that collect other modules
            pass


@pytest.mark.parametrize("inference", ["ffg", "fcg"])
def test_output_shapes(inference):
    net = nn.Sequential(nn.Linear(5, 4), nn.ReLU(), nn.Linear(4, 3))
    bnet = copy.deepcopy(net)
    bayesianize_(bnet, inference)

    x = torch.randn(1, 5)
    assert net(x).shape == bnet(x).shape


@pytest.mark.parametrize("inference", ["ffg", "fcg"])
def test_incorrect_input_error(inference):
    bnet = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    bayesianize_(bnet, inference)
    x = torch.randn(1, 5)
    with pytest.raises(RuntimeError):
        bnet(x)


def test_bayesianize_last_layer():
    bnet = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    bayesianize_(bnet, inference={"2": "ffg"})
    # explicitly comparing the class for the first layer, since an FFGLinear object is an instance of nn.Linear
    assert bnet[0].__class__ == nn.Linear
    assert isinstance(bnet[2], bnn.nn.FFGLinear)


def test_bayesianize_last_layer_index():
    bnet = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    bayesianize_(bnet, inference={-1: "fcg"})
    assert bnet[0].__class__ == nn.Linear
    assert isinstance(bnet[2], bnn.nn.FCGLinear)


def test_bayesianize_class_name():
    bnet = bnn.nn.nets.CNN(channels=[3, 3], lin_sizes=[9, 10, 2], maxpool_freq=0)
    bayesianize_(bnet, inference={"Conv2d": "ffg", "Linear": "fcg"})
    assert isinstance(bnet[0], bnn.nn.FFGConv2d)
    assert isinstance(bnet[3][0], bnn.nn.FCGLinear)
    assert isinstance(bnet[3][2], bnn.nn.FCGLinear)


def test_module_name_priority_over_class_name():
    bnet = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    bayesianize_(bnet, inference={"0": "fcg", "Linear": "ffg"})
    assert isinstance(bnet[0], bnn.nn.FCGLinear)
    assert isinstance(bnet[2], bnn.nn.FFGLinear)


def test_initialize_ffg():
    ref_net = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    bnn = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    bayesianize_(bnn, reference_state_dict=ref_net.state_dict(), inference="ffg")
    for m, bm in zip(ref_net.modules(), bnn.modules()):
        if isinstance(m, nn.Linear):
            assert torch.allclose(m.weight, bm.weight_mean)
            assert torch.allclose(m.bias, bm.bias_mean)


def test_initialize_fcg():
    ref_net = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    bnn = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    bayesianize_(bnn, reference_state_dict=ref_net.state_dict(), inference="fcg")
    for m, bm in zip(ref_net.modules(), bnn.modules()):
        if isinstance(m, nn.Linear):
            assert torch.allclose(torch.cat([m.weight.flatten(), m.bias]), bm.mean)

from typing import List, Union

import torch.nn as nn
import torchvision


def make_network(architecture: str, *args, **kwargs):
    if architecture == "fcn":
        return FCN(**kwargs)
    elif architecture == "cnn":
        return CNN(**kwargs)
    elif architecture.startswith("resnet"):
        net = getattr(torchvision.models, architecture)(num_classes=kwargs["out_features"])
        if "kernel_size" in kwargs:
            kernel_size = kwargs["kernel_size"]
            stride = kwargs.get("stride", 1)
            padding = kwargs.get("padding", kernel_size // 2)
            in_channels = kwargs.get("in_channels", 3)
            bias = net.conv1.bias is not None
            net.conv1 = nn.Conv2d(in_channels, net.conv1.out_channels, kernel_size, stride, padding, bias=bias,)
        if kwargs.get("remove_maxpool", False):
            net.maxpool = nn.Identity()
        return net
    else:
        raise ValueError("Unrecognized network architecture:", architecture)


class FCN(nn.Sequential):
    """Basic fully connected network class."""

    def __init__(
        self, sizes: List[int], nonlinearity: Union[str, type] = "ReLU", bn: bool = False, **layer_kwargs,
    ):
        super().__init__()
        nonl_class = getattr(nn, nonlinearity) if isinstance(nonlinearity, str) else nonlinearity

        layer_kwargs.setdefault("bias", not bn)
        for i, (s0, s1) in enumerate(zip(sizes[:-1], sizes[1:])):
            self.add_module(f"Linear{i}", nn.Linear(s0, s1, **layer_kwargs))
            if bn:
                self.add_module(f"BN{i}", nn.BatchNorm1d(s1))
            if i < len(sizes) - 2:
                self.add_module(f"Nonlinarity{i}", nonl_class())


class CNN(nn.Sequential):
    """Basic CNN class with Conv/BN/Nonl/Maxpool blocks followed by a fully connected net. Batchnorm and maxpooling
    are optional and the latter can also only be included after every nth block."""

    def __init__(
        self,
        channels: List[int],
        lin_sizes: List[int],
        nonlinearity: Union[str, type] = "ReLU",
        maxpool_freq: int = 1,
        conv_bn: bool = False,
        linear_bn: bool = False,
        kernel_size: int = 3,
        **conv_kwargs,
    ):
        super().__init__()
        nonl_class = getattr(nn, nonlinearity) if isinstance(nonlinearity, str) else nonlinearity
        conv_kwargs.setdefault("bias", not conv_bn)
        for i, (c0, c1) in enumerate(zip(channels[:-1], channels[1:])):
            self.add_module(f"Conv{i}", nn.Conv2d(c0, c1, kernel_size, **conv_kwargs))
            if conv_bn:
                self.add_module(f"ConvBN{i}", nn.BatchNorm2d(c1))
            self.add_module(f"ConvNonlinearity{i}", nonl_class())
            if maxpool_freq and (i + 1) % maxpool_freq == 0:
                self.add_module(f"Maxpool{i//maxpool_freq}", nn.MaxPool2d(2, 2))
        self.add_module("Flatten", nn.Flatten())

        self.add_module("fc", FCN(lin_sizes, nonlinearity=nonlinearity, bn=linear_bn))

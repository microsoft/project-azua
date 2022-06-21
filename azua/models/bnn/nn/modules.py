"""This module creates the actual Bayesian layers classes. Currently
this is done by hand, but it might be worth defining a list of modules
of which Bayesian variants should exist (currently Linear and Conv1-3d)
and generate the corresponding subclasses in combination with all Mixin
classes, which could be found programmatically."""

import torch.nn as nn

from .mixins import *

__all__ = [
    "FFGLinear",
    "FFGConv1d",
    "FFGConv2d",
    "FFGConv3d",
    "FCGLinear",
    "FCGConv1d",
    "FCGConv2d",
    "FCGConv3d",
    "InducingDeterministicLinear",
    "InducingDeterministicConv1d",
    "InducingDeterministicConv2d",
    "InducingDeterministicConv3d",
    "InducingLinear",
    "InducingConv1d",
    "InducingConv2d",
    "InducingConv3d",
]


class FFGLinear(FFGMixin, nn.Linear):
    pass


class FFGConv1d(FFGMixin, nn.Conv1d):
    pass


class FFGConv2d(FFGMixin, nn.Conv2d):
    pass


class FFGConv3d(FFGMixin, nn.Conv3d):
    pass


class FCGLinear(FCGMixin, nn.Linear):
    pass


class FCGConv1d(FCGMixin, nn.Conv1d):
    pass


class FCGConv2d(FCGMixin, nn.Conv2d):
    pass


class FCGConv3d(FCGMixin, nn.Conv3d):
    pass


class InducingDeterministicLinear(InducingDeterministicMixin, nn.Linear):
    pass


class InducingDeterministicConv1d(InducingDeterministicMixin, nn.Conv1d):
    pass


class InducingDeterministicConv2d(InducingDeterministicMixin, nn.Conv2d):
    pass


class InducingDeterministicConv3d(InducingDeterministicMixin, nn.Conv3d):
    pass


class InducingLinear(InducingMixin, nn.Linear):
    pass


class InducingConv1d(InducingMixin, nn.Conv1d):
    pass


class InducingConv2d(InducingMixin, nn.Conv2d):
    pass


class InducingConv3d(InducingMixin, nn.Conv3d):
    pass

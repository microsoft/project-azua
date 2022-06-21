import pytest
import torch

from azua.models.mask_net import MaskNet


@pytest.fixture(scope="function")
def mask_net():
    return MaskNet(input_dim=5, device="cpu")


def test_mask_net_output_dim(mask_net):
    batch_size = 100
    input_dim = 5
    data = torch.ones((batch_size, input_dim))
    assert data.shape == mask_net(data).shape

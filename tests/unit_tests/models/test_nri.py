import pytest
import torch

from azua.models.nri import NRI

params_dict = {
    "seed": 42,
    "encoder_hidden": 256,
    "decoder_hidden": 256,
    "temp": 0.5,
    "num_atoms": 5,
    "encoder_dropout": 0.0,
    "decoder_dropout": 0.0,
    "edge_types": 2,
    "dims": 4,
    "timesteps": 49,
    "skip_first": False,
    "var": 5e-5,
    "prior": False,
}
num_sims = 500
num_edges = params_dict["num_atoms"] * (params_dict["num_atoms"] - 1)


@pytest.fixture(scope="function")
def nri():
    return NRI(params_dict=params_dict, device=torch.device("cpu"), save_dir="")


def test_encoder(nri):
    input_shape = (
        num_sims,
        params_dict["num_atoms"],
        params_dict["timesteps"],
        params_dict["dims"],
    )
    input = torch.randn(input_shape)
    output_shape = (num_sims, num_edges, params_dict["edge_types"])
    output = nri.encoder(input, nri.rel_rec, nri.rel_send)
    assert tuple(output.shape) == output_shape


def test_decoder(nri):
    input_shape = (
        num_sims,
        params_dict["num_atoms"],
        params_dict["timesteps"],
        params_dict["dims"],
    )
    input = torch.randn(input_shape)
    rel_type_shape = (num_sims, num_edges, params_dict["edge_types"])
    rel_type = torch.randn(rel_type_shape)
    output_shape = (
        num_sims,
        params_dict["num_atoms"],
        params_dict["timesteps"] - 1,
        params_dict["dims"],
    )
    output = nri.decoder(input, rel_type, nri.rel_rec, nri.rel_send, pred_steps=1)
    assert tuple(output.shape) == output_shape

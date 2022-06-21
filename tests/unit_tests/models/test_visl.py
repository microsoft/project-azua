import torch

from azua.models.visl import VISL
from azua.datasets.variables import Variable, Variables


def test_reconstruct(tmpdir_factory):

    variables = Variables(
        [
            Variable("continuous_input_1", True, "continuous", 0, 1),
            Variable("continuous_input_2", True, "continuous", 0, 1),
        ]
    )

    model_config = {
        "n_in_node": 1,
        "num_edge_types": 2,
        "gnn_iters": 3,
        "shared_init_and_final_mappings": True,
        "random_seed": [0],
        "embedding_dim": 16,
        "init_prob": 0.5,
    }

    model = VISL.create(
        model_id="model_id",
        variables=variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        model_config_dict=model_config,
        device="cpu",
    )
    model.hard = False

    d = len(variables)
    n = 100

    data = torch.rand(n, d)
    mask = torch.rand(n, d).round()
    (data_reconstructed, _), _, (_, _) = model.reconstruct(data, mask, count=1)

    assert data_reconstructed.shape == data.shape

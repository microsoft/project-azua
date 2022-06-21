import numpy as np
import pytest
import torch

from azua.models.graph_neural_network import GraphNeuralNetwork
from azua.datasets.variables import Variables

num_nodes = 10
num_edges = 100
node_input_dim = 2
edge_input_dim = 1
prediction_metadata_dim = 0
n_layer = 1
node_dim = 3
message_dim = 3
hidden_dims = [5]
dropout_rates = 0.0


@pytest.fixture(scope="function")
def graph_neural_network(tmpdir_factory):
    return GraphNeuralNetwork(
        model_id="test_model",
        variables=Variables([]),
        save_dir=tmpdir_factory.mktemp("save_dir"),
        device=torch.device("cpu"),
        aggregate_function="mean",
        normalize_embeddings=True,
        node_input_dim=node_input_dim,
        edge_input_dim=edge_input_dim,
        prediction_metadata_dim=prediction_metadata_dim,
        node_dim=node_dim,
        message_dim=message_dim,
        n_layer=n_layer,
        node_update_dropout=dropout_rates,
        node_update_dims=hidden_dims,
        prediction_dropout=dropout_rates,
        prediction_dims=hidden_dims,
        prediction_activation="sigmoid",
        node_init="random",
        separate_msg_channels_by_labels=False,
        update_edge_embeddings=False,
        num_heads=1,
        item_meta_dim=0,
        use_transformer=False,
        corgi_attention_method="NONE",
        max_transformer_length=0,
        is_inductive_task=False,
        aggregation_type="CONV",
    )


def test_forward(graph_neural_network):
    x = torch.rand((num_nodes, node_input_dim))
    edge_attr = torch.randint(2, size=(num_edges, edge_input_dim)).float()
    edge_index = torch.randint(num_nodes, size=(2, num_edges))
    adj = (edge_index, torch.arange(num_edges), (num_nodes, num_nodes))
    adjs = [adj] * n_layer
    sentences = np.array([""] * num_nodes)
    attention_mask = np.array([])
    output = graph_neural_network.forward(x, sentences, attention_mask, edge_attr, adjs, 0, num_nodes, False)
    output_shape = (num_nodes, node_dim)
    assert tuple(output.shape) == output_shape

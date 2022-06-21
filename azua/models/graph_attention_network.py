from .graph_neural_network import GraphNeuralNetwork


class GraphAttentionNetwork(GraphNeuralNetwork):
    """
    Graph Attention Network (GAT)

    Graph Attention Networks.
    https://arxiv.org/abs/1710.10903
    P. Velickovic, G. Cucurull, A. Casanova, A. Romero, P. Lio, and Y. Bengio.
    In ICLR, 2018.
    """

    @classmethod
    def name(self):
        return "graph_attention_network"

    @classmethod
    def _create(cls, model_id, variables, save_dir, device, **model_config_dict):
        assert model_config_dict["use_transformer"] is False
        assert model_config_dict["max_transformer_length"] == 0
        assert model_config_dict["aggregation_type"] == "ATTENTION"
        assert model_config_dict["corgi_attention_method"] == "NONE"

        return super(GraphAttentionNetwork, cls)._create(model_id, variables, save_dir, device, **model_config_dict)

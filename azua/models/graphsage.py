from .graph_neural_network import GraphNeuralNetwork


class GraphSAGE(GraphNeuralNetwork):
    """
    GraphSAGE

    Inductive representation learning on large graphs.
    https://arxiv.org/abs/1706.02216
    W. L. Hamilton, R. Ying, and J. Leskovec.
    In NeurIPS, 2017.

    GraphSAGE extends GCN by allowing the model to be trained on the part of the graph,
    making the model to be used in inductive settings.
    """

    @classmethod
    def name(self):
        return "graphsage"

    @classmethod
    def _create(cls, model_id, variables, save_dir, device, **model_config_dict):
        assert model_config_dict["update_edge_embeddings"] is False
        assert model_config_dict["use_discrete_edge_value"] is False
        assert model_config_dict["separate_msg_channels_by_labels"] is False
        assert model_config_dict["use_transformer"] is False
        assert model_config_dict["max_transformer_length"] == 0
        assert model_config_dict["aggregation_type"] == "CONV"
        assert model_config_dict["corgi_attention_method"] == "NONE"

        return super(GraphSAGE, cls)._create(model_id, variables, save_dir, device, **model_config_dict)

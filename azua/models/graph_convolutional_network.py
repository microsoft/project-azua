from .graph_neural_network import GraphNeuralNetwork


class GCN(GraphNeuralNetwork):
    """
    Graph Convolutional Network (GCN)

    Semi-supervised classification with graph convolutional networks.
    https://arxiv.org/abs/1609.02907
    T. Kipf and M. Welling.
    In ICLR, 2017.
    """

    @classmethod
    def name(self):
        return "graph_convolutional_network"

    @classmethod
    def _create(cls, model_id, variables, save_dir, device, **model_config_dict):
        assert model_config_dict["update_edge_embeddings"] is False
        assert model_config_dict["use_discrete_edge_value"] is False
        assert model_config_dict["separate_msg_channels_by_labels"] is False
        assert model_config_dict["use_transformer"] is False
        assert model_config_dict["max_transformer_length"] == 0
        assert model_config_dict["aggregation_type"] == "CONV"
        assert model_config_dict["corgi_attention_method"] == "NONE"

        return super(GCN, cls)._create(model_id, variables, save_dir, device, **model_config_dict)

from .graph_neural_network import GraphNeuralNetwork


class GRAPE(GraphNeuralNetwork):
    """
    GRAPE

    Handling missing data with graph representation learning.
    https://arxiv.org/abs/2010.16418
    J. You, X. Ma, D. Y. Ding, M. Kochenderfer, and J. Leskovec.
    In NeurIPS, 2020.

    GNN model that employs edge embeddings.
    Also, it adopts edge dropouts that are applied throughout all message-passing layers.
    Compared to the GRAPE proposed in the paper, because of the memory issue,
    we do not initialize nodes with one-hot vectors nor constants (ones).
    """

    @classmethod
    def name(self):
        return "grape"

    @classmethod
    def _create(cls, model_id, variables, save_dir, device, **model_config_dict):
        assert model_config_dict["update_edge_embeddings"] is True
        assert model_config_dict["use_discrete_edge_value"] is False
        assert model_config_dict["separate_msg_channels_by_labels"] is False
        assert model_config_dict["use_transformer"] is False
        assert model_config_dict["max_transformer_length"] == 0
        assert model_config_dict["aggregation_type"] == "CONV"
        assert model_config_dict["corgi_attention_method"] == "NONE"

        return super(GRAPE, cls)._create(model_id, variables, save_dir, device, **model_config_dict)

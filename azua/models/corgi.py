from .graph_neural_network import GraphNeuralNetwork


class CoRGi(GraphNeuralNetwork):
    """
    CORGI: Content-Rich Graph Neural Networks with Attention.

    J. Kim, A. Lamb, S. Woodhead, S. Peyton Jones, C. Zhang, M. Allamanis.
    In submission, 2021

    CoRGi is a GNN model that considers the rich data within nodes in the context of their neighbors.
    This is achieved by endowing CORGI’s message passing with a personalized attention
    mechanism over the content of each node. This way, CORGI assigns user-item-specific
    attention scores with respect to the words that appear in items.
    """

    @classmethod
    def name(self):
        return "corgi"

    @classmethod
    def _create(cls, model_id, variables, save_dir, device, **model_config_dict):
        assert model_config_dict["update_edge_embeddings"] is True
        assert model_config_dict["use_discrete_edge_value"] is False
        assert model_config_dict["separate_msg_channels_by_labels"] is False
        assert model_config_dict["use_transformer"] is True
        assert model_config_dict["max_transformer_length"] != 0
        assert model_config_dict["aggregation_type"] == "CONV"
        assert model_config_dict["corgi_attention_method"] in ("dot-product", "concat")

        return super(CoRGi, cls)._create(model_id, variables, save_dir, device, **model_config_dict)

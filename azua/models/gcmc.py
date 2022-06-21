from .graph_neural_network import GraphNeuralNetwork


class GCMC(GraphNeuralNetwork):
    """
    Graph Convolutional Matrix Completion (GC-MC)

    Graph convolutional matrix completion.
    https://arxiv.org/abs/1706.02263
    R. v. d. Berg, T. Kipf, and M. Welling
    In KDD Deep Learning Day Workshop, 2017.

    Compared to GCN, this model has a single message-passing layer.
    Also, For classification, each label is endowed with a separate message passing channel.
    Here, we do not implement the weight sharing.
    """

    @classmethod
    def name(self):
        return "gcmc"

    @classmethod
    def _create(cls, model_id, variables, save_dir, device, **model_config_dict):
        assert model_config_dict["n_layer"] == 1
        assert model_config_dict["update_edge_embeddings"] is False
        assert model_config_dict["use_discrete_edge_value"] is True
        assert model_config_dict["separate_msg_channels_by_labels"] is True
        assert model_config_dict["use_transformer"] is False
        assert model_config_dict["max_transformer_length"] == 0
        assert model_config_dict["aggregation_type"] == "CONV"
        assert model_config_dict["corgi_attention_method"] == "NONE"

        return super(GCMC, cls)._create(model_id, variables, save_dir, device, **model_config_dict)

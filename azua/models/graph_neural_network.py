# This is required in python 3 to allow return types of the same class.
from __future__ import annotations

import logging
import os
from tqdm import trange  # type: ignore
from typing import Dict, List, Optional, Callable, Any, TypeVar, Tuple

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import ReLU, Linear, Sigmoid, Parameter
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch_geometric.data import NeighborSampler
from torch_geometric.data import Data as GraphData
from torch_geometric.typing import Adj
from torch_geometric.utils.dropout import dropout_adj
from transformers import BertModel
from transformers import BertTokenizer

from ..experiment.azua_context import AzuaContext
from ..models.torch_model import TorchModel
from ..models.graph_convs import ConvModel, GATModel
from ..utils.torch_utils import generate_fully_connected
from ..utils.io_utils import save_json
from ..datasets.variables import Variables
from ..datasets.dataset import GraphDataset
from dependency_injector.wiring import inject, Provide

# Create type variable with upper bound of `GraphNeuralNetwork`, in order to precisely specify return types of create/load
# methods as the subclass on which they are called
T = TypeVar("T", bound="GraphNeuralNetwork")


class GraphNeuralNetwork(TorchModel):
    """
    Graph Neural Network for recommendation
    """

    __model_config_path = "model_config.json"
    __model_type_path = "model_type.txt"
    __variables_path = "variables.json"

    __model_file = "model.pt"

    def __init__(
        self,
        model_id: str,
        variables: Variables,
        save_dir: str,
        device: torch.device,
        aggregate_function: str,
        normalize_embeddings: bool,
        node_input_dim: int,
        edge_input_dim: int,
        prediction_metadata_dim: int,
        node_dim: int,
        message_dim: int,
        n_layer: int,
        node_update_dropout: float,
        node_update_dims: List[int],
        prediction_dropout: float,
        prediction_dims: List[int],
        prediction_activation: str,
        node_init: str,
        separate_msg_channels_by_labels: bool,
        update_edge_embeddings: bool,
        num_heads: int,
        item_meta_dim: int,
        use_transformer: bool,
        corgi_attention_method: str,
        max_transformer_length: int,
        is_inductive_task: bool,
        aggregation_type: str,
        **kwargs,
    ) -> None:
        """
        Args:
            model_id: Unique model ID for referencing this model instance.
            variables: Information about variables/features used
                by this model.
            save_dir: Location to save any information about this model, including training data.
                be created if it doesn't exist.
            device: Name of Torch device to create the model on. Valid options are 'cpu', 'gpu', or a device ID
                (e.g. 0 or 1 on a two-GPU machine).
            aggregate_function: Aggregate function for the message passing.  Valid options are 'add', 'mean', and 'max'.
            normalize_embeddings: Whether to normalize the node embeddings or not during message passing.
            node_input_dim: Dimensionality of the input nodes.
            edge_input_dim: Dimensionality of the input edges. For categorical labels, the value corresponds
                to the number of categories. For continuous or binary labels, the value is set to 1.
            prediction_metadata_dim: Dimensionality of edge metadata that is used in prediction MLP.
            node_dim: Dimensionality of the node embeddings. 
            message_dim: Dimensionality of the edge embeddings.
            n_layer: Number of the message passing layers for training and inference.
            node_update_dropout: Dropout rate for the MLP that updates the node embeddings after the message passing.
            node_update_dims: Layer numbers and sizes for the MLP that updates the node embeddings
                after the message passing.
            prediction_dropout: Dropout rate for the MLP that is used for the label (edge) prediction.
            prediction_dims: Layer numbers and sizes for the MLP that is used for the label (edge) prediction.
            prediction_activation: Activation function for the edge prediction MLP. Valid options are 'sigmoid', 'softmax', and 'none'.
            node_init: Node initialization method for graph representation of data. Current options are 'random', 'grape'.
            separate_msg_channels_by_labels: When set to True, let multiple messages flow between two nodes to deal with discrete (multiple) edge values.
            update_edge_embeddings: Whether to do message updates or not. If not, edge_attr is always going to be used for all
                msg. passing layers for the message() function. Setting this to True drastically increases the GPU memory usage.
            num_heads: Number of attention heads for GAT.
            item_meta_dim: Dimensionality of item metadata embeddings. When node_init is text_init, then this corresponds to the vocab size.
                This can also be the dimensionality of transformer model outputs.
            use_transformer: Whether or not to use transformer to compute context-aware word embeddings for item nodes.
            corgi_attention_method: Choose from ("NONE", "concat", "dot-product") for computing user-word attentions.
                Corresponds to equation (6) in the paper. When use_transformer is set to True, corgi_attention_method
                is forced to be chosen from ("concat", "dot-product"). If use_transformer is set to False, 
                corgi_attention_method needs to be "NONE".
            max_transformer_length: Maximum length of tokens (words) the transformer model outputs. Words outwide of range(0, max_transformer_length) will be truncated.
            is_inductive_task: Boolean that signifies whether the task is inductive or not (transductive).
                Make sure to set split_type = 'rows' in the training config when is_inductive_task = True, 
                as we need to evaluate on nodes that didn't appear during training. split_type='elements', 
                then 'transductive', therefore is_inductive_task = False.
            aggregation_type: Message aggregation type, chosen from ("CONV", "ATTENTION"). 
                Choosing option "CONV" corresponds to GCN and "ATTENTION" to GAT.
        """
        super().__init__(model_id, variables, save_dir, device)

        transformer_model_id = "bert-base-cased"
        self._device = device

        self.edge_input_dim = edge_input_dim
        message_dim = message_dim if update_edge_embeddings else edge_input_dim
        self.message_dim = message_dim

        self.prediction_metadata_dim = prediction_metadata_dim

        self.normalize_embeddings = normalize_embeddings
        self.node_init = node_init
        self.aggregate_function = aggregate_function

        self.node_update_dropout = node_update_dropout
        self.prediction_dropout = prediction_dropout

        self.n_layer = n_layer

        self.separate_msg_channels_by_labels = separate_msg_channels_by_labels
        self.node_dim = node_dim
        self.node_input_dim = node_input_dim
        self.update_edge_embeddings = update_edge_embeddings

        self.item_meta_dim = item_meta_dim
        self.num_heads = num_heads
        self.use_transformer = use_transformer
        self.corgi_attention_method = corgi_attention_method
        assert corgi_attention_method in ("NONE", "concat", "dot-product")
        self.max_transformer_length = max_transformer_length
        self.is_inductive_task = is_inductive_task
        self.aggregation_type = aggregation_type

        # Message passing layers and message update layers
        if separate_msg_channels_by_labels:
            convs = nn.ModuleList()
            message_updates = nn.ModuleList()

            num_edge_types = edge_input_dim
            for msg_type in range(num_edge_types):
                convs_small, message_updates_small = self._build_convs_and_message_updates(
                    separate_msg_channels_by_labels
                )
                convs.append(convs_small)
                message_updates.append(message_updates_small)
        else:
            convs, message_updates = self._build_convs_and_message_updates(separate_msg_channels_by_labels)

        self.convs = convs
        self.message_updates = message_updates

        # MLP that updates node embeddings
        node_dim_mult_val = edge_input_dim if separate_msg_channels_by_labels else 1
        self.node_embedding_mlp = generate_fully_connected(
            node_dim * node_dim_mult_val,
            node_dim,
            np.array(node_update_dims) * node_dim_mult_val,
            ReLU,
            None,
            device,
            node_update_dropout,
        )

        # MLP for predicting edge values
        output_activation = Sigmoid if prediction_activation == "sigmoid" else nn.Identity
        prediction_edge_input_dim = 1
        self.prediction_mlp = generate_fully_connected(
            2 * node_dim + self.prediction_metadata_dim,
            prediction_edge_input_dim,
            prediction_dims,
            ReLU,
            output_activation,
            device,
            prediction_dropout,
        )

        # Introduce another Linear that reduces the node dimensionality for the methods below:
        if node_init in ["text_init", "sbert_init", "neural_bow_init", "bert_cls_init", "bert_avg_init"]:
            self.item_x_encoder = nn.Sequential(
                nn.Linear(item_meta_dim, node_input_dim), ReLU(), nn.Dropout(node_update_dropout)
            )

        if use_transformer:
            assert self.corgi_attention_method in ("concat", "dot-product")
            self.tokenizer = BertTokenizer.from_pretrained(transformer_model_id)
            self.transformer_model = BertModel.from_pretrained(transformer_model_id)

            self.W_source = Linear(node_dim, message_dim)
            self.W_target = nn.Sequential(Linear(item_meta_dim, message_dim), nn.Dropout(node_update_dropout),)

            if corgi_attention_method == "concat":
                self.W_attention = Parameter(torch.empty(1, 1, 2 * message_dim, device=self._device))
                nn.init.xavier_uniform_(self.W_attention)

            def init_weights(m):
                if type(m) == nn.Linear:
                    torch.nn.init.xavier_uniform_(m.weight)
                    m.bias.data.fill_(0.01)

            self.W_source.apply(init_weights)
            self.W_target.apply(init_weights)
        else:
            assert self.corgi_attention_method == "NONE"

        self.to(self._device)

    # CLASS METHODS #
    @classmethod
    def name(cls) -> str:
        return "graph_neural_network"

    def forward(
        self,
        x: Tensor,
        sentences_encoded: Tensor,
        padding_mask: Tensor,
        edge_attr: Tensor,
        adjs: List[Adj],
        edge_dropout: float,
        num_nodes: int,
        is_inductive: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass of graph neural network.

        Args:
            x: Shape (number_of_nodes, node_input_dim). Data to be used for the
                forward pass. number_of_nodes corresponds to n_user + n_items.
            sentences_encoded: Transformer output from the text metadata for items. 
                Non-item nodes are also assigned with zeros tensors.
            padding_mask: Mask for the sentences_encoded. 
            edge_attr: Shape (number_of_sampled_edges, edge_input_dim). Edge attributes
                used for training. number_of_sampled_edges corresponds to the total number of edges that are
                connecting the sampled nodes with the specified minibatch size.
            adjs: Each adj contains a 3-tuple: (edge_index_minibatch, original_edge_index, sizes).
                In edge_index_minibatch, the indices are adjusted to fit into the sampled minibatch nodes.
                Sizes is another 2-tuple with the first item being the total number of subgraph nodes and the
                second item being the number of target nodes.
            edge_dropout: Edge dropout rate set for message passing (during training).
                For prediction MLP, this rate is neglected.
            num_nodes: number of nodes in the data.
            is_inductive: boolean telling whether the current computation is for inductive task.
                When True, we observe nodes that do not appear in training sets.
        """
        if self.separate_msg_channels_by_labels:
            X = []
            num_edge_types = self.edge_input_dim
            for msg_type in range(num_edge_types):
                convs = self.convs[msg_type]
                message_updates = self.message_updates[msg_type]
                messages = edge_attr[:, msg_type][:, None]
                x_cloned = self._compute_node_embeddings(
                    x,
                    messages,
                    convs,
                    message_updates,
                    adjs,
                    edge_dropout,
                    num_nodes,
                    sentences_encoded,
                    padding_mask,
                    is_inductive,
                )
                X.append(x_cloned)
            x = torch.cat(X, dim=1)
        else:
            x = self._compute_node_embeddings(
                x,
                edge_attr,
                self.convs,
                self.message_updates,
                adjs,
                edge_dropout,
                num_nodes,
                sentences_encoded,
                padding_mask,
                is_inductive,
            )
        final_embedding = self.node_embedding_mlp(x)
        return final_embedding

    def run_train(  # type: ignore[override]
        self,
        dataset: GraphDataset,
        train_config_dict: Dict[str, Any] = {},
        report_progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> None:
        """
        Train the model.
        Training results will be saved.

        Args:
            data: Graph data holding attributes such as x, edge_index, edge_attr for
                both training and val sets.
            train_config_dict: Any other parameters needed by a specific concrete class. Of
                the form {arg_name: arg_value}. e.g. {"learning_rate": 1e-3, "epochs": 100}
            report_progress_callback: Function to report model progress for API.
        """
        assert isinstance(dataset, GraphDataset)
        data = dataset.get_graph_data_object()
        train_output_dir = self._create_train_output_dir_and_save_config(train_config_dict)
        self.train_output_dir = train_output_dir

        # Save graph data
        graph_data_save_path = os.path.join(train_output_dir, "graph_data.pt")
        torch.save(data, graph_data_save_path)

        # Run the training.
        train_results = self._train(data, train_output_dir, report_progress_callback, **train_config_dict)

        # Save train results.
        if train_results is not None:
            train_results_save_path = os.path.join(self.save_dir, "training_results_dict.json")
            save_json(train_results, train_results_save_path)

        # Reload best saved model into this class.
        self = self.load(self.model_id, self.save_dir, self._device)

    def _encode_sentence_using_transformer(self, sentences: np.ndarray):
        """
        encode a list of sentences using transformer model.

        Args:
            sentences: List of sentences
        """

        padding_mask = torch.zeros(len(sentences), self.max_transformer_length, device=self._device, dtype=torch.long)
        sentences_encoded = torch.zeros(
            (len(sentences), self.max_transformer_length, self.item_meta_dim), device=self._device
        )

        nonempty_idxs = [i for i, v in enumerate(sentences) if v != ""]

        if len(nonempty_idxs) != 0:
            transformer_tokenized = self.tokenizer(
                list(sentences[nonempty_idxs]),
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_transformer_length,
            )
            with torch.no_grad():
                self.transformer_model.eval()
                transformer_output = self.transformer_model(**transformer_tokenized.to(self._device))[0]

            padding_mask_ = transformer_tokenized["attention_mask"]
            sentences_encoded[nonempty_idxs] = transformer_output
            padding_mask[nonempty_idxs] = padding_mask_
        return sentences_encoded, padding_mask

    def _compute_node_embeddings(
        self,
        x: Tensor,
        edge_attr: Tensor,
        convs: nn.ModuleList,
        message_updates: nn.ModuleList,
        adjs: List[Adj],
        edge_dropout: float,
        num_nodes: int,
        sentences_encoded: Tensor,
        padding_mask: Tensor,
        is_inductive: bool = False,
    ) -> Tensor:
        """
        Compute node embeddings through message passing.
        Optionally, go through message update and word attention calculation using transformer.

        Args:
            x: Shape (number_of_nodes, node_input_dim). Data to be used for the
                forward pass. number_of_nodes corresponds to n_user + n_items.
            edge_attr: Shape (number_of_sampled_edges, edge_input_dim). Edge attributes
                used for training. number_of_sampled_edges corresponds to the total number of edges that are
                connecting the sampled nodes with the specified minibatch size.
            convs: GCN or GAT convolutions for different msg. passing layers.
            message_updates: message updates for different msg. passing layers.
            adjs: Each adj contains a 3-tuple: (edge_index_minibatch, original_edge_index, sizes).
                In edge_index_minibatch, the indices are adjusted to fit into the sampled minibatch nodes. 
                Sizes is another 2-tuple with the first item being the total number of subgraph nodes and the
                second item being the number of target nodes.
            edge_dropout: Edge dropout rate set for message passing (during training).
                For prediction MLP, this rate is neglected.
            num_nodes: number of nodes in the data.
            sentences_encoded: Shape (number_of_nodes, max_transformer_length, message_dim). Transformer outputs of
                sentences, with dimension compression using W_target.
            padding_mask: Shape (number_of_nodes, max_transformer_length). Mask that tells whether each entry of
                sentences_encoded is actual transformer output or placeholder.
            is_inductive: boolean telling whether the current computation is for inductive task.
                When True, we observe nodes that do not appear in training sets.
        """
        messages = edge_attr

        for level, (conv, message_update, adj) in enumerate(zip(convs, message_updates, adjs)):
            edge_index_minibatch, original_edge_index, sizes = adj
            edge_index_minibatch = edge_index_minibatch.to(self._device)

            messages_minibatch = messages[original_edge_index].to(self._device)

            if self.update_edge_embeddings:
                embedding_source = x[edge_index_minibatch[0]]
                messages_minibatch = message_update(torch.cat((embedding_source, messages_minibatch), dim=-1))

            if (level == len(self.convs) - 1) and self.use_transformer:
                messages_updated, attention = self._message_update_using_transformer(
                    messages_minibatch, edge_index_minibatch, x, sentences_encoded, padding_mask,
                )
                self.message_cache[original_edge_index] = messages_updated.cpu()
                self.attention[original_edge_index] = attention

            # Update the edge embeddings with the edge embeddings computed using content-attentions (CA).
            # i.e., e_{ij}^{(l)} = e_{ij}^{(l)\prime} + e_{ij, \text{CA}}^{(l)} following the notations from the CoRGi paper.
            if self.use_transformer:
                messages_minibatch = messages_minibatch + self.message_cache[original_edge_index].to(self._device)

            edge_index_minibatch, messages_minibatch = dropout_adj(
                edge_index_minibatch, messages_minibatch, edge_dropout, num_nodes=num_nodes
            )

            x_target = x[: sizes[1]]
            if is_inductive:
                x_target = conv((x, x_target), edge_index_minibatch, messages_minibatch)
                x[: sizes[1]] = x_target
            else:
                x = conv((x, x_target), edge_index_minibatch, messages_minibatch)
        if is_inductive:
            x = x_target
        return x

    def _build_convs_and_message_updates(self, separate_msg_channels_by_labels: bool):
        """
        Build GCN or GAT convolutions and message updates.

        Args:
            separate_msg_channels_by_labels: If set to true, build different convs for different labels.
        """
        convs = nn.ModuleList()
        message_updates = nn.ModuleList()

        if separate_msg_channels_by_labels:
            edge_input_dim = 1
        else:
            edge_input_dim = self.edge_input_dim

        for current_layer in range(self.n_layer):
            message_update_activation = ReLU

            conv_input_dim = self.node_dim if current_layer != 0 else self.node_input_dim

            conv_message_dim = self.message_dim

            if self.aggregation_type == "CONV":
                conv_model = ConvModel
            elif self.aggregation_type == "ATTENTION":
                conv_model = GATModel
            else:
                raise ValueError("aggregation_type should be either CONV or ATTENTION.")
            conv = conv_model(
                conv_input_dim,
                self.node_dim,
                conv_message_dim,
                self.aggregate_function,
                self.normalize_embeddings,
                self.node_update_dropout,
                # concat=concat,
                # heads=num_heads,
            )
            convs.append(conv)
            message_update = nn.Sequential(
                nn.Linear(self.node_dim + edge_input_dim, self.message_dim), message_update_activation(),
            )
            message_updates.append(message_update)
        return convs, message_updates

    def _message_update_using_transformer(
        self,
        messages_minibatch: Tensor,
        edge_index_minibatch: Tensor,
        x: Tensor,
        sentences_encoded: Tensor,
        padding_mask: Tensor,
        negative_slope: float = 0.2,
        LARGE_NEGATIVE_NUMBER: int = -1000,
        eps: float = 1e-30,
    ):
        """
        Update messages (or, edge attributes) by the weighted sum of word embeddings from transformers
        with its corresponding word attention values.
        This part of the code corresponds to  Equations (4-6) in the CoRGi paper.

        Args:
            edge_index: Edge indices of shape (2, num_edges).
            x: Shape (number_of_nodes, node_input_dim). Data to be used for the
                forward pass. number_of_nodes corresponds to n_user + n_items.
            sentences_encoded: Shape (number_of_nodes, max_transformer_length, message_dim). Transformer outputs of
                sentences, with dimension compression using W_target.
            padding_mask: Shape (number_of_nodes, max_transformer_length). Mask that tells whether each entry of
                sentences_encoded is actual transformer output or placeholder.
            negative_slope: LeakyReLU negative slope.
            LARGE_NEGATIVE_NUMBER: A large negative integer used in softmax calculation for numerical stability.
            eps: A small float used in softmax calculation.

        """
        target_size: int = int(torch.max(edge_index_minibatch[1]).item()) + 1

        sentences_encoded = sentences_encoded[:target_size]
        padding_mask = padding_mask[:target_size]

        max_length = sentences_encoded.shape[1]
        messages_updated = torch.zeros(edge_index_minibatch.shape[1], self.message_dim, device=self._device)
        attention_score_ = torch.zeros(edge_index_minibatch.shape[1], max_length)

        nonempty_idxs = torch.where(padding_mask[:, 0])[0]

        def torch_in1d(ar1: Tensor, ar2: Tensor):
            if not len(ar1) or not len(ar2):
                return torch.tensor([])
            mask = ar2.new_zeros((max(ar1.max(), ar2.max()) + 1,), dtype=torch.bool)  # type: ignore
            mask[ar2.unique()] = True
            return mask[ar1]

        edge_index_nonempty_idx = torch.where(torch_in1d(edge_index_minibatch[1], nonempty_idxs))[0]
        if len(edge_index_nonempty_idx) != 0:
            edge_index_minibatch_nonempty = edge_index_minibatch[:, edge_index_nonempty_idx]

            edge_index_target = edge_index_minibatch_nonempty[1]

            x = self.W_source(x)

            x_source = x[edge_index_minibatch_nonempty[0]]
            sentences_target = sentences_encoded[edge_index_target]
            padding_mask_target = padding_mask[edge_index_target]

            if self.corgi_attention_method == "concat":
                att_x_source = self.W_attention[:, :, : self.message_dim] * x_source.unsqueeze(1)
                att_sentences_target = self.W_attention[:, :, self.message_dim :] * sentences_target
                attention_coefficient = torch.sum(att_x_source, dim=-1) + torch.sum(att_sentences_target, dim=-1)

            else:
                attention_coefficient = torch.sum(
                    x_source.unsqueeze(1).expand(-1, max_length, -1) * sentences_target, dim=-1
                )
            attention_coefficient = F.leaky_relu(attention_coefficient, negative_slope)

            attention_coefficient = attention_coefficient - torch.max(attention_coefficient, dim=1)[0].unsqueeze(1)
            attention_coefficient = attention_coefficient + (1 - padding_mask_target) * LARGE_NEGATIVE_NUMBER

            attention_coefficient = torch.exp(attention_coefficient)
            attention_score = attention_coefficient / (torch.sum(attention_coefficient, dim=1) + eps).unsqueeze(1)
            messages_updated_nonempty_only = torch.sum(attention_score.unsqueeze(-1) * sentences_target, dim=1)
            messages_updated[edge_index_nonempty_idx] = messages_updated_nonempty_only
            attention_score_[edge_index_nonempty_idx] = attention_score.detach().cpu()

        return messages_updated, attention_score_

    def _compute_accuracy(
        self,
        X_embeddings: Tensor,
        edge_index: Tensor,
        sample_size: int,
        prediction_edge_metadata: Tensor,
        labels: Tensor,
    ):
        """
        Compute accuracies for train, test, or validation datasets.

        Args:
            X_embeddings: Shape (number_of_nodes, node_dim)
            edge_index: Edge indices of shape (2, num_edges).
            sample_size: Subsample size of edge index. Set this number to smaller value when GPU memory error occurs.
            prediction_edge_metadata: Shape (num_edges, prediction_metadata_dim). Tensor used for prediction MLP.
            labels: Shape (num_edges, ) of edge labels.
        """
        with torch.no_grad():
            self.eval()

            inference_rand_idx = torch.tensor(np.random.choice(range(edge_index.shape[1]), sample_size, replace=False,))
            if self.prediction_metadata_dim == 0:
                pred = self.prediction_mlp(
                    torch.cat(
                        [
                            X_embeddings[edge_index[0, inference_rand_idx], :],
                            X_embeddings[edge_index[1, inference_rand_idx], :],
                        ],
                        -1,
                    )
                )
            else:
                pred = self.prediction_mlp(
                    torch.cat(
                        [
                            X_embeddings[edge_index[0, inference_rand_idx], :],
                            X_embeddings[edge_index[1, inference_rand_idx], :],
                            prediction_edge_metadata[inference_rand_idx].to(self._device),
                        ],
                        -1,
                    )
                )

            pred = pred.reshape(-1).cpu()
            predicted_labels = torch.zeros(pred.shape)
            predicted_labels[torch.where(pred > 0.5)] = 1.0
            labels_for_inference = torch.cat((labels, labels))[inference_rand_idx]

            if self.loss_function == "MSE":
                loss = F.mse_loss(pred, labels_for_inference, reduction="mean").item()
                return -loss

            binary_inference_labels_idx = torch.where((labels_for_inference == 0.0) | (labels_for_inference == 1.0))[0]

            acc = (
                1
                - torch.mean(
                    torch.abs(
                        predicted_labels[binary_inference_labels_idx]
                        - labels_for_inference[binary_inference_labels_idx]
                    )
                ).item()
            )
        return acc

    def _update_X_embeddings(
        self,
        X_embeddings: Tensor,
        bs: int,
        n_id: Tensor,
        adjs: List[Adj],
        data: GraphData,
        x: Tensor,
        item_x: Tensor,
        edge_dropout: float,
        sentences_encoded: Tensor,
        padding_mask: Tensor,
        edge_attr: Tensor,
        is_train: bool,
        is_inductive: bool = False,
    ):
        """
        Update a computed node embeddings to X_embeddings (Eqn. (2) in the CoRGi paper). Differentiates the training and eval modes.

        Args:
            X_embeddings: Shape (number_of_nodes, node_dim)
            bs: batch size from the NeighborSampler.
            n_id: node IDs from the NeighborSampler.
            adjs: Each adj contains a 3-tuple: (edge_index_minibatch, original_edge_index, sizes).
                In edge_index_minibatch, the indices are adjusted to fit into the sampled minibatch nodes. 
                Sizes is another 2-tuple with the first item being the total number of subgraph nodes and the
                second item being the number of target nodes.
            data: GraphData.
            x: Shape (number_of_nodes, node_input_dim). Data to be used for the
                forward pass. number_of_nodes corresponds to n_user + n_items.
            item_x: Item node init embeddings from pre-trained sequence embedders. It is passed to self.item_x_encoder for
                dim. reduction and then incorporated in x.
            edge_dropout: Edge dropout rate.
            sentences_encoded: Transformer output from the text metadata for items. 
                Non-item nodes are also assigned with zeros tensors.
            padding_mask: Mask for the sentences_encoded. 
            is_train: boolean telling whether to use the training or evaluation mode.
            is_inductive: boolean telling whether the current computation is for inductive task. 
                When True, we observe nodes that do not appear in training sets.
        """
        if len(adjs[-1][1]) == 0:
            return
        if is_inductive:
            x = X_embeddings
        with torch.set_grad_enabled(is_train):
            if not is_train:
                self.eval()
            if self.node_init in ["text_init", "sbert_init", "neural_bow_init", "bert_cls_init", "bert_avg_init"]:
                encoded_item_x = self.item_x_encoder(item_x)
                x[data.num_users : data.num_users + data.num_items] = encoded_item_x.cpu()
            x_small = x[n_id].to(self._device)

            if self.use_transformer:
                sentences_encoded_small = torch.zeros(
                    len(n_id), sentences_encoded.shape[1], self.message_dim, device=self._device
                )
                item_ids = torch.where(n_id > data.num_users)[0]
                sentences_encoded_small_items = sentences_encoded[n_id[item_ids] - data.num_users]
                sentences_encoded_small_items = self.W_target(sentences_encoded_small_items.to(self._device))
                sentences_encoded_small[item_ids] = sentences_encoded_small_items

                padding_mask_small = padding_mask[n_id].to(self._device)
            else:
                sentences_encoded_small = padding_mask_small = torch.tensor([])

            x_embeddings = self(
                x_small,
                sentences_encoded_small,
                padding_mask_small,
                edge_attr,
                adjs,
                edge_dropout,
                data.num_nodes,
                is_inductive,
            )
            sampled_node_idx = n_id[:bs]
            X_embeddings[sampled_node_idx] = x_embeddings
        return

    def _update_X_embeddings_for_inductive_task(
        self,
        X_embeddings: Tensor,
        data: GraphData,
        x: Tensor,
        item_x: Tensor,
        sentences_encoded: Tensor,
        padding_mask: Tensor,
        dataset_type: str,
    ):
        """
        Computed node embeddings for the nodes that are unseen during training and only appear in
        test and validation sets. The updates and stored in X_embeddings.
        This method is called only during the inference, hence with self.eval().

        Args:
            X_embeddings: Shape (number_of_nodes, node_dim)
            data: GraphData.
            x: Shape (number_of_nodes, node_input_dim). Data to be used for the
                forward pass. number_of_nodes corresponds to n_user + n_items.
            item_x: Item node init embeddings from pre-trained sequence embedders. It is passed to self.item_x_encoder for
                dim. reduction and then incorporated in x.
            sentences_encoded: Transformer output from the text metadata for items. 
                Non-item nodes are also assigned with zeros tensors.
            padding_mask: Mask for the sentences_encoded. 
            dataset_type: Whether the inductive task is targeted for test or validation set.
        """
        self.eval()
        assert dataset_type in ("test", "val")
        if dataset_type == "test":
            edge_index = data.test_edge_index[:, ~self.test_heldout_idx]
            edge_attr = data.test_edge_attr[~self.test_heldout_idx]
        else:
            edge_index = data.val_edge_index[:, ~self.val_heldout_idx]
            edge_attr = data.val_edge_attr[~self.val_heldout_idx]

        unique_nodes = edge_index.unique()
        unique_user_nodes = unique_nodes[unique_nodes < data.num_users]

        minibatch_loader_inductive = NeighborSampler(
            edge_index,
            node_idx=unique_user_nodes,
            sizes=[-1],
            batch_size=data.num_nodes,
            shuffle=False,
            num_nodes=data.num_nodes,
            num_workers=0,
        )
        bs_inductive, n_id_inductive, adjs_inductive = next(iter(minibatch_loader_inductive))
        adjs_inductive = [adjs_inductive] * len(self.convs)

        self._update_X_embeddings(
            X_embeddings,
            bs_inductive,
            n_id_inductive,
            adjs_inductive,
            data,
            x,
            item_x,
            0,
            sentences_encoded,
            padding_mask,
            edge_attr,
            False,
            True,
        )

    @inject
    def _train(
        self,
        data: GraphData,
        train_output_dir: str,
        report_progress_callback: Optional[Callable[[str, int, int], None]],
        learning_rate: float,
        n_split: int,
        epochs: int,
        loss_function: str,
        neighbor_sampling_sizes: list,
        inference_at_every: int,
        edge_dropout: float,
        test_val_heldout_ratio: float,
        azua_context: AzuaContext = Provide[AzuaContext],
    ) -> Dict[str, List[float]]:
        """
        Train the model using the given data.

        Args:
            data: Graph data holding attributes such as x, edge_index, edge_attr for
                both training and val sets.
            train_output_dir: Path to save any training information to, including tensorboard summary
                files.
            report_progress_callback: Function to report model progress for API.
            learning_rate: Learning rate for Adam optimiser.
            n_split: Number of minibatches to split the training nodes into.
            epochs: Number of epochs to train for.
            loss_function: Loss function for training. Currently, only MSE and BCE are supported.
            neighbor_sampling_sizes: Node sampling sizes for each message passing layer for training. 
                List length should be the same as the n_layer.
            inference_at_every: Do inference at every this number with respect to the total iteration.
            edge_dropout: Edge dropout rate set for message passing (during training).
                For prediction MLP, this rate is neglected.
            test_val_heldout_ratio: The ratio of the heldout data over the rest of the datapoints
                for test and validation datasets. Only effective when self.is_inductive_task = True, and during inference.

        Returns:
            train_results: Train loss for each epoch as a dictionary.
        """
        # Put PyTorch into train mode.
        self.train()

        assert loss_function in ["BCE", "MSE"], NotImplementedError(
            "Only BCE and MSE are supported as loss functions for graph neural network."
        )

        self.loss_function = loss_function

        x = data.x
        item_x = data.item_x.to(self._device)
        train_labels = data.train_labels
        train_labels_double = torch.cat((train_labels, train_labels))
        writer = SummaryWriter(os.path.join(train_output_dir, "summary"), flush_secs=1)
        logger = logging.getLogger()
        metrics_logger = azua_context.metrics_logger()

        if self.use_transformer:
            self.message_cache = torch.zeros(data.train_edge_attr.shape[0], self.message_dim)
            self.attention = torch.zeros(data.train_edge_attr.shape[0], self.max_transformer_length)

        results_dict: Dict[str, List] = {
            "training_loss": [],
            "training_acc": [],
            "val_acc": [],
            "best_test_acc": [],
        }

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        best_train_loss = np.nan
        best_training_acc = np.nan
        best_val_acc = np.nan

        is_quiet = logger.level > logging.INFO
        train_edge_index = data.train_edge_index

        X_embeddings = torch.randn((data.num_nodes, self.node_dim), device=self._device)
        X_embeddings_inference = torch.randn((data.num_nodes, self.node_dim), device=self._device)

        if self.use_transformer:
            if data.sentences_encoded is None:
                sentences_encoded, padding_mask = self._encode_sentence_using_transformer(data.sentences)
            else:
                sentences_encoded, padding_mask = data.sentences_encoded, data.attention_mask
        else:
            sentences_encoded = padding_mask = torch.tensor([])

        total_iteration = 0

        for epoch in trange(epochs, desc="Epochs", disable=is_quiet):
            minibatch_loader = NeighborSampler(
                train_edge_index,
                node_idx=None,
                sizes=neighbor_sampling_sizes,
                batch_size=data.num_nodes // n_split,
                shuffle=True,
                num_nodes=data.num_nodes,
                num_workers=0,
            )
            for iteration, (bs, n_id, adjs) in enumerate(minibatch_loader):
                if len(neighbor_sampling_sizes) == 1:
                    adjs = [adjs]
                self.train()
                optimizer.zero_grad()

                self._update_X_embeddings(
                    X_embeddings,
                    bs,
                    n_id,
                    adjs,
                    data,
                    x,
                    item_x,
                    edge_dropout,
                    sentences_encoded,
                    padding_mask,
                    data.train_edge_attr,
                    True,
                )

                index_of_edge_index_within_batch = adjs[-1][1]
                edge_index_within_batch = train_edge_index[:, index_of_edge_index_within_batch]
                if self.prediction_metadata_dim == 0:
                    prediction_input = torch.cat(
                        [X_embeddings[edge_index_within_batch[0]], X_embeddings[edge_index_within_batch[1]]], -1
                    )
                else:
                    prediction_metadata_input = data.prediction_train_edge_metadata[
                        index_of_edge_index_within_batch
                    ].to(self._device)
                    prediction_input = torch.cat(
                        [
                            X_embeddings[edge_index_within_batch[0]],
                            X_embeddings[edge_index_within_batch[1]],
                            prediction_metadata_input,
                        ],
                        -1,
                    )
                predictions = self.prediction_mlp(prediction_input)
                train_labels_minibatch = train_labels_double[index_of_edge_index_within_batch].to(self._device)
                binary_train_labels_idx = torch.where(
                    (train_labels_minibatch == 0.0) | (train_labels_minibatch == 1.0)
                )[0]
                if loss_function == "BCE":
                    loss = F.binary_cross_entropy(
                        predictions[binary_train_labels_idx, 0],
                        train_labels_minibatch[binary_train_labels_idx],
                        reduction="mean",
                    )
                else:
                    loss = F.mse_loss(predictions[:, 0], train_labels_minibatch, reduction="mean")
                loss_val = loss.item()
                loss.backward()
                optimizer.step()
                X_embeddings.detach_()
                x.detach_()
                if self.use_transformer:
                    self.message_cache.detach_()

                if total_iteration % inference_at_every == 0:
                    if edge_dropout == 0 and self.node_update_dropout == 0 and self.prediction_dropout == 0:
                        X_embeddings_inference = X_embeddings
                    else:
                        for iteration_inference, (bs, n_id, adjs) in enumerate(minibatch_loader):
                            if len(neighbor_sampling_sizes) == 1:
                                adjs = [adjs]
                            self._update_X_embeddings(
                                X_embeddings_inference,
                                bs,
                                n_id,
                                adjs,
                                data,
                                x,
                                item_x,
                                0,
                                sentences_encoded,
                                padding_mask,
                                data.train_edge_attr,
                                False,
                            )

                    if self.is_inductive_task:
                        test_heldout_idx = torch.bernoulli(
                            torch.ones(data.test_edge_index.shape[1] // 2) * test_val_heldout_ratio
                        ).to(torch.bool)
                        val_heldout_idx = torch.bernoulli(
                            torch.ones(data.val_edge_index.shape[1] // 2) * test_val_heldout_ratio
                        ).to(torch.bool)
                        self.test_heldout_idx = torch.cat((test_heldout_idx, test_heldout_idx))
                        self.val_heldout_idx = torch.cat((val_heldout_idx, val_heldout_idx))

                        self._update_X_embeddings_for_inductive_task(
                            X_embeddings_inference, data, x, item_x, sentences_encoded, padding_mask, "val"
                        )
                        self._update_X_embeddings_for_inductive_task(
                            X_embeddings_inference, data, x, item_x, sentences_encoded, padding_mask, "test"
                        )

                        if self.prediction_metadata_dim != 0:
                            data.prediction_test_edge_metadata = data.prediction_test_edge_metadata[
                                self.test_heldout_idx
                            ]
                            data.prediction_val_edge_metadata = data.prediction_val_edge_metadata[self.val_heldout_idx]

                    training_acc = self._compute_accuracy(
                        X_embeddings_inference,
                        train_edge_index,
                        int(data.val_edge_index.shape[1]),
                        data.prediction_train_edge_metadata,
                        train_labels,
                    )
                    val_acc = self._compute_accuracy(
                        X_embeddings_inference,
                        data.val_edge_index[:, self.val_heldout_idx],
                        int(data.val_edge_index[:, self.val_heldout_idx].shape[1]),
                        data.prediction_val_edge_metadata,
                        data.val_labels[self.val_heldout_idx[: len(data.val_labels)]],
                    )

                    # average loss over most recent epoch
                    if np.isnan(best_val_acc) or val_acc > best_val_acc:
                        best_train_loss = loss_val
                        best_training_acc = training_acc
                        best_val_acc = val_acc
                        best_epoch = epoch
                        best_test_acc = self._compute_accuracy(
                            X_embeddings_inference,
                            data.test_edge_index[:, self.test_heldout_idx],
                            int(data.test_edge_index[:, self.test_heldout_idx].shape[1]),
                            data.prediction_test_edge_metadata,
                            data.test_labels[self.test_heldout_idx[: len(data.test_labels)]],
                        )
                        # Save model.
                        self.save()
                        if self.use_transformer:
                            torch.save(self.attention, os.path.join(self.train_output_dir, "train_attention.pt"))

                    # AzureML
                    metrics_logger.log_dict({"train/acc-val": val_acc})
                    metrics_logger.log_dict({"train/acc-train": training_acc})
                    metrics_logger.log_dict({"train/acc-test-best": best_test_acc})
                    metrics_logger.log_dict({"train/loss-train": loss_val})

                    # Save useful quantities.
                    writer.add_scalar("train/loss-train", loss_val, total_iteration // inference_at_every)
                    writer.add_scalar("train/train-acc", training_acc, total_iteration // inference_at_every)
                    writer.add_scalar("train/val-acc", val_acc, total_iteration // inference_at_every)
                    writer.add_scalar("train/best-test-acc", best_test_acc, total_iteration // inference_at_every)
                    results_dict["training_loss"].append(loss_val)
                    results_dict["training_acc"].append(training_acc)
                    results_dict["val_acc"].append(val_acc)
                    results_dict["best_test_acc"].append(best_test_acc)

                    if report_progress_callback:
                        report_progress_callback(self.model_id, epoch + 1, epochs)

                    if np.isnan(loss_val):
                        logger.info("Training loss is NaN. Exiting early.")
                        break
                total_iteration += 1
        metrics_logger.log_dict({"test_data.all.Accuracy": best_test_acc})
        logger.info(
            "Best model found at epoch %d, with train_loss %.4f, training_acc %.4f, val_acc %.4f, test_acc %.4f"
            % (best_epoch, best_train_loss, best_training_acc, best_val_acc, best_test_acc)
        )
        writer.close()

        return results_dict

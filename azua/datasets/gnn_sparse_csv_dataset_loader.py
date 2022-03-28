import ast
import logging
import os
from typing import Dict, Optional, Tuple, Union, Any, cast, List

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from scipy.sparse.csr import csr_matrix
from torch import Tensor
from torch.nn.functional import one_hot
from torch_geometric.data import Data

from ..datasets.dataset import SparseDataset, GraphDataset
from ..datasets.sparse_csv_dataset_loader import SparseCSVDatasetLoader

logger = logging.getLogger(__name__)


class GNNSparseCSVDatasetLoader(SparseCSVDatasetLoader):
    """
    Load a dataset from a sparse CSV file as graphs for GNN, where each row entry has a form (row_id, col_id, value).
    """

    def split_data_and_load_dataset(
        self,
        test_frac: float,
        val_frac: float,
        random_state: Union[int, Tuple[int, int]],
        max_num_rows: Optional[int] = None,
        negative_sample: bool = False,
        split_type: str = "rows",
        model_config: Dict = None,
        **kwargs,
    ) -> GraphDataset:
        """
        Load the data from memory and make the train/val/test split to instantiate a dataset.
        The data is split deterministically given the random state. If the given random state is a pair of integers,
        the first is used to extract test set and the second is used to extract the validation set from the remaining data.
        If only a single integer is given as random state it is used for both.

        Args:
            test_frac: Fraction of data to put in the test set.
            val_frac: Fraction of data to put in the validation set.
            random_state: An integer or a tuple of integers to be used as the splitting random state.
            max_num_rows: Maximum number of rows to include when reading data files.
            negative_sample: Whether to perform negative sampling after loading the dataset. Negative sampling requires
                a file negative_sampling_levels.csv in the dataset folder, and negative samples for each row will be
                drawn from features of a greater level than all those observed in the row.
            split_type: Manner in which the dataset has been split: "rows" indicates a split by rows of the matrix,
                "elements" indicates a split by individual elements of the matrix, so that different elements of a row
                can appear in different data splits.
            model_config: model_config

        Returns:
            dataset: GraphDataset object, holding the data and variable metadata as well as
            the torch_geometric Data object as one of it's attributes.
        """
        assert isinstance(model_config, Dict), "model_config should be a Dict."
        assert (
            model_config["is_inductive_task"] if split_type == "rows" else not model_config["is_inductive_task"]
        ), "is_inductive_task needs to be True (False) when split_type is 'rows' ('elements')."
        dataset = super().split_data_and_load_dataset(
            test_frac, val_frac, random_state, max_num_rows, negative_sample, split_type
        )

        logger.info("Create graph dataset.")
        graph_data = self._get_graph_data(dataset, model_config)
        graph_dataset = dataset.to_graph(graph_data)
        return graph_dataset

    def load_predefined_dataset(
        self,
        max_num_rows: Optional[int] = None,
        negative_sample: bool = False,
        split_type: str = "rows",
        model_config: Dict = None,
        **kwargs,
    ) -> GraphDataset:
        """
        Load the data from memory and use the predefined train/val/test split to instantiate a dataset.

        Args:
            max_num_rows: Maximum number of rows to include when reading data files.
            negative_sample: Whether to perform negative sampling after loading the dataset. Negative sampling requires
                a file negative_sampling_levels.csv in the dataset folder, and negative samples for each row will be
                drawn from features of a greater level than all those observed in the row.
            split_type: Manner in which the dataset has been split: "rows" indicates a split by rows of the matrix,
                "elements" indicates a split by individual elements of the matrix, so that different elements of a row
                can appear in different data splits.
            model_config: model_config

        Returns:
            dataset: GraphDataset object, holding the data and variable metadata.
        """
        # Add assertion to avoid mypy error: None is not indexable
        assert isinstance(model_config, Dict), "model_config should be a Dict."
        assert (
            model_config["is_inductive_task"] if split_type == "rows" else not model_config["is_inductive_task"]
        ), "is_inductive_task needs to be True (False) when split_type is 'rows' ('elements')."
        dataset = super().load_predefined_dataset(max_num_rows, negative_sample, split_type)

        logger.info("Create graph dataset.")
        graph_data = self._get_graph_data(dataset, model_config)
        graph_dataset = dataset.to_graph(graph_data)

        return graph_dataset

    def _get_graph_data(self, dataset: SparseDataset, model_config: Optional[Dict[Any, Any]]) -> Data:
        """
        Create torch_gemetric.data.Data from SparseDataset.

        Args:
            dataset: SparseDataset containing train, test, val data with user, item entries.
            model_config: model_config file for node_init, item_metadata_path.
        
        Returns:
            torch_geometric.data.Data
        """

        assert dataset.variables is not None
        used_cols = dataset.variables.used_cols

        data_split: dict = cast(dict, dataset.data_split)

        train_df = self._create_df_with_recalibrated_row_idxs(dataset.train_data_and_mask, data_split["train_idxs"])
        test_df = self._create_df_with_recalibrated_row_idxs(dataset.test_data_and_mask, data_split["test_idxs"])
        val_df = self._create_df_with_recalibrated_row_idxs(dataset.val_data_and_mask, data_split["val_idxs"])

        if val_df is None:
            all_df = pd.concat([train_df, test_df], ignore_index=True)
        else:
            all_df = pd.concat([train_df, test_df, val_df], ignore_index=True)

        graph_data = self._create_graph_data(all_df, train_df, test_df, val_df, used_cols, model_config)

        return graph_data

    def _create_graph_data(
        self,
        all_df: DataFrame,
        train_df: DataFrame,
        test_df: DataFrame,
        val_df: DataFrame,
        used_cols: List[int],
        model_config: Optional[Dict[Any, Any]],
    ) -> Data:
        """
        Create torch_geometric Data that will be passed as an attribute of GraphDataset object.

        Args:
            all_df, train_df, test_df, val_df: DataFrames that will be converted to the graph structures.
            used_cols: Original indexes in original dataset's columns.
            model_config: model configuration dictionary that is also partially used in the main model class: GraphNeuralNetwork.

        Returns:
            torch_geometric.data.Data object.
        """
        assert model_config is not None
        random_seed = model_config["random_seed"]
        if isinstance(random_seed, List):
            torch.manual_seed(random_seed[0])
        else:
            assert isinstance(random_seed, int)
            torch.manual_seed(random_seed)

        if model_config is None:
            raise TypeError("Model config needs to be correctly specified.")

        NODE_INIT_OPTIONS = (
            "random",
            "grape",
            "topic_init",
            "topic_sep",
            "topic_sep_tree",
            "topic_sep_tree_leaf",
            "text_init",
            "sbert_init",
            "neural_bow_init",
            "bert_cls_init",
            "bert_average_init",
        )

        node_init = model_config["node_init"]
        use_transformer = model_config["use_transformer"]

        assert (
            node_init in NODE_INIT_OPTIONS
        ), f"Option `{node_init}` not recognized. Valid options: `{NODE_INIT_OPTIONS}`"

        assert len(used_cols) == len(train_df.col.unique())
        self._used_cols = used_cols

        item_metadata_df = None
        meta_nodes_to_index = {}
        n_meta = 0

        if node_init in ("topic_init", "topic_sep", "topic_sep_tree", "topic_sep_tree_leaf"):
            item_metadata_file = "metadata/item_metadata_topic.csv"
            item_metadata_path = os.path.join(self._dataset_dir, item_metadata_file)

            if os.path.exists(item_metadata_path) and os.path.splitext(item_metadata_path)[-1] == ".csv":
                item_metadata_df = pd.read_csv(item_metadata_path)
                item_metadata_df = self._match_df_col_idxs(item_metadata_df, "QuestionId")
            else:
                raise TypeError("item_metadata_path needs to be correctly specified.")

        if node_init in ["text_init", "sbert_init", "neural_bow_init", "bert_cls_init", "bert_average_init"]:
            item_metadata_array_path = os.path.join(self._dataset_dir, f"metadata/item_metadata_{node_init[:-5]}.npy")

            assert os.path.exists(item_metadata_array_path), f"Item metadata array for option `{node_init}` not found."

            item_metadata_embedding = np.load(item_metadata_array_path)
            item_metadata_embedding = item_metadata_embedding[self._used_cols]

        if node_init in ["topic_sep", "topic_sep_tree", "topic_sep_tree_leaf"]:
            assert (
                model_config["use_discrete_edge_value"] is True
            ), "Current implementation only allows that when setting node_init to 'topic_sep', 'topic_sep_tree' or 'topic_sep_tree_leaf', use_discrete_edge_value has to be set to True."

            assert (
                model_config["use_edge_metadata"] is False
            ), "Current implementation only supports 'use_edge_metadata' to be False when node_init is set to 'topic_sep', 'topic_sep_tree' or 'topic_sep_tree_leaf'."
        if model_config["use_edge_metadata_for_prediction"] is True:
            assert (
                model_config["use_edge_metadata"] is True
            ), "When use_edge_metadata_for_prediction is set to True, then use_edge_metadata also needs to be set to True."

        n_user = all_df["row"].max() + 1
        n_item = all_df["col"].max() + 1

        if node_init == "grape":
            x_user = torch.ones((n_user, n_item))
            x_item = torch.eye(n_item)
            x = torch.cat((x_user, x_item), dim=0).float()

        elif node_init in [
            "random",
            "text_init",
            "sbert_init",
            "neural_bow_init",
            "bert_cls_init",
            "bert_average_init",
        ]:
            x = torch.randn(int(n_user + n_item), int(model_config["node_input_dim"]))
        elif node_init == "topic_init":
            x_item = self._init_with_topic_metadata(n_item, item_metadata_df)
            x_user = torch.randn((n_user, x_item.shape[1]))
            x = torch.cat((x_user, x_item), dim=0)
        else:
            assert node_init in ["topic_sep", "topic_sep_tree", "topic_sep_tree_leaf"]
            assert isinstance(item_metadata_df, DataFrame)
            meta_lists = [ast.literal_eval(v) for v in item_metadata_df["SubjectId"].unique()]
            meta_nodes = sorted(set(item for meta_list in meta_lists for item in meta_list))
            meta_nodes_to_index = {v: i for i, v in enumerate(meta_nodes)}
            n_meta = len(meta_nodes)
            x = torch.randn(int(n_user + n_item + n_meta), int(model_config["node_input_dim"]))

        edge_index, num_edges_meta, num_edges_tree = self._df2edge_index(
            all_df, n_user, n_item, node_init, item_metadata_df, meta_nodes_to_index,
        )
        train_edge_index, num_edges_meta, num_edges_tree = self._df2edge_index(
            train_df, n_user, n_item, node_init, item_metadata_df, meta_nodes_to_index,
        )
        test_edge_index, num_edges_meta, num_edges_tree = self._df2edge_index(
            test_df, n_user, n_item, node_init, item_metadata_df, meta_nodes_to_index,
        )
        val_edge_index, num_edges_meta, num_edges_tree = self._df2edge_index(
            val_df, n_user, n_item, node_init, item_metadata_df, meta_nodes_to_index,
        )

        edge_attr = torch.tensor(np.expand_dims(np.array(all_df)[:, 2], -1))
        train_edge_attr = torch.tensor(np.expand_dims(np.array(train_df)[:, 2], -1))
        test_edge_attr = torch.tensor(np.expand_dims(np.array(test_df)[:, 2], -1))
        val_edge_attr = torch.tensor(np.expand_dims(np.array(val_df)[:, 2], -1))

        train_labels = torch.tensor(np.array(train_df)[:, 2])
        test_labels = torch.tensor(np.array(test_df)[:, 2])
        val_labels = torch.tensor(np.array(val_df)[:, 2])

        if node_init in ["topic_sep", "topic_sep_tree", "topic_sep_tree_leaf"]:
            if node_init == "topic_sep":
                assert num_edges_meta == edge_index.shape[1] // 2 - edge_attr.shape[0]
                assert num_edges_tree == 0
            label_for_meta = len(edge_attr.unique())
            edge_attr_meta = torch.tensor([label_for_meta] * num_edges_meta)
            label_for_tree = label_for_meta + 1
            edge_attr_tree = torch.tensor([label_for_tree] * num_edges_tree, dtype=torch.int)

            edge_attr = torch.cat((edge_attr, edge_attr_meta[:, None], edge_attr_tree[:, None]))
            train_edge_attr = torch.cat((train_edge_attr, edge_attr_meta[:, None], edge_attr_tree[:, None]))
            test_edge_attr = torch.cat((test_edge_attr, edge_attr_meta[:, None], edge_attr_tree[:, None]))
            val_edge_attr = torch.cat((val_edge_attr, edge_attr_meta[:, None], edge_attr_tree[:, None]))

            train_labels = torch.cat((train_labels, edge_attr_meta, edge_attr_tree))
            test_labels = torch.cat((test_labels, edge_attr_meta, edge_attr_tree))
            val_labels = torch.cat((val_labels, edge_attr_meta, edge_attr_tree))

        if model_config["use_edge_metadata"] is True:
            all_df_merged = self._expand_df_using_edge_metadata(all_df, "all")
            train_df_merged = self._expand_df_using_edge_metadata(train_df, "train")
            test_df_merged = self._expand_df_using_edge_metadata(test_df, "test")
            val_df_merged = self._expand_df_using_edge_metadata(val_df, "val")

            edge_attr_extra = torch.tensor(np.array(all_df_merged)[:, 3:])
            train_edge_attr_extra = torch.tensor(np.array(train_df_merged)[:, 3:])
            test_edge_attr_extra = torch.tensor(np.array(test_df_merged)[:, 3:])
            val_edge_attr_extra = torch.tensor(np.array(val_df_merged)[:, 3:])

            edge_attr = torch.cat((edge_attr, edge_attr_extra), dim=1)
            train_edge_attr = torch.cat((train_edge_attr, train_edge_attr_extra), dim=1)
            test_edge_attr = torch.cat((test_edge_attr, test_edge_attr_extra), dim=1)
            val_edge_attr = torch.cat((val_edge_attr, val_edge_attr_extra), dim=1)

        edge_attr = torch.cat((edge_attr, edge_attr))
        train_edge_attr = torch.cat((train_edge_attr, train_edge_attr))
        test_edge_attr = torch.cat((test_edge_attr, test_edge_attr))
        val_edge_attr = torch.cat((val_edge_attr, val_edge_attr))

        if model_config["use_discrete_edge_value"] is True:
            edge_attr = self._to_onehot(edge_attr)
            train_edge_attr = self._to_onehot(train_edge_attr)
            test_edge_attr = self._to_onehot(test_edge_attr)
            val_edge_attr = self._to_onehot(val_edge_attr)

        edge_attr = edge_attr.float()
        train_edge_attr = train_edge_attr.float()
        test_edge_attr = test_edge_attr.float()
        val_edge_attr = val_edge_attr.float()

        train_labels = train_labels.float()
        test_labels = test_labels.float()
        val_labels = val_labels.float()

        if model_config["use_edge_metadata_for_prediction"] is True:
            prediction_metadata_dim = model_config["prediction_metadata_dim"]
            assert prediction_metadata_dim == edge_attr_extra.shape[1]

            prediction_edge_metadata = edge_attr[:, -prediction_metadata_dim:]
            edge_attr = edge_attr[:, :-prediction_metadata_dim]
            prediction_train_edge_metadata = train_edge_attr[:, -prediction_metadata_dim:]
            train_edge_attr = train_edge_attr[:, :-prediction_metadata_dim]
            prediction_test_edge_metadata = test_edge_attr[:, -prediction_metadata_dim:]
            test_edge_attr = test_edge_attr[:, :-prediction_metadata_dim]
            prediction_val_edge_metadata = val_edge_attr[:, -prediction_metadata_dim:]
            val_edge_attr = val_edge_attr[:, :-prediction_metadata_dim]
        else:
            prediction_edge_metadata = (
                prediction_train_edge_metadata
            ) = prediction_test_edge_metadata = prediction_val_edge_metadata = torch.tensor([], dtype=torch.float)

        if node_init in ["text_init", "sbert_init", "neural_bow_init", "bert_cls_init", "bert_average_init"]:
            item_x = torch.from_numpy(item_metadata_embedding).float()
            assert len(item_x) == n_item, "The length of data.item_x should be equal to n_item."
        else:
            item_x = torch.tensor([], dtype=torch.float)

        sentences_path = os.path.join(self._dataset_dir, "metadata/sentences_array.npy")
        if os.path.exists(sentences_path):
            sentences = np.load(sentences_path)
            sentences = np.concatenate((np.array([""] * n_user), sentences[self._used_cols]))
        else:
            sentences = None

        sentences_encoded = None
        attention_mask = None
        if use_transformer:
            sentences_encoded_path = os.path.join(self._dataset_dir, "metadata/sentences_encoded.pt")
            if os.path.exists(sentences_encoded_path):
                sentences_encoded = torch.load(sentences_encoded_path)
                sentences_encoded = sentences_encoded[self._used_cols]

                attention_mask = torch.load(os.path.join(self._dataset_dir, "metadata/attention_mask.pt"))
                attention_mask = attention_mask[self._used_cols]
                attention_mask = torch.cat((torch.zeros(n_user, attention_mask.shape[1]), attention_mask))
            else:
                print(
                    "Pre-computed sentence_encoded.py file does not exist. This will now be computed in manually in graph_neural_network.py"
                )
        """
        x: (num_items, node_init_dim).
        edge_index: (2, num_edges)
        train_edge_index: (2, num_edges_train)
        test_edge_index: (2, num_edges_test)
        val_edge_index: (2, num_edges_val)
        edge_attr: (num_edges, 1) if not use_discrete_edge_value else (num_edges_train, num_labels)
        train_edge_attr: (num_edges_train, 1) if not use_discrete_edge_value else (num_edges_train, num_labels)
        test_edge_attr: (num_edges_test, 1) if not use_discrete_edge_value else (num_edges_train, num_labels)
        val_edge_attr: (num_edges_val, 1) if not use_discrete_edge_value else (num_edges_train, num_labels)
        train_labels: (num_edges_undirected_train). Here, num_edges_undirected_train*2 == num_edges_train
        test_labels: (num_edges_undirected_test). Here, num_edges_undirected_test*2 == num_edges_test
        val_labels: (num_edges_undirected_val). Here, num_edges_undirected_val*2 == num_edges_val
        num_users, num_items, num_metas: (int)
        class_values: (Tensor)
        prediction_(train or test or val)_edge_metadata: (num_edges, prediction_metadata_dim)
        item_x: (num_item, vocab_size) when node_init is text_init, sbert_init, neural_bow_init, bert_cls_init, bert_average_init
        sentences: (num_nodes, ) sentences as content information used as metadata. Those nodes with
            no sentence information gets empty strings
        sentences_encoded: (num_nodes, max_transformer_length, item_metadata_dim). Word embeddings
            from pre-trained transformers. Use the pre-computed values when it is already saved in the dataset directory.
        attention_mask: (num_nodes, max_transformer_length). Binary values with 1 indicating the actual word occurance
            and 0 indicating the padding.
        """
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            train_edge_index=train_edge_index,
            test_edge_index=test_edge_index,
            val_edge_index=val_edge_index,
            train_edge_attr=train_edge_attr,
            test_edge_attr=test_edge_attr,
            val_edge_attr=val_edge_attr,
            train_labels=train_labels,
            test_labels=test_labels,
            val_labels=val_labels,
            num_users=n_user,
            num_items=n_item,
            num_metas=n_meta,
            class_values=torch.tensor(np.array([0, 1]), dtype=torch.float),
            prediction_edge_metadata=prediction_edge_metadata,
            prediction_train_edge_metadata=prediction_train_edge_metadata,
            prediction_test_edge_metadata=prediction_test_edge_metadata,
            prediction_val_edge_metadata=prediction_val_edge_metadata,
            item_x=item_x,
            sentences=sentences,
            sentences_encoded=sentences_encoded,
            attention_mask=attention_mask,
        )

        return data

    def _create_df_with_recalibrated_row_idxs(
        self, data_and_mask: Tuple[csr_matrix, csr_matrix], idxs: List[int],
    ) -> Optional[DataFrame]:
        """
        Create train, test, or val dataframe with recalibrated row idxs.
        In the newly created dataframe, the row IDs that appeared in the original
        all.csv file are re-used, since the graph_data needs keep track of the row IDs
        as well as the column IDs.

        Args:
            data_and_mask: data and mask tuples that originally appear in Dataset, e.g., train_data_and_mask
            idxs: train, test, val idxs that origianlly appear in SparseDataset, e.g., dataset.data_split["train_idxs"]
        
        Returns:
            Train, test, or val dataframe with recalibrated row indexes if we have the data, else, return None.
        """
        data, _ = data_and_mask
        if data is None:
            return None
        else:
            coo = data.tocoo()
            assert np.array_equal(coo.row, np.sort(coo.row))
            assert idxs == sorted(idxs)
            map_coo_row_to_idxs = {v1: v2 for v1, v2 in zip(np.sort(np.unique(coo.row)), idxs)}
            df_row = np.vectorize(map_coo_row_to_idxs.get)(coo.row)
            df = pd.DataFrame(
                {"row": df_row, "col": coo.col, "val": coo.data.astype(np.float64)}, columns=["row", "col", "val"],
            )
            return df

    def _match_df_col_idxs(self, df: DataFrame, column_name: str):
        """
        Modifies the DataFrame's original column index IDs to the ones that's been updated
        in SparseCSVDatasetLoader.

        Args:
            df: DataFrame whose column indexes will be updated.
            column_name: column name in the given DataFrame.
                The name differs for different types of item metadata.
        """
        map_used_cols = {v: i for i, v in enumerate(self._used_cols)}
        df[column_name] = df[column_name].map(map_used_cols, na_action=None)
        df = df[~df[column_name].isna()]
        return df

    @classmethod
    def _df2edge_index(
        cls,
        df: DataFrame,
        n_user: int,
        n_item: int,
        node_init: str,
        item_metadata_df: Optional[DataFrame],
        meta_nodes_to_index: Dict,
        meta_colname: str = "SubjectId",
    ):
        """
        Make the DataFrame to graph edge index tensor that will be used in the torchgeometric.data.Data.

        Args:
            df: DataFrame that contain edge indices.
            n_user, n_item: Number of users and items.
            node_init: Node initialization options.
            item_metadata_df: Topic metadata DataFrame that will be used as additional nodes when node_init
                            is set to: topic_sep. 
            meta_nodes_to_index: mapping of the metadata nodes graph node indices when node_init = topic_sep.
            meta_colname: column name for metadata lists in item_metadata_df.
        
        Returns:
            edge_index Tensor
        """
        edge_index1 = np.array(df[["row", "col"]]).T
        edge_index1 = edge_index1 + np.array([0, n_user])[:, np.newaxis]

        if node_init in ["topic_sep", "topic_sep_tree", "topic_sep_tree_leaf"]:
            assert item_metadata_df is not None
            assert len(meta_nodes_to_index)
            meta_edge_index = []
            topic_tree = []
            topic_links_set = set()
            for i in item_metadata_df.index:
                row = item_metadata_df.loc[i]
                if isinstance(row[meta_colname], str):
                    topiclist = ast.literal_eval(row[meta_colname])
                else:
                    topiclist = row[meta_colname]
                for j, topic in enumerate(topiclist):
                    meta_index = meta_nodes_to_index[topic]
                    if node_init == "topic_sep_tree_leaf":
                        if j == len(topiclist) - 1:
                            meta_edge_index.append([row.QuestionId, meta_index])
                    else:
                        meta_edge_index.append([row.QuestionId, meta_index])
                    if node_init in ["topic_sep_tree", "topic_sep_tree_leaf"]:
                        if j != len(topiclist) - 1:
                            child_topic = topiclist[j + 1]
                            meta_child_index = meta_nodes_to_index[child_topic]
                            topic_link = [meta_index, meta_child_index]
                            if tuple(topic_link) not in topic_links_set:
                                topic_tree.append(topic_link)
                                topic_links_set.add(tuple(topic_link))
            meta_edge_index_array = np.array(meta_edge_index).T
            meta_edge_index_array = meta_edge_index_array + np.array([n_user, n_user + n_item])[:, np.newaxis]
            if node_init in ["topic_sep_tree", "topic_sep_tree_leaf"]:
                topic_tree_array = np.array(topic_tree).T + n_user + n_item
                edge_index1 = np.concatenate((edge_index1, meta_edge_index_array, topic_tree_array), axis=1)
                num_edges_tree = len(topic_tree)
            else:
                edge_index1 = np.concatenate((edge_index1, meta_edge_index_array), axis=1)
                num_edges_tree = len(topic_tree)
            num_edges_meta = len(meta_edge_index)
        else:
            num_edges_meta = num_edges_tree = 0

        edge_index2 = edge_index1.copy()
        edge_index2[[0, 1], :] = edge_index2[[1, 0], :]

        edge_index = np.concatenate((edge_index1, edge_index2), axis=1)
        return (torch.tensor(edge_index).type(torch.long), num_edges_meta, num_edges_tree)

    @classmethod
    def _init_with_topic_metadata(cls, n_item: int, topic_metadata_df: DataFrame):
        """
        Create node embedding initializations when model_config["node_init"] is set to topic_init.

        Args:
            n_item: number of items in the bypartite graph.
            topic_metadata_df: Topic metadata DataFrame that will be used as additional nodes when node_init
                            is set to: topic_sep. 

        Returns:
            node embeddings initialized with metadata.
        """
        topics_to_index = {v: k for k, v in enumerate(topic_metadata_df.SubjectId.unique())}
        x_item = torch.zeros((n_item, len(topics_to_index)))
        for i in range(len(topic_metadata_df)):
            row = topic_metadata_df.loc[i]
            x_item[row.QuestionId][topics_to_index[row.SubjectId]] = 1.0
        return x_item

    @classmethod
    def _to_onehot(cls, input_tensor: Tensor):
        """
        Convert integer entries to one-hot vectors using torch.nn.functional.one_hot method.
        """
        input_tensor_int = input_tensor.to(torch.int64)
        if torch.sum(input_tensor_int - input_tensor) != 0:
            raise ValueError(
                "Edge attributes should not have non-zero floating points when 'use_discrete_edge_value' is True."
            )
        input_tensor_flatten = input_tensor_int[:, 0]
        num_classes = len(input_tensor_flatten.unique())
        assert num_classes == torch.max(input_tensor_flatten).item() + 1
        encoded = one_hot(input_tensor_flatten, num_classes=num_classes)
        output_tensor = torch.cat((encoded, input_tensor_int[:, 1:]), dim=1)
        return output_tensor

    def _expand_df_using_edge_metadata(self, df: DataFrame, dataset_type: str):
        """
        Expand all, train, test, val datasets with edge metadata information such as 
        the timestamps (ts) and user confidences (cf).

        Args:
            df: all, train, test, val datasets whose columns need to be expanded with edge metadata.
            dataset_type: all, train, test, val.
        """

        col_names = ["row", "col", "val", "ts", "cf_0.0", "cf_25.0", "cf_50.0", "cf_75.0", "cf_100.0"]

        expanded_df_path = os.path.join(self._dataset_dir, "{}_expanded.csv".format(dataset_type))
        use_saved_df = False
        if os.path.exists(expanded_df_path):
            df_expanded = pd.read_csv(expanded_df_path, header=None, names=col_names)
            if len(df_expanded.columns) == len(col_names) and len(df_expanded.dropna()) != 0:
                use_saved_df = True
        if not use_saved_df:
            all_df_expanded_path = os.path.join(self._dataset_dir, "metadata/task_expanded.csv")
            all_df_expanded = pd.read_csv(all_df_expanded_path)
            edge_meta_path = os.path.join(self._dataset_dir, "metadata/edge_metadata.csv")

            all_df_expanded = all_df_expanded.rename(columns={"QuestionId": "col", "UserId": "row", "IsCorrect": "val"})
            all_df_expanded = all_df_expanded[["row", "col", "val", "AnswerId"]]

            all_df_expanded = self._match_df_col_idxs(all_df_expanded, "col")

            df_merged = df.merge(all_df_expanded, on=["row", "col", "val"])
            assert len(df) == len(df_merged)

            edge_meta_df = pd.read_csv(edge_meta_path)
            edge_meta_df["ts"] = pd.to_datetime(edge_meta_df["DateAnswered"]).values.astype(np.float_)
            ts_min, ts_max = edge_meta_df["ts"].min(), edge_meta_df["ts"].max()
            edge_meta_df["ts"] = (edge_meta_df["ts"] - ts_min) / (ts_max - ts_min)

            df_merged = df_merged.merge(edge_meta_df, on=["AnswerId"])
            df_merged = df_merged.drop(columns=["DateAnswered"])
            y = pd.get_dummies(df_merged.Confidence, prefix="cf", columns=["one", "two", "three", "four"])
            df_merged[y.columns] = y
            df_merged = df_merged.drop(columns=["Confidence"])
            df_expanded = df_merged[col_names]
            df_expanded.to_csv(expanded_df_path, header=None, index=None)
        return df_expanded

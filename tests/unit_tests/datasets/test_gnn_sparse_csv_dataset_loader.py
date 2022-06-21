import os

import numpy as np
import pandas as pd
import pytest
from torch import Tensor
from torch_geometric.data import Data

from azua.datasets.gnn_sparse_csv_dataset_loader import GNNSparseCSVDatasetLoader
from azua.datasets.dataset import GraphDataset

default_data = np.array(
    [
        [0, 0, 1],  # Full data: 10 rows
        [1, 0, 0],
        [2, 0, 1],
        [3, 0, 1],
        [4, 0, 0],
        [0, 2, 1],
        [1, 2, 0],
        [4, 2, 1],
        [3, 2, 1],
        [2, 2, 0],
    ],
)

default_model_config = {
    "metadata_path": "NONE",
    "node_init": "random",
    "use_discrete_edge_value": False,
    "node_input_dim": 3,
    "use_edge_metadata": False,
    "use_edge_metadata_for_prediction": False,
    "prediction_metadata_dim": 0,
    "random_seed": 0,
    "use_transformer": False,
    "is_inductive_task": True,
}

default_test_val_frac = (0.2, 0.2)


@pytest.fixture
def gnn_sparse_csv_dataset_loader(tmpdir_factory, data=default_data):
    dataset_dir = tmpdir_factory.mktemp("dataset_dir")
    pd.DataFrame(data).to_csv(os.path.join(dataset_dir, "all.csv"), header=None, index=None)

    return GNNSparseCSVDatasetLoader(dataset_dir=dataset_dir)


@pytest.mark.parametrize(
    "model_config, test_val_frac",
    [(default_model_config, default_test_val_frac)],
)
def test_split_data_and_load_dataset_types(gnn_sparse_csv_dataset_loader, model_config, test_val_frac):
    test_frac = test_val_frac[0]
    val_frac = test_val_frac[1]

    dataset = gnn_sparse_csv_dataset_loader.split_data_and_load_dataset(
        test_frac=test_frac, val_frac=val_frac, random_state=0, max_num_rows=None, model_config=model_config
    )
    graph_data = dataset.get_graph_data_object()

    assert isinstance(dataset, GraphDataset)
    assert isinstance(graph_data, Data)

    assert isinstance(graph_data.x, Tensor)
    assert isinstance(graph_data.edge_index, Tensor)
    assert isinstance(graph_data.train_edge_index, Tensor)
    assert isinstance(graph_data.test_edge_index, Tensor)
    assert isinstance(graph_data.val_edge_index, Tensor)
    assert isinstance(graph_data.edge_attr, Tensor)
    assert isinstance(graph_data.train_edge_attr, Tensor)
    assert isinstance(graph_data.test_edge_attr, Tensor)
    assert isinstance(graph_data.val_edge_attr, Tensor)
    assert isinstance(graph_data.train_labels, Tensor)
    assert isinstance(graph_data.test_labels, Tensor)
    assert isinstance(graph_data.val_labels, Tensor)
    assert isinstance(graph_data.num_users, np.int64)
    assert isinstance(graph_data.num_items, np.int64)
    assert isinstance(graph_data.num_metas, int)


@pytest.mark.parametrize(
    "model_config, test_val_frac",
    [(default_model_config, default_test_val_frac)],
)
def test_split_data_and_load_dataset_sizes(
    gnn_sparse_csv_dataset_loader,
    model_config,
    test_val_frac,
    expected_num_nodes=0,
    expected_num_edges=0,
    expected_num_train_edges=0,
    expected_num_test_edges=0,
    expected_num_val_edges=0,
):
    test_frac = test_val_frac[0]
    val_frac = test_val_frac[1]

    dataset = gnn_sparse_csv_dataset_loader.split_data_and_load_dataset(
        test_frac=test_frac, val_frac=val_frac, random_state=0, max_num_rows=None, model_config=model_config
    )

    graph_data = dataset.get_graph_data_object()
    used_cols = gnn_sparse_csv_dataset_loader._used_cols

    expected_num_nodes = graph_data.num_users + graph_data.num_items
    expected_num_edges = graph_data.edge_attr.shape[0]
    expected_num_train_edges = graph_data.train_edge_attr.shape[0]
    expected_num_test_edges = graph_data.test_edge_attr.shape[0]
    expected_num_val_edges = graph_data.val_edge_attr.shape[0]

    assert graph_data.x.shape == (7, 3)
    assert graph_data.num_nodes == 7  # 5 users, 2 items
    assert graph_data.num_edges == 20
    assert graph_data.num_nodes == expected_num_nodes
    assert graph_data.num_metas == 0
    assert expected_num_edges == expected_num_train_edges + expected_num_test_edges + expected_num_val_edges
    assert expected_num_edges == graph_data.edge_index.shape[1]
    assert expected_num_train_edges == graph_data.train_edge_index.shape[1]
    assert expected_num_test_edges == graph_data.test_edge_index.shape[1]
    assert expected_num_val_edges == graph_data.val_edge_index.shape[1]
    assert len(used_cols)


@pytest.mark.parametrize(
    "use_discrete_edge_value, test_val_frac, edge_input_dim",
    [
        (True, (0.1, 0.3), 2),
        (False, (0.1, 0.3), 1),
        (True, (0.3, 0.1), 2),
        (False, (0.3, 0.1), 1),
    ],
)
def test_split_data_and_load_dataset_sizes_with_varying_model_configs_and_test_val_fracs(
    gnn_sparse_csv_dataset_loader,
    use_discrete_edge_value,
    test_val_frac,
    edge_input_dim,
    expected_num_nodes=0,
    expected_num_edges=0,
    expected_num_train_edges=0,
    expected_num_test_edges=0,
    expected_num_val_edges=0,
):
    test_frac = test_val_frac[0]
    val_frac = test_val_frac[1]

    default_model_config["use_discrete_edge_value"] = use_discrete_edge_value

    dataset = gnn_sparse_csv_dataset_loader.split_data_and_load_dataset(
        test_frac=test_frac, val_frac=val_frac, random_state=0, max_num_rows=None, model_config=default_model_config
    )
    graph_data = dataset.get_graph_data_object()

    expected_num_edges = graph_data.edge_attr.shape[0]
    expected_num_train_edges = graph_data.train_edge_attr.shape[0]
    expected_num_test_edges = graph_data.test_edge_attr.shape[0]
    expected_num_val_edges = graph_data.val_edge_attr.shape[0]
    expected_edge_input_dim = graph_data.edge_attr.shape[1]

    assert graph_data.num_nodes == graph_data.num_users + graph_data.num_items
    assert graph_data.num_metas == 0
    assert expected_num_edges == expected_num_train_edges + expected_num_test_edges + expected_num_val_edges
    assert expected_num_edges == graph_data.edge_index.shape[1]
    assert expected_num_train_edges == graph_data.train_edge_index.shape[1]
    assert expected_num_test_edges == graph_data.test_edge_index.shape[1]
    assert expected_num_val_edges == graph_data.val_edge_index.shape[1]

    assert expected_edge_input_dim == edge_input_dim


@pytest.mark.parametrize(
    "model_config, test_val_frac, expected_num_nodes, expected_num_edges, expected_num_train_edges, expected_num_test_edges, expected_num_val_edges",
    [(default_model_config, default_test_val_frac, 0, 0, 0, 0, 0)],
)
def test_load_predefined_dataset(
    model_config,
    test_val_frac,
    expected_num_nodes,
    expected_num_edges,
    expected_num_train_edges,
    expected_num_test_edges,
    expected_num_val_edges,
    tmpdir_factory,
):
    dataset_dir = tmpdir_factory.mktemp("dataset_dir")

    train_data = np.array([[0, 0, 1], [2, 1, 0]])
    val_data = np.array([[1, 0, 0]])
    test_data = np.array([[1, 1, 0], [1, 2, 1]])

    pd.DataFrame(train_data).to_csv(os.path.join(dataset_dir, "train.csv"), header=None, index=None)
    pd.DataFrame(val_data).to_csv(os.path.join(dataset_dir, "val.csv"), header=None, index=None)
    pd.DataFrame(test_data).to_csv(os.path.join(dataset_dir, "test.csv"), header=None, index=None)

    dataset_loader = GNNSparseCSVDatasetLoader(dataset_dir=dataset_dir)
    dataset = dataset_loader.load_predefined_dataset(model_config=model_config)
    used_cols = dataset_loader._used_cols

    graph_data = dataset.get_graph_data_object()
    expected_num_nodes = graph_data.num_users + graph_data.num_items
    expected_num_edges = graph_data.edge_attr.shape[0]
    expected_num_train_edges = graph_data.train_edge_attr.shape[0]
    expected_num_test_edges = graph_data.test_edge_attr.shape[0]
    expected_num_val_edges = graph_data.val_edge_attr.shape[0]

    assert graph_data.x.shape == (5, 3)
    assert graph_data.num_nodes == 5  # 3 users, 2 items
    assert graph_data.num_edges == 8
    assert graph_data.num_nodes == expected_num_nodes
    assert graph_data.num_metas == 0
    assert expected_num_edges == expected_num_train_edges + expected_num_test_edges + expected_num_val_edges
    assert expected_num_edges == graph_data.edge_index.shape[1]
    assert expected_num_train_edges == graph_data.train_edge_index.shape[1]
    assert expected_num_test_edges == graph_data.test_edge_index.shape[1]
    assert expected_num_val_edges == graph_data.val_edge_index.shape[1]
    assert len(used_cols)
    assert np.array_equal(np.array(used_cols), np.array([0, 1]))


@pytest.mark.parametrize(
    "model_config, test_val_frac",
    [(default_model_config, default_test_val_frac)],
)
def test_match_df_col_idxs_from_load_predefined_dataset(
    model_config,
    test_val_frac,
    tmpdir_factory,
):
    dataset_dir = tmpdir_factory.mktemp("dataset_dir")

    train_data = np.array([[0, 0, 1], [2, 1, 0], [0, 3, 1], [0, 10, 1]])
    val_data = np.array([[1, 0, 0]])
    test_data = np.array([[1, 1, 0], [1, 2, 1]])

    pd.DataFrame(train_data).to_csv(os.path.join(dataset_dir, "train.csv"), header=None, index=None)
    pd.DataFrame(val_data).to_csv(os.path.join(dataset_dir, "val.csv"), header=None, index=None)
    pd.DataFrame(test_data).to_csv(os.path.join(dataset_dir, "test.csv"), header=None, index=None)

    expected_previous_num_df_rows = 5
    expected_updated_num_df_rows = 4
    expected_col_idxs = np.array([0, 1, 2, 3])

    item_metadata = np.array([[0, [10, 20, 30, 40]], [1, [30, 60, 10]], [2, [10, 20, 100]], [3, [70]], [10, [33]]])
    item_metadata_df = pd.DataFrame(item_metadata, columns=["TestColumnId", "TestItemMetadata"])

    dataset_loader = GNNSparseCSVDatasetLoader(dataset_dir=dataset_dir)
    _ = dataset_loader.load_predefined_dataset(model_config=model_config)

    updated_item_metadata_df = dataset_loader._match_df_col_idxs(item_metadata_df, "TestColumnId")

    assert len(item_metadata_df) == expected_previous_num_df_rows
    assert len(updated_item_metadata_df) == expected_updated_num_df_rows
    assert np.array_equal(expected_col_idxs, np.sort(updated_item_metadata_df["TestColumnId"].unique()))


@pytest.mark.parametrize(
    "model_config, test_val_frac",
    [(default_model_config, default_test_val_frac)],
)
def test_metadata_methods(gnn_sparse_csv_dataset_loader, model_config, test_val_frac):
    test_frac = test_val_frac[0]
    val_frac = test_val_frac[1]

    dataset = gnn_sparse_csv_dataset_loader.split_data_and_load_dataset(
        test_frac=test_frac, val_frac=val_frac, random_state=0, max_num_rows=None, model_config=model_config
    )

    train_data_and_mask = dataset.train_data_and_mask
    train_idxs = dataset.data_split["train_idxs"]
    train_df = gnn_sparse_csv_dataset_loader._create_df_with_recalibrated_row_idxs(train_data_and_mask, train_idxs)

    test_data_and_mask = dataset.test_data_and_mask
    test_idxs = dataset.data_split["test_idxs"]
    test_df = gnn_sparse_csv_dataset_loader._create_df_with_recalibrated_row_idxs(test_data_and_mask, test_idxs)

    val_data_and_mask = dataset.val_data_and_mask
    val_idxs = dataset.data_split["val_idxs"]
    val_df = gnn_sparse_csv_dataset_loader._create_df_with_recalibrated_row_idxs(val_data_and_mask, val_idxs)

    all_df = pd.concat([train_df, test_df, val_df], ignore_index=True)

    assert len(all_df.row.unique()) == 5
    assert len(all_df.col.unique()) == 2

    assert np.array_equal(np.sort(all_df.row.unique()), np.array([0, 1, 2, 3, 4]))
    assert np.array_equal(np.sort(all_df.col.unique()), np.array([0, 1]))

    item_metadata = np.array([[0, [10, 20, 30, 40]], [1, [30, 60, 10]], [2, [10, 20, 100, 300]], [3, [70]], [10, [33]]])
    item_metadata_df = pd.DataFrame(item_metadata, columns=["QuestionId", "TestItemMetadata"])
    item_metadata_df = gnn_sparse_csv_dataset_loader._match_df_col_idxs(item_metadata_df, "QuestionId")

    assert np.array_equal(np.sort(item_metadata_df["QuestionId"].unique()), np.array([0, 1]))
    assert np.array_equal(np.sort(np.array(item_metadata_df.index)), np.array([0, 2]))

    meta_lists = [v for v in item_metadata_df["TestItemMetadata"]]

    # meta_nodes corresponds to e.g., unique word IDs if we are using text metadata, or, unique topic IDs if we are using topic metadata.
    meta_nodes = sorted(list(set([item for meta_list in meta_lists for item in meta_list])))
    meta_nodes_to_index = {v: i for i, v in enumerate(meta_nodes)}
    n_meta = len(meta_nodes)

    assert meta_nodes == [10, 20, 30, 40, 100, 300]
    assert n_meta == 6

    # These numbers are confirmed from the previous tests.
    n_user = 5
    n_item = 2

    node_init1 = "topic_init"
    edge_index, num_edges_meta, num_edges_tree = gnn_sparse_csv_dataset_loader._df2edge_index(
        all_df, n_user, n_item, node_init1, item_metadata_df, meta_nodes_to_index
    )

    assert edge_index.shape == (2, 20)  # 10 rows * 2 because of bidirectional edges
    assert num_edges_meta == 0
    assert num_edges_tree == 0

    node_init2 = "topic_sep"
    edge_index, num_edges_meta, num_edges_tree = gnn_sparse_csv_dataset_loader._df2edge_index(
        all_df, n_user, n_item, node_init2, item_metadata_df, meta_nodes_to_index, "TestItemMetadata"
    )

    assert num_edges_meta == 8  # 4 edges from item 0, 4 edges from item 2
    assert edge_index.shape == (
        2,
        36,
    )  # 10 rows * 2 because of bidirectional edges + (4+4)*2 bidirectinoal question-metadata edges
    assert num_edges_tree == 0

    node_init3 = "topic_sep_tree_leaf"
    edge_index, num_edges_meta, num_edges_tree = gnn_sparse_csv_dataset_loader._df2edge_index(
        all_df, n_user, n_item, node_init3, item_metadata_df, meta_nodes_to_index, "TestItemMetadata"
    )

    assert edge_index.shape == (2, 34)  # {10 rows + 2 leaf connections + (2 + 3) item-item connections} * 2 = 34

    node_init4 = "topic_sep_tree"
    edge_index, num_edges_meta, num_edges_tree = gnn_sparse_csv_dataset_loader._df2edge_index(
        all_df, n_user, n_item, node_init4, item_metadata_df, meta_nodes_to_index, "TestItemMetadata"
    )

    assert edge_index.shape == (2, 46)  # 36 (topic_sep) + {(2 + 3) item-item connections} * 2 (bidirectionality)

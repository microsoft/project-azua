import os

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from azua.datasets.sparse_csv_dataset_loader import SparseCSVDatasetLoader
from azua.datasets.dataset import SparseDataset
from azua.utils.io_utils import read_json_as, save_json


def test_split_data_and_load_dataset(tmpdir_factory):
    dataset_dir = tmpdir_factory.mktemp("dataset_dir")
    data = np.array(
        [
            [0, 0, 2.1],  # Full data: 10 rows
            [1, 0, 2.1],
            [2, 0, 2.1],
            [3, 0, 2.1],
            [4, 0, 2.1],
            [5, 0, 2.1],
            [6, 0, 2.1],
            [7, 0, 2.1],
            [8, 0, 2.1],
            [9, 0, 2.1],
        ]
    )

    pd.DataFrame(data).to_csv(os.path.join(dataset_dir, "all.csv"), header=None, index=None)
    dataset_loader = SparseCSVDatasetLoader(dataset_dir=dataset_dir)
    dataset = dataset_loader.split_data_and_load_dataset(test_frac=0.5, val_frac=0.2, random_state=0, max_num_rows=None)

    expected_train_data = np.array([[2.1], [2.1], [2.1]])  # Train data: 3 rows
    expected_train_mask = np.array([[1], [1], [1]])
    expected_val_data = np.array([[2.1], [2.1]])  # Val data: 2 rows
    expected_val_mask = np.array([[1], [1]])
    expected_test_data = np.array([[2.1], [2.1], [2.1], [2.1], [2.1]])  # Test data: 5 rows
    expected_test_mask = np.array([[1], [1], [1], [1], [1]])

    assert type(dataset) == SparseDataset
    assert np.array_equal(expected_train_data, dataset.train_data_and_mask[0].toarray())
    assert np.array_equal(expected_train_mask, dataset.train_data_and_mask[1].toarray())
    assert np.array_equal(expected_val_data, dataset.val_data_and_mask[0].toarray())
    assert np.array_equal(expected_val_mask, dataset.val_data_and_mask[1].toarray())
    assert np.array_equal(expected_test_data, dataset.test_data_and_mask[0].toarray())
    assert np.array_equal(expected_test_mask, dataset.test_data_and_mask[1].toarray())


def test_split_data_and_load_dataset_zero_val_frac(tmpdir_factory):
    dataset_dir = tmpdir_factory.mktemp("dataset_dir")
    data = np.array(
        [
            [0, 0, 2.1],  # Full data: 10 rows
            [1, 0, 2.1],
            [2, 0, 2.1],
            [3, 0, 2.1],
            [4, 0, 2.1],
            [5, 0, 2.1],
            [6, 0, 2.1],
            [7, 0, 2.1],
            [8, 0, 2.1],
            [9, 0, 2.1],
        ]
    )

    pd.DataFrame(data).to_csv(os.path.join(dataset_dir, "all.csv"), header=None, index=None)
    dataset_loader = SparseCSVDatasetLoader(dataset_dir=dataset_dir)
    dataset = dataset_loader.split_data_and_load_dataset(test_frac=0.1, val_frac=0.0, random_state=0, max_num_rows=None)

    expected_train_data = np.array(
        [[2.1], [2.1], [2.1], [2.1], [2.1], [2.1], [2.1], [2.1], [2.1]]
    )  # Train data: 9 rows
    expected_train_mask = np.array([[1], [1], [1], [1], [1], [1], [1], [1], [1]])
    expected_val_data = None  # Val data: 0 rows -> None
    expected_val_mask = None
    expected_test_data = np.array([[2.1]])  # Test data: 1 row
    expected_test_mask = np.array([[1]])

    assert type(dataset) == SparseDataset
    assert np.array_equal(expected_train_data, dataset.train_data_and_mask[0].toarray())
    assert np.array_equal(expected_train_mask, dataset.train_data_and_mask[1].toarray())
    assert np.array_equal(expected_val_data, dataset.val_data_and_mask[0])
    assert np.array_equal(expected_val_mask, dataset.val_data_and_mask[1])
    assert np.array_equal(expected_test_data, dataset.test_data_and_mask[0].toarray())
    assert np.array_equal(expected_test_mask, dataset.test_data_and_mask[1].toarray())


def test_split_data_and_load_dataset_save_data_split(tmpdir_factory):
    dataset_dir = tmpdir_factory.mktemp("dataset_dir")
    data = np.array(
        [
            [0, 0, 0],  # Full data: 10 rows
            [1, 0, 1],
            [2, 0, 2],
            [3, 0, 3],
            [4, 0, 4],
            [5, 0, 5],
            [6, 0, 6],
            [7, 0, 7],
            [8, 0, 8],
            [9, 0, 9],
        ]
    )

    pd.DataFrame(data).to_csv(os.path.join(dataset_dir, "all.csv"), header=None, index=None)
    dataset_loader = SparseCSVDatasetLoader(dataset_dir=dataset_dir)
    dataset = dataset_loader.split_data_and_load_dataset(test_frac=0.3, val_frac=0.2, random_state=0, max_num_rows=None)

    save_dir = tmpdir_factory.mktemp("save_dir")
    dataset.save_data_split(save_dir=str(save_dir))

    saved_train_data = dataset.train_data_and_mask[0].toarray()
    saved_val_data = dataset.val_data_and_mask[0].toarray()
    saved_test_data = dataset.test_data_and_mask[0].toarray()

    data_split = read_json_as(os.path.join(save_dir, "data_split.json"), dict)

    assert data_split["train_idxs"] == list(saved_train_data[:, 0])
    assert data_split["val_idxs"] == list(saved_val_data[:, 0])
    assert data_split["test_idxs"] == list(saved_test_data[:, 0])


@pytest.mark.parametrize(
    "test_frac, val_frac", [(None, 0.25), (0.25, None), (None, None), (1, 0.25), (0.25, 1), (0.5, 0.5)]
)
def test_split_data_and_load_dataset_invalid_test_frac_val_frac_raises_error(tmpdir_factory, test_frac, val_frac):
    dataset_dir = tmpdir_factory.mktemp("dataset_dir")
    data = np.array([[0, 0, 1.0], [1, 0, 1.0], [2, 0, 1.0], [3, 0, 1.0]])
    pd.DataFrame(data).to_csv(os.path.join(dataset_dir, "all.csv"), header=None, index=None)
    with pytest.raises(AssertionError):
        dataset_loader = SparseCSVDatasetLoader(dataset_dir=dataset_dir)
        _ = dataset_loader.split_data_and_load_dataset(
            test_frac=test_frac, val_frac=val_frac, random_state=0, max_num_rows=None
        )


@pytest.mark.parametrize("random_state", [(0), ((0, 1))])
def test_split_data_and_load_dataset_deterministic(tmpdir_factory, random_state):
    dataset_dir = tmpdir_factory.mktemp("dataset_dir")
    data = np.array(
        [
            [0, 1, 0],
            [0, 2, 0],
            [0, 3, 0],
            [1, 0, 1],
            [1, 2, 1],
            [1, 3, 1],
            [2, 0, 2],
            [2, 1, 2],
            [2, 3, 2],
            [3, 0, 3],
            [3, 1, 3],
            [3, 2, 3],
        ]
    )

    pd.DataFrame(data).to_csv(os.path.join(dataset_dir, "all.csv"), header=None, index=None)
    dataset_loader = SparseCSVDatasetLoader(dataset_dir=dataset_dir)
    dataset1 = dataset_loader.split_data_and_load_dataset(
        test_frac=0.25, val_frac=0.25, random_state=random_state, max_num_rows=None
    )
    dataset2 = dataset_loader.split_data_and_load_dataset(
        test_frac=0.25, val_frac=0.25, random_state=random_state, max_num_rows=None
    )

    assert np.array_equal(dataset1.train_data_and_mask[0].toarray(), dataset2.train_data_and_mask[0].toarray())
    assert np.array_equal(dataset1.train_data_and_mask[1].toarray(), dataset2.train_data_and_mask[1].toarray())
    assert np.array_equal(dataset1.val_data_and_mask[0].toarray(), dataset2.val_data_and_mask[0].toarray())
    assert np.array_equal(dataset1.val_data_and_mask[1].toarray(), dataset2.val_data_and_mask[1].toarray())
    assert np.array_equal(dataset1.test_data_and_mask[0].toarray(), dataset2.test_data_and_mask[0].toarray())
    assert np.array_equal(dataset1.test_data_and_mask[1].toarray(), dataset2.test_data_and_mask[1].toarray())


def test_split_data_and_load_dataset_deterministic_test_set(tmpdir_factory):
    dataset_dir = tmpdir_factory.mktemp("dataset_dir")
    data = np.array(
        [
            [0, 1, 0],
            [0, 2, 0],
            [0, 3, 0],
            [1, 0, 1],
            [1, 2, 1],
            [1, 3, 1],
            [2, 0, 2],
            [2, 1, 2],
            [2, 3, 2],
            [3, 0, 3],
            [3, 1, 3],
            [3, 2, 3],
        ]
    )

    pd.DataFrame(data).to_csv(os.path.join(dataset_dir, "all.csv"), header=None, index=None)
    dataset_loader = SparseCSVDatasetLoader(dataset_dir=dataset_dir)
    dataset1 = dataset_loader.split_data_and_load_dataset(
        test_frac=0.25, val_frac=0.25, random_state=(0, 1), max_num_rows=None
    )
    dataset2 = dataset_loader.split_data_and_load_dataset(
        test_frac=0.25, val_frac=0.25, random_state=(0, 2), max_num_rows=None
    )

    assert not np.array_equal(dataset1.train_data_and_mask[0].toarray(), dataset2.train_data_and_mask[0].toarray())
    assert not np.array_equal(dataset1.train_data_and_mask[1].toarray(), dataset2.train_data_and_mask[1].toarray())
    assert not np.array_equal(dataset1.val_data_and_mask[0].toarray(), dataset2.val_data_and_mask[0].toarray())
    assert not np.array_equal(dataset1.val_data_and_mask[1].toarray(), dataset2.val_data_and_mask[1].toarray())
    assert np.array_equal(dataset1.test_data_and_mask[0].toarray(), dataset2.test_data_and_mask[0].toarray())
    assert np.array_equal(dataset1.test_data_and_mask[1].toarray(), dataset2.test_data_and_mask[1].toarray())


def test_split_data_and_load_dataset_with_duplicates(tmpdir_factory):
    # Check that only the last entry is taken for each (row, col, val) triple.
    dataset_dir = tmpdir_factory.mktemp("dataset_dir")
    data = np.array(
        [
            [0, 0, 1.1],  # Full data with duplicates
            [1, 0, 2.1],
            [2, 0, 2.1],
            [3, 0, 2.1],
            [4, 0, 2.1],
            [5, 0, 2.1],
            [6, 0, 2.1],
            [7, 0, 1.1],
            [8, 0, 2.1],
            [9, 0, 2.1],
            [0, 0, 2.1],
            [7, 0, 2.1],
        ]
    )

    pd.DataFrame(data).to_csv(os.path.join(dataset_dir, "all.csv"), header=None, index=None)
    dataset_loader = SparseCSVDatasetLoader(dataset_dir=dataset_dir)
    dataset = dataset_loader.split_data_and_load_dataset(test_frac=0.5, val_frac=0.2, random_state=0, max_num_rows=None)

    expected_train_data = np.array([[2.1], [2.1], [2.1]])  # Train data: 3 rows
    expected_train_mask = np.array([[1], [1], [1]])
    expected_val_data = np.array([[2.1], [2.1]])  # Val data: 2 rows
    expected_val_mask = np.array([[1], [1]])
    expected_test_data = np.array([[2.1], [2.1], [2.1], [2.1], [2.1]])  # Test data: 5 rows
    expected_test_mask = np.array([[1], [1], [1], [1], [1]])

    assert type(dataset) == SparseDataset
    assert np.array_equal(expected_train_data, dataset.train_data_and_mask[0].toarray())
    assert np.array_equal(expected_train_mask, dataset.train_data_and_mask[1].toarray())
    assert np.array_equal(expected_val_data, dataset.val_data_and_mask[0].toarray())
    assert np.array_equal(expected_val_mask, dataset.val_data_and_mask[1].toarray())
    assert np.array_equal(expected_test_data, dataset.test_data_and_mask[0].toarray())
    assert np.array_equal(expected_test_mask, dataset.test_data_and_mask[1].toarray())


def test_load_predefined_dataset(tmpdir_factory):
    dataset_dir = tmpdir_factory.mktemp("dataset_dir")
    train_data = np.array([[0, 0, 2.1], [1, 1, 2.2]])
    val_data = np.array([[1, 0, 3.1]])
    test_data = np.array([[1, 1, 4.1], [1, 2, 4.2]])
    pd.DataFrame(train_data).to_csv(os.path.join(dataset_dir, "train.csv"), header=None, index=None)
    pd.DataFrame(val_data).to_csv(os.path.join(dataset_dir, "val.csv"), header=None, index=None)
    pd.DataFrame(test_data).to_csv(os.path.join(dataset_dir, "test.csv"), header=None, index=None)

    dataset_loader = SparseCSVDatasetLoader(dataset_dir=dataset_dir)
    dataset = dataset_loader.load_predefined_dataset(max_num_rows=None)

    expected_train_data = np.array([[2.1, 0.0], [0.0, 2.2]])
    expected_train_mask = np.array([[1, 0], [0, 1]])
    expected_val_data = np.array([[3.1, 0.0]])
    expected_val_mask = np.array([[1, 0]])
    expected_test_data = np.array([[0.0, 4.1]])
    expected_test_mask = np.array([[0, 1]])

    assert type(dataset) == SparseDataset
    assert np.array_equal(expected_train_data, dataset.train_data_and_mask[0].toarray())
    assert np.array_equal(expected_train_mask, dataset.train_data_and_mask[1].toarray())
    assert np.array_equal(expected_val_data, dataset.val_data_and_mask[0].toarray())
    assert np.array_equal(expected_val_mask, dataset.val_data_and_mask[1].toarray())
    assert np.array_equal(expected_test_data, dataset.test_data_and_mask[0].toarray())
    assert np.array_equal(expected_test_mask, dataset.test_data_and_mask[1].toarray())


def test_load_predefined_dataset_save_data_split(tmpdir_factory):
    dataset_dir = tmpdir_factory.mktemp("dataset_dir")
    train_data = np.array([[0, 0, 2.1]])
    val_data = np.array([[0, 0, 2.1]])
    test_data = np.array([[0, 0, 4.1]])
    pd.DataFrame(train_data).to_csv(os.path.join(dataset_dir, "train.csv"), header=None, index=None)
    pd.DataFrame(val_data).to_csv(os.path.join(dataset_dir, "val.csv"), header=None, index=None)
    pd.DataFrame(test_data).to_csv(os.path.join(dataset_dir, "test.csv"), header=None, index=None)

    dataset_loader = SparseCSVDatasetLoader(dataset_dir=dataset_dir)
    dataset = dataset_loader.load_predefined_dataset(max_num_rows=None)

    save_dir = tmpdir_factory.mktemp("save_dir")
    dataset.save_data_split(save_dir=str(save_dir))

    expected_data_split = {
        "train_idxs": [0],
        "test_idxs": [0],
        "val_idxs": [0],
    }

    assert dataset.data_split == expected_data_split

    data_split = read_json_as(os.path.join(save_dir, "data_split.json"), dict)
    assert data_split == expected_data_split


def test_load_data_max_num_rows_raises_error(tmpdir_factory):
    dataset_dir = tmpdir_factory.mktemp("dataset_dir")
    dataset_loader = SparseCSVDatasetLoader(dataset_dir=dataset_dir)
    with pytest.raises(NotImplementedError):
        _ = dataset_loader.load_predefined_dataset(max_num_rows=1)
    with pytest.raises(NotImplementedError):
        _ = dataset_loader.split_data_and_load_dataset(test_frac=0.3, val_frac=0.2, random_state=0, max_num_rows=1)


def test_load_variables_dict(tmpdir_factory):
    dataset_dir = tmpdir_factory.mktemp("dataset_dir")
    variables_dict = {"metadata_variables": [], "variables": [{"id": 0}]}
    variables_dict_path = str(os.path.join(dataset_dir, "variables.json"))
    save_json(data=variables_dict, path=variables_dict_path)

    dataset_loader = SparseCSVDatasetLoader(dataset_dir=dataset_dir)
    loaded_variables_dict = dataset_loader._load_variables_dict(used_cols=[0])

    expected_variables_dict = {"metadata_variables": [], "variables": [{"id": 0}], "used_cols": [0]}

    assert loaded_variables_dict == expected_variables_dict


def test_load_variables_dict_file_doesnt_exist(tmpdir_factory):
    dataset_dir = tmpdir_factory.mktemp("dataset_dir")
    dataset_loader = SparseCSVDatasetLoader(dataset_dir=dataset_dir)
    loaded_variables_dict = dataset_loader._load_variables_dict(used_cols=[0])

    expected_variables_dict = {"used_cols": [0]}

    assert loaded_variables_dict == expected_variables_dict


@pytest.mark.parametrize(
    "used_cols, expected_used_cols, expected_used_rows, expected_data, expected_mask, expected_shape",
    [
        (
            None,
            [0, 1, 3],
            [1, 2],
            np.array([[1.1, 0, 2.1], [0, 3.1, 0]]),
            np.array([[1, 0, 1], [0, 1, 0]]),
            (2, 3),
        ),  # Column 2 removed, Row 0 removed
        (
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [1, 2],
            np.array([[1.1, 0, 0, 2.1], [0, 3.1, 0, 0]]),
            np.array([[1, 0, 0, 1], [0, 1, 0, 0]]),
            (2, 4),
        ),  # No columns removed, Row 0 removed
        (
            [0, 1, 2],
            [0, 1, 2],
            [1, 2],
            np.array([[1.1, 0, 0], [0, 3.1, 0]]),
            np.array([[1, 0, 0], [0, 1, 0]]),
            (2, 3),
        ),  # Column 3 removed, Row 0 removed
        (
            [0, 2, 3],
            [0, 2, 3],
            [1],
            np.array([[1.1, 0, 2.1]]),
            np.array([[1, 0, 1]]),
            (1, 3),
        ),  # Column 1 removed, Rows 0 and 2 removed
        (
            [0, 2],
            [0, 2],
            [1],
            np.array([[1.1, 0]]),
            np.array([[1, 0]]),
            (1, 2),
        ),  # Columns 1, 3 removed, Rows 0 and 2 removed
    ],
)
def test_read_sparse_csv_from_file(
    tmpdir_factory, used_cols, expected_used_cols, expected_used_rows, expected_data, expected_mask, expected_shape
):
    # [[   ,    ,    ,    ],
    #  [1.1,    ,    , 2.1],
    #  [   , 3.1,    ,    ]])
    data = np.array([[1, 0, 1.1], [1, 3, 2.1], [2, 1, 3.1]])
    dataset_dir = tmpdir_factory.mktemp("dataset_dir")
    pd.DataFrame(data).to_csv(os.path.join(dataset_dir, "all.csv"), header=None, index=None)
    path = str(os.path.join(dataset_dir, "all.csv"))

    (
        processed_data,
        processed_mask,
        processed_used_cols,
        processed_used_rows,
    ) = SparseCSVDatasetLoader.read_sparse_csv_from_file(path, used_cols=used_cols)

    assert np.array_equal(processed_data.toarray(), expected_data)
    assert np.array_equal(processed_mask.toarray(), expected_mask)
    assert processed_used_cols == expected_used_cols
    assert processed_used_rows == expected_used_rows
    assert processed_data.shape == expected_shape


@pytest.mark.parametrize(
    "used_cols, expected_data, expected_mask, expected_shape",
    [
        (
            [0, 1, 2, 3],
            np.array([[0, 0, 0, 0], [1.1, 0, 0, 2.1], [0, 3.1, 0, 0]]),
            np.array([[0, 0, 0, 0], [1, 0, 0, 1], [0, 1, 0, 0]]),
            (3, 4),
        ),  # No columns removed, no rows removed
        (
            [0, 1, 2],
            np.array([[0, 0, 0], [1.1, 0, 0], [0, 3.1, 0]]),
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            (3, 3),
        ),  # Column 3 removed, no rows removed
        (
            [0, 2, 3],
            np.array([[0, 0, 0], [1.1, 0, 2.1], [0, 0, 0]]),
            np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]]),
            (3, 3),
        ),  # Column 1 removed, no rows removed
        (
            [0, 2],
            np.array([[0, 0], [1.1, 0], [0, 0]]),
            np.array([[0, 0], [1, 0], [0, 0]]),
            (3, 2),
        ),  # Columns 1, 3 removed, no rows removed
    ],
)
def test_read_sparse_csv_from_dicts(used_cols, expected_data, expected_mask, expected_shape):
    # [[   ,    ,    ,    ],
    #  [1.1,    ,    , 2.1],
    #  [   , 3.1,    ,    ]])
    dicts = [{}, {"0": 1.1, "3": 2.1}, {"1": 3.1}]

    processed_data, processed_mask, used_rows = SparseCSVDatasetLoader.read_sparse_csv_from_dicts(
        dicts, used_cols=used_cols
    )

    assert np.array_equal(processed_data.toarray(), expected_data)
    assert np.array_equal(processed_mask.toarray(), expected_mask)
    assert processed_data.shape == expected_shape
    assert used_rows == [0, 1, 2]


@pytest.mark.parametrize(
    "used_cols, expected_used_cols, expected_used_rows, expected_data, expected_mask, expected_shape",
    [
        (
            None,
            [0, 1, 3],
            [1, 2],
            np.array([[1.1, 0, 2.1], [0, 3.1, 0]]),
            np.array([[1, 0, 1], [0, 1, 0]]),
            (2, 3),
        ),  # Column 2 removed, Row 0 removed
        (
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [1, 2],
            np.array([[1.1, 0, 0, 2.1], [0, 3.1, 0, 0]]),
            np.array([[1, 0, 0, 1], [0, 1, 0, 0]]),
            (2, 4),
        ),  # No columns removed, Row 0 removed
        (
            [0, 1, 2],
            [0, 1, 2],
            [1, 2],
            np.array([[1.1, 0, 0], [0, 3.1, 0]]),
            np.array([[1, 0, 0], [0, 1, 0]]),
            (2, 3),
        ),  # Column 3 removed, Row 0 removed
        (
            [0, 2, 3],
            [0, 2, 3],
            [1],
            np.array([[1.1, 0, 2.1]]),
            np.array([[1, 0, 1]]),
            (1, 3),
        ),  # Column 1 removed, Rows 0 and 2 removed
        (
            [0, 2],
            [0, 2],
            [1],
            np.array([[1.1, 0]]),
            np.array([[1, 0]]),
            (1, 2),
        ),  # Columns 1, 3 removed, Rows 0 and 2 removed
    ],
)
def test_process_sparse_data_used_cols_specified(
    used_cols, expected_used_cols, expected_used_rows, expected_data, expected_mask, expected_shape
):
    # [[   ,    ,    ,    ],
    #  [1.1,    ,    , 2.1],
    #  [   , 3.1,    ,    ]])
    data = sparse.coo_matrix(([1.1, 2.1, 3.1], ([1, 1, 2], [0, 3, 1])))

    (
        processed_data,
        processed_mask,
        processed_used_cols,
        processed_used_rows,
    ) = SparseCSVDatasetLoader._process_sparse_data(data, used_cols=used_cols)

    assert np.array_equal(processed_data.toarray(), expected_data)
    assert np.array_equal(processed_mask.toarray(), expected_mask)
    assert processed_used_cols == expected_used_cols
    assert processed_used_rows == expected_used_rows
    assert processed_data.shape == expected_shape


@pytest.mark.parametrize(
    "used_cols, expected_used_cols, expected_used_rows, expected_data, expected_mask, expected_shape",
    [
        (
            None,
            [0, 1, 3],
            [0, 1, 2],
            np.array([[0, 0, 0], [1.1, 0, 2.1], [0, 3.1, 0]]),
            np.array([[0, 0, 0], [1, 0, 1], [0, 1, 0]]),
            (3, 3),
        ),  # Column 2 removed, no rows removed
        (
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2],
            np.array([[0, 0, 0, 0], [1.1, 0, 0, 2.1], [0, 3.1, 0, 0]]),
            np.array([[0, 0, 0, 0], [1, 0, 0, 1], [0, 1, 0, 0]]),
            (3, 4),
        ),  # No columns removed, no rows removed
        (
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            np.array([[0, 0, 0], [1.1, 0, 0], [0, 3.1, 0]]),
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            (3, 3),
        ),  # Column 3 removed, no rows removed
        (
            [0, 2, 3],
            [0, 2, 3],
            [0, 1, 2],
            np.array([[0, 0, 0], [1.1, 0, 2.1], [0, 0, 0]]),
            np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]]),
            (3, 3),
        ),  # Column 1 removed, no rows removed
        (
            [0, 2],
            [0, 2],
            [0, 1, 2],
            np.array([[0, 0], [1.1, 0], [0, 0]]),
            np.array([[0, 0], [1, 0], [0, 0]]),
            (3, 2),
        ),  # Columns 1, 3 removed, no rows removed
    ],
)
def test_process_sparse_data_used_cols_specified_drop_rows_false(
    used_cols, expected_used_cols, expected_used_rows, expected_data, expected_mask, expected_shape
):
    # [[   ,    ,    ,    ],
    #  [1.1,    ,    , 2.1],
    #  [   , 3.1,    ,    ]])
    data = sparse.coo_matrix(([1.1, 2.1, 3.1], ([1, 1, 2], [0, 3, 1])))

    (
        processed_data,
        processed_mask,
        processed_used_cols,
        processed_used_rows,
    ) = SparseCSVDatasetLoader._process_sparse_data(data, used_cols=used_cols, drop_rows=False)

    assert np.array_equal(processed_data.toarray(), expected_data)
    assert np.array_equal(processed_mask.toarray(), expected_mask)
    assert processed_used_cols == expected_used_cols
    assert processed_used_rows == expected_used_rows
    assert processed_data.shape == expected_shape


def test_process_sparse_data_used_cols_unsorted_raises_error():
    # [[   ,    ,    ,    ],
    #  [1.1,    ,    , 2.1],
    #  [   , 3.1,    ,    ]])
    data = sparse.coo_matrix(([1.1, 2.1, 3.1], ([1, 1, 2], [0, 3, 1])))

    with pytest.raises(ValueError):
        _ = SparseCSVDatasetLoader._process_sparse_data(data, used_cols=[0, 2, 1, 3])


def test_validate_train_data_valid_data():
    # [[1.1,    , 2.1],
    #  [   , 3.1,    ]])
    train_data = sparse.csr_matrix(([1.1, 2.1, 3.1], ([0, 0, 1], [0, 2, 1])))
    used_cols = [0, 1, 2]
    SparseCSVDatasetLoader._validate_train_data(train_data, used_cols)


def test_validate_train_data_invalid_data():
    # [[1.1,    ,    ],
    #  [   , 3.1,    ]])
    train_data = sparse.csr_matrix(([1.1, 3.1], ([1, 2], [0, 1])))
    used_cols = [0, 1, 2]
    with pytest.raises(ValueError):
        SparseCSVDatasetLoader._validate_train_data(train_data, used_cols)


def test_negative_sample_sparse(tmpdir):
    data = np.eye(5)
    data[0, 1] = 1
    mask = np.eye(5, dtype=bool)
    mask[0, 1] = 1
    mask[1, 2] = 1
    data = sparse.csr_matrix(data)
    mask = sparse.csr_matrix(mask)
    expected_data = data.copy()  # Make copy of original data before we apply negative sampling
    levels = {i: i for i in range(5)}

    sparse_dataset_loader = SparseCSVDatasetLoader(dataset_dir=tmpdir)
    data, mask = sparse_dataset_loader.negative_sample(data, mask, levels)

    assert data.shape == mask.shape
    assert mask.dtype == bool
    # Shouldn't add any positive samples so data should stay the same
    assert np.all(data.toarray() == expected_data.toarray())

    # Check original observed elements in mask unchanged
    for i in range(5):
        assert mask[i, i] == 1
    assert mask[0, 1] == 1
    assert mask[1, 2] == 1

    # Should sample 2 extra negative samples for row 0, 0 for row 1 (num positive = num negative), 1 for rows 2 + 3 and
    # 0 for row 4 (no candidates).
    assert mask[0].sum() == 4
    assert mask[1].sum() == 2
    assert mask[2].sum() == 2
    assert mask[3].sum() == 2
    assert mask[4].sum() == 1

    # Check no negative samples chosen below the main diagonal (since levels match the column IDs and row i always has
    # a positive element in col i).
    assert np.all(np.tril(mask.toarray(), -1) == 0)

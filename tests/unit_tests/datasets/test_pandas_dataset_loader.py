import os

import numpy as np
import pandas as pd
import pytest

from azua.datasets.pandas_dataset_loader import PandasDatasetLoader
from azua.datasets.dataset import Dataset
from azua.utils.io_utils import read_json_as


def test_split_data_and_load_dataset(tmpdir_factory):
    data = np.array(
        [
            [np.nan, 2.1],  # Full data: 10 rows
            [np.nan, 2.1],
            [np.nan, 2.1],
            [np.nan, 2.1],
            [np.nan, 2.1],
            [np.nan, 2.1],
            [np.nan, 2.1],
            [np.nan, 2.1],
            [np.nan, 2.1],
            [np.nan, 2.1],
        ]
    )

    df = pd.DataFrame(data)
    dataset_loader = PandasDatasetLoader(dataset_dir="")
    dataset = dataset_loader.split_data_and_load_dataset(
        df, test_frac=0.5, val_frac=0.2, random_state=0, max_num_rows=None
    )

    expected_train_data = np.array([[0.0, 2.1], [0.0, 2.1], [0.0, 2.1]])  # Train data: 3 rows
    expected_train_mask = np.array([[0, 1], [0, 1], [0, 1]])
    expected_val_data = np.array([[0.0, 2.1], [0.0, 2.1]])  # Val data: 2 rows
    expected_val_mask = np.array([[0, 1], [0, 1]])
    expected_test_data = np.array([[0.0, 2.1], [0.0, 2.1], [0.0, 2.1], [0.0, 2.1], [0.0, 2.1]])  # Test data: 5 rows
    expected_test_mask = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])

    assert type(dataset) == Dataset
    assert np.array_equal(expected_train_data, dataset.train_data_and_mask[0])
    assert np.array_equal(expected_train_mask, dataset.train_data_and_mask[1])
    assert np.array_equal(expected_val_data, dataset.val_data_and_mask[0])
    assert np.array_equal(expected_val_mask, dataset.val_data_and_mask[1])
    assert np.array_equal(expected_test_data, dataset.test_data_and_mask[0])
    assert np.array_equal(expected_test_mask, dataset.test_data_and_mask[1])
    assert not np.isnan(dataset.train_data_and_mask[0].astype(float)).any()
    assert not np.isnan(dataset.val_data_and_mask[0].astype(float)).any()
    assert not np.isnan(dataset.test_data_and_mask[0].astype(float)).any()


def test_split_data_and_load_dataset_zero_val_frac(tmpdir_factory):
    data = np.array(
        [
            [np.nan, 2.1],  # Full data: 10 rows
            [np.nan, 2.1],
            [np.nan, 2.1],
            [np.nan, 2.1],
            [np.nan, 2.1],
            [np.nan, 2.1],
            [np.nan, 2.1],
            [np.nan, 2.1],
            [np.nan, 2.1],
            [np.nan, 2.1],
        ]
    )

    df = pd.DataFrame(data)
    dataset_loader = PandasDatasetLoader(dataset_dir="")
    dataset = dataset_loader.split_data_and_load_dataset(
        df, test_frac=0.1, val_frac=0.0, random_state=0, max_num_rows=None
    )

    expected_train_data = np.array(
        [[0.0, 2.1], [0.0, 2.1], [0.0, 2.1], [0.0, 2.1], [0.0, 2.1], [0.0, 2.1], [0.0, 2.1], [0.0, 2.1], [0.0, 2.1]]
    )  # Train data: 9 rows
    expected_train_mask = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])
    expected_val_data = None  # Val data: 0 rows -> None
    expected_val_mask = None
    expected_test_data = np.array([[0.0, 2.1]])  # Test data: 1 row
    expected_test_mask = np.array([[0, 1]])

    assert type(dataset) == Dataset
    assert np.array_equal(expected_train_data, dataset.train_data_and_mask[0])
    assert np.array_equal(expected_train_mask, dataset.train_data_and_mask[1])
    assert np.array_equal(expected_val_data, dataset.val_data_and_mask[0])
    assert np.array_equal(expected_val_mask, dataset.val_data_and_mask[1])
    assert np.array_equal(expected_test_data, dataset.test_data_and_mask[0])
    assert np.array_equal(expected_test_mask, dataset.test_data_and_mask[1])
    assert not np.isnan(dataset.train_data_and_mask[0].astype(float)).any()
    assert not np.isnan(dataset.test_data_and_mask[0].astype(float)).any()


def test_split_data_and_load_dataset_max_num_rows_specified(tmpdir_factory):
    data = np.array(
        [
            [np.nan, 2.1],  # Full data: 11 rows - Last row is ignored
            [np.nan, 2.1],
            [np.nan, 2.1],
            [np.nan, 2.1],
            [np.nan, 2.1],
            [np.nan, 2.1],
            [np.nan, 2.1],
            [np.nan, 2.1],
            [np.nan, 2.1],
            [np.nan, 2.1],
            [np.nan, 2.2],
        ]
    )

    df = pd.DataFrame(data)
    dataset_loader = PandasDatasetLoader(dataset_dir="")
    dataset = dataset_loader.split_data_and_load_dataset(
        df, test_frac=0.5, val_frac=0.2, random_state=0, max_num_rows=10
    )

    expected_train_data = np.array([[0.0, 2.1], [0.0, 2.1], [0.0, 2.1]])  # Train data: 3 rows
    expected_train_mask = np.array([[0, 1], [0, 1], [0, 1]])
    expected_val_data = np.array([[0.0, 2.1], [0.0, 2.1]])  # Val data: 2 rows
    expected_val_mask = np.array([[0, 1], [0, 1]])
    expected_test_data = np.array([[0.0, 2.1], [0.0, 2.1], [0.0, 2.1], [0.0, 2.1], [0.0, 2.1]])  # Test data: 5 rows
    expected_test_mask = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])

    assert type(dataset) == Dataset
    assert np.array_equal(expected_train_data, dataset.train_data_and_mask[0])
    assert np.array_equal(expected_train_mask, dataset.train_data_and_mask[1])
    assert np.array_equal(expected_val_data, dataset.val_data_and_mask[0])
    assert np.array_equal(expected_val_mask, dataset.val_data_and_mask[1])
    assert np.array_equal(expected_test_data, dataset.test_data_and_mask[0])
    assert np.array_equal(expected_test_mask, dataset.test_data_and_mask[1])
    assert not np.isnan(dataset.train_data_and_mask[0].astype(float)).any()
    assert not np.isnan(dataset.val_data_and_mask[0].astype(float)).any()
    assert not np.isnan(dataset.test_data_and_mask[0].astype(float)).any()


def test_split_data_and_load_dataset_more_rows(tmpdir_factory):
    data = np.ones((20, 2))  # Full data: 20 rows

    df = pd.DataFrame(data)
    dataset_loader = PandasDatasetLoader(dataset_dir="")
    dataset = dataset_loader.split_data_and_load_dataset(
        df, test_frac=0.5, val_frac=0.2, random_state=0, max_num_rows=None
    )

    expected_train_data = np.ones((6, 2))  # Train data: 6 rows
    expected_train_mask = np.ones((6, 2))
    expected_val_data = np.ones((4, 2))  # Val data: 4 rows
    expected_val_mask = np.ones((4, 2))
    expected_test_data = np.ones((10, 2))  # Test data: 10 rows
    expected_test_mask = np.ones((10, 2))

    assert type(dataset) == Dataset
    assert np.array_equal(expected_train_data, dataset.train_data_and_mask[0])
    assert np.array_equal(expected_train_mask, dataset.train_data_and_mask[1])
    assert np.array_equal(expected_val_data, dataset.val_data_and_mask[0])
    assert np.array_equal(expected_val_mask, dataset.val_data_and_mask[1])
    assert np.array_equal(expected_test_data, dataset.test_data_and_mask[0])
    assert np.array_equal(expected_test_mask, dataset.test_data_and_mask[1])
    assert not np.isnan(dataset.train_data_and_mask[0].astype(float)).any()
    assert not np.isnan(dataset.val_data_and_mask[0].astype(float)).any()
    assert not np.isnan(dataset.test_data_and_mask[0].astype(float)).any()


def test_split_data_and_load_dataset_save_data_split(tmpdir_factory):
    data = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]])

    df = pd.DataFrame(data)
    dataset_loader = PandasDatasetLoader(dataset_dir="")
    dataset = dataset_loader.split_data_and_load_dataset(
        df, test_frac=0.3, val_frac=0.2, random_state=0, max_num_rows=None
    )

    save_dir = tmpdir_factory.mktemp("save_dir")
    dataset.save_data_split(save_dir=str(save_dir))

    saved_train_data = dataset.train_data_and_mask[0]
    saved_val_data = dataset.val_data_and_mask[0]
    saved_test_data = dataset.test_data_and_mask[0]

    data_split = read_json_as(os.path.join(save_dir, "data_split.json"), dict)

    assert data_split["train_idxs"] == list(saved_train_data[:, 0])
    assert data_split["val_idxs"] == list(saved_val_data[:, 0])
    assert data_split["test_idxs"] == list(saved_test_data[:, 0])


@pytest.mark.parametrize(
    "test_frac, val_frac", [(None, 0.25), (0.25, None), (None, None), (1, 0.25), (0.25, 1), (0.5, 0.5)]
)
def test_split_data_and_load_dataset_invalid_test_frac_val_frac_raises_error(tmpdir_factory, test_frac, val_frac):
    data = np.ones((4, 5))
    df = pd.DataFrame(data)
    with pytest.raises(AssertionError):
        dataset_loader = PandasDatasetLoader(dataset_dir="")
        _ = dataset_loader.split_data_and_load_dataset(
            df, test_frac=test_frac, val_frac=val_frac, random_state=0, max_num_rows=None
        )


@pytest.mark.parametrize("random_state", [(0), ((0, 1))])
def test_split_data_and_load_dataset_deterministic(tmpdir_factory, random_state):
    data = np.array(
        [
            [0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, 1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, 2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, 3, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, 4, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 5, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 6, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 8, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 9],
        ]
    )
    df = pd.DataFrame(data)
    dataset_loader = PandasDatasetLoader(dataset_dir="")
    dataset1 = dataset_loader.split_data_and_load_dataset(
        df, test_frac=0.3, val_frac=0.2, random_state=random_state, max_num_rows=None
    )
    dataset2 = dataset_loader.split_data_and_load_dataset(
        df, test_frac=0.3, val_frac=0.2, random_state=random_state, max_num_rows=None
    )

    assert np.array_equal(dataset1.train_data_and_mask[0], dataset2.train_data_and_mask[0])
    assert np.array_equal(dataset1.train_data_and_mask[1], dataset2.train_data_and_mask[1])
    assert np.array_equal(dataset1.val_data_and_mask[0], dataset2.val_data_and_mask[0])
    assert np.array_equal(dataset1.val_data_and_mask[1], dataset2.val_data_and_mask[1])
    assert np.array_equal(dataset1.test_data_and_mask[0], dataset2.test_data_and_mask[0])
    assert np.array_equal(dataset1.test_data_and_mask[1], dataset2.test_data_and_mask[1])


def test_split_data_and_load_dataset_deterministic_test_set(tmpdir_factory):
    data = np.array(
        [
            [0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, 1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, 2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, 3, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, 4, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 5, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 6, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 8, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 9],
        ]
    )
    df = pd.DataFrame(data)
    dataset_loader = PandasDatasetLoader(dataset_dir="")
    dataset1 = dataset_loader.split_data_and_load_dataset(
        df, test_frac=0.3, val_frac=0.2, random_state=(0, 1), max_num_rows=None
    )
    dataset2 = dataset_loader.split_data_and_load_dataset(
        df, test_frac=0.3, val_frac=0.2, random_state=(0, 2), max_num_rows=None
    )

    assert not np.array_equal(dataset1.train_data_and_mask[0], dataset2.train_data_and_mask[0])
    assert not np.array_equal(dataset1.train_data_and_mask[1], dataset2.train_data_and_mask[1])
    assert not np.array_equal(dataset1.val_data_and_mask[0], dataset2.val_data_and_mask[0])
    assert not np.array_equal(dataset1.val_data_and_mask[1], dataset2.val_data_and_mask[1])
    assert np.array_equal(dataset1.test_data_and_mask[0], dataset2.test_data_and_mask[0])
    assert np.array_equal(dataset1.test_data_and_mask[1], dataset2.test_data_and_mask[1])


def test_process_data():
    data = np.array([[np.nan, 1.1, 2.1], [3.1, np.nan, 5.1]])

    processed_data, processed_mask = PandasDatasetLoader._process_data(data)

    expected_data = np.array([[0.0, 1.1, 2.1], [3.1, 0.0, 5.1]])
    expected_mask = np.array([[0, 1, 1], [1, 0, 1]])

    assert np.array_equal(processed_data, expected_data)
    assert np.array_equal(processed_mask, expected_mask)
    assert not np.isnan(processed_mask).any()


def test_process_data_with_txt():
    data = np.array(
        [[np.nan, "I would like to leave on monday .", 1.1, 2.1], [3.1, "", np.nan, 5.1], [3.1, "NaN", np.nan, 5.1]],
        dtype=object,
    )

    processed_data, processed_mask = PandasDatasetLoader._process_data(data)

    expected_data = np.array(
        [[0.0, "I would like to leave on monday .", 1.1, 2.1], [3.1, "", 0.0, 5.1], [3.1, "NaN", 0.0, 5.1]],
        dtype=object,
    )
    expected_mask = np.array([[0, 1, 1, 1], [1, 0, 0, 1], [1, 1, 0, 1]])

    print(processed_data)

    assert np.array_equal(processed_data, expected_data)
    assert np.array_equal(processed_mask, expected_mask)
    assert not np.isnan(processed_mask).any()


def test_is_value_present():
    assert PandasDatasetLoader._is_value_present("Train leaves on Monday")
    assert not PandasDatasetLoader._is_value_present("")
    assert PandasDatasetLoader._is_value_present("NaN")
    assert PandasDatasetLoader._is_value_present("uknown")
    assert PandasDatasetLoader._is_value_present(3.4)
    assert not PandasDatasetLoader._is_value_present(float("NaN"))

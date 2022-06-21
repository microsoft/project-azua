import numpy as np

from azua.utils.data_mask_utils import argsort_rows_exclude_nan, matrix_to_list


def test_matrix_to_list():
    data = np.arange(12).reshape(4, 3)
    row_ids = [1, 3, 5, 9]
    col_ids = [2, 4, 8]
    df = matrix_to_list(data, row_ids, col_ids)
    assert np.all(df["row_id"].values == np.repeat(row_ids, 3))
    assert np.all(df["col_id"].values == np.tile(col_ids, 4))
    assert np.all(df["value"].values == np.arange(12))


def test_argsort_rows_exclude_nan():
    array = np.array(
        [[1.2, 3.2, np.nan, -0.2, -0.7], [np.nan, 0.8, np.nan, np.nan, 0.2], [np.nan, np.nan, np.nan, np.nan, np.nan]]
    )
    output = argsort_rows_exclude_nan(array)
    expected = [[4, 3, 0, 1], [4, 1], []]
    assert output == expected


def test_argsort_rows_exclude_nan_descending():
    array = np.array(
        [[1.2, 3.2, np.nan, -0.2, -0.7], [np.nan, 0.8, np.nan, np.nan, 0.2], [np.nan, np.nan, np.nan, np.nan, np.nan]]
    )
    output = argsort_rows_exclude_nan(array, ascending=False)
    expected = [[1, 0, 3, 4], [1, 4], []]
    assert output == expected


def test_argsort_rows_exclude_nan_max_qs_per_row():
    array = np.array(
        [[1.2, 3.2, np.nan, -0.2, -0.7], [np.nan, 0.8, np.nan, np.nan, 0.2], [np.nan, np.nan, np.nan, np.nan, np.nan]]
    )
    output = argsort_rows_exclude_nan(array, max_qs_per_row=2)
    expected = [[4, 3], [4, 1], []]
    assert output == expected

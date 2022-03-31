import logging
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from ..datasets.dataset import SparseDataset
from ..datasets.dataset_loader import DatasetLoader
from ..datasets.variables import Variables

logger = logging.getLogger(__name__)


class SparseCSVDatasetLoader(DatasetLoader):
    """
    Load a dataset from a sparse CSV file, where each row entry has a form (row_id, col_id, value).
    """

    _all_data_file = "all.csv"
    _train_data_file = "train.csv"
    _val_data_file = "val.csv"
    _test_data_file = "test.csv"

    def split_data_and_load_dataset(
        self,
        test_frac: float,
        val_frac: float,
        random_state: Union[int, Tuple[int, int]],
        max_num_rows: Optional[int] = None,
        negative_sample: bool = False,
        split_type: str = "rows",
        **kwargs,
    ) -> SparseDataset:
        """
        Load the data from disk and make the train/val/test split to instantiate a dataset.
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
        Returns:
            dataset: SparseDataset object, holding the data and variable metadata.
        """
        logger.info(f"Splitting data to load the dataset: test fraction: {test_frac}, validation fraction: {val_frac}.")

        if max_num_rows is not None:
            raise NotImplementedError("Maximum number of rows cannot currently be enforced with sparse CSV data.")

        data_path = os.path.join(self._dataset_dir, self._all_data_file)
        self._download_data_if_necessary(self._dataset_dir)

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"The required data file not found: {data_path}.")

        data, mask, used_cols, _ = self.read_sparse_csv_from_file(data_path, used_cols=None)

        rows = list(set(data.tocoo().row))
        train_rows, val_rows, test_rows, data_split = self._generate_data_split(rows, test_frac, val_frac, random_state)

        train_data = data[train_rows, :]
        train_mask = mask[train_rows, :]
        self._validate_train_data(train_data, used_cols)

        test_data = data[test_rows, :]
        test_mask = mask[test_rows, :]

        if len(val_rows) == 0:
            val_data, val_mask = None, None
        else:
            val_data = data[val_rows, :]
            val_mask = mask[val_rows, :]

        variables_dict = self._load_variables_dict(used_cols)
        variables = Variables.create_from_data_and_dict(train_data.toarray(), train_mask.toarray(), variables_dict)

        if negative_sample:
            train_data, train_mask, val_data, val_mask, test_data, test_mask = self._apply_negative_sampling(
                variables, train_data, train_mask, val_data, val_mask, test_data, test_mask
            )

        return SparseDataset(
            train_data=train_data,
            train_mask=train_mask,
            val_data=val_data,
            val_mask=val_mask,
            test_data=test_data,
            test_mask=test_mask,
            variables=variables,
            data_split=data_split,
        )

    def load_predefined_dataset(
        self, max_num_rows: Optional[int] = None, negative_sample: bool = False, split_type: str = "rows", **kwargs
    ) -> SparseDataset:
        """
        Load the data from disk and use the predefined train/val/test split to instantiate a dataset.
        Args:
            max_num_rows: Maximum number of rows to include when reading data files.
            negative_sample: Whether to perform negative sampling after loading the dataset. Negative sampling requires
                a file negative_sampling_levels.csv in the dataset folder, and negative samples for each row will be 
                drawn from features of a greater level than all those observed in the row.
            split_type: Manner in which the dataset has been split: "rows" indicates a split by rows of the matrix, 
                "elements" indicates a split by individual elements of the matrix, so that different elements of a row
                can appear in different data splits.
        Returns:
            dataset: SparseDataset object, holding the data and variable metadata.
        """
        logger.info("Using a predefined data split to load the dataset.")
        assert split_type in ["rows", "elements"], "Split type must be one of 'rows' or 'elements'"

        if max_num_rows is not None:
            raise NotImplementedError("Maximum number of rows cannot currently be enforced with sparse CSV data.")

        # Download data
        train_data_path = os.path.join(self._dataset_dir, self._train_data_file)
        test_data_path = os.path.join(self._dataset_dir, self._test_data_file)
        val_data_path = os.path.join(self._dataset_dir, self._val_data_file)
        self._download_data_if_necessary(self._dataset_dir)

        # Loading train and test data - raise an error if not found
        train_test_files_exist = all([os.path.exists(train_data_path), os.path.exists(test_data_path)])
        if not train_test_files_exist:
            raise FileNotFoundError(
                f"At least one of the required data files not found: {[train_data_path, test_data_path]}."
            )
        # We do not want to drop empty rows from train/test data if using an elementwise split, since we require the
        # number of rows to match and a row could be empty in one part of the split but not another.
        drop_rows = split_type == "rows"

        train_data, train_mask, used_cols, train_rows = self.read_sparse_csv_from_file(
            train_data_path, used_cols=None, drop_rows=drop_rows
        )
        test_data, test_mask, _, test_rows = self.read_sparse_csv_from_file(
            test_data_path, used_cols=used_cols, drop_rows=drop_rows
        )

        # Loading val data - make a warning if not found
        if not os.path.exists(val_data_path):
            val_data, val_mask = None, None
            val_rows: List[int] = []
            warnings.warn(f"Validation data file not found: {val_data_path}.", UserWarning)
        else:
            val_data, val_mask, _, val_rows = self.read_sparse_csv_from_file(val_data_path, used_cols=used_cols)

        if split_type == "elements":
            # For an elementwise data split, we require the train, val and test matrices to be the same shape so that we
            # can combine these matrices to perform imputation evaluation correctly.
            # Since the CSR matrix created for each split will only contain rows up to the largest row ID in the split,
            # we resize the smaller matrices to include 0s on these truncated rows if necessary..
            if val_data is not None and val_mask is not None:
                if train_rows != val_rows or train_rows != test_rows or val_rows != test_rows:
                    num_rows = max(train_data.shape[0], val_data.shape[0], test_data.shape[0])
                    num_cols = len(used_cols)
                    train_data.resize((num_rows, num_cols))
                    val_data.resize((num_rows, num_cols))
                    test_data.resize((num_rows, num_cols))
                    # resize the masks
                    train_mask.resize((num_rows, num_cols))
                    test_mask.resize((num_rows, num_cols))
                    val_mask.resize((num_rows, num_cols))
            else:
                if train_rows != test_rows:
                    num_rows = max(train_data.shape[0], test_data.shape[0])
                    num_cols = len(used_cols)
                    train_data.resize((num_rows, num_cols))
                    test_data.resize((num_rows, num_cols))
                    train_mask.resize((num_rows, num_cols))
                    test_mask.resize((num_rows, num_cols))

            # Remove rows that are not used in train/val/test data: this is necessary if the row IDs are not
            # [0, ..., num_rows-1], since the sparse matrices will be created with as many rows as the largest row ID.
            used_rows_set = set(train_data.tocoo().row).union(set(test_data.tocoo().row))
            if val_data is not None:
                used_rows_set = used_rows_set.union(set(val_data.tocoo().row))
            used_rows = list(used_rows_set)

            train_data = train_data[used_rows, :]
            train_mask = train_mask[used_rows, :]
            test_data = test_data[used_rows, :]
            test_mask = test_mask[used_rows, :]
            if val_data is not None and val_mask is not None:
                val_data = val_data[used_rows, :]
                val_mask = val_mask[used_rows, :]

        variables_dict = self._load_variables_dict(used_cols)
        variables = Variables.create_from_data_and_dict(train_data.toarray(), train_mask.toarray(), variables_dict)

        if negative_sample:
            train_data, train_mask, val_data, val_mask, test_data, test_mask = self._apply_negative_sampling(
                variables, train_data, train_mask, val_data, val_mask, test_data, test_mask
            )

        data_split = {
            "train_idxs": [int(id) for id in train_rows],
            "val_idxs": [int(id) for id in val_rows],
            "test_idxs": [int(id) for id in test_rows],
        }

        return SparseDataset(
            train_data=train_data,
            train_mask=train_mask,
            val_data=val_data,
            val_mask=val_mask,
            test_data=test_data,
            test_mask=test_mask,
            variables=variables,
            data_split=data_split,
        )

    def _load_variables_dict(self, used_cols: List[int]) -> Dict[str, List[Any]]:  # type: ignore
        """
        Load variables info object from a file if it exists including the list of used columns.
        Args:
            dataset_dir: Directory in which the dataset files are contained, or will be saved if not present.
            used_cols: A list of observed columns that were used.
        Returns:
            variables_dict: If not None, dictionary containing metadata for each variable (column) in the input data.
        """
        variables_dict = super()._load_variables_dict()
        variables_dict = {} if variables_dict is None else variables_dict
        variables_dict["used_cols"] = used_cols
        return variables_dict

    @classmethod
    def read_sparse_csv_from_file(
        cls, path: str, used_cols: Optional[List[int]] = None, drop_rows=True
    ) -> Tuple[csr_matrix, csr_matrix, List[int], List[int]]:
        """
        Read the sparse csv file to generate a sparse data and mask matrices.
        Drop the columns and rows that were not observed.
        Args:
            paths: Path to CSV file with three columns corresponding to (row ID, col ID, value). The file should not
                include a header.
            used_cols: A sorted list of observed columns that were used.
            drop_rows: Whether to remove all empty rows (True), or include all rows between 0 and max(row_id) (False).
        Returns:
            data: Sparse data matrix with unused rows and columns dropped.
            mask: Corresponding mask, where observed values are 1 and unobserved values are 0.
            used_cols: A list of observed columns that were used.
            used_rows: A list of observed rows that were used.
        """
        df = pd.read_csv(path, header=None, names=["row", "col", "val"])

        # Drop duplicate (row, col) pairs before creating the sparse matrix, since the constructor will sum all such
        # duplicates which is not the desired behaviour. Instead just keep the last entry for each (row, col) pair,
        # assuming that this is the most up-to-date entry.
        df.drop_duplicates(subset=["row", "col"], keep="last", inplace=True)

        rows = df["row"].to_numpy(dtype=np.int_)
        cols = df["col"].to_numpy(dtype=np.int_)
        vals = df["val"].to_numpy(dtype=np.float_)
        data = csr_matrix((vals, (rows, cols)), dtype=np.float_)
        return cls._process_sparse_data(data, used_cols, drop_rows)

    @classmethod
    def read_sparse_csv_from_dicts(
        cls, dicts: List[Dict[str, int]], used_cols: List[int]
    ) -> Tuple[csr_matrix, csr_matrix, List[int]]:
        """
        Read the list of dicts to generate a sparse data and mask matrices.
        Drop the columns that were not observed.
        Args:
<<<<<<< HEAD
            records: List of dicts, where each dict is one data row represented as a mapping from columns
=======
            dicts: List of dicts, where each dict is one data row represented as a mapping from columns
>>>>>>> master
                to values.
            used_cols: A sorted list of observed columns that were used.
        Returns:
            data: Sparse data matrix with unused rows and columns dropped.
            mask: Corresponding mask, where observed values are 1 and unobserved values are 0.
            used_rows: IDs of rows included in the output array, where the IDs are assumed to be [0, ..., n-1] since for
                this input format we have no user-specified row IDs.
        """
        assert used_cols is not None
        rows, cols, vals = [], [], []
        max_col = max(used_cols)
        for row, entries in enumerate(dicts):
            for col_str, val in entries.items():
                col = int(col_str)
                if col <= max_col:
                    rows.append(row)
                    cols.append(col)
                    vals.append(val)
        n_rows = len(dicts)
        n_cols = max_col + 1
        data = csr_matrix((vals, (rows, cols)), dtype=np.float_, shape=(n_rows, n_cols))
        data, mask, _, used_rows = cls._process_sparse_data(data, used_cols, drop_rows=False)
        return data, mask, used_rows

    @classmethod
    def _process_sparse_data(
        cls, data: csr_matrix, used_cols: Optional[List[int]], drop_rows: bool = True
    ) -> Tuple[csr_matrix, csr_matrix, List[int], List[int]]:
        """
        If not given, use the sparse data matrix to generate the list of column ids with no observed values.
        Drop the columns and rows that were not observed. 
        Args:
            data: Sparse data matrix.
            used_cols: A sorted list of observed columns that were used.
            drop_rows: Whether or not to remove empty rows from the processed data.
        Returns:
            data: Sparse data matrix with unused rows and columns dropped.
            mask: Corresponding mask, where observed values are 1 and unobserved values are 0.
            used_cols: A list of observed columns that were used.
            used_rows: A list of observed rows that were used.
        """
        if used_cols is None:
            used_cols = list(set(data.tocoo().col))
            used_cols.sort()
        else:
            if sorted(used_cols) != used_cols:
                raise ValueError("The list of used columns must be sorted in the ascending order.")
            data.resize(data.shape[0], max(used_cols) + 1)

        data = data.tocsc()[:, used_cols].tocsr()

        if drop_rows:
            used_rows = list(set(data.tocoo().row))
            data = data[used_rows, :]

        else:
            used_rows = list(range(data.shape[0]))

        data_coo = data.tocoo()
        mask = csr_matrix(
            (np.ones_like(data.data, dtype=bool), (data_coo.row, data_coo.col)), dtype=bool, shape=data.shape
        )
        return data, mask, used_cols, used_rows

    @classmethod
    def _validate_train_data(cls, train_data: csr_matrix, used_cols: List[int]) -> None:
        """
        Check that for each feature there is at least one observed value in the train data.
        Args:
           train_data: Sparse train data matrix.
           used_cols: A list of observed columns that were used.
        """
        if len(set(train_data.tocoo().col)) != len(used_cols):
            raise ValueError("Some columns have no observed values in the generated train set.")

import os
import numpy as np

import logging
from ..datasets.causal_csv_dataset_loader import CausalCSVDatasetLoader
from ..datasets.dataset import TemporalDataset
from typing import Optional, Tuple, List, Dict


logger = logging.getLogger(__name__)


class TemporalCausalCSVDatasetLoader(CausalCSVDatasetLoader):
    """
    Load a dataset from a CSV file in tabular format, i.e. where each row is an individual datapoint and each
    column is a feature. Load an adjacency matrix from a CSV file contained in the same data directory. 
    Load a variable number of intervention vectors together with their corresponding intervened data
    from CSV files contained within the same data directory.
    """

    _intervention_data_file = "interventions.csv"
    _adjacency_data_file = "adj_matrix.npy"
    _transition_matrix_file = "transition_matrix.npy"

    def split_data_and_load_dataset(
        self,
        test_frac: float,
        val_frac: float,
        max_num_rows: Optional[int] = None,
        negative_sample: bool = False,
        **kwargs,
    ) -> TemporalDataset:
        """
        Load the data from memory and make the train/val/test split to instantiate a dataset.
        The data is split deterministically along the time axis.

        Args:
            test_frac: Fraction of data to put in the test set.
            val_frac: Fraction of data to put in the validation set.
            max_num_rows: Maximum number of rows to include when reading data files.
            negative_sample: Whether to perform negative sampling after loading the dataset. Negative sampling requires
                a file negative_sampling_levels.csv in the dataset folder, and negative samples for each row will be
                drawn from features of a greater level than all those observed in the row.

        Returns:
            temporal_dataset: TemporalDataset object, holding the data and variable metadata as well as
            the transition matrix as a np.ndarray, the adjacency matrix as a np.ndarray and a list of IntervetionData
            objects, each containing an intervention vector and samples. 
        """

        dataset = super(CausalCSVDatasetLoader, self).split_data_and_load_dataset(
            test_frac, val_frac, 0, max_num_rows, negative_sample
        )

        logger.info("Create temporal dataset.")

        adjacency_data = self._get_adjacency_data()
        intervention_data = self._get_intervention_data(max_num_rows)
        transition_matrix = self._get_transition_matrix()
        temporal_dataset = dataset.to_temporal(adjacency_data, intervention_data, transition_matrix)
        return temporal_dataset

    def load_predefined_dataset(
        self, max_num_rows: Optional[int] = None, negative_sample: bool = False, **kwargs,
    ) -> TemporalDataset:
        """
        Load the data from memory and use the predefined train/val/test split to instantiate a dataset.

        Args:
            max_num_rows: Maximum number of rows to include when reading data files.
            negative_sample: Whether to perform negative sampling after loading the dataset. Negative sampling requires
                a file negative_sampling_levels.csv in the dataset folder, and negative samples for each row will be
                drawn from features of a greater level than all those observed in the row.

        Returns:
            temporal_dataset: TemporalDataset object, holding the data and variable metadata as well as
            the transition matrix as a np.ndarray, the adjacency matrix as a np.ndarray and a list of IntervetionData
            objects, each containing an intervention vector and samples. 
        """
        dataset = super(CausalCSVDatasetLoader, self).load_predefined_dataset(max_num_rows, negative_sample)

        logger.info("Create temporal dataset.")

        adjacency_data = self._get_adjacency_data()
        intervention_data = self._get_intervention_data(max_num_rows)
        transition_matrix = self._get_transition_matrix()
        temporal_dataset = dataset.to_temporal(adjacency_data, intervention_data, transition_matrix)
        return temporal_dataset

    @classmethod
    def _generate_data_split(
        cls, rows: List[int], test_frac: float, val_frac: float, *args, **kwargs
    ) -> Tuple[List[int], List[int], List[int], Dict[str, List[int]]]:
        """
        Split the given list of row indices into three lists using the given test and validation fraction.
        The data is split deterministically along the time axis.

        Args:
            rows: List of row indices to be split.
            test_frac: Fraction of rows to put in the test set.
            val_frac: Fraction of rows to put in the validation set.
        Returns:
            train_rows: List of row indices to assigned to the train set.
            val_rows: List of row indices to assigned to the validation set.
            test_rows: List of row indices to assigned to the test set.
            data_split: Dictionary record about how the row indices were split.
        """
        cls._validate_val_frac_test_frac(test_frac, val_frac)

        rows.sort()
        num_samples = len(rows)

        num_test = int(num_samples * test_frac)

        val_frac = val_frac / (1 - test_frac)
        num_val = int(num_samples * val_frac)

        if num_test > 0:
            test_rows = rows[-num_test:]
        else:
            test_rows = []

        if num_val > 0:
            val_rows = rows[:-num_test][-num_val:]
        else:
            val_rows = []

        train_rows = rows[: -(num_test + num_val)]

        train_rows.sort()
        val_rows.sort()
        test_rows.sort()
        data_split = {
            "train_idxs": [int(id) for id in train_rows],
            "val_idxs": [int(id) for id in val_rows],
            "test_idxs": [int(id) for id in test_rows],
        }

        return train_rows, val_rows, test_rows, data_split

    def _get_adjacency_data(self):

        adjacency_data_path = os.path.join(self._dataset_dir, self._adjacency_data_file)

        adjacency_file_exists = all([os.path.exists(adjacency_data_path)])

        if not adjacency_file_exists:
            logger.info("DAG adjacency matrix not found.")
            adjacency_data = None
        else:
            logger.info("DAG adjacency matrix found.")
            adjacency_data = np.load(adjacency_data_path)

        return adjacency_data

    def _get_transition_matrix(self):

        transition_matrix_path = os.path.join(self._dataset_dir, self._transition_matrix_file)

        transition_matrix_file_exists = all([os.path.exists(transition_matrix_path)])

        if not transition_matrix_file_exists:
            logger.info("Transition matrix not found.")
            transition_matrix = None
        else:
            logger.info("Transition matrix found.")
            transition_matrix = np.load(transition_matrix_path)

        return transition_matrix

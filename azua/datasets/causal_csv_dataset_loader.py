import os
import numpy as np

import logging
from ..datasets.csv_dataset_loader import CSVDatasetLoader
from ..datasets.dataset import CausalDataset, IntervetionData
from typing import Optional, Tuple, Union


logger = logging.getLogger(__name__)


class CausalCSVDatasetLoader(CSVDatasetLoader):
    """
    Load a dataset from a CSV file in tabular format, i.e. where each row is an individual datapoint and each
    column is a feature. Load an adjacency matrix from a CSV file contained in the same data directory. 
    Load a variable number of intervention vectors together with their corresponding intervened data
    from CSV files contained within the same data directory.
    """

    _intervention_data_file = "interventions.csv"
    _adjacency_data_file = "adj_matrix.csv"

    def split_data_and_load_dataset(
        self,
        test_frac: float,
        val_frac: float,
        random_state: Union[int, Tuple[int, int]],
        max_num_rows: Optional[int] = None,
        negative_sample: bool = False,
        **kwargs,
    ) -> CausalDataset:
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

        Returns:
            causal_dataset: CausalDataset object, holding the data and variable metadata as well as
            the adjacency matrix as a np.ndarray and a list of IntervetionData objects, each containing an intervention
            vector and samples. 
        """

        dataset = super().split_data_and_load_dataset(test_frac, val_frac, random_state, max_num_rows, negative_sample)

        logger.info("Create causal dataset.")

        adjacency_data = self._get_adjacency_data()
        intervention_data = self._get_intervention_data(max_num_rows)
        causal_dataset = dataset.to_causal(adjacency_data, intervention_data)
        return causal_dataset

    def load_predefined_dataset(
        self, max_num_rows: Optional[int] = None, negative_sample: bool = False, **kwargs,
    ) -> CausalDataset:
        """
        Load the data from memory and use the predefined train/val/test split to instantiate a dataset.

        Args:
            max_num_rows: Maximum number of rows to include when reading data files.
            negative_sample: Whether to perform negative sampling after loading the dataset. Negative sampling requires
                a file negative_sampling_levels.csv in the dataset folder, and negative samples for each row will be
                drawn from features of a greater level than all those observed in the row.

        Returns:
            causal_dataset: CausalDataset object, holding the data and variable metadata as well as
            the adjacency matrix as a np.ndarray and a list of IntervetionData objects, each containing an intervention
            vector and samples. 
        """
        dataset = super().load_predefined_dataset(max_num_rows, negative_sample)

        logger.info("Create causal dataset.")

        adjacency_data = self._get_adjacency_data()
        intervention_data = self._get_intervention_data(max_num_rows)
        causal_dataset = dataset.to_causal(adjacency_data, intervention_data)
        return causal_dataset

    def _get_adjacency_data(self):

        adjacency_data_path = os.path.join(self._dataset_dir, self._adjacency_data_file)

        adjacency_file_exists = all([os.path.exists(adjacency_data_path)])

        if not adjacency_file_exists:
            logger.info("DAG adjacency matrix not found.")
            adjacency_data = None
        else:
            logger.info("DAG adjacency matrix found.")
            adjacency_data, mask = self.read_csv_from_file(adjacency_data_path)
            # check there are no missing values in the adjacency matrix
            assert np.all(mask == 1)

        return adjacency_data

    def _get_intervention_data(self, max_num_rows):

        intervention_data_path = os.path.join(self._dataset_dir, self._intervention_data_file)
        intervention_file_exists = all([os.path.exists(intervention_data_path)])

        if not intervention_file_exists:
            logger.info("Intervention data not found.")
            intervention_data = None
        else:
            logger.info("Intervention data found.")
            raw_intervention_data, mask = self.read_csv_from_file(intervention_data_path, max_num_rows=max_num_rows)
            intervention_data = self._process_intervention_data(raw_intervention_data, mask)

        return intervention_data

    @classmethod
    def _process_intervention_data(cls, raw_intervention_data, mask):
        """
        TODO: re-structure this method into smaller sub-methods to increase readability

           Parse the raw data from the interventions.csv file, separating the intervened variables, their intervened values and samples from the intervened distribution.
           Also, if they exist, retrieves indinces of effect variables, reference variables, data generated with reference interventions, conditioning indices and conditioning variables.
           If they do not exist, those fields of the IntervetionData object are populated with None.
           Expects format of interventions.csv to be 5xN_vars columns. The order is [conditioning_cols, intervention_cols, reference_cols, effect_mask_cols, data_cols].
           It is infered automatically which rows correspond to the same intervention.

            Args:
                raw_intervention_data: np.ndarray read directly from interventions.csv
                mask: Corresponding mask, where observed values are 1 and np.nan values (representing non-intervened variables) are 0.

            Returns:
                causal_dataset: List of instances of IntervetionData, one per each intervention.
                
            """

        Ncols = int(raw_intervention_data.shape[1] / 5)
        Nrows = raw_intervention_data.shape[0]

        # Split into rows that contain conditioning vectors, intervention vectors, referrence vectors, effect vectors, and rows that contain samples
        conditioning_cols = raw_intervention_data[:, :Ncols].astype(float)
        conditioning_mask_cols = mask[:, :Ncols]

        intervention_cols = raw_intervention_data[:, Ncols : 2 * Ncols].astype(float)
        intervention_mask_cols = mask[:, Ncols : 2 * Ncols]

        reference_cols = raw_intervention_data[:, 2 * Ncols : 3 * Ncols].astype(float)
        reference_mask_cols = mask[:, 2 * Ncols : 3 * Ncols]

        effect_mask_cols = mask[:, 3 * Ncols : 4 * Ncols]

        sample_cols = raw_intervention_data[:, -Ncols:].astype(float)

        # Iterate over file rows, checking if they contain the start of a new intervention
        intervention_data = []
        intervention_start_row = 0
        has_ref = False

        # Process first row

        # identify conditionioning variable indices and their values
        conditioning_idxs = np.where(conditioning_mask_cols[0, :] == 1)[0]
        conditioning_values = conditioning_cols[0, conditioning_idxs]

        # identify intervention variable indices and their values
        intervention_idxs = np.where(intervention_mask_cols[0, :] == 1)[0]
        intervention_values = intervention_cols[0, intervention_idxs]

        # identify reference variable indices and their values
        reference_idxs = np.where(reference_mask_cols[0, :] == 1)[0]
        reference_values = reference_cols[0, reference_idxs]
        assert len(reference_idxs) == 0, "reference identified in data without previous intervention"

        # identify effect variable indices and their values
        effect_idxs = np.where(effect_mask_cols[0, :] == 1)[0]

        # Process all remaining rows
        for n_row in range(1, Nrows):

            next_conditioning_idxs = np.where(conditioning_mask_cols[n_row, :] == 1)[0]
            next_conditioning_values = conditioning_cols[n_row, next_conditioning_idxs]

            next_intervention_idxs = np.where(intervention_mask_cols[n_row, :] == 1)[0]
            next_intervention_values = intervention_cols[n_row, next_intervention_idxs]

            next_reference_idxs = np.where(reference_mask_cols[n_row, :] == 1)[0]
            next_reference_values = reference_cols[n_row, next_reference_idxs]

            next_effect_idxs = np.where(effect_mask_cols[n_row, :] == 1)[0]

            intervention_change = (
                next_intervention_idxs != intervention_idxs or next_intervention_values != intervention_values
            )

            ref_start = len(reference_idxs) == 0 and len(next_reference_idxs) > 0

            # check for the start of reference data for an intervention
            if ref_start:
                assert not has_ref, "there must be no more than one reference dataset per intervention dataset"
                assert (
                    n_row > intervention_start_row
                ), "there must be interevention test data for there to be reference data"
                has_ref = True
                intervention_end_row = n_row

                reference_idxs = next_reference_idxs
                reference_values = next_reference_values

            # decide data for a given intervention has finished based on where the intervened indices or values change
            if intervention_change:

                # Ensure that we dont intervene, condition or measure effect on same variable
                assert not (set(intervention_idxs) & set(conditioning_idxs) & set(effect_idxs))
                # Ensure that reference incides are empty or match the treatment indices
                assert reference_idxs == intervention_idxs or len(reference_idxs) == 0

                # Check for references, conditioning and effects. Set to None if they are not present in the data
                if has_ref:
                    reference_data = sample_cols[intervention_end_row:n_row]
                    has_ref = False
                else:
                    intervention_end_row = n_row
                    reference_data = None
                    reference_values = None

                if len(effect_idxs) == 0:
                    effect_idxs = None
                if len(conditioning_idxs) == 0:
                    conditioning_values = None
                if len(conditioning_idxs) == 0:
                    conditioning_idxs = None

                intervention_data.append(
                    IntervetionData(
                        conditioning_idxs=conditioning_idxs,
                        conditioning_values=conditioning_values,
                        effect_idxs=effect_idxs,
                        intervention_idxs=intervention_idxs,
                        intervention_values=intervention_values,
                        intervention_reference=reference_values,
                        test_data=sample_cols[intervention_start_row:intervention_end_row],
                        reference_data=reference_data,
                    )
                )

                intervention_start_row = n_row

                intervention_idxs = next_intervention_idxs
                intervention_values = next_intervention_values

                conditioning_idxs = next_conditioning_idxs
                conditioning_values = next_conditioning_values

                effect_idxs = next_effect_idxs

                reference_idxs = next_reference_idxs
                reference_values = next_reference_values

        # Process final intervention
        if has_ref:
            reference_data = sample_cols[intervention_end_row:]
        else:
            intervention_end_row = n_row + 1
            reference_data = None
            reference_values = None

        if len(effect_idxs) == 0:
            effect_idxs = None
        if len(conditioning_idxs) == 0:
            conditioning_values = None
        if len(conditioning_idxs) == 0:
            conditioning_idxs = None

        intervention_data.append(
            IntervetionData(
                conditioning_idxs=conditioning_idxs,
                conditioning_values=conditioning_values,
                effect_idxs=effect_idxs,
                intervention_idxs=intervention_idxs,
                intervention_values=intervention_values,
                intervention_reference=reference_values,
                test_data=sample_cols[intervention_start_row:intervention_end_row],
                reference_data=reference_data,
            )
        )
        return intervention_data

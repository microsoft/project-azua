from ..datasets.sentence_transformer_model import SentenceTransformerModel
from ..datasets.itext_embedding_model import ITextEmbeddingModel
import logging
import warnings
from typing import Tuple, List, overload, Union, Optional

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
from scipy.sparse import issparse, csr_matrix
from tqdm import tqdm
import torch

from ..datasets.variables import Variables
from ..datasets.dataset import Dataset, SparseDataset

EPSILON = 1e-5
logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(
        self,
        variables: Variables,
        squash_input: Optional[bool] = None,
        text_embedder: Optional[ITextEmbeddingModel] = None,
    ):
        """
        Args:
            variables (Variables): Information about variables/features used
                by this model.
            squash_input (bool): squash VAE input or not
            text_embedder (ITextEmbeddingModel): text embedding model for processing text variables. If text variables present, it defaults to SentenceTransformerModel
        """
        self._variables = variables
        if squash_input is not None:
            self._squash_input = squash_input
        else:
            self._squash_input = True
        # Call unprocessed columns unproc_cols, processed columns proc_cols
        unproc_cols_by_type = self._variables.unprocessed_cols_by_type
        proc_cols_by_type = self._variables.processed_cols_by_type

        def flatten(lists):
            # Flatten proc_cols for continuous and binary unproc_cols, since they will be of form [[1], [2], ...]
            return [i for sublist in lists for i in sublist]

        if "binary" in unproc_cols_by_type:
            self._bin_unproc_cols = unproc_cols_by_type["binary"]
            self._bin_proc_cols = flatten(proc_cols_by_type["binary"])

            # Save contiguous regions containig binary features to allow for more efficient processing via slicing
            self._bin_unproc_regions = self._split_contiguous_sublists(self._bin_unproc_cols)
            self._bin_proc_regions = self._split_contiguous_sublists(self._bin_proc_cols)
            assert len(self._bin_unproc_regions) == len(self._bin_proc_regions)
        else:
            self._bin_unproc_cols, self._bin_proc_cols = [], []

        if "continuous" in unproc_cols_by_type:
            self._cts_unproc_cols = unproc_cols_by_type["continuous"]
            self._cts_proc_cols = flatten(proc_cols_by_type["continuous"])

            # Save contiguous regions containig continuous features to allow for more efficient processing via slicing
            if all(x.overwrite_processed_dim is None for x in self._variables):
                self._cts_unproc_regions = self._split_contiguous_sublists(self._cts_unproc_cols)
                self._cts_proc_regions = self._split_contiguous_sublists(self._cts_proc_cols)
            else:
                # For VAEM, we can only take single variable as region
                # to allow for processing/reverting mask
                self._cts_unproc_regions = [[col_id] for col_id in unproc_cols_by_type["continuous"]]
                self._cts_proc_regions = proc_cols_by_type["continuous"]

            assert len(self._cts_unproc_regions) == len(self._cts_proc_regions)
        else:
            self._cts_unproc_cols, self._cts_proc_cols = [], []

        if "categorical" in unproc_cols_by_type:
            self._cat_unproc_cols = unproc_cols_by_type["categorical"]
            self._cat_proc_cols = flatten(proc_cols_by_type["categorical"])
            self._cat_proc_cols_grouped = proc_cols_by_type["categorical"]

            def get_lower(idx):
                return self._variables[idx].lower

            def get_upper(idx):
                return self._variables[idx].upper

            var_categories = [
                np.arange(int(get_lower(var_idx)), int(get_upper(var_idx)) + 1) for var_idx in self._cat_unproc_cols
            ]
            self._one_hot_encoder = OneHotEncoder(categories=var_categories, sparse=False, handle_unknown="ignore")
            # Fit on dummy data due to an issue in sklearn where the encoder needs to be fitted to data even if the
            # categories are specified upon creation.
            self._one_hot_encoder.fit(np.array([categories[0] for categories in var_categories]).reshape(1, -1))
        else:
            self._cat_unproc_cols, self._cat_proc_cols = [], []

        if "text" in unproc_cols_by_type:
            if text_embedder is None:
                text_embedder = SentenceTransformerModel()
            # TODO: add assert on whether hidden space's dimensions agree
            self._txt_unproc_cols = unproc_cols_by_type["text"]
            self._txt_proc_cols = flatten(proc_cols_by_type["text"])
            self._txt_proc_cols_grouped = proc_cols_by_type["text"]

            self._text_embedder = text_embedder
        else:
            self._txt_unproc_cols, self._txt_proc_cols = [], []

        self._num_processed_cols = sum([var.processed_dim for var in self._variables])

    @overload
    def process_data_and_masks(
        self, data: np.ndarray, data_mask: np.ndarray, *extra_masks: np.ndarray, batch_size: int = 1000,
    ) -> Tuple[np.ndarray, ...]:
        ...

    @overload
    def process_data_and_masks(
        self, data: csr_matrix, data_mask: csr_matrix, *extra_masks: csr_matrix, batch_size: int = 1000,
    ) -> Tuple[csr_matrix, ...]:
        ...

    def process_data_and_masks(self, data, data_mask, *extra_masks, batch_size=1000):
        """
        Process and validate data, data mask and optionally any number of additional masks. These masks will all be applied
        to the data when performing data range validation, in case of e.g. dummy zero data that is masked out by an
        additional obs_mask.

        Args:
            data: Unprocessed data array
            data_mask: Data indicating which values in `data` are observed. Can be any dtype provided all values are
                either 0 or 1.
            extra_masks: Additional masks to be processed, if any. Can be any dtype provided all values are either 0 or 
                1.
            batch_size: Batch size used during data preprocessing for sparse matrices.
        Returns:
            processed_data: Data with categorical variables expanded to a one-hot encoding, and features normalised.
            processed_data_mask: Boolean mask with categorical variables expanded to a one-hot encoding.
            processed_extra_masks: Any additional boolean masks with categorical variables expanded to a one-hot 
                encoding.
        """

        if not issparse(data):
            (proc_data, proc_data_mask, *proc_extra_masks,) = self._process_and_check_dense(
                data, data_mask, *extra_masks
            )
        else:
            # Break sparse data into smaller batches and preprocess each as a dense array. Somewhat inefficient but
            # allows us to reuse our preprocessing functions and keeps memory usage manageable.
            proc_data_list: List[csr_matrix] = []
            proc_data_mask_list: List[csr_matrix] = []
            proc_extra_masks_lists: Tuple[List[csr_matrix], ...] = tuple([] for mask in extra_masks)
            num_rows = data.shape[0]
            for start_idx in tqdm(range(0, num_rows, batch_size), desc="Data preprocessing"):
                stop_idx = min(start_idx + batch_size, num_rows)
                data_batch = data[start_idx:stop_idx].toarray()
                data_mask_batch = data_mask[start_idx:stop_idx].toarray()
                extra_masks_batch = tuple(mask[start_idx:stop_idx].toarray() for mask in extra_masks)

                # TODO: we will currently lose sparsity for rescaled continuous data here, since 0 will be mapped to
                # another value. We could multiply by the mask to zero out unobserved data but we need to make sure this
                # doesn't have any unintended consequences for cases with more complex masking, e.g. active learning
                (proc_data_batch, proc_data_mask_batch, *proc_extra_masks_batch,) = self._process_and_check_dense(
                    data_batch, data_mask_batch, *extra_masks_batch
                )
                proc_data_list.append(csr_matrix(proc_data_batch))
                proc_data_mask_list.append(csr_matrix(proc_data_mask_batch))
                for mask_list, mask in zip(proc_extra_masks_lists, proc_extra_masks_batch):
                    mask_list.append(csr_matrix(mask))

            proc_data = sparse.vstack(proc_data_list, format="csr")
            proc_data_mask = sparse.vstack(proc_data_mask_list, format="csr")
            proc_extra_masks = tuple(
                sparse.vstack(proc_mask_list, format="csr") for proc_mask_list in proc_extra_masks_lists
            )

        return (proc_data, proc_data_mask, *proc_extra_masks)

    def _process_and_check_dense(self, data: np.ndarray, data_mask: np.ndarray, *extra_masks: np.ndarray):
        """
        Check validity of dense data and masks and process them.
        """
        combined_mask = data_mask
        for mask in extra_masks:
            combined_mask = combined_mask * mask
        self.check_data(data, combined_mask)
        self.check_mask(data_mask)
        for mask in extra_masks:
            self.check_mask(mask)
        proc_data = self.process_data(data)
        proc_data_mask = self.process_mask(data_mask)
        proc_extra_masks = tuple(self.process_mask(mask) for mask in extra_masks)
        return (proc_data, proc_data_mask, *proc_extra_masks)

    def process_dataset(self, dataset: Union[Dataset, SparseDataset]) -> Union[Dataset, SparseDataset]:
        train_data, train_mask = self.process_data_and_masks(*dataset.train_data_and_mask)
        val_data, _ = dataset.val_data_and_mask
        if val_data is not None:
            val_data, val_mask = self.process_data_and_masks(*dataset.val_data_and_mask)
        else:
            val_data, val_mask = None, None
        test_data, _ = dataset.test_data_and_mask
        if test_data is not None:
            test_data, test_mask = self.process_data_and_masks(*dataset.test_data_and_mask)
        else:
            test_data, test_mask = None, None
        return type(dataset)(
            train_data, train_mask, val_data, val_mask, test_data, test_mask, variables=dataset.variables
        )

    def check_mask(self, mask: np.ndarray) -> None:
        """
        Check mask contains 1 and 0 only
        """
        if len(mask.shape) != 2 or mask.shape[1] != len(self._variables):
            raise ValueError(
                "Mask must be 2D with shape (row_count, feature_count + aux_count)."
                "Mask has shape %s and feature_count is %d." % (str(mask.shape), len(self._variables))
            )
        bool_mask = mask.astype(bool)

        if not np.array_equal(mask, bool_mask):
            raise ValueError("Mask must contain 1 and 0 only.")

    def check_data(self, data: np.ndarray, mask: np.ndarray) -> None:
        """
        Check that each column of the data is valid with respect to the given variable definition.
        Raise an error if a discrete variable (binary or categorical) is not an integer or not within the specified range.
        Make a warning if a continuous variable is not within the specified range.
        Note that only observed values are checked.

        Args:
            variables: Variables object for data
            data: Unprocessed data array with shape (num_rows, num_features)
            mask: Mask indicting observed variables with shape (num_rows, num_features). 1 is observed, 0 is un-observed.
        """
        lower = np.array([var.lower for var in self._variables])
        upper = np.array([var.upper for var in self._variables])

        # Continuous variables
        cts_idxs = self._variables.continuous_idxs
        if len(cts_idxs) > 0:
            self.check_continuous_data(
                data=data[:, cts_idxs],
                mask=mask[:, cts_idxs],
                lower=lower[cts_idxs],
                upper=upper[cts_idxs],
                epsilon=EPSILON,
            )

        # Discrete variables
        disc_idxs = self._variables.discrete_idxs
        if len(disc_idxs) > 0:
            self.check_discrete_data(
                data=data[:, disc_idxs],
                mask=mask[:, disc_idxs],
                lower=lower[disc_idxs],
                upper=upper[disc_idxs],
                epsilon=EPSILON,
            )

    def check_continuous_data(
        self, data: np.ndarray, mask: np.ndarray, lower: np.ndarray, upper: np.ndarray, epsilon: float,
    ) -> None:
        """
        Check if values in each column of the given continuous data are in the specified range. Make a warning
        if there is at least one value outside of the specified range. Note that only observed values are checked.

        Args:
            data: Unprocessed data array with shape (num_rows, num_features)
            mask: Mask indicting observed variables with shape (num_rows, num_features). 1 is observed, 0 is un-observed.
            lower: Array of column lower bounds with shape (num_features,)
            upper: Array of column upper bounds with shape (num_features,)
            epsilon: How close to the specified range we require values to be
        """
        lower_diff = data - lower
        higher_diff = data - upper
        too_low_cols = np.any(lower_diff * mask < -1 * epsilon, axis=0)
        too_high_cols = np.any(higher_diff * mask > epsilon, axis=0)

        too_low = np.any(too_low_cols)
        too_high = np.any(too_high_cols)

        if too_low:
            warnings.warn(
                f"Data too low for continous variables {np.where(too_low_cols)[0]}", UserWarning,
            )
        if too_high:
            warnings.warn(
                f"Data too high for continous variables {np.where(too_high_cols)[0]}", UserWarning,
            )

    def check_discrete_data(
        self, data: np.ndarray, mask: np.ndarray, lower: np.ndarray, upper: np.ndarray, epsilon: float,
    ) -> None:
        """
        Check if values in each column of the given discrete (binary and categorical) data are in the specified range.
        Raise an error if there is at least one value outside of the specified range.
        Additionally, assert that all the given values are integers. Note that only observed values are checked.

        Args:
            data: Unprocessed data array with shape (num_rows, num_features)
            mask: Mask indicting observed variables with shape (num_rows, num_features). 1 is observed, 0 is un-observed.
            lower: Array of column lower bounds with shape (num_features,)
            upper: Array of column upper bounds with shape (num_features,)
            epsilon: How close to the specified range we require values to be
        """
        lower_diff = data - lower
        higher_diff = data - upper
        too_low_cols = np.any(lower_diff * mask < -1 * epsilon, axis=0)
        too_high_cols = np.any(higher_diff * mask > epsilon, axis=0)

        too_low = np.any(too_low_cols)
        too_high = np.any(too_high_cols)

        if too_low and too_high:
            raise ValueError(
                f"Data too low for discrete variables {np.where(too_low_cols)[0]} \n"
                f"Data too high for discrete variables {np.where(too_high_cols)[0]}"
            )
        if too_low:
            raise ValueError(f"Data too low for discrete variables {np.where(too_low_cols)[0]}")
        if too_high:
            raise ValueError(f"Data too high for discrete variables {np.where(too_high_cols)[0]}")

        # Check all unmasked values are integer-valued.
        observed_data = data * mask
        is_integer = np.floor_divide(observed_data, 1) == observed_data
        assert np.all(is_integer)

    def process_data(self, data: np.ndarray) -> np.ndarray:
        """
        Args:
            data: Array of shape (num_rows, feature_count + aux_count)
        Returns:
            processed_data: Array of shape (num_rows, num_processed_cols)
        """

        def squash(vals, lower, upper):
            return (vals - lower) / (upper - lower)

        num_rows, _ = data.shape

        # If all features are binary, no processing required so short-circuit here
        if len(self._cts_unproc_cols) == 0 and len(self._cat_unproc_cols) == 0:
            return data.astype(float)

        processed_data = np.full((num_rows, self._num_processed_cols), fill_value=np.nan)

        def get_lower(var_idx):
            return self._variables[var_idx].lower

        def get_upper(var_idx):
            return self._variables[var_idx].upper

        # Iterate through each contiguous subarray of features of each type. Can guarantee that these regions will line
        # up between processed and unprocessed arrays since we don't change the feature order. We do this since
        # accessing/writing slices is much more efficient in NumPy than fancy indexing.
        # TODO: if we can sort/unsort features by type during processing without breaking anything, then we can simply
        # do one slice of the array per feature type and not need all this extra complexity.

        if self._bin_unproc_cols:
            for unproc_region, proc_region in zip(self._bin_unproc_regions, self._bin_proc_regions):
                unproc_start, unproc_end = unproc_region[0], unproc_region[-1] + 1
                proc_start, proc_end = proc_region[0], proc_region[-1] + 1
                processed_data[:, proc_start:proc_end] = data[:, unproc_start:unproc_end].astype(float)

        if self._cts_unproc_cols:
            for unproc_region, proc_region in zip(self._cts_unproc_regions, self._cts_proc_regions):
                unproc_start, unproc_end = unproc_region[0], unproc_region[-1] + 1
                proc_start, proc_end = proc_region[0], proc_region[-1] + 1
                lower_vals = np.array([get_lower(var_id) for var_id in unproc_region])
                upper_vals = np.array([get_upper(var_id) for var_id in unproc_region])
                cts_unproc_data = data[:, unproc_start:unproc_end].astype(float)
                if self._squash_input:
                    processed_data[:, proc_start:proc_end] = squash(cts_unproc_data, lower_vals, upper_vals)
                else:
                    processed_data[:, proc_start:proc_end] = cts_unproc_data

        if self._cat_unproc_cols:
            # Don't currently split into separate contiguous subarrays for categorical vars since we only want a single
            # one-hot encoder for simplicity.
            cat_unproc_data = data[:, self._cat_unproc_cols].astype(float)
            processed_data[:, self._cat_proc_cols] = self._one_hot_encoder.transform(cat_unproc_data)

        if self._txt_unproc_cols:
            processed_data[:, self._txt_proc_cols] = self._text_embedder.encode(data[:, self._txt_unproc_cols])

        return processed_data

    @overload
    def process_mask(self, mask: np.ndarray) -> np.ndarray:
        ...

    @overload
    def process_mask(self, mask: torch.Tensor) -> torch.Tensor:
        ...

    def process_mask(self, mask):
        """
        Args:
            mask: Array/Tensor of shape (num_rows, feature_count + aux_count) taking values 0 or 1
        Returns:
            processed_mask: Boolean array of shape (num_rows, num_processed_cols)
        """
        num_rows, _ = mask.shape

        if isinstance(mask, np.ndarray):  # If numpy array opperate on bools
            processed_mask = np.zeros((num_rows, self._num_processed_cols), dtype=bool)
        elif isinstance(mask, torch.Tensor):  # If torch tensors operate on floats
            processed_mask = torch.zeros((num_rows, self._num_processed_cols), dtype=mask.dtype, device=mask.device)
        else:
            raise ValueError("Wrong type of mask object")

        if self._bin_unproc_cols:
            for unproc_region, proc_region in zip(self._bin_unproc_regions, self._bin_proc_regions):
                unproc_start, unproc_end = unproc_region[0], unproc_region[-1] + 1
                proc_start, proc_end = proc_region[0], proc_region[-1] + 1
                processed_mask[:, proc_start:proc_end] = mask[:, unproc_start:unproc_end]

        if self._cts_unproc_cols:
            for unproc_region, proc_region in zip(self._cts_unproc_regions, self._cts_proc_regions):
                unproc_start, unproc_end = unproc_region[0], unproc_region[-1] + 1
                proc_start, proc_end = proc_region[0], proc_region[-1] + 1
                processed_mask[:, proc_start:proc_end] = mask[:, unproc_start:unproc_end]

        if self._cat_unproc_cols:
            for var, proc_cols in zip(self._cat_unproc_cols, self._cat_proc_cols_grouped):
                # Index with var:var+1 to return 2D array rather than 1D to allow broadcasting.
                processed_mask[:, proc_cols] = mask[:, var : var + 1]

        if self._txt_unproc_cols:
            for var, proc_cols in zip(self._txt_unproc_cols, self._txt_proc_cols_grouped):
                # Index with var:var+1 to return 2D array rather than 1D to allow broadcasting.
                processed_mask[:, proc_cols] = mask[:, var : var + 1]

        return processed_mask

    @overload
    def revert_mask(self, mask: np.ndarray) -> np.ndarray:
        ...

    @overload
    def revert_mask(self, mask: torch.Tensor) -> torch.Tensor:
        ...

    def revert_mask(self, mask):
        """
        Revert processed mask into unprocessed form (i.e. squash categorical/text var indices).

        Args:
            variables:
            mask: Numpy array/Torch tensor with shape (num_rows, input_count)

        Returns:
            data: Numpy array/Torch tensor with shape (num_rows, feature_count + aux_count)
        """
        proc_cols_to_delete = []
        for idx, var in enumerate(self._variables):
            if var.type not in ("categorical", "text") and var.overwrite_processed_dim is not None:
                continue
            cols = self._variables.processed_cols[idx]
            # Delete all columns except for first one
            proc_cols_to_delete += cols[1:]
        proc_cols_to_stay = [col for col in range(mask.shape[1]) if col not in proc_cols_to_delete]
        return mask[:, proc_cols_to_stay]

    def revert_data(self, data: np.ndarray) -> np.ndarray:
        """
        Undo processing to return output in the same form as the input. Sort-of-inverse of process_data.
        This involves reversing the squash operation for continuous variables, changing one-hot
        categorical variables into a single natural number and reordering data.

        Args:
            data: Numpy array with shape (num_rows, input_count)

        Returns:
            data: Numpy array with shape (num_rows, feature_count + aux_count)
        """
        # revert_data() is only called on imputed data, which is inherently dense, so we assume a sparse matrix is never
        # passed into this method.

        def unsquash(vals, lower, upper):
            return (vals * (upper - lower)) + lower

        def get_lower(var_idx):
            return self._variables[var_idx].lower

        def get_upper(var_idx):
            return self._variables[var_idx].upper

        num_rows, _ = data.shape

        unprocessed_data = np.empty((num_rows, self._variables.num_unprocessed_cols), dtype=object)

        if self._bin_unproc_cols:
            for unproc_region, proc_region in zip(self._bin_unproc_regions, self._bin_proc_regions):
                unproc_start, unproc_end = unproc_region[0], unproc_region[-1] + 1
                proc_start, proc_end = proc_region[0], proc_region[-1] + 1
                unprocessed_data[:, unproc_start:unproc_end] = data[:, proc_start:proc_end]

        if self._cts_unproc_cols:
            for unproc_region, proc_region in zip(self._cts_unproc_regions, self._cts_proc_regions):
                unproc_start, unproc_end = unproc_region[0], unproc_region[-1] + 1
                proc_start, proc_end = proc_region[0], proc_region[-1] + 1
                lower_vals = np.array([get_lower(var_id) for var_id in unproc_region])
                upper_vals = np.array([get_upper(var_id) for var_id in unproc_region])
                if self._squash_input:
                    unprocessed_data[:, unproc_start:unproc_end] = unsquash(
                        data[:, proc_start:proc_end], lower_vals, upper_vals
                    )
                else:
                    unprocessed_data[:, unproc_start:unproc_end] = data[:, proc_start:proc_end]

        if self._cat_unproc_cols:
            unprocessed_data[:, self._cat_unproc_cols] = self._one_hot_encoder.inverse_transform(
                data[:, self._cat_proc_cols]
            )

        if self._txt_unproc_cols:
            unprocessed_data[:, self._txt_unproc_cols] = self._text_embedder.decode(data[:, self._txt_proc_cols])

        return unprocessed_data

    @staticmethod
    def _split_contiguous_sublists(ints: List[int]) -> List[List[int]]:
        """
        Map from list of ints to list of contiguous sublists. E.g. [1,2,4,6,7] -> [[1,2],[4],[6,7]]. Assumes input list
        is sorted.
        """
        out: List[List[int]] = []
        for i in ints:
            if len(out) == 0:
                out.append([i])
            elif i == out[-1][-1] + 1:
                out[-1].append(i)
            else:
                out.append([i])
        return out
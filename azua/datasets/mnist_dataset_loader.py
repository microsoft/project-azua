import logging
from typing import Optional, Tuple, Union

import numpy as np
from torchvision.datasets import MNIST

from ..datasets.dataset import Dataset
from ..datasets.dataset_loader import DatasetLoader
from ..datasets.variables import Variables
from ..utils.helper_functions import maintain_random_state
from ..utils.torch_utils import set_random_seeds

logger = logging.getLogger(__name__)


class MNISTDatasetLoader(DatasetLoader):
    """
    Load the MNIST dataset from torchvision using the train/test split provided externally.
    """

    # Where to split to MNIST into a binary classification task; 0-4 inclusive are 0, 5-9 inclusive are 1
    _binary_split = 4
    _image_shape = (28, 28)

    def split_data_and_load_dataset(
        self,
        test_frac: Optional[float],  # type: ignore[override]
        val_frac: float,
        random_state: Union[int, Tuple[int, int]],
        max_num_rows: Optional[int] = None,
        **kwargs,
    ) -> Dataset:
        """

        The canonical MNIST split is a two-way train/test split. In order to make our results comparable with others,
        we always use the canonical test set, but we generate different random splits of the canonical training set
        into train and val.

        Args:
            test_frac: Ignored (allowed for consistency of interface with other dataset loaders).
            val_frac: Fraction of data to put in the validation set.
            random_state: Used as the splitting random state.
            max_num_rows: Maximum number of rows to include when reading data files.
        Returns:
            dataset: Dataset object, holding the data and variable metadata.
        """
        logger.info(f"Splitting train data to load the dataset: validation fraction: {val_frac}.")
        if test_frac is not None:
            logger.info(
                f"Ignoring test_frac {test_frac}. Using predefined train/test split "
                "and splitting predefined training data into train and val"
            )
        fixed_dataset = self.load_predefined_dataset(max_num_rows, **kwargs)

        train_and_val_data, train_and_val_mask = fixed_dataset.train_data_and_mask
        test_data, test_mask = fixed_dataset.test_data_and_mask
        train_rows, val_rows, _, data_split = self._generate_data_split(
            list(range(train_and_val_data.shape[0])), val_frac=val_frac, test_frac=0, random_state=random_state
        )
        data_split["test_idxs"] = "predefined"  # type: ignore
        train_data = train_and_val_data[train_rows]
        train_mask = train_and_val_mask[train_rows]
        val_data = train_and_val_data[val_rows]
        val_mask = train_and_val_mask[val_rows]

        return Dataset(
            train_data=train_data,
            train_mask=train_mask,
            val_data=val_data,
            val_mask=val_mask,
            test_data=test_data,
            test_mask=test_mask,
            variables=fixed_dataset.variables,
            data_split=data_split,
        )

    def load_predefined_dataset(  # type: ignore
        self,
        max_num_rows: Optional[int],
        use_targets: bool = False,
        binary: bool = True,
        split_between: Optional[Tuple[int, int]] = None,
        test_limit: Optional[int] = None,
        task_info: Optional[Tuple[int, int]] = None,
        var_binary: bool = False,
        **kwargs,
    ) -> Dataset:
        """
        Load the data from disk and use the predefined train/test split to instantiate a dataset.
        Args:
            max_num_rows: Maximum number of rows to include when reading data files.
            use_targets: Whether we include the targets (i.e., class) or not.
            binary: Turns MNIST into a binary dataset of [0-4] and [5-9].
            split_between: Used for continual/multitask learning settings; splits the MNIST data set between two
                classes. i.e., split_between = (0,1) means we produce a binary MNIST subset of just 0 and 1.
            test_limit: How many test data samples to return.
            task_info: Used to add task information, (1,5) means we are on task 1 out of 5, and will append a 1-hot
                encoding of [1,0,0,0,0].
            var_binary: Sets the variable type to 'binary' not continuous if True.
        Returns:
            dataset: Dataset object, holding the data and variable metadata.
        """
        logger.info("Using a predefined data split to load the dataset.")

        if max_num_rows is not None and test_limit is not None:
            test_limit = min(test_limit, max_num_rows)

        if binary and split_between:
            print(
                "Producing binary MNIST dataset, classifying between {} and {}".format(
                    split_between[0], split_between[1]
                )
            )
        elif binary:
            print("Producing binary MNIST dataset, classifying whether greater than {}".format(self._binary_split))
        elif split_between:
            raise Exception("Error: split_between ints are passed, but binary target not specified")
        else:
            print("Producing Full MNIST dataset")

        self._download_data_if_necessary(self._dataset_dir)

        # This will not download data if it already exists.
        train_dataset = MNIST(self._dataset_dir, train=True, download=False)
        test_dataset = MNIST(self._dataset_dir, train=False, download=False)

        if max_num_rows is None:
            train_data = train_dataset.data.numpy()
            test_data = test_dataset.data.numpy()
        else:
            train_data = train_dataset.data.numpy()[:max_num_rows]
            test_data = test_dataset.data.numpy()[:max_num_rows]

        batch, height, width = train_data.shape
        train_data = train_data.reshape((batch, height * width))

        batch, height, width = test_data.shape
        test_data = test_data.reshape((batch, height * width))

        # Normalise values to be between 0 and 1.
        # Original range is 0 to 255.
        x_min, x_max = (0, 255)

        train_data = (train_data - x_min) / (x_max - x_min)
        test_data = (test_data - x_min) / (x_max - x_min)

        train_targets = train_dataset.targets.numpy()
        test_targets = test_dataset.targets.numpy()

        train_data, train_targets = self._get_data_and_targets(train_data, train_targets, split_between, binary)
        test_data, test_targets = self._get_data_and_targets(test_data, test_targets, split_between, binary)

        if use_targets:
            # Append targets onto the end
            train_data = np.hstack((train_data, train_targets))
            test_data = np.hstack((test_data, test_targets))

        if task_info:
            # Get the task one hot encoding and append
            task_ohe = np.zeros((1, task_info[1]))
            task_ohe[0, task_info[0] - 1] = 1
            task_ohe_train = np.repeat(task_ohe, train_data.shape[0], axis=0)
            task_ohe_test = np.repeat(task_ohe, test_data.shape[0], axis=0)
            train_data = np.hstack((train_data, task_ohe_train))
            test_data = np.hstack((test_data, task_ohe_test))

        if test_limit is not None:
            with maintain_random_state():
                set_random_seeds(0)
                test_idxs = np.random.choice(len(test_data), test_limit, replace=False)
                test_data = test_data[test_idxs, :]

        # All values are observed
        train_mask = np.ones_like(train_data, dtype=bool)
        test_mask = np.ones_like(test_data, dtype=bool)

        variables = self._create_variables(use_targets, var_binary, binary, task_info)

        return Dataset(
            train_data=train_data,
            train_mask=train_mask,
            val_data=None,
            val_mask=None,
            test_data=test_data,
            test_mask=test_mask,
            variables=variables,
            data_split=self._predefined_data_split,
        )

    @classmethod
    def _get_data_and_targets(
        cls, data: np.ndarray, targets: np.ndarray, split_between: Optional[Tuple[int, int]], binary: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get target values, split data into subsets and convert to binary if any of these operations are specified.
        """
        if not split_between:
            # no need to subset if we aren't splitting between two digits
            data_out = data
            if binary:
                targets_out = np.where(targets > cls._binary_split, 1, 0)
            else:
                # if not binary, then it's just a standard multi-class classification task
                targets_out = targets
        elif binary and split_between:
            subset = np.logical_or(targets == split_between[0], targets == split_between[1])
            data_out, targets_out = data[subset], targets[subset]
            targets_out = np.where(targets_out == split_between[0], 1, 0)
        else:
            raise Exception("Cannot have split_between specified but not binary")

        targets_out = np.expand_dims(targets_out, axis=1)
        return data_out, targets_out

    @classmethod
    def _create_variables(
        cls, use_targets: bool, var_binary: bool, binary: bool, task_info: Optional[Tuple[int, int]]
    ) -> Variables:
        """
        Create variables object for MNIST dataset with specified options.
        """
        height, width = cls._image_shape
        min_val, max_val = (0, 1)  # We have normalised MNIST
        variable_info = []

        # Add a variable for each MNIST pixel.
        for row in range(height):
            for col in range(width):
                id = row * width + col
                var = {
                    "id": id,  # Feature index, 0 to pixel count - 1
                    "query": True,  # All features are query features.
                    "type": "binary" if var_binary else "continuous",  # Feature type is continuous/binary.
                    "name": "(%d, %d)" % (row, col),  # Short variable description
                    "lower": min_val,  # Min pixel value
                    "upper": max_val,  # Max pixel value
                }
                variable_info.append(var)

        if use_targets:
            # Add a variable for the predicted class.
            id = height * width
            predicted_class_var = {
                "id": id,  # Feature index, pixel count
                "query": False,  # Target feature, not query feature
                "type": "binary" if binary else "categorical",  # Feature type determined by binary flag.
                "name": "Class",  # Short variable description
                "lower": 0,  # Minimum class value is 0
                "upper": 1 if binary else 9,  # Maximum class value is 1 if binary, otherwise 9.
            }

            variable_info.append(predicted_class_var)

        if task_info:
            # Add variables pertaining to the task index.
            for i in range(task_info[1]):
                id = height * width + 1 + i
                task_var = {
                    "id": id,  # Feature index, starting from pixel count + 1
                    "query": False,  # Target feature, not query feature
                    "type": "binary",  # Feature type determined by binary flag.
                    "name": "Task",  # Short variable description
                    "lower": 0,  # Minimum class value is 0
                    "upper": 1,  # Maximum class value is 1 if binary, otherwise 9.
                }
                variable_info.append(task_var)

        return Variables.create_from_dict({"variables": variable_info, "metadata_variables": []})

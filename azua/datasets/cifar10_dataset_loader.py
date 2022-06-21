from typing import Optional, Tuple, Union

import numpy as np
from torchvision.datasets import CIFAR10

from ..datasets.dataset import Dataset
from ..datasets.dataset_loader import DatasetLoader
from ..datasets.variables import Variables


class CIFAR10DatasetLoader(DatasetLoader):
    """
    Load the CIFAR10 dataset from torchvision using the train/test split provided externally.
    """

    _image_shape = (3, 32, 32)

    def split_data_and_load_dataset(
        self,
        test_frac: float,
        val_frac: float,
        random_state: Union[int, Tuple[int, int]],
        max_num_rows: Optional[int] = None,
        **kwargs,
    ) -> Dataset:
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
        Returns:
            dataset: Dataset object, holding the data and variable metadata.
        """
        raise NotImplementedError("CIFAR10 cannot currently be split - specify use_predefined_dataset=True.")

    def load_predefined_dataset(self, max_num_rows: Optional[int], use_targets: bool = False, **kwargs) -> Dataset:  # type: ignore
        """
        Load the data from disk and use the predefined train/val/test split to instantiate a dataset.
        Args:
            max_num_rows: Maximum number of rows to include when reading data files.
            use_targets: Whether we include the targets (i.e., class) or not.
        Returns:
            dataset: Dataset object, holding the data and variable metadata.
        """
        # This will not download data if it already exists.
        train_all = CIFAR10(self._dataset_dir, train=True, download=True)
        test_all = CIFAR10(self._dataset_dir, train=False, download=True)

        train_data = train_all.data / 255
        test_data = test_all.data / 255

        # permute dimensions to NCHW before flattening
        train_data = train_data.transpose(0, 3, 1, 2).reshape(len(train_data), -1)
        test_data = test_data.transpose(0, 3, 1, 2).reshape(len(test_data), -1)

        if max_num_rows is not None:
            train_data = train_data[0:max_num_rows]
            test_data = test_data[0:max_num_rows]

        # append targets to data if using
        if use_targets:
            train_targets = self._get_targets(train_all)
            train_data = np.concatenate((train_data, train_targets), 1)

            test_targets = self._get_targets(test_all)
            test_data = np.concatenate((test_data, test_targets), 1)

        # all values are observed
        train_mask = np.ones_like(train_data, dtype=bool)
        test_mask = np.ones_like(test_data, dtype=bool)

        variables = self._create_variables(use_targets)

        # TODO: Add validation set
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

    @staticmethod
    def _get_targets(dataset: CIFAR10) -> np.ndarray:
        """
        Get targets from torch dataset and return as separate numpy array, ensuring a dimension isn't dropped.
        """
        targets = np.array(dataset.targets)
        return np.expand_dims(targets, axis=1)

    @classmethod
    def _create_variables(cls, use_targets: bool) -> Variables:
        """
        Create variables object for CIFAR10 dataset with specified options.
        """
        channels, height, width = cls._image_shape
        min_val, max_val = (0, 1)
        variable_info = []

        # Add a variable for each pixel.
        for channel in range(channels):
            for row in range(height):
                for col in range(width):
                    id = channel * height * width + row * width + col
                    var = {
                        "id": id,  # Feature index, 0 to pixel count - 1
                        "query": True,  # All features are query features.
                        "type": "continuous",  # Feature type is continuous.
                        "name": f"({channel:d}, {row:d}, {col:d})",  # Short variable description
                        "lower": min_val,  # Min pixel value
                        "upper": max_val,  # Max pixel value
                    }
                    variable_info.append(var)

        if use_targets:
            # Add a variable for the predicted class.
            id = channels * height * width
            predicted_class_var = {
                "id": id,  # Feature index, pixel count
                "query": False,  # Target feature, not query feature
                "type": "categorical",  # Feature type determined by binary flag.
                "name": "Class",  # Short variable description
                "lower": 0,  # Min class value is 0
                "upper": 9,  # Max class value is 1 if binary, otherwise 9
            }

            variable_info.append(predicted_class_var)

        return Variables.create_from_dict({"variables": variable_info, "metadata_variables": []})

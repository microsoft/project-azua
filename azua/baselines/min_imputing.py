from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from sklearn.impute import SimpleImputer

from ..datasets.dataset import Dataset
from ..datasets.variables import Variables
from .sk_learn_imputer import SKLearnImputer


class MinImputing(SKLearnImputer):
    def __init__(self, model_id: str, variables: Variables, save_dir: str, random_seed: int = 0):
        imputer = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)
        super().__init__(model_id, variables, save_dir, imputer)
        self._mins = np.array([var.lower for var in variables])
        self._mins = np.expand_dims(self._mins, axis=0)  # Shape (1, feature_count)

        maxs = np.array([var.upper for var in variables])
        maxs = np.expand_dims(maxs, axis=0)  # Shape (1, feature_count)
        self._ranges = np.subtract(maxs, self._mins)  # Shape (1, feature_count)

    @classmethod
    def name(cls) -> str:
        return "min_imputing"

    def run_train(
        self,
        dataset: Dataset,
        train_config_dict: Dict[str, Any] = {},
        report_progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> None:
        data, mask = dataset.train_data_and_mask
        data = self.fill_mask(data, mask)
        scaled_data = self._scale_data(data)
        self._imputer.fit(scaled_data)

    def impute(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        impute_config_dict: Optional[Dict[str, int]] = None,
        vamp_prior_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        average: bool = True,
    ) -> np.ndarray:

        data = self.fill_mask(data, mask)
        scaled_data = self._scale_data(data)
        scaled_imputed = self._imputer.transform(scaled_data)
        imputed = self._unscale_data(scaled_imputed)
        if not average:
            # Add extra dimension that would be used for sampling
            imputed = np.expand_dims(imputed, axis=0)
        return imputed

    def _scale_data(self, data):
        scaled_data = (data - self._mins) / self._ranges
        return scaled_data

    def _unscale_data(self, scaled_data):
        data = (scaled_data * self._ranges) + self._mins
        return data

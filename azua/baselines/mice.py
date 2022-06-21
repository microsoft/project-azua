from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

# Explicitly enable experimental IterativeImputer (new in scikit-learn 0.22.2)
from sklearn.experimental import enable_iterative_imputer  # noqa F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

from ..datasets.dataset import Dataset
from ..datasets.variables import Variables
from .sk_learn_imputer import SKLearnImputer


class MICE(SKLearnImputer):
    def __init__(
        self,
        model_id: str,
        variables: Variables,
        save_dir,
        max_iter=10,
        initial_strategy="mean",
        random_seed=0,
        test_sample_count=50,
    ):
        imputer = IterativeImputer(
            max_iter=max_iter,
            initial_strategy=initial_strategy,
            random_state=random_seed,
            estimator=BayesianRidge(),
            sample_posterior=True,
        )
        super().__init__(model_id, variables, save_dir, imputer)

        self._sample_count = test_sample_count

    @classmethod
    def name(cls) -> str:
        return "mice"

    def run_train(
        self,
        dataset: Dataset,
        train_config_dict: Optional[Dict[str, Any]] = None,
        report_progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> None:
        data, mask = dataset.train_data_and_mask
        data = self.fill_mask(data, mask)
        self._imputer.fit(data)

    def impute(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        impute_config_dict: Optional[Dict[str, int]] = None,
        vamp_prior_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        average: bool = True,
    ) -> np.ndarray:

        data = self.fill_mask(data, mask)

        row_count, feature_count = data.shape
        imputed = np.zeros((self._sample_count, row_count, feature_count))
        # Sample posterior N times.
        for sample_idx in range(self._sample_count):
            imputed[sample_idx, :, :] = self._imputer.transform(data)

        # Take average across sample dimension
        if average:
            imputed = imputed.mean(axis=0)  # Shape (row_count, feature_count)
        return imputed

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

# Explicitly enable experimental IterativeImputer (new in scikit-learn 0.22.2)
from sklearn.experimental import enable_iterative_imputer  # noqa F401
from sklearn.impute import IterativeImputer

from ..datasets.dataset import Dataset
from ..datasets.variables import Variables
from .sk_learn_imputer import SKLearnImputer


class MissForest(SKLearnImputer):
    def __init__(
        self, model_id: str, variables: Variables, save_dir, max_iter=10, initial_strategy="mean", random_seed=0
    ):
        imputer = IterativeImputer(
            max_iter=max_iter,
            initial_strategy=initial_strategy,
            random_state=random_seed,
            estimator=ExtraTreesRegressor(),
        )
        super().__init__(model_id, variables, save_dir, imputer)

    @classmethod
    def name(cls) -> str:
        return "missforest"

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

        imputed = self._imputer.transform(data)
        if not average:
            # Add extra dimension that would be used for sampling
            imputed = np.expand_dims(imputed, axis=0)

        return imputed

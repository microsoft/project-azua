from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from sklearn.impute import SimpleImputer

from ..datasets.dataset import Dataset
from ..datasets.variables import Variables
from .sk_learn_imputer import SKLearnImputer


class MajorityVote(SKLearnImputer):
    def __init__(self, model_id: str, variables: Variables, save_dir: str, random_seed: int = 0):
        imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
        super().__init__(model_id, variables, save_dir, imputer)

    @classmethod
    def name(cls) -> str:
        return "majority_vote"

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
        imputed = self._imputer.transform(data)
        if not average:
            # Add extra dimension that would be used for sampling
            imputed = np.expand_dims(imputed, axis=0)
        return imputed

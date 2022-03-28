from __future__ import annotations  # Needed to support method returning instance of its parent class until Python 3.10
from typing import Any, Dict, Union, TypeVar, Type, Optional, Callable


import numpy as np
import os

from ..models.model import Model
from ..models.imodel import IModelForCausalInference
from ..datasets.variables import Variables
from ..utils.io_utils import save_json
from ..datasets.dataset import Dataset


BASELINES = [
    "notears_linear",
    "notears_mlp",
    "notears_sob",
    "notears_grandag",
    "pc",
]

T = TypeVar("T", bound="CastleCausalLearner")


class CastleCausalLearner(Model, IModelForCausalInference):
    "TODO: deduplicate these file paths between DoWhy, CastleCausalLearner and SkLearnImputer"
    _model_config_path = "model_config.json"
    _model_type_path = "model_type.txt"
    _variables_path = "variables.json"
    _model_file = "model.pt"

    def __init__(self, model_id: str, variables: Variables, save_dir: str, causal_learner: T) -> None:
        super().__init__(model_id, variables, save_dir)
        self._causal_learner = causal_learner

    def get_adj_matrix(self, nsamples=1):
        return self._causal_learner.causal_matrix.astype(np.float64)

    def run_train(
        self,
        dataset: Dataset,
        train_config_dict: Dict[str, Any] = {},
        report_progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> None:
        data, _ = dataset.train_data_and_mask
        # This is needed by these methods
        # TODO #20930: Use baselines with double precision (nonlinear notears and grandag)
        # See https://msrcambridge.visualstudio.com/MinimumDataAI/_workitems/edit/20930
        data = data.copy().astype(np.float32)
        self._causal_learner.learn(data)

    @classmethod
    def create(
        cls: Type[T],
        model_id: str,
        save_dir: str,
        variables: Variables,
        model_config_dict: Dict[str, Any],
        device: Union[str, int],
    ) -> T:

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save model config to save dir.
        model_config_save_path = os.path.join(save_dir, cls._model_config_path)
        save_json(model_config_dict, model_config_save_path)

        return cls(model_id, variables, save_dir, **model_config_dict)

    @classmethod
    def load(cls: Type[T], model_id: str, save_dir: str, device: Union[str, int], **model_config_dict) -> T:
        raise NotImplementedError()

    def save(self) -> None:
        raise NotImplementedError()

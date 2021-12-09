from __future__ import annotations  # Needed to support method returning instance of its parent class until Python 3.10
from typing import Any, Dict, Union, TypeVar, Type

import numpy as np
import os

from ..models.model import Model
from ..models.imodel import IModelForImputation
from ..datasets.variables import Variables
from ..utils.io_utils import read_pickle, save_pickle, save_json

BASELINES = [
    "mean_imputing",
    "majority_vote",
    "zero_imputing",
    "min_imputing",
    "mice",
    "missforest",
    "pvae",
    "bayesian_pvae",
    "deep_matrix_factorization",
    "graph_neural_network",
]

T = TypeVar("T", bound="SKLearnImputer")


class SKLearnImputer(Model, IModelForImputation):
    "TODO: deduplicate these file paths between DoWhy, CastleCausalLearner and SkLearnImputer"
    _model_config_path = "model_config.json"
    _model_type_path = "model_type.txt"
    _variables_path = "variables.json"
    _model_file = "model.pt"

    def __init__(self, model_id: str, variables: Variables, save_dir: str, imputer) -> None:
        super().__init__(model_id, variables, save_dir)
        self._imputer = imputer

    @staticmethod
    def fill_mask(data, mask, fill_value=np.nan):
        # Fill values this way rather than e.g. data * mask + fill_value * ~mask to handle case of sparse mask, where
        # * is matrix multiplication rather than elementwise multiplication.
        data_masked = np.zeros_like(data)
        data_masked[mask.nonzero()] = data[mask.nonzero()]
        data_masked[(~mask).nonzero()] = fill_value
        return data_masked

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
        variables_path = os.path.join(save_dir, cls._variables_path)
        variables = Variables.create_from_json(variables_path)

        model = cls(model_id, variables, save_dir, **model_config_dict)
        params = read_pickle(os.path.join(save_dir, "parameters.pkl"))
        model._imputer.set_params(**params)

        return model

    def save(self) -> None:
        self.variables.save(os.path.join(self.save_dir, self._variables_path))
        params = self._imputer.get_params()
        save_pickle(params, os.path.join(self.save_dir, "parameters.pkl"))

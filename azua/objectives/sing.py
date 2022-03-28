import numpy as np

from ..datasets.variables import Variables
from ..models.imodel import IModelForObjective
from ..objectives.eddi_base import EDDIBaseObjective
from typing import List


class SINGObjective(EDDIBaseObjective):
    """"
    SING objective.
    """

    def __init__(self, model: IModelForObjective, sample_count: int, use_vamp_prior: bool = False, **kwargs):
        """
        Args:
            model (Model): Trained `Model` class to use.
            sample_count (int): Number of imputation samples to use.
            use_vamp_prior (bool): Whether or not to use vamp prior method.
        """
        super().__init__(model, sample_count, use_vamp_prior, **kwargs)
        # TODO: This empty data may cause data out-of-range warnings when it is checked against the variables metadata.
        # Expand create_empty_data to create values in range
        empty_data = Variables.create_empty_data(model.variables)

        # Assume that all data can be queried and nothing is currently observed
        data_mask = np.ones_like(empty_data, dtype=bool)
        obs_mask = np.zeros_like(empty_data, dtype=bool)
        self._info_array = self.get_information_gain(empty_data, data_mask, obs_mask)[0]
        # Sort the info gain by descending value.
        self._info_gain_idxs_sorted = np.argsort(-self._info_array).tolist()

    @classmethod
    def name(cls):
        return "sing"

    def get_next_questions(self, _, data_mask: np.ndarray, obs_mask: np.ndarray, question_count=1, as_array=False):  # type: ignore[override]
        # TODO: this can probably be optimised.

        data_mask_row = data_mask[0, :]
        obs_mask_row = obs_mask[0, :]
        observable_groups = self._model.variables.get_observable_groups(np.ones_like(data_mask_row), obs_mask_row)
        info_gain_list_copy = self._info_gain_idxs_sorted.copy()
        next_qs: List[int] = []
        while len(next_qs) < question_count and info_gain_list_copy:
            next_q_id = info_gain_list_copy.pop(0)
            if next_q_id in observable_groups:
                next_qs.append(next_q_id)
        next_question_idxs = [next_qs] * data_mask.shape[0]

        if as_array:
            rewards = self._info_array
        else:
            rewards = {idx: float(val) for idx, val in enumerate(self._info_array)}

        return next_question_idxs, rewards

import numpy as np

from ..objectives.eddi_base import EDDIBaseObjective
from ..utils.data_mask_utils import argsort_rows_exclude_nan


class EDDIObjective(EDDIBaseObjective):
    """
    EDDI objective.
    """

    @classmethod
    def name(cls):
        return "eddi"

    def get_next_questions(self, data, data_mask, obs_mask, question_count=1, as_array=False, **model_kwargs):
        rewards = self.get_information_gain(data, data_mask, obs_mask, **model_kwargs)
        next_question_idxs = argsort_rows_exclude_nan(rewards, ascending=False, max_qs_per_row=question_count)

        if not as_array:
            rewards = [{idx: float(val) for idx, val in enumerate(row) if not np.isnan(val)} for row in rewards]

        return next_question_idxs, rewards

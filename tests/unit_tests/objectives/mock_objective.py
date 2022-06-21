import numpy as np

from azua.objectives.objective import Objective
from azua.utils.data_mask_utils import argsort_rows_exclude_nan


class MockObjective(Objective):
    @classmethod
    def name(cls):
        """
        Name of the objective implemented in abstract class.
        """
        return "MockObjective"

    def get_next_questions(
        self,
        data: np.ndarray,
        data_mask: np.ndarray,
        obs_mask: np.ndarray,
        question_count: int = 1,
        as_array: bool = False,
    ):
        rewards = np.arange(len(self._model.variables.group_idxs), dtype=float)
        rewards = np.repeat(rewards[np.newaxis, :], data.shape[0], axis=0)
        next_question_idxs = argsort_rows_exclude_nan(rewards, ascending=False, max_qs_per_row=question_count)

        if as_array:
            return next_question_idxs, rewards
        else:
            rewards_return = [{idx: float(val) for idx, val in enumerate(row) if not np.isnan(val)} for row in rewards]
            return next_question_idxs, rewards_return

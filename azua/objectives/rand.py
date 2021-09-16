import numpy as np

from ..objectives.objective import Objective

from ..models.imodel import IModelForObjective


class RandomObjective(Objective):
    """"
    Random objective.
    """

    def __init__(self, model: IModelForObjective, **kwargs):
        super().__init__(model)

    @classmethod
    def name(cls):
        return "rand"

    def get_next_questions(self, data: np.ndarray, data_mask: np.ndarray, obs_mask, question_count=1):
        # For each row, pick up to N random query groups that are unobserved.
        # Map them back to a query group index.
        next_question_idxs = []
        for data_mask_row, obs_mask_row in zip(data_mask, obs_mask):
            observable_groups = self._model.variables.get_observable_groups(data_mask_row, obs_mask_row)
            if len(observable_groups) < question_count:
                question_idxs = observable_groups
            else:
                question_idxs = np.random.choice(observable_groups, question_count, replace=False).tolist()

            next_question_idxs.append(question_idxs)
        return next_question_idxs, None

from abc import ABC, abstractmethod

import numpy as np

from ..models.imodel import IModelForObjective


class Objective(ABC):
    """
    Abstract objective class.

    To instantiate this class, these functions need to be implemented:
    get_next_questions: Returns next best feature id to query.
    name: Name of objective, to use when finding objective to use from string.

    In order for the factory methods to work, this class needs to be placed in the `objectives` directory.

    """

    # TODO: This is currently defined in objective and in PVAEBaseModel, we should think where it should be sitting
    _vamp_prior_info_gain_path = "vamp_prior_info_gain.json"

    def __init__(self, model: IModelForObjective):
        self._model = model

    @classmethod
    @abstractmethod
    def name(cls):
        """
        Name of the objective implemented in abstract class.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_next_questions(
        self,
        data: np.ndarray,
        data_mask: np.ndarray,
        obs_mask: np.ndarray,
        question_count: int = 1,
    ):
        """
        Get the id of the next best question(s) to query, using a trained model.
        Data should be provided in unprocessed form.

        Args:
            data (numpy array of shape (batch_size, input_dim)): partially observed data in processed form.
            data_mask (numpy array of shape (batch_size, input_dim)): Contains mask where 1 is observed in the
                underlying data, 0 is missing.
            obs_mask (numpy array of shape (batch_size, input_dim)): Contains mask where 1 indicates a feature that has
                been observed before or during active learning, 0 a feature that has not yet been observed and could be
                queried (if the value also exists in the underlying dataset).
            question_count (int): The maximum number of next questions to return for each row. Defaults to 1.

        Returns:
            next_questions (list of list of variable ids): A list of length (batch_size), each
                containing a list of length (question_count) of the next best questions to query, for the
                given objective.
            other_info: Any other information useful. e.g. for the EDDIObjective, this will return a
                list of length (batch_size) of the information rewards for each row. Returns None if no
                extra information is needed.
        """
        raise NotImplementedError()

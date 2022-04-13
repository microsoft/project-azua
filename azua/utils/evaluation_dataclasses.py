from dataclasses import dataclass

import numpy as np


@dataclass
class IteEvaluationResults:
    """ Dataclass to hold individual treatment effect (ITE) evaluation results """
    average_ite_rmse: float
    per_intervention_average_ite_rmse: np.ndarray

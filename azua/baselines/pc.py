import numpy as np
from ..baselines.castle_causal_learner import CastleCausalLearner
from ..datasets.variables import Variables
from ..utils.causality_utils import cpdag2dags
from castle.algorithms import PC as PC_alg


class PC(CastleCausalLearner):
    """
    Child class of CastleCausalLearner which specifies PC_alg should be used for discovery
    """

    def __init__(self, model_id: str, variables: Variables, save_dir: str, random_seed: int = 0):
        super().__init__(model_id, variables, save_dir, PC_alg())

    def get_adj_matrix(self, samples=100, round=None, squeeze=False, **kwargs):
        """
        Draws a series of DAG samples from markov equivalence class. If enough samples are specified, all the DAGs in the equivalence class will be returned.
        """
        graph_samples = cpdag2dags(self._causal_learner.causal_matrix.astype(np.float64), samples=samples)

        if samples == 1 and squeeze:
            return graph_samples[0]
        else:
            return graph_samples

    @classmethod
    def name(cls) -> str:
        return "pc"

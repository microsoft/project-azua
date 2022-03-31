from ..baselines.castle_causal_learner import CastleCausalLearner
from ..datasets.variables import Variables

from castle.algorithms import GraNDAG as GraNDAG_alg


class GraNDAG(CastleCausalLearner):
    def __init__(self, model_id: str, variables: Variables, save_dir: str, random_seed: int = 0, iterations: int = 10000):
        causal_learner = GraNDAG_alg(variables.num_processed_cols, iterations=iterations)
        super().__init__(model_id, variables, save_dir, causal_learner)

    @classmethod
    def name(cls) -> str:
        return "grandag"

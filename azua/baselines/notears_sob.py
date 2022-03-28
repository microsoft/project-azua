from ..baselines.castle_causal_learner import CastleCausalLearner
from ..datasets.variables import Variables

from castle.algorithms import NotearsSob as NotearsSob_alg


class NotearsSob(CastleCausalLearner):
    def __init__(self, model_id: str, variables: Variables, save_dir: str, random_seed: int = 0):
        super().__init__(model_id, variables, save_dir, NotearsSob_alg())

    @classmethod
    def name(cls) -> str:
        return "notears_sob"

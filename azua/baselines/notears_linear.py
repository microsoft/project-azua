from ..baselines.castle_causal_learner import CastleCausalLearner
from ..datasets.variables import Variables

from castle.algorithms import Notears


class NotearsLinear(CastleCausalLearner):
    def __init__(self, model_id: str, variables: Variables, save_dir: str, random_seed: int = 0):
        super().__init__(model_id, variables, save_dir, Notears())

    @classmethod
    def name(cls) -> str:
        return "notears_linear"

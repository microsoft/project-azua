from dataclasses import dataclass, fields
from typing import NamedTuple, Optional
import torch


class LossConfig(NamedTuple):
    # TODO add KL coeff, categorical likelihood coeff, and IWAE vs. ELBO. Remove them from model config.
    max_p_train_dropout: Optional[float] = None
    score_reconstruction: Optional[bool] = None
    score_imputation: Optional[bool] = None


@dataclass(frozen=True)
class LossResults:
    loss: torch.Tensor
    mask_sum: torch.Tensor  # TODO: consider using int as type here


@dataclass(frozen=True)
class VAELossResults(LossResults):
    kl: torch.Tensor  # KL divergence
    nll: torch.Tensor  # negative log-likelihood


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    mask_sum: float
    inner_epoch_time: float

    def __add__(self, other):
        return type(self)(*(getattr(self, k.name) + getattr(other, k.name) for k in fields(self)))


@dataclass(frozen=True)
class VAEEpochMetrics(EpochMetrics):
    kl: float
    nll: float

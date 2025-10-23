from logging import getLogger
from typing import Literal, NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor

from trm.model.trm import TRMTrainPrediction

logger = getLogger(__name__)


class TRMLoss(NamedTuple):
    token: Tensor
    halt: Tensor
    halt_weight: float = 1.0

    @property
    def total(self) -> Tensor:
        return self.token + self.halt_weight * self.halt

    def reduce(self, reduction: Literal["mean", "sum"] = "mean") -> Tensor:
        match reduction:
            case "mean":
                return self.total.mean()
            case "sum":
                return self.total.sum()
            case _:
                msg = f"Unknown reduction: {reduction}"
                raise ValueError(msg)


def trm_step_loss(
    preds: TRMTrainPrediction, targets: Tensor, halt_weight: float = 1.0
) -> TRMLoss:
    token_loss = F.cross_entropy(
        preds.logits.flatten(0, 1), targets.flatten(), reduction="none"
    ).view(targets.size(0), targets.size(1))

    with torch.no_grad():
        should_halt = (preds.logits.argmax(-1) == targets).all(-1).float()
    halt_loss = F.binary_cross_entropy_with_logits(
        preds.halt_logits, should_halt, reduction="none"
    )

    return TRMLoss(token=token_loss.mean(1), halt=halt_loss, halt_weight=halt_weight)

from logging import getLogger

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from trm.model.trm import TinyRecursiveModel, TRMState, trm_step_loss

__all__ = ["Trainer"]


logger = getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: TinyRecursiveModel,
        opt: torch.optim.Optimizer,
        sched: torch.optim.lr_scheduler.LRScheduler,
        ema_model: torch.optim.swa_utils.AveragedModel | None = None,
    ) -> None:
        self.model = model
        self.opt = opt
        self.sched = sched
        self.ema_model = ema_model

    def train_one_recursive_step(
        self,
        inputs: Tensor,
        targets: Tensor,
        reasoning_steps: int,
        halt_logit_thresh: float,
    ) -> float:
        total_loss = 0.0
        state: TRMState | None = None
        for step in range(1, reasoning_steps + 1):
            logger.debug("Reasoning Step %d forward pass", step)
            preds = self.model(inputs, state)
            loss = trm_step_loss(preds, targets)

            logger.debug("Backward pass")
            reduced_loss = loss.reduce()
            reduced_loss.backward()
            py_reduced_loss = reduced_loss.item()
            total_loss += py_reduced_loss

            logger.debug("Update step (Loss: %10.04e)", py_reduced_loss)
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            self.sched.step()

            if self.ema_model is not None:
                logger.debug("EMA update")
                self.ema_model.update_parameters(self.model)

            logger.debug("Halt check")
            state = preds.state
            halt_mask = preds.halt_logits >= halt_logit_thresh
            if halt_mask.any():
                logger.debug("%d elements halted", int(halt_mask.sum().item()))
                keep_mask = ~halt_mask
                inputs = inputs[keep_mask]
                targets = targets[keep_mask]
                state = state.masked(keep_mask)

            if halt_mask.all() or inputs.size(0) == 0:
                # Early-exit if all samples have halted
                logger.debug("All elements halted, exiting reasoning loop")
                break
        else:
            logger.debug("Reached maximum reasoning steps")

        return total_loss

    def train_one_epoch(
        self,
        dl: DataLoader,
        reasoning_steps: int = 12,
        halt_logit_thresh: float = 0.0,
    ) -> float:
        logger.info(
            "Epoch start. Reasoning Steps: %d, Halt Threshold: %.01f",
            reasoning_steps,
            halt_logit_thresh,
        )
        self.model.train()

        epoch_total_loss = 0.0
        for inputs, targets in dl:
            epoch_total_loss += self.train_one_recursive_step(
                inputs,
                targets,
                reasoning_steps=reasoning_steps,
                halt_logit_thresh=halt_logit_thresh,
            )
        avg_loss = epoch_total_loss / len(dl)

        logger.info("Epoch end. Average Loss: %10.04e", avg_loss)
        return avg_loss

    def train(
        self,
        dl: DataLoader,
        num_epochs: int,
        reasoning_steps: int = 12,
        halt_logit_thresh: float = 0.0,
    ) -> None:
        logger.info(
            "Beginning training. Epochs: %d, Reasoning Steps: %d, Halt Threshold: %.01f",
            num_epochs,
            reasoning_steps,
            halt_logit_thresh,
        )
        for _epoch in range(num_epochs):
            _avg_loss = self.train_one_epoch(
                dl,
                reasoning_steps=reasoning_steps,
                halt_logit_thresh=halt_logit_thresh,
            )
        logger.info("Training complete.")

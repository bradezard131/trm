from logging import getLogger
from typing import Final

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from trm.loss import TRMLoss, trm_step_loss
from trm.model.trm import TinyRecursiveModel, TRMState

__all__ = ["Trainer"]


logger = getLogger(__name__)
CPU_DEVICE: Final[torch.device] = torch.device("cpu")


class Trainer:
    def __init__(
        self,
        model: TinyRecursiveModel,
        opt: torch.optim.Optimizer,
        sched: torch.optim.lr_scheduler.LRScheduler,
        ema_model: torch.optim.swa_utils.AveragedModel | None = None,
        device: torch.device = CPU_DEVICE,
    ) -> None:
        self.model = model
        self.opt = opt
        self.sched = sched
        self.ema_model = ema_model
        self.device = device  # pyrefly: ignore[read-only]

    def _get_exploration_steps(
        self, inputs: Tensor, reasoning_steps: int, exploration_rate: float = 0.1
    ) -> Tensor:
        """Force some percentage of inputs to explore a minimum number of steps.
        To help prevent early convergence to quick halting."""
        explore_prob = torch.rand(inputs.size(0), device=inputs.device)
        explore_steps = torch.randint(
            1, reasoning_steps + 1, (inputs.size(0),), device=inputs.device
        )
        explore_steps[explore_prob > exploration_rate] = 0
        return explore_steps

    def _backwards(self, loss: TRMLoss) -> float:
        logger.debug("Backward pass")
        reduced_loss = loss.reduce()
        reduced_loss.backward()
        loss_item = reduced_loss.item()
        logger.debug("Loss: %10.04e", loss_item)
        return loss_item

    def _update(self) -> None:
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        self.sched.step()
        if self.ema_model is not None:
            logger.debug("EMA update")
            self.ema_model.update_parameters(self.model)

    def train_one_recursive_step(
        self,
        inputs: Tensor,
        targets: Tensor,
        reasoning_steps: int,
        halt_logit_thresh: float,
    ) -> float:
        total_loss = 0.0
        state: TRMState | None = None
        explore_steps = self._get_exploration_steps(inputs, reasoning_steps)
        for step in range(1, reasoning_steps + 1):
            logger.debug("Reasoning Step %d forward pass", step)
            preds = self.model(inputs, state)
            loss = trm_step_loss(preds, targets)

            total_loss += self._backwards(loss)
            self._update()
            state = preds.state

            logger.debug("Halt check")
            halt_mask = torch.logical_and(
                preds.halt_logits >= halt_logit_thresh,
                explore_steps > step,
            )
            if halt_mask.any():
                logger.debug("%d elements halted", int(halt_mask.sum().item()))
                keep_mask = ~halt_mask
                inputs = inputs[keep_mask]
                targets = targets[keep_mask]
                explore_steps = explore_steps[keep_mask]
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
        for inputs, targets, _difficulties in dl:
            epoch_total_loss += self.train_one_recursive_step(
                inputs.to(self.device, non_blocking=True),
                targets.to(self.device, non_blocking=True),
                reasoning_steps=reasoning_steps,
                halt_logit_thresh=halt_logit_thresh,
            )
        avg_loss = epoch_total_loss / len(dl)

        logger.info("Epoch end. Average Loss: %10.04e", avg_loss)
        return avg_loss

    def _get_eval_model(self) -> TinyRecursiveModel:
        if self.ema_model is None:
            return self.model.eval()
        assert isinstance(self.ema_model.module, TinyRecursiveModel)
        return self.ema_model.module.eval()

    def val_one_epoch(self, dl: DataLoader) -> None:
        model = self._get_eval_model()

        cell_count, cell_correct = 0, 0
        puzzle_count, puzzle_correct = 0, 0
        missing_correct = 0
        with torch.inference_mode():
            for inputs_cpu, targets_cpu, _difficulties in dl:
                inputs = inputs_cpu.to(self.device, non_blocking=True)
                mask = inputs == 0
                pred = model.predict(inputs).predictions
                targets = targets_cpu.to(self.device, non_blocking=True)
                cell_count += mask.sum().item()
                puzzle_count += inputs.size(0)

                correct = pred == targets
                cell_correct += correct[mask].sum().item()
                puzzle_correct += correct.all(-1).sum().item()
                missing_correct += (correct | ~mask).all(-1).sum().item()

        cell_acc = cell_correct / cell_count * 100
        puzzle_acc = puzzle_correct / puzzle_count * 100
        missing_acc = missing_correct / puzzle_count * 100
        logger.info(
            "Validation epoch end. "
            "Average Cell Acc: %5.02f%%, "
            "Puzzle Acc: %5.02f%%, "
            "Missing Only Acc: %5.02f%%",
            cell_acc,
            puzzle_acc,
            missing_acc,
        )

    def train(
        self,
        dl: DataLoader,
        num_epochs: int,
        reasoning_steps: int = 12,
        halt_logit_thresh: float = 0.0,
        val_dl: DataLoader | None = None,
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
            if val_dl is not None:
                self.val_one_epoch(val_dl)

        logger.info("Training complete.")

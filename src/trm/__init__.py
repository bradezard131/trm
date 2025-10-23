import os
from concurrent.futures import ProcessPoolExecutor
from logging import basicConfig, getLogger
from math import ceil

import torch
from torch.utils.data import DataLoader, random_split

from trm.data import make_dummy_sudoku_dataset, make_sudoku_dataset
from trm.engine.train import Trainer
from trm.loss import trm_step_loss
from trm.model import make_mlp_tiny_recursive_model

logger = getLogger(__name__)


def rescale_learning_rate(lr: float, recurrent_steps: int) -> float:
    return lr / recurrent_steps


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.xpu.is_available():
        return torch.device("xpu")
    logger.warning("No accelerator is available, using SLOW cpu backend")
    return torch.device("cpu")


def main(  # noqa: PLR0913
    *,
    trm_dim: int = 384,
    mlp_depth: int = 2,
    registers: int = 0,
    refinement_iters: int = 3,
    latent_refinement_iters: int = 6,
    learning_rate: float = 3.33e-5,
    warmup_steps: int = 100,
    batch_size: int = 128,
    recurrent_steps: int = 12,
    epochs: int = 100,
    rng_seed: int = 42,
) -> None:
    device = _get_device()
    basicConfig(level=os.environ.get("TRM_LOG_LEVEL", "INFO"))

    with ProcessPoolExecutor() as tpx:
        train_dataset = make_sudoku_dataset(
            num_puzzles=128 * 25,
            difficulty=32,
            rng=rng_seed,
            executor=tpx,
            mode="train",
        )
        val_dataset = make_sudoku_dataset(
            num_puzzles=128 * 5,
            difficulty=32,
            rng=rng_seed + 1,
            executor=tpx,
            mode="val",
        )
    logger.info(
        "Dataset built. Train size: %d, Val size: %d",
        len(train_dataset),
        len(val_dataset),
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    warmup_epochs = ceil(warmup_steps / len(train_dataloader))
    rest_epochs = epochs - warmup_epochs

    logger.info("Initialising model...")
    model = make_mlp_tiny_recursive_model(
        dim=trm_dim,
        num_tokens=10,  # 1-9 for Sudoku, 0 indicates empty cell
        max_seq_len=81,  # 9x9 Sudoku grid
        inner_depth=mlp_depth,
        num_register_tokens=registers,
        refinement_iters=refinement_iters,
        latent_refinement_iters=latent_refinement_iters,
    ).to(device)

    logger.info("Setting up optimizer and scheduler...")
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=rescale_learning_rate(learning_rate, recurrent_steps),
        weight_decay=1.0,
        betas=(0.9, 0.95),
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: min((step + 1) / warmup_steps, 1.0)
    )

    logger.info("Initializing trainer...")
    trainer = Trainer(model, opt, sched, None, device)
    logger.info("Starting training loop...")
    trainer.train(train_dataloader, epochs, recurrent_steps, val_dl=val_dataloader)
    trainer.train(train_dataloader, rest_epochs, recurrent_steps, val_dl=val_dataloader)
    logger.info("Training complete!")


def overfit() -> None:
    device = _get_device()
    basicConfig(level=os.environ.get("TRM_LOG_LEVEL", "INFO"))

    model = make_mlp_tiny_recursive_model(
        dim=512,
        num_tokens=10,  # 1-9 for Sudoku, 0 indicates empty cell
        max_seq_len=81,  # 9x9 Sudoku grid
        inner_depth=2,
        num_register_tokens=0,
        refinement_iters=3,
        latent_refinement_iters=6,
    ).to(device)

    adjusted_lr = rescale_learning_rate(1e-4, 12)
    logger.info("LR: 1e-4 -> %10.04e (12 recurrent steps)", adjusted_lr)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=adjusted_lr,
        weight_decay=1.0,
        betas=(0.9, 0.95),
    )
    input()

    logger.info("Building dataset...")
    dataset = make_dummy_sudoku_dataset(num_puzzles=1, difficulty=4, rng=42)
    element = dataset[0]

    inputs = torch.from_numpy(element.puzzle).to(dtype=torch.long, device=device)
    targets = torch.from_numpy(element.answer).to(dtype=torch.long, device=device)
    inputs, targets = inputs.unsqueeze(0), targets.unsqueeze(0)
    for outer_step in range(100):
        state = None
        for inner_step in range(1, 13):
            preds = model(inputs, state)
            loss = trm_step_loss(preds, targets)

            reduced_loss = loss.reduce()
            reduced_loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

            state = preds.state
            logger.info(
                "%d.%d: Loss: %10.04e", outer_step, inner_step, reduced_loss.item()
            )

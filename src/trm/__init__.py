from logging import basicConfig, getLogger

import torch
from torch.utils.data import DataLoader, Dataset

from trm.engine.train import Trainer
from trm.model.mixer import Mixer1D
from trm.model.trm import TinyRecursiveModel, TRMEmbeddings

logger = getLogger(__name__)


class MockDataset(Dataset):
    def __len__(self) -> int:
        return 16

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.randint(0, 256, (256,)),
            torch.randint(0, 256, (256,)),
        )


def rescale_learning_rate(lr: float, batch_size: int, recurrent_steps: int) -> float:
    return lr / (batch_size * recurrent_steps)


def main() -> None:
    basicConfig(level="INFO")
    logger.info("Building dataset, model, optimizer, and trainer...")
    dataset = MockDataset()
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    model = TinyRecursiveModel(
        embeddings=TRMEmbeddings(
            dim=16,
            num_tokens=256,
        ),
        inner_model=Mixer1D(
            dim=16,
            tokens=256,
            depth=2,
        ),
    )
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=rescale_learning_rate(1e-4, batch_size=8, recurrent_steps=12),
        weight_decay=1.0,
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: min((step + 1) / 1000, 1.0)
    )

    logger.info("Initializing trainer...")
    trainer = Trainer(model, opt, sched, None)
    logger.info("Starting training loop...")
    trainer.train(dataloader, num_epochs=10, reasoning_steps=12)
    logger.info("Training complete!")

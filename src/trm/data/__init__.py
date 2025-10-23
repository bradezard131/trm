from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from concurrent.futures import Executor

    import numpy as np

from trm.data import sudoku, sudoku_dataset

__all__ = ["make_sudoku_dataset", "sudoku", "sudoku_dataset"]


logger = getLogger(__name__)


def make_sudoku_dataset(
    num_puzzles: int,
    difficulty: int | tuple[int, int],
    rng: np.random.Generator | int | None = None,
    executor: Executor | None = None,
    *,
    mode: Literal["train", "val"] = "val",
) -> sudoku_dataset.SudokuDataset:
    answers = sudoku_dataset.generate_answers(num_puzzles, rng=rng, executor=executor)
    puzzle_generator = sudoku_dataset.SudokuPuzzleGenerator(
        difficulty=difficulty, rng=rng
    )
    ds = sudoku_dataset.SudokuDataset(
        answers=answers,
        puzzle_generator=puzzle_generator,
        rng=(rng if mode == "train" else None),
    )
    logger.debug("Example dataset answer: %s", str(ds[0].answer))
    logger.debug("Example dataset problem: %s", str(ds[0].puzzle))
    return ds


def make_dummy_sudoku_dataset(
    num_puzzles: int,
    difficulty: int | tuple[int, int],
    rng: np.random.Generator | int | None = None,
    executor: Executor | None = None,
) -> sudoku_dataset.SudokuDataset:
    answers = sudoku_dataset.generate_answers(1, rng, executor=executor)
    answers = answers * num_puzzles
    puzzle_generator = sudoku_dataset.SudokuPuzzleGenerator(
        difficulty=difficulty, rng=rng
    )
    return sudoku_dataset.SudokuDataset(
        answers=answers, puzzle_generator=puzzle_generator
    )

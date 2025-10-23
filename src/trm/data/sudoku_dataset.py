from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import numpy.typing as npt
from torch.utils.data import Dataset

from trm.data.sudoku import generate_sudoku, remove_numbers_inplace

if TYPE_CHECKING:
    from concurrent.futures.thread import Executor


__all__ = [
    "SudokuDataset",
    "SudokuPuzzleGenerator",
    "SudokuSample",
    "generate_answers",
]


def _split_rng_seeds(rng: np.random.Generator, count: int) -> tuple[int, ...]:
    return tuple(rng.integers(0, 2**32 - 1, size=count).tolist())


def generate_answers(
    count: int,
    rng: np.random.Generator | int | None = None,
    *,
    executor: Executor | None = None,
) -> tuple[npt.NDArray[np.int8], ...]:
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    seeds = _split_rng_seeds(rng, count)
    if executor is None:
        return tuple(map(generate_sudoku, seeds))
    return tuple(executor.map(generate_sudoku, seeds))


class SudokuSample(NamedTuple):
    puzzle: npt.NDArray[np.int64]
    answer: npt.NDArray[np.int64]
    difficulty: int


class SudokuPuzzleGenerator:
    def __init__(
        self,
        difficulty: int | tuple[int, int],
        rng: np.random.Generator | int | None = None,
    ) -> None:
        super().__init__()
        self.difficulties = (
            (difficulty,)
            if isinstance(difficulty, int)
            else tuple(range(difficulty[0], difficulty[1]))
        )
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)
        self.rng = rng

    def __call__(self, answer: np.ndarray) -> SudokuSample:
        difficulty = self.rng.choice(self.difficulties)
        answer = generate_sudoku(rng=self.rng).astype(np.int64)
        puzzle = answer.copy()
        remove_numbers_inplace(puzzle, difficulty, self.rng)
        return SudokuSample(
            puzzle=puzzle.flatten(), answer=answer.flatten(), difficulty=difficulty
        )


class SudokuDataset(Dataset[SudokuSample]):
    def __init__(
        self,
        answers: tuple[npt.NDArray[np.int8], ...],
        puzzle_generator: SudokuPuzzleGenerator,
    ) -> None:
        if not all(p.shape == (9, 9) for p in answers):
            msg = "All puzzles must have shape (9, 9)."
            raise ValueError(msg)

        self.answers = answers
        self.puzzle_generator = puzzle_generator

    def __len__(self) -> int:
        return len(self.answers)

    def __getitem__(self, index: int) -> SudokuSample:
        return self.puzzle_generator(self.answers[index])

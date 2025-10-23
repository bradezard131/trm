import numpy as np
import numpy.typing as npt


def _shuffle_inner_columns_inplace(grid: npt.NDArray, rng: np.random.Generator) -> None:
    """Shuffle the inner columns of each 3-column block in the Sudoku grid."""
    for block_col in range(3):
        cols = [block_col * 3 + i for i in range(3)]
        shuffled_cols = rng.permutation(cols)
        grid[..., cols] = grid[..., shuffled_cols]


def _shuffle_inner_rows_inplace(grid: npt.NDArray, rng: np.random.Generator) -> None:
    """Shuffle the inner rows of each 3-row block in the Sudoku grid."""
    for block_row in range(3):
        rows = [block_row * 3 + i for i in range(3)]
        shuffled_rows = rng.permutation(rows)
        grid[..., rows, :] = grid[..., shuffled_rows, :]


def augment_sudoku_batch_inplace(
    puzzles: npt.NDArray, rng: np.random.Generator
) -> None:
    """Apply random transformations to a batch of Sudoku puzzles."""
    _shuffle_inner_columns_inplace(puzzles, rng)
    _shuffle_inner_rows_inplace(puzzles, rng)

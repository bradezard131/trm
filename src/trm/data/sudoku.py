import numpy as np
import numpy.typing as npt

__all__ = [
    "generate_sudoku",
    "remove_numbers_inplace",
]


def generate_sudoku(
    rng: np.random.Generator | int | None = None,
    max_attempts: int = 100,
) -> npt.NDArray[np.int8]:
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    for _attempt in range(max_attempts):
        grid = np.zeros((9, 9), dtype=np.int8)
        if _fill_grid(grid, rng):
            return grid

    msg = f"Failed to generate Sudoku puzzle after {max_attempts} attempts."
    raise RuntimeError(msg)


def _fill_grid(grid: npt.NDArray[np.int8], rng: np.random.Generator) -> bool:
    empty = _find_empty(grid)
    if empty is None:
        return True

    row, col = empty
    numbers = rng.permutation(9) + 1  # Shuffle 1-9

    for num in numbers:
        if _is_valid(grid, row, col, num):
            grid[row, col] = num
            if _fill_grid(grid, rng):
                return True
            grid[row, col] = 0

    return False


def _find_empty(grid: npt.NDArray[np.int8]) -> tuple[int, int] | None:
    """Find the first empty cell (value = 0)."""
    zeros = np.argwhere(grid == 0)
    return tuple(zeros[0]) if len(zeros) > 0 else None  # pyrefly: ignore[bad-return]


def _is_valid(grid: npt.NDArray[np.int8], row: int, col: int, num: int) -> bool:
    if num in grid[row, :] or num in grid[:, col]:
        return False

    box_row, box_col = 3 * (row // 3), 3 * (col // 3)
    return num not in grid[box_row : box_row + 3, box_col : box_col + 3]


def remove_numbers_inplace(
    grid: npt.NDArray[np.int8], count: int, rng: np.random.Generator
) -> None:
    positions = rng.choice(81, count, replace=False)

    for pos in positions:
        row, col = divmod(pos, 9)
        grid[row, col] = 0

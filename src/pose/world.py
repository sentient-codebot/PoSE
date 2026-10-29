"""Model world for the data-scaling study.

A teacher prior trained on all real days is the true distribution. Draws
cover several virtual copies of the 2024 calendar. The first copy holds every
split (training, validation, test); later copies add training snapshots
only. Training sets of different sizes are masks over the same draws, so the
validation and test snapshots, and with them the paired observations, are
identical for every size.
"""

import numpy as np

from .data import SPLITS, calendar_arrays


def world_calendar(years: int) -> dict[str, np.ndarray]:
    """The 2024 calendar repeated ``years`` times, with a ``year`` array.
    day_of_year stays 0-365, so it identifies a test day within year 0."""
    cal = calendar_arrays()
    out = {k: np.tile(v, years) for k, v in cal.items()}
    out["year"] = np.repeat(np.arange(years), len(cal["split"]))
    return out


def world_valid_mask(
    cal: dict[str, np.ndarray],
    years_used: int,
    train_fraction: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Year 0: every validation and test snapshot, and a random
    ``train_fraction`` of its training snapshots; years 1 .. years_used - 1:
    training snapshots only; later years: none."""
    year, split = cal["year"], cal["split"]
    train = split == SPLITS["train"]
    valid = (year == 0) & ~train
    year0_train = np.flatnonzero((year == 0) & train)
    keep = np.random.default_rng(seed).permutation(year0_train)
    keep = keep[: round(train_fraction * len(year0_train))]
    valid[keep] = True
    valid |= train & (year >= 1) & (year < years_used)
    return valid

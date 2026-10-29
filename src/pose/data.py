"""Snapshot data set for DSSE on IEEE 33-bus: injections, context and split.

A snapshot is one 15-minute interval of 2024 on the fixed UTC+1 clock of the
Fluvius preprocessing (CET without daylight saving). ``p_kw[t, i]`` is the
net active power drawn at node i (positive = consumption), the scaled sum of
the homes assigned to that node (scripts/data/assemble_fluvius.py).

Context c = (month 0-11, quarter-hour of day 0-95, day type 0 weekday /
1 weekend). Split by day of month: days 8-14 test, days 22-28 validation,
all other days training.
"""

import datetime as dt
import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np

from .feeder import NUM_NODES, P_NOMINAL_KW, PV_HEAVY_BUSES_BASE, node_of_bus


YEAR = 2024
STEPS_PER_DAY = 96
NUM_STEPS = 366 * STEPS_PER_DAY
SPLITS = {"train": 0, "val": 1, "test": 2}
TEST_DAYS = range(8, 15)
VAL_DAYS = range(22, 29)
# hour bins on the fixed UTC+1 clock, [start, end) in hours
HOUR_BINS = {"night": (0, 5), "midday": (10, 14), "evening": (17, 21)}


def calendar_arrays() -> dict[str, np.ndarray]:
    """Calendar features for every 15-minute step of 2024 (UTC+1 clock)."""
    start = dt.date(YEAR, 1, 1)
    days = [start + dt.timedelta(days=d) for d in range(366)]
    month = np.repeat([d.month - 1 for d in days], STEPS_PER_DAY)
    day = np.repeat([d.day for d in days], STEPS_PER_DAY)
    weekend = np.repeat([int(d.weekday() >= 5) for d in days], STEPS_PER_DAY)
    day_of_year = np.repeat(np.arange(366), STEPS_PER_DAY)
    qh = np.tile(np.arange(STEPS_PER_DAY), 366)
    split = np.full(NUM_STEPS, SPLITS["train"])
    split[np.isin(day, TEST_DAYS)] = SPLITS["test"]
    split[np.isin(day, VAL_DAYS)] = SPLITS["val"]
    return {
        "month": month.astype(np.int64),
        "quarter_hour": qh.astype(np.int64),
        "weekend": weekend.astype(np.int64),
        "day_of_month": day.astype(np.int64),
        "day_of_year": day_of_year.astype(np.int64),
        "split": split.astype(np.int64),
    }


def hour_bin(quarter_hour: np.ndarray) -> np.ndarray:
    """Hour-bin label per snapshot: 'night', 'midday', 'evening' or 'other'."""
    hours = np.asarray(quarter_hour) / 4.0
    labels = np.full(hours.shape, "other", dtype=object)
    for name, (lo, hi) in HOUR_BINS.items():
        labels[(hours >= lo) & (hours < hi)] = name
    return labels


def split_hash() -> str:
    """SHA-256 of the split protocol (split label of every step)."""
    return hashlib.sha256(calendar_arrays()["split"].tobytes()).hexdigest()


@dataclass
class SnapshotData:
    p_kw: np.ndarray  # [T, 32] float64, NaN where invalid
    valid: np.ndarray  # [T] bool
    context: dict[str, np.ndarray]
    meta: dict = field(default_factory=dict)

    def indices(self, split: str) -> np.ndarray:
        mask = self.valid & (self.context["split"] == SPLITS[split])
        return np.flatnonzero(mask)

    def context_of(self, idx: np.ndarray) -> dict[str, np.ndarray]:
        keys = ("month", "quarter_hour", "weekend")
        return {k: self.context[k][idx] for k in keys}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(path, p_kw=self.p_kw, valid=self.valid, **self.context)
        with open(os.path.splitext(path)[0] + "_meta.json", "w") as f:
            json.dump(self.meta, f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "SnapshotData":
        arrays = np.load(path)
        context = {k: arrays[k] for k in calendar_arrays()}
        meta_path = os.path.splitext(path)[0] + "_meta.json"
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        return cls(arrays["p_kw"], arrays["valid"].astype(bool), context, meta)


def select_test_snapshots(
    data: SnapshotData, design: str, seed: int = 20261016
) -> np.ndarray:
    """Test snapshot indices for the two stored designs.

    "pilot": 100 each in the night, midday and evening bins, spread evenly
    over the months. "extended": 1,000 stratified by hour of day (round-robin
    over the 24 hours, months spread within each hour).
    """
    rng = np.random.default_rng(seed)
    test = data.indices("test")
    qh = data.context["quarter_hour"][test]
    if design == "pilot":
        groups = [test[hour_bin(qh) == name] for name in HOUR_BINS]
        quota = [100] * len(groups)
    elif design == "extended":
        groups = [test[qh // 4 == h] for h in range(24)]
        quota = [1000 // 24 + (1 if h < 1000 % 24 else 0) for h in range(24)]
    else:
        raise ValueError(design)
    chosen = []
    for members, n in zip(groups, quota, strict=True):
        m = data.context["month"][members]
        per_month = [rng.permutation(members[m == k]) for k in range(12)]
        picked, k = [], 0
        while len(picked) < n and any(len(p) for p in per_month):
            if len(per_month[k % 12]):
                picked.append(per_month[k % 12][0])
                per_month[k % 12] = per_month[k % 12][1:]
            k += 1
        chosen.extend(picked)
    return np.sort(np.asarray(chosen, dtype=np.int64))


def synthetic_snapshots(seed: int = 0) -> SnapshotData:
    """Synthetic year for tests and smoke runs: residential shape plus
    bimodal PV.

    Node loads follow a daily shape with a weekend shift and noise correlated
    across nodes; PV-heavy nodes get a clear-sky bump that is either sunny or
    cloudy each day (two modes).
    """
    rng = np.random.default_rng(seed)
    cal = calendar_arrays()
    hours = cal["quarter_hour"] / 4.0
    p_nom = np.asarray(P_NOMINAL_KW)
    shape = 0.35 + 0.25 * np.exp(-0.5 * ((hours - 19.0) / 2.0) ** 2)
    shape += 0.15 * np.exp(-0.5 * ((hours - 8.0) / 1.5) ** 2)
    shape += 0.05 * cal["weekend"]
    common = rng.standard_normal(NUM_STEPS)
    common = np.convolve(common, np.ones(8) / np.sqrt(8), mode="same")
    noise = 0.6 * common[:, None] + 0.8 * rng.standard_normal((NUM_STEPS, NUM_NODES))
    load = p_nom * (shape[:, None] * (1.0 + 0.15 * noise))
    season = 0.6 + 0.4 * np.cos(2 * np.pi * (cal["day_of_year"] - 172) / 366)
    sun = np.clip(np.cos(np.pi * (hours - 12.7) / 9.0), 0, None) ** 1.5
    sunny = rng.random(366) < 0.5
    clear = np.where(sunny[cal["day_of_year"]], 1.0, 0.25)
    pv_share = np.full(NUM_NODES, 0.3)
    pv_share[[node_of_bus(b) for b in PV_HEAVY_BUSES_BASE]] = 0.8
    pv = (1.6 * p_nom * pv_share)[None, :] * (season * sun * clear)[:, None]
    p_kw = load - pv * (1.0 + 0.05 * rng.standard_normal((NUM_STEPS, NUM_NODES)))
    meta = {"dataset": "synthetic", "seed": seed}
    return SnapshotData(p_kw, np.ones(NUM_STEPS, dtype=bool), cal, meta)

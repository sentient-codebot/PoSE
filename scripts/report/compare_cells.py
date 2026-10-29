"""Pairing check: a rerun cell against a stored cell, snapshot by snapshot.

The snapshot IDs and their calendar columns must match exactly. The check
passes if the per-snapshot tables are bit-identical, or if every scored
column agrees within --rtol (relative, NaN equal to NaN). max_pf_mismatch, a
power-flow residual of about 1e-16, is reported but not scored. Exit status 1
if the check fails.

    python scripts/report/compare_cells.py --new CELL --stored CELL --output OUT.json"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from pose.experiment import file_sha256, git_state, write_json


ID_COLS = ("snapshot", "day_of_year", "quarter_hour", "hour_bin")
NOT_SCORED = ("max_pf_mismatch",)


def compare_tables(new: pd.DataFrame, stored: pd.DataFrame, rtol: float) -> dict:
    out: dict = {"num_new": len(new), "num_stored": len(stored)}
    if list(new.columns) != list(stored.columns):
        out.update(passed=False, reason="columns differ")
        out["columns_only_new"] = sorted(set(new.columns) - set(stored.columns))
        out["columns_only_stored"] = sorted(set(stored.columns) - set(new.columns))
        return out
    if len(new) != len(stored) or not all(
        new[c].equals(stored[c]) for c in ID_COLS if c in new
    ):
        out.update(passed=False, reason="snapshot IDs or calendar columns differ")
        return out
    columns, ok = {}, True
    for c in new.columns:
        if c in ID_COLS:
            continue
        a = new[c].to_numpy(dtype=np.float64)
        b = stored[c].to_numpy(dtype=np.float64)
        same = (a == b) | (np.isnan(a) & np.isnan(b))
        close = np.isclose(a, b, rtol=rtol, atol=0.0, equal_nan=True)
        diff = np.abs(a - b)
        rel = diff / np.maximum(np.abs(b), np.finfo(float).tiny)
        finite = ~(np.isnan(a) & np.isnan(b))
        columns[c] = {
            "identical": bool(same.all()),
            "within_rtol": bool(close.all()),
            "max_abs_diff": float(np.nanmax(diff[finite])) if finite.any() else 0.0,
            "max_rel_diff": float(np.nanmax(rel[finite])) if finite.any() else 0.0,
            "scored": c not in NOT_SCORED,
        }
        if c not in NOT_SCORED and not close.all():
            ok = False
    out["columns"] = columns
    out["bit_identical"] = all(v["identical"] for v in columns.values())
    out["scored_within_rtol"] = ok
    out["passed"] = ok
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--new", required=True)
    p.add_argument("--stored", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--rtol", type=float, default=1e-9)
    args = p.parse_args()
    paths = {
        k: os.path.join(d, "per_snapshot.csv")
        for k, d in (("new", args.new), ("stored", args.stored))
    }
    result = compare_tables(
        pd.read_csv(paths["new"]), pd.read_csv(paths["stored"]), args.rtol
    )
    sha = {k: file_sha256(v) for k, v in paths.items()}
    info = {}
    for k, d in (("new", args.new), ("stored", args.stored)):
        with open(os.path.join(d, "metrics.json")) as f:
            m = json.load(f)
        info[k] = {
            "dir": d,
            "git": m.get("git"),
            "data_sha256": m.get("data", {}).get("sha256"),
            "hardware": m.get("hardware"),
            "per_snapshot_sha256": sha[k],
        }
    result.update(
        {
            "rtol": args.rtol,
            "files_identical": sha["new"] == sha["stored"],
            "cells": info,
            "git": git_state(),
        }
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    write_json(args.output, result)
    print(
        f"pairing check {'PASSED' if result['passed'] else 'FAILED'}: "
        f"bit-identical {result.get('bit_identical')}, "
        f"files identical {result['files_identical']}"
        + ("" if result["passed"] else f" ({result.get('reason', 'scored diff')})"),
        flush=True,
    )
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()

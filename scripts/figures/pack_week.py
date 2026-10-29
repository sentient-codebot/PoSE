"""Pack the example-week outputs into one compact archive (numpy only).

Reads the week cells (samples.npz with keys samples, truth, y, snapshots;
metrics.json; per_snapshot.csv), the prior samples and selected_ids.json and
writes fig_week_data.npz plus a JSON record with sha256 of every input.
S2 cells are kept for the whole week; S1 and S3 only for the midday ids.
Checks that all cells share the same snapshots and truths.

Usage: python pack_week.py OUT_DIR"""

import hashlib
import json
import os
import sys

import numpy as np


CELLS = {  # key in the archive: (cell directory, keep)
    "pose_S2": ("hpv_S2_nominal_M2-euler-meas", "week"),
    "wls_S2": ("hpv_S2_nominal_B1", "week"),
    "corr_S2": ("hpv_S2_nominal_B2", "week"),
    "pooled_S2": ("hpv_S2_nominal_B7", "week"),
    "pose_S1": ("hpv_S1_nominal_M2-euler-meas", "midday"),
    "pose_S3": ("hpv_S3_nominal_M2-euler-meas", "midday"),
}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    out_dir = sys.argv[1]
    with open(os.path.join(out_dir, "selected_ids.json")) as fh:
        sel = json.load(fh)
    week = np.array(sel["ids"]["week"])
    midday = np.array(sel["ids"]["midday"])
    arrays = {"week_snapshots": week, "midday_snapshots": midday}
    record = {"selected_week": sel["selected_week"], "inputs": {}}
    truth_ref = None
    for key, (cell, keep) in CELLS.items():
        d = os.path.join(out_dir, "cells", cell)
        z = np.load(os.path.join(d, "samples.npz"))
        snaps = z["snapshots"]
        assert np.array_equal(snaps, week), (
            cell,
            "snapshots differ from the selected week",
        )
        rows = np.arange(len(week)) if keep == "week" else np.searchsorted(week, midday)
        if truth_ref is None:
            truth_ref = z["truth"]
            arrays["truth"] = z["truth"].astype(np.float64)
        assert np.array_equal(z["truth"], truth_ref), (cell, "truth differs")
        arrays[f"{key}_samples"] = z["samples"][rows].astype(np.float32)
        arrays[f"{key}_y"] = z["y"][rows]
        for f in ("samples.npz", "metrics.json", "per_snapshot.csv"):
            record["inputs"][f"{cell}/{f}"] = sha(os.path.join(d, f))
        with open(os.path.join(d, "metrics.json")) as fh:
            record[f"{key}_metrics"] = json.load(fh)
    zp = np.load(os.path.join(out_dir, "prior", "prior.npz"))
    assert np.array_equal(zp["snapshots"], midday)
    arrays["prior_midday_samples"] = zp["samples"].astype(np.float32)
    record["inputs"]["prior/prior.npz"] = sha(
        os.path.join(out_dir, "prior", "prior.npz")
    )
    path = os.path.join(out_dir, "fig_week_data.npz")
    np.savez_compressed(path, **arrays)
    record["output_sha256"] = sha(path)
    record["shapes"] = {k: list(v.shape) for k, v in arrays.items()}
    with open(os.path.join(out_dir, "fig_week_data.json"), "w") as fh:
        json.dump(record, fh, indent=1, default=str)
    print(f"wrote {path}: {os.path.getsize(path) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

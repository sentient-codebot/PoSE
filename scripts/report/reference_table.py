"""CRPS(method) - CRPS(reference) with paired day-block bootstrap intervals
(2,000 resamples, seed 0; as extended_report.py) for one measurement set and
noise level, per scenario. Used for the WLS-referenced main table.

    python scripts/report/reference_table.py --root results/ausgrid \
        --scenarios ausgrid ausgrid_hpv --mset S2 --noise nominal --reference B1 \
        --output results/ausgrid/wls_reference.csv"""

import argparse
import glob
import os

import pandas as pd
from extended_report import ci, load


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", required=True)
    p.add_argument("--scenarios", nargs="+", required=True)
    p.add_argument("--mset", default="S2")
    p.add_argument("--noise", default="nominal")
    p.add_argument("--reference", default="B1")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    rows = []
    for sc in args.scenarios:
        stem = os.path.join(args.root, "cells", sc, f"{sc}_{args.mset}_{args.noise}_")
        _ref_info, ref = load(stem + args.reference)
        ref = ref.set_index("snapshot")
        for cell in sorted(glob.glob(stem + "*")):
            got = load(cell)
            if got is None:
                continue
            info, t = got
            if len(t) != len(ref) or set(t["snapshot"]) != set(ref.index):
                raise SystemExit(f"{cell}: snapshots differ from the reference")
            a = t.set_index("snapshot").loc[ref.index]
            if not (a["day_of_year"].to_numpy() == ref["day_of_year"].to_numpy()).all():
                raise SystemExit(f"{cell}: day metadata differ")
            r = ci(t, ref)
            rows.append(
                {
                    "scenario": sc,
                    "mset": args.mset,
                    "noise": args.noise,
                    "method": info["label"],
                    "reference": args.reference,
                    "crps_kw": float(t["crps_kw"].mean()),
                    "diff_kw": r["mean_diff"],
                    "ci_low": r["ci_low"],
                    "ci_high": r["ci_high"],
                    "num_blocks": r["num_blocks"],
                    "num_snapshots": r["num_snapshots"],
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()

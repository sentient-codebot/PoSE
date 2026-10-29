"""Pairing checks of the reconfigured-feeder study.

The base-topology cells are compared, snapshot by snapshot, with
compare_cells.compare_tables (rtol 1e-6):

    A  base B1, B2, B3, B4, B7, M0-euler, M1-euler against a stored set
    B  base M2-euler-meas against a stored set
    C  base B5, B6 against a stored set (recorded only)
    D  base B5-frozen, B6-frozen against base B5, B6

A and B require a strict pass; D requires a strict pass or agreement of the
cell means at reporting precision. Writes one JSON with every pair and the
per-snapshot SHA-256 of both sides. Exit status 1 if a required pair fails or
a cell is missing.

    python scripts/reconf_pairing.py --out $OUT --root $ROOT --cv1 $CV1 \
        --cv2 $CV2 --output $OUT/check/pairing_check.json"""

import argparse
import json
import os
import sys

import pandas as pd
from compare_cells import compare_tables

from pose.experiment import file_sha256, git_state, write_json


SCENARIOS = ("ausgrid", "ausgrid_hpv")
MSETS = ("S1", "S2")
GROUP_A = ("B1", "B2", "B3", "B4", "B7", "M0-euler", "M1-euler")
RTOL = 1e-6
# agreement of cell means at reporting precision (absolute)
PRECISION = {
    "crps_kw": 1e-3,
    "crps_q_kvar": 1e-3,
    "vm_rmse_pu": 1e-5,
    "branch_err_pct": 1e-2,
    "coverage90": 1e-3,
}


def new_cell(out, sc, mset, label):
    return os.path.join(out, "cells", "base", sc, f"{sc}_{mset}_nominal_{label}")


def stored_cell(root, sc, mset, label):
    return os.path.join(
        root,
        "results",
        "dsse",
        "ausgrid_pq",
        "cells",
        sc,
        f"{sc}_{mset}_nominal_{label}",
    )


def pairs(args):
    """(group, name, new dir, stored dir) for every pair of Section 6.3."""
    out = []
    for sc in SCENARIOS:
        for mset in MSETS:
            for lab in GROUP_A:
                out.append(
                    (
                        "A",
                        f"{sc}_{mset}_{lab}",
                        new_cell(args.out, sc, mset, lab),
                        stored_cell(args.root, sc, mset, lab),
                    )
                )
    for sc, mset, cv in (
        ("ausgrid", "S2", args.cv2),
        ("ausgrid_hpv", "S1", args.cv1),
        ("ausgrid_hpv", "S2", args.cv2),
    ):
        lab = "M2-euler-meas"
        out.append(
            (
                "B",
                f"{sc}_{mset}_{lab}",
                new_cell(args.out, sc, mset, lab),
                os.path.join(cv, f"{sc}_{mset}_nominal_{lab}"),
            )
        )
    for sc in SCENARIOS:
        for mset in MSETS:
            for lab in ("B5", "B6"):
                out.append(
                    (
                        "C",
                        f"{sc}_{mset}_{lab}",
                        new_cell(args.out, sc, mset, lab),
                        stored_cell(args.root, sc, mset, lab),
                    )
                )
    for sc in SCENARIOS:
        for mset in MSETS:
            for lab in ("B5", "B6"):
                out.append(
                    (
                        "D",
                        f"{sc}_{mset}_{lab}-frozen_vs_{lab}",
                        new_cell(args.out, sc, mset, f"{lab}-frozen"),
                        new_cell(args.out, sc, mset, lab),
                    )
                )
    return out


def precision_check(new: pd.DataFrame, stored: pd.DataFrame) -> dict:
    """Cell means within PRECISION; columns absent from both are skipped."""
    cols = {}
    for c, tol in PRECISION.items():
        if c not in new and c not in stored:
            continue
        if c not in new or c not in stored:
            cols[c] = {"present": False, "within": False}
            continue
        a, b = float(new[c].mean()), float(stored[c].mean())
        cols[c] = {
            "new": a,
            "stored": b,
            "abs_diff": abs(a - b),
            "tol": tol,
            "within": abs(a - b) <= tol,
        }
    return {"columns": cols, "passed": all(v["within"] for v in cols.values())}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True, help="study output root ($OUT)")
    p.add_argument("--root", required=True, help="stored inputs ($ROOT)")
    p.add_argument("--cv1", required=True)
    p.add_argument("--cv2", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    results, missing, failed = [], [], []
    for group, name, new_dir, stored_dir in pairs(args):
        entry = {"group": group, "name": name, "new": new_dir, "stored": stored_dir}
        paths = {
            k: os.path.join(d, "per_snapshot.csv")
            for k, d in (("new", new_dir), ("stored", stored_dir))
        }
        absent = [
            d
            for k, d in (("new", new_dir), ("stored", stored_dir))
            if not os.path.exists(paths[k])
        ]
        if absent:
            entry.update(missing=absent, passed=False, test="missing")
            missing += absent
            results.append(entry)
            if group != "C":
                failed.append(name)
            continue
        new, stored = pd.read_csv(paths["new"]), pd.read_csv(paths["stored"])
        strict = compare_tables(new, stored, RTOL)
        entry["strict"] = strict
        entry["per_snapshot_sha256"] = {k: file_sha256(v) for k, v in paths.items()}
        entry["files_identical"] = (
            entry["per_snapshot_sha256"]["new"]
            == entry["per_snapshot_sha256"]["stored"]
        )
        info = {}
        for k, d in (("new", new_dir), ("stored", stored_dir)):
            with open(os.path.join(d, "metrics.json")) as f:
                m = json.load(f)
            info[k] = {"git": m.get("git"), "hardware": m.get("hardware")}
        entry["cells"] = info
        if group in ("C", "D"):
            entry["precision"] = precision_check(new, stored)
        if strict["passed"]:
            entry.update(passed=True, test="strict")
        elif group in ("C", "D") and entry["precision"]["passed"]:
            entry.update(passed=True, test="reporting precision")
        else:
            entry.update(passed=False, test="none")
        if group != "C" and not entry["passed"]:
            failed.append(name)
        results.append(entry)
        print(
            f"{group} {name}: {entry['test']}"
            + ("" if entry["passed"] else " FAILED")
            + (f" ({strict.get('reason')})" if strict.get("reason") else ""),
            flush=True,
        )
    summary = {
        g: {
            "pairs": sum(r["group"] == g for r in results),
            "strict": sum(r["group"] == g and r["test"] == "strict" for r in results),
            "precision": sum(
                r["group"] == g and r["test"] == "reporting precision" for r in results
            ),
            "failed": sum(r["group"] == g and not r["passed"] for r in results),
        }
        for g in "ABCD"
    }
    passed = not failed and not missing
    write_json(
        args.output,
        {
            "passed": passed,
            "rtol": RTOL,
            "precision": PRECISION,
            "summary": summary,
            "failed": failed,
            "missing": missing,
            "pairs": results,
            "git": git_state(),
        },
    )
    print(json.dumps(summary), flush=True)
    print(f"pairing check {'PASSED' if passed else 'FAILED'}", flush=True)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

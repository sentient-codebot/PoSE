"""Report of the main cell matrix.

For every PV scenario, measurement set and noise level under
--root/cells/<scenario>/<scenario>_<mset>_<noise>_<label>: per-method means
(CRPS, energy score, voltage RMSE, branch error, coverage, chi-squared,
runtime), paired block-bootstrap 95% CIs of CRPS(method) - CRPS(B2) over all
snapshots and at midday, and the DSO metrics. Then the seed spread of
M2-Euler in S2 nominal (--root/cells_seed<k>/) and the step curve. Cells
that are missing are listed, not skipped silently. A last table gives
CRPS(M2-Euler) - CRPS(method) for every method and setting, with paired
block-bootstrap 95% CIs over all snapshots and at midday."""

import argparse
import json
import os

import pandas as pd

from pose.experiment import git_state, write_json
from pose.metrics import (
    attainable_coverage,
    dso_summary,
    dso_table_lines,
    paired_block_bootstrap,
)


METHODS = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "M0-euler", "M1-euler", "M2-euler")
MEANS = (
    "crps_kw",
    "energy_score_kw",
    "vm_rmse_pu",
    "branch_err_pct",
    "coverage90",
    "chi2_norm",
)
OPTIONAL = ("crps_q_kvar", "coverage90_q")  # joint (P, Q) state only
DSO_METHODS = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "M2-euler")


def load(cell_dir):
    path = os.path.join(cell_dir, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        info = json.load(f)
    return info, pd.read_csv(os.path.join(cell_dir, "per_snapshot.csv"))


def ci(t, ref, mask=None):
    t = t.set_index("snapshot").loc[ref.index]
    m = slice(None) if mask is None else (t["hour_bin"] == mask).to_numpy()
    return paired_block_bootstrap(
        t["crps_kw"].to_numpy()[m],
        ref["crps_kw"].to_numpy()[m],
        t["day_of_year"].to_numpy()[m],
    )


def fmt(r):
    return f"{r['mean_diff']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", required=True)
    p.add_argument("--scenarios", nargs="+", required=True)
    p.add_argument("--msets", nargs="+", default=["S1", "S2", "S3"])
    p.add_argument("--noises", nargs="+", default=["nominal", "high"])
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    rows, missing, versus = [], [], []
    lines = ["# Extended experiments: Ausgrid testbed", ""]
    for sc in args.scenarios:
        for mset in args.msets:
            for noise in args.noises:
                stem = os.path.join(args.root, "cells", sc, f"{sc}_{mset}_{noise}")
                cells = {m: load(f"{stem}_{m}") for m in METHODS}
                missing += [
                    f"{sc}_{mset}_{noise}_{m}" for m, c in cells.items() if c is None
                ]
                cells = {m: c for m, c in cells.items() if c is not None}
                if "B2" not in cells:
                    continue
                ref = cells["B2"][1].set_index("snapshot")
                lines += [f"## {sc}, {mset}, {noise} noise", ""]
                lines += [
                    (
                        "| method | CRPS (kW) | - B2, all (95% CI) | - B2, midday | energy score | V RMSE (p.u.) "
                        "| branch err (%) | cov90 (attainable) | chi2 | s/snapshot | Q CRPS (kvar) |"
                    ),
                    "|---" * 11 + "|",
                ]
                dso = {}
                for m, (info, t) in cells.items():
                    k = info.get("num_samples") or 1
                    row = {"scenario": sc, "mset": mset, "noise": noise, "method": m}
                    row.update({c: float(t[c].mean()) for c in MEANS})
                    row.update({c: float(t[c].mean()) for c in OPTIONAL if c in t})
                    row["runtime_s"] = info["runtime_per_snapshot_s"]
                    row["attainable"] = attainable_coverage(k)
                    if m != "B2":
                        a, d = ci(t, ref), ci(t, ref, "midday")
                        row.update(
                            diff_all=a["mean_diff"],
                            diff_all_lo=a["ci_low"],
                            diff_all_hi=a["ci_high"],
                            diff_mid=d["mean_diff"],
                            diff_mid_lo=d["ci_low"],
                            diff_mid_hi=d["ci_high"],
                        )
                        da, dm = fmt(a), fmt(d)
                    else:
                        da = dm = "reference"
                    rows.append(row)
                    att = (
                        "n/a"
                        if row["attainable"] is None
                        else f"{row['attainable']:.3f}"
                    )
                    cov = "n/a" if k < 2 else f"{row['coverage90']:.3f}"
                    lines.append(
                        f"| {m} | {row['crps_kw']:.3f} | {da} | {dm} | {row['energy_score_kw']:.2f} "
                        f"| {row['vm_rmse_pu']:.5f} | {row['branch_err_pct']:.2f} | {cov} ({att}) "
                        f"| {row['chi2_norm']:.2f} | {row['runtime_s']:.3f} "
                        f"| {row.get('crps_q_kvar', float('nan')):.3f} |"
                    )
                    if m in DSO_METHODS:
                        dso[m] = dso_summary(t, k)
                if "M2-euler" in cells:
                    m2 = cells["M2-euler"][1].set_index("snapshot")
                    for m, (_, t) in cells.items():
                        if m == "M2-euler":
                            continue
                        other = t.set_index("snapshot")
                        a, d = (
                            ci(m2.reset_index(), other),
                            ci(m2.reset_index(), other, "midday"),
                        )
                        versus.append(
                            (
                                f"{sc}, {mset}, {noise}",
                                m,
                                fmt(a),
                                fmt(d),
                                a["ci_high"] < 0,
                            )
                        )
                lines += ["", "DSO metrics:", ""] + dso_table_lines(dso) + [""]

        s2 = os.path.join(args.root, "cells", sc, f"{sc}_S2_nominal")
        b2 = load(f"{s2}_B2")
        if b2 is None:
            continue
        ref = b2[1].set_index("snapshot")
        lines += [f"## {sc}, S2 nominal: seeds, sampler and steps", ""]
        lines += [
            "| run | CRPS (kW) | - B2, all (95% CI) | cov90 | s/snapshot |",
            "|---" * 5 + "|",
        ]
        extra = [("seed 0", f"{s2}_M2-euler")]
        extra += [
            (
                f"seed {s}",
                os.path.join(
                    args.root, f"cells_seed{s}", sc, f"{sc}_S2_nominal_M2-euler"
                ),
            )
            for s in (1, 2)
        ]
        extra += [("ancestral", f"{s2}_M2-ancestral")]
        extra += [(f"{n} steps", f"{s2}_M2-euler-s{n}") for n in (50, 100, 500)]
        for name, d in extra:
            c = load(d)
            if c is None:
                missing.append(f"{sc}: {name}")
                continue
            info, t = c
            rows.append(
                {
                    "scenario": sc,
                    "mset": "S2",
                    "noise": "nominal",
                    "method": f"M2 {name}",
                    **{m: float(t[m].mean()) for m in MEANS},
                    "runtime_s": info["runtime_per_snapshot_s"],
                }
            )
            lines.append(
                f"| {name} | {t['crps_kw'].mean():.3f} | {fmt(ci(t, ref))} | {t['coverage90'].mean():.3f} "
                f"| {info['runtime_per_snapshot_s']:.3f} |"
            )
        lines.append("")
    if versus:
        lines += [
            "## CRPS(M2-Euler) - CRPS(method), kW (95% CI)",
            "",
            "| setting | method | all | midday | M2 better (CI below zero) |",
            "|---" * 5 + "|",
        ]
        lines += [
            f"| {s} | {m} | {a} | {d} | {'yes' if b else 'no'} |"
            for s, m, a, d, b in versus
        ]
        for m in METHODS[:-1]:
            wins = [b for _, mm, _, _, b in versus if mm == m]
            if wins:
                lines.append(
                    f"\nM2 better than {m} in {sum(wins)} of {len(wins)} settings."
                )
        lines.append("")
    if missing:
        lines += ["## Missing cells", ""] + [f"- {m}" for m in missing]
    os.makedirs(args.output_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        os.path.join(args.output_dir, "extended_summary.csv"), index=False
    )
    with open(os.path.join(args.output_dir, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    write_json(
        os.path.join(args.output_dir, "report.json"),
        {"rows": rows, "missing": missing, "git": git_state()},
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()

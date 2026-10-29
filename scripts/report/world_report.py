"""Report of the data-scaling study: PoSE against B2 as the training set grows.

For every scenario, measurement set and training size under --cells_root
(<scenario>/<size>/world-<scenario>_<mset>_nominal_<label>), the report gives
CRPS(M2-Euler) - CRPS(B2) and CRPS(B1) - CRPS(B2) with paired block
bootstrap 95% CIs over test days, M2 coverage next to its attainable value,
the other accuracy metrics and the DSO metrics. Reading rule: the method
helps with rich data if, at the largest size, M2 beats B2 on CRPS with the CI
below zero in both S1 and S2 and its coverage is within 0.02 of the
attainable value."""

import argparse
import json
import os

import pandas as pd

from pose.data import HOUR_BINS
from pose.experiment import git_state, write_json
from pose.metrics import (
    attainable_coverage,
    dso_summary,
    dso_table_lines,
    paired_block_bootstrap,
)


SIZES = ("n2k", "n19k", "n190k")
MEANS = ("crps_kw", "energy_score_kw", "vm_rmse_pu", "branch_err_pct", "chi2_norm")


def load(cell_dir):
    with open(os.path.join(cell_dir, "metrics.json")) as f:
        info = json.load(f)
    return info, pd.read_csv(os.path.join(cell_dir, "per_snapshot.csv"))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cells_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--scenarios", nargs="+", default=["base", "high_pv"])
    p.add_argument("--msets", nargs="+", default=["S1", "S2"])
    p.add_argument("--sizes", nargs="+", default=list(SIZES))
    p.add_argument("--noise", default="nominal")
    p.add_argument(
        "--prefix", default="world-", help="setting prefix ('' for real data)"
    )
    p.add_argument("--no_rule", action="store_true", help="skip the reading rule")
    p.add_argument(
        "--title",
        default="Design A: model world, M2 against B2 as the training set grows",
    )
    args = p.parse_args()

    rows = []
    lines = [f"# {args.title}", ""]
    for sc in args.scenarios:
        for mset in args.msets:
            dso = {}
            for size in args.sizes:
                stem = os.path.join(
                    args.cells_root, sc, size, f"{args.prefix}{sc}_{mset}_{args.noise}"
                )
                cells = {m: load(f"{stem}_{m}") for m in ("B1", "B2", "M2-euler")}
                b2 = cells["B2"][1].set_index("snapshot")
                row = {"scenario": sc, "mset": mset, "size": size}
                for m in ("M2-euler", "B1"):
                    t = cells[m][1].set_index("snapshot").loc[b2.index]
                    for b in ["all"] + list(HOUR_BINS):
                        mask = (
                            (t["hour_bin"] == b).to_numpy()
                            if b != "all"
                            else slice(None)
                        )
                        r = paired_block_bootstrap(
                            t["crps_kw"].to_numpy()[mask],
                            b2["crps_kw"].to_numpy()[mask],
                            t["day_of_year"].to_numpy()[mask],
                        )
                        key = "m2" if m == "M2-euler" else "b1"
                        row[f"{key}_minus_b2_{b}"] = r["mean_diff"]
                        row[f"{key}_minus_b2_{b}_lo"] = r["ci_low"]
                        row[f"{key}_minus_b2_{b}_hi"] = r["ci_high"]
                for m, (_, t) in cells.items():
                    for k in MEANS:
                        row[f"{m}:{k}"] = float(t[k].mean())
                    row[f"{m}:coverage90"] = float(t["coverage90"].mean())
                k = cells["M2-euler"][0]["num_samples"]
                row["attainable"] = attainable_coverage(k)
                rows.append(row)
                dso[f"{size} B2"] = dso_summary(
                    cells["B2"][1], cells["B2"][0]["num_samples"]
                )
                dso[f"{size} M2"] = dso_summary(cells["M2-euler"][1], k)
            lines += [f"## {sc}, {mset}, DSO metrics", ""] + dso_table_lines(dso) + [""]
    table = pd.DataFrame(rows)
    os.makedirs(args.output_dir, exist_ok=True)
    table.to_csv(os.path.join(args.output_dir, "learning_curve.csv"), index=False)

    head = [
        "## CRPS against B2 (kW, 95% CI) and M2 coverage",
        "",
        (
            "| scenario | set | size | M2 - B2, all | M2 - B2, midday | B1 - B2, all "
            "| CRPS B2 | CRPS M2 | cov M2 (attainable) | cov B2 |"
        ),
        "|---" * 10 + "|",
    ]
    for _, r in table.iterrows():
        head.append(
            f"| {r.scenario} | {r.mset} | {r['size']} "
            f"| {r.m2_minus_b2_all:+.3f} [{r.m2_minus_b2_all_lo:+.3f}, {r.m2_minus_b2_all_hi:+.3f}] "
            f"| {r.m2_minus_b2_midday:+.3f} [{r.m2_minus_b2_midday_lo:+.3f}, {r.m2_minus_b2_midday_hi:+.3f}] "
            f"| {r.b1_minus_b2_all:+.3f} | {r['B2:crps_kw']:.3f} | {r['M2-euler:crps_kw']:.3f} "
            f"| {r['M2-euler:coverage90']:.3f} ({r.attainable:.3f}) | {r['B2:coverage90']:.3f} |"
        )
    largest = args.sizes[-1]
    verdict = {}
    for sc in args.scenarios:
        sub = table[(table.scenario == sc) & (table["size"] == largest)]
        checks = {
            r.mset: bool(
                r.m2_minus_b2_all_hi < 0
                and abs(r["M2-euler:coverage90"] - r.attainable) <= 0.02
            )
            for _, r in sub.iterrows()
        }
        verdict[sc] = {
            "per_set": checks,
            "helps": all(checks.values()) and len(checks) == 2,
        }
    rule = [
        "",
        f"## Reading rule at {largest}",
        "",
        (
            "M2 beats B2 on CRPS with the 95% CI below zero in both S1 and S2, and its "
            "coverage is within 0.02 of the attainable value."
        ),
        "",
    ]
    rule += [
        f"- {sc}: helps = {v['helps']} (per set: {v['per_set']})"
        for sc, v in verdict.items()
    ]
    if args.no_rule:
        rule, verdict = [], None
    lines = lines[:2] + head + rule + [""] + lines[2:]
    with open(os.path.join(args.output_dir, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    write_json(
        os.path.join(args.output_dir, "report.json"),
        {"verdict": verdict, "rows": rows, "git": git_state()},
    )
    print("\n".join(head + rule))


if __name__ == "__main__":
    main()

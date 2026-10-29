"""Report of the reconfigured-feeder study.

Reads the 384 cells under --root/cells/<topology>/<scenario>/
<scenario>_<mset>_nominal_<label> (8 topologies x 2 PV scenarios x S1, S2 x
12 methods) and writes to --output_dir:

    reconf_summary.csv          cell means, runtimes and DSO metrics per
                                (topology, scenario, mset, method)
    reconf_contrasts.csv        R1-R5 CRPS contrasts with paired day-block
                                bootstrap 95% CIs (extended_report.ci: 2,000
                                resamples, seed 0), over all snapshots and at
                                midday
    reconf_wls_reference.csv    CRPS(method) - CRPS(B1) per setting
    report.md, report.json

Contrasts (PoSE-fast = M2-euler-meas):
    R1  PoSE-fast - B2, PoSE-fast - B7
    R2  PoSE-fast - B5, B6 (retrained) and - B5-frozen, B6-frozen
    R3  B5-frozen - B5, B6-frozen - B6
    R4  method(topology) - method(base), every method (descriptive)
    R5  PoSE-fast - B1, B3, M0-euler, M1-euler, B4
R1-R3 and R5 are computed on every setting (base included as reference); the
win counts are over the 28 reconfigured settings. Missing cells: exit 1
unless --allow_missing, which lists them.

    python scripts/report/reconf_report.py --root $OUT --output_dir $OUT/report"""

import argparse
import os
import sys

import pandas as pd
from extended_report import ci, load

from pose.experiment import git_state, write_json
from pose.feeder import RECONFIGURATIONS


TOPOLOGIES = tuple(RECONFIGURATIONS)
RECONF = tuple(t for t in TOPOLOGIES if t != "base")
STRESS = ("reconf_t36",)
SCENARIOS = ("ausgrid", "ausgrid_hpv")
MSETS = ("S1", "S2")
POSE = "M2-euler-meas"
METHODS = (
    "B1",
    "B2",
    "B7",
    "B3",
    "B5",
    "B6",
    "B5-frozen",
    "B6-frozen",
    "M0-euler",
    "M1-euler",
    "B4",
    POSE,
)
DISPLAY = {POSE: "PoSE-fast"}
SUMMARY_MEANS = (
    "crps_kw",
    "crps_q_kvar",
    "energy_score_kw",
    "vm_rmse_pu",
    "branch_err_pct",
    "coverage90",
    "coverage90_q",
    "chi2_norm",
)
TIMES = ("runtime_per_snapshot_s", "setup_time_s", "topology_update_s")
# report column -> path in metrics.json "dso"["all"]
DSO_KEYS = {
    "v_hit_0p5": ("v", "hit_rate50"),
    "v_fa_0p5": ("v", "false_alarm_rate50"),
    "v_brier": ("v", "brier"),
    "i_hit_0p5": ("i", "hit_rate50"),
    "i_fa_0p5": ("i", "false_alarm_rate50"),
    "i_brier": ("i", "brier"),
    "max_v_err_p95": ("vm_max_abs_pu_p95",),
    "max_i_err_p95": ("branch_max_err_pct_p95",),
}
CONTRASTS = (
    [("R1", POSE, m) for m in ("B2", "B7")]
    + [("R2", POSE, m) for m in ("B5", "B6", "B5-frozen", "B6-frozen")]
    + [("R3", "B5-frozen", "B5"), ("R3", "B6-frozen", "B6")]
    + [("R5", POSE, m) for m in ("B1", "B3", "M0-euler", "M1-euler", "B4")]
)
RATINGS_NOTE = (
    "Branch ratings: the rating (branch current at nominal "
    "loading) is applied to each operating topology, so percent-of-rating errors "
    "and overload metrics are relative to that topology's own nominal loading. "
    "Branch k is the in-service branch that feeds bus k + 1 in the current tree; "
    "branch metrics compare methods within a topology, not branches across "
    "topologies."
)


def name(m):
    return DISPLAY.get(m, m)


def topo_name(t):
    return f"{t} (stress case)" if t in STRESS else t


def cell_dir(root, t, sc, mset, m):
    return os.path.join(root, "cells", t, sc, f"{sc}_{mset}_nominal_{m}")


def num(x, spec):
    return "n/a" if x is None or pd.isna(x) else format(x, spec)


def need(d: dict, *keys, where=""):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            raise SystemExit(f"{where}: metrics.json has no {'/'.join(keys)}")
        d = d[k]
    return d


def summary_row(t, sc, mset, m, info, table):
    where = f"{t}/{sc}_{mset}_{m}"
    row = {"topology": t, "scenario": sc, "mset": mset, "method": m}
    s_all = need(info, "summary", "all", where=where)
    for c in SUMMARY_MEANS:
        row[c] = need(s_all, c, where=where)
    d_all = need(info, "dso", "all", where=where)
    row["attainable"] = need(d_all, "coverage90_attainable", where=where)
    for c in TIMES:
        row[c] = need(info, c, where=where)
    for c, keys in DSO_KEYS.items():
        row[c] = need(d_all, *keys, where=where)
    row["num_snapshots"] = len(table)
    return row


def contrast(a, b, subset):
    """Paired day-block bootstrap of CRPS(a) - CRPS(b); a, b per-snapshot tables."""
    ref = b.set_index("snapshot")
    if set(a["snapshot"]) != set(ref.index) or len(a) != len(ref):
        raise SystemExit("cells to contrast hold different snapshots")
    return ci(a, ref, None if subset == "all" else subset)


def fmt(r):
    return f"{r['mean']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]"


def verdict(r):
    if r["hi"] < 0:
        return "below 0"
    if r["lo"] > 0:
        return "above 0"
    return "includes 0"


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--allow_missing", action="store_true")
    args = p.parse_args()

    cells, missing = {}, []
    for t in TOPOLOGIES:
        for sc in SCENARIOS:
            for mset in MSETS:
                for m in METHODS:
                    c = load(cell_dir(args.root, t, sc, mset, m))
                    if c is None:
                        missing.append(f"{t}/{sc}_{mset}_nominal_{m}")
                    else:
                        cells[t, sc, mset, m] = c
    if missing and not args.allow_missing:
        print("missing cells:\n" + "\n".join(missing), flush=True)
        sys.exit(1)

    rows = [summary_row(*key, info, table) for key, (info, table) in cells.items()]
    summary = pd.DataFrame(rows)

    con, wls = [], []
    for (t, sc, mset, m), (_, table) in cells.items():
        for g, a, b in CONTRASTS:
            if m != a or (t, sc, mset, b) not in cells:
                continue
            for subset in ("all", "midday"):
                r = contrast(table, cells[t, sc, mset, b][1], subset)
                con.append(
                    {
                        "topology": t,
                        "scenario": sc,
                        "mset": mset,
                        "contrast": f"{g}: {name(a)} - {name(b)}",
                        "subset": subset,
                        "mean": r["mean_diff"],
                        "lo": r["ci_low"],
                        "hi": r["ci_high"],
                    }
                )
        if t != "base" and ("base", sc, mset, m) in cells:
            for subset in ("all", "midday"):
                r = contrast(table, cells["base", sc, mset, m][1], subset)
                con.append(
                    {
                        "topology": t,
                        "scenario": sc,
                        "mset": mset,
                        "contrast": f"R4: {name(m)} {t} - base",
                        "subset": subset,
                        "mean": r["mean_diff"],
                        "lo": r["ci_low"],
                        "hi": r["ci_high"],
                    }
                )
        if (t, sc, mset, "B1") in cells:
            ref_table = cells[t, sc, mset, "B1"][1]
            r = contrast(table, ref_table, "all")
            wls.append(
                {
                    "topology": t,
                    "scenario": sc,
                    "mset": mset,
                    "noise": "nominal",
                    "method": m,
                    "reference": "B1",
                    "crps_kw": float(table["crps_kw"].mean()),
                    "reference_crps_kw": float(ref_table["crps_kw"].mean()),
                    "diff_crps_kw": r["mean_diff"],
                    "ci_low_kw": r["ci_low"],
                    "ci_high_kw": r["ci_high"],
                    "num_snapshots": r["num_snapshots"],
                    "num_blocks": r["num_blocks"],
                }
            )
    contrasts = pd.DataFrame(
        con,
        columns=[
            "topology",
            "scenario",
            "mset",
            "contrast",
            "subset",
            "mean",
            "lo",
            "hi",
        ],
    )
    wls = pd.DataFrame(wls)

    # win counts over the reconfigured settings, all snapshots
    wins = {}
    for g, a, b in CONTRASTS:
        label = f"{g}: {name(a)} - {name(b)}"
        sub = contrasts[
            (contrasts["contrast"] == label)
            & (contrasts["subset"] == "all")
            & (contrasts["topology"] != "base")
        ]
        counts = {"below 0": 0, "above 0": 0, "includes 0": 0}
        for _, r in sub.iterrows():
            counts[verdict(r)] += 1
        wins[label] = {**counts, "settings": len(sub)}

    lines = [
        "# Reconfigured IEEE 33-bus feeders: PoSE-fast without retraining",
        "",
        (
            "PoSE-fast = M2-euler-meas (measurement-space covariance). Nominal "
            "noise, 1,000 extended-design test snapshots, K = 100. `reconf_t36` "
            "raises losses and lowers V_min relative to base: a stress case, not a "
            "loss-reducing reconfiguration."
        ),
        "",
        (
            "Contrasts are CRPS differences in kW with paired day-block bootstrap "
            "95% CIs; negative favours the first method."
        ),
        "",
    ]
    if missing:
        lines += ["## Missing cells", ""] + [f"- {m}" for m in missing] + [""]

    # cross-topology table
    lines += [
        "## CRPS (kW) across topologies",
        "",
        "Base value, then mean, min and max over the 7 reconfigured topologies.",
        "",
    ]
    for sc in SCENARIOS:
        for mset in MSETS:
            lines += [
                f"### {sc}, {mset}",
                "",
                "| method | base | mean (7 reconf.) | min | max |",
                "|---" * 5 + "|",
            ]
            for m in METHODS:
                s = (
                    summary[
                        (summary["scenario"] == sc)
                        & (summary["mset"] == mset)
                        & (summary["method"] == m)
                    ]
                    if len(summary)
                    else summary
                )
                base = s[s["topology"] == "base"]["crps_kw"]
                rec = s[s["topology"] != "base"]["crps_kw"]
                b = f"{base.iloc[0]:.3f}" if len(base) else "n/a"
                if len(rec):
                    lines.append(
                        f"| {name(m)} | {b} | {rec.mean():.3f} | {rec.min():.3f} "
                        f"| {rec.max():.3f} |"
                    )
                else:
                    lines.append(f"| {name(m)} | {b} | n/a | n/a | n/a |")
            lines.append("")

    # R1-R3 tables and win counts
    for g, title in (
        ("R1", "R1 (primary): PoSE-fast against B2 and B7"),
        ("R2", "R2: PoSE-fast against the supervised estimators"),
        ("R3", "R3: frozen minus retrained supervised estimators"),
        ("R5", "R5: PoSE-fast against the other comparators"),
    ):
        labels = [f"{gg}: {name(a)} - {name(b)}" for gg, a, b in CONTRASTS if gg == g]
        lines += [f"## {title}", ""]
        lines += [
            "| topology | scenario | mset | contrast | all (95% CI) | midday (95% CI) |",
            "|---" * 6 + "|",
        ]
        for t in TOPOLOGIES:
            for sc in SCENARIOS:
                for mset in MSETS:
                    for lab in labels:
                        sel = contrasts[
                            (contrasts["topology"] == t)
                            & (contrasts["scenario"] == sc)
                            & (contrasts["mset"] == mset)
                            & (contrasts["contrast"] == lab)
                        ]
                        if not len(sel):
                            continue
                        by = {r["subset"]: r for _, r in sel.iterrows()}
                        lines.append(
                            f"| {topo_name(t)} | {sc} | {mset} | {lab} "
                            f"| {fmt(by['all'])} | {fmt(by['midday'])} |"
                        )
        lines += ["", "Win count over the reconfigured settings (all snapshots):", ""]
        lines += ["| contrast | CI below 0 | CI above 0 | CI includes 0 | settings |"]
        lines += ["|---" * 5 + "|"]
        for lab in labels:
            w = wins[lab]
            lines.append(
                f"| {lab} | {w['below 0']} | {w['above 0']} | {w['includes 0']} "
                f"| {w['settings']} |"
            )
        lines.append("")

    # R4, descriptive
    lines += [
        "## R4 (descriptive): CRPS(method, topology) - CRPS(method, base), all snapshots",
        "",
        (
            "Not a pass criterion: reconfiguration also changes how informative "
            "the fixed meters are, for every method."
        ),
        "",
        "| method | scenario | mset | " + " | ".join(RECONF) + " |",
        "|---" * (3 + len(RECONF)) + "|",
    ]
    r4 = contrasts[
        contrasts["contrast"].str.startswith("R4") & (contrasts["subset"] == "all")
    ]
    for m in METHODS:
        for sc in SCENARIOS:
            for mset in MSETS:
                vals = []
                for t in RECONF:
                    sel = r4[
                        (r4["topology"] == t)
                        & (r4["scenario"] == sc)
                        & (r4["mset"] == mset)
                        & (r4["contrast"] == f"R4: {name(m)} {t} - base")
                    ]
                    vals.append(fmt(sel.iloc[0]) if len(sel) else "n/a")
                lines.append(
                    f"| {name(m)} | {sc} | {mset} | " + " | ".join(vals) + " |"
                )
    lines.append("")

    # per-setting tables
    lines += ["## Per-setting results", "", RATINGS_NOTE, ""]
    for t in TOPOLOGIES:
        for sc in SCENARIOS:
            for mset in MSETS:
                s = (
                    summary[
                        (summary["topology"] == t)
                        & (summary["scenario"] == sc)
                        & (summary["mset"] == mset)
                    ]
                    if len(summary)
                    else summary
                )
                if not len(s):
                    continue
                lines += [
                    f"### {topo_name(t)}, {sc}, {mset}",
                    "",
                    (
                        "| method | CRPS (kW) | Q CRPS (kvar) | energy score "
                        "| V RMSE (p.u.) | branch err (%) | cov90 (attainable) "
                        "| Q cov90 | chi2 | s/snapshot | setup (s) "
                        "| topology update (s) | V hit | V FA | I hit | I FA "
                        "| max V err p95 | max I err p95 (%) |"
                    ),
                    "|---" * 18 + "|",
                ]
                for m in METHODS:
                    r = s[s["method"] == m]
                    if not len(r):
                        continue
                    r = r.iloc[0]

                    lines.append(
                        f"| {name(m)} | {r['crps_kw']:.3f} | {r['crps_q_kvar']:.3f} "
                        f"| {r['energy_score_kw']:.2f} | {r['vm_rmse_pu']:.5f} "
                        f"| {r['branch_err_pct']:.2f} | {num(r['coverage90'], '.3f')} "
                        f"({num(r['attainable'], '.3f')}) | {num(r['coverage90_q'], '.3f')} "
                        f"| {r['chi2_norm']:.2f} | {r['runtime_per_snapshot_s']:.3f} "
                        f"| {r['setup_time_s']:.1f} | {r['topology_update_s']:.2f} "
                        f"| {num(r['v_hit_0p5'], '.3f')} | {num(r['v_fa_0p5'], '.4f')} "
                        f"| {num(r['i_hit_0p5'], '.3f')} | {num(r['i_fa_0p5'], '.4f')} "
                        f"| {r['max_v_err_p95']:.5f} | {r['max_i_err_p95']:.2f} |"
                    )
                lines.append("")

    os.makedirs(args.output_dir, exist_ok=True)
    summary.to_csv(os.path.join(args.output_dir, "reconf_summary.csv"), index=False)
    contrasts.to_csv(os.path.join(args.output_dir, "reconf_contrasts.csv"), index=False)
    wls.to_csv(os.path.join(args.output_dir, "reconf_wls_reference.csv"), index=False)
    with open(os.path.join(args.output_dir, "report.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    topo_update = (
        summary.groupby("method")["topology_update_s"].agg(["mean", "min", "max"])
        if len(summary)
        else pd.DataFrame()
    )
    write_json(
        os.path.join(args.output_dir, "report.json"),
        {
            "rows": rows,
            "contrasts": contrasts.to_dict(orient="records"),
            "win_counts": wins,
            "topology_update_s": topo_update.to_dict(orient="index"),
            "ratings_convention": RATINGS_NOTE,
            "stress_cases": list(STRESS),
            "missing": missing,
            "git": git_state(),
        },
    )
    print("\n".join(lines[:200]))
    print(f"report written to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

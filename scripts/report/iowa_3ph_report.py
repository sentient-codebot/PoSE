"""Report of the three-phase Iowa test.

Reads the five three-phase cells under --root/cells/iowa_S2-3ph_nominal_<label>
(B1, B2, B7, B3, PoSE-fast = M2-euler-meas) and the stored balanced cells
(--balanced LABEL=DIR), and writes to --output_dir:

    iowa_3ph_summary.csv     cell means, coverage with its attainable value,
                             DSO metrics and runtimes per method (three-phase)
    iowa_3ph_contrasts.csv   R1-R3 paired differences with day-block bootstrap
                             95% CIs (metrics.paired_block_bootstrap: 2,000
                             resamples, seed 0), over all snapshots and at
                             midday
    report.md, report.json

    R1  CRPS(PoSE-fast) - CRPS(X), X in B1, B2, B7, B3 (primary)
    R2  the same for crps_q_kvar, vm_rmse_pu and branch_err_pct; win count
        across the summary metrics and every metric where a baseline is better
    R3  three-phase minus balanced per method (paired per snapshot) for
        crps_kw, crps_q_kvar and coverage90 (descriptive). The three-phase set
        measures per-phase head flows, so it carries more information than the
        balanced set; R3 is not a physics-only effect.

Physics facts: channel count, node-phase and branch-phase counts, excluded
branch-phases, per-channel std of the linearization error against the noise
std for both models, the true voltage range over node-phases and the number
of true violations and overloads. With --stage1 present, the probe numbers
are quoted.

The per-snapshot vm_* and branch_* columns of the three-phase cells run over
node-phases and the kept (masked) branch-phases.

    python scripts/report/iowa_3ph_report.py --root $OUT --data $DATA \
        --physics $OUT/feeder3ph/iowa_3ph.json --balanced B1=DIR ... \
        --output_dir $OUT"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from pose.data import SnapshotData
from pose.experiment import file_sha256, git_state, write_json
from pose.feeder import NOISE_LEVELS, Feeder
from pose.metrics import paired_block_bootstrap
from pose.powerflow import RadialPowerFlow
from pose.threephase import (
    ThreePhaseFeeder,
    ThreePhasePowerFlow,
    ratings_3ph,
)


POSE = "M2-euler-meas"
METHODS = ("B1", "B2", "B7", "B3", POSE)
BASELINES = ("B1", "B2", "B7", "B3")
DISPLAY = {
    "B1": "WLS",
    "B2": "correlated Gaussian",
    "B7": "pooled Gaussian",
    "B3": "Gaussian mixture",
    POSE: "PoSE-fast",
}
SUMMARY_MEANS = (
    "crps_kw",
    "crps_q_kvar",
    "vm_rmse_pu",
    "branch_err_pct",
    "coverage90",
    "coverage90_q",
    "chi2_norm",
)
TIMES = ("runtime_per_snapshot_s", "setup_time_s")
# report column -> path in metrics.json "dso"["all"]
DSO_KEYS = {
    "v_hit_0p5": ("v", "hit_rate50"),
    "v_fa_0p5": ("v", "false_alarm_rate50"),
    "v_hit_0p1": ("v", "hit_rate10"),
    "v_fa_0p1": ("v", "false_alarm_rate10"),
    "v_brier": ("v", "brier"),
    "i_hit_0p5": ("i", "hit_rate50"),
    "i_fa_0p5": ("i", "false_alarm_rate50"),
    "i_hit_0p1": ("i", "hit_rate10"),
    "i_fa_0p1": ("i", "false_alarm_rate10"),
    "i_brier": ("i", "brier"),
    "max_v_err_p95": ("vm_max_abs_pu_p95",),
    "max_i_err_p95": ("branch_max_err_pct_p95",),
}
# direction of "better" for the win count; "attainable": closest to the
# attainable coverage, "one": closest to 1
BETTER = {
    "crps_kw": "lower",
    "crps_q_kvar": "lower",
    "vm_rmse_pu": "lower",
    "branch_err_pct": "lower",
    "coverage90": "attainable",
    "coverage90_q": "attainable",
    "chi2_norm": "one",
    "v_hit_0p5": "higher",
    "v_fa_0p5": "lower",
    "v_hit_0p1": "higher",
    "v_fa_0p1": "lower",
    "i_hit_0p5": "higher",
    "i_fa_0p5": "lower",
    "i_hit_0p1": "higher",
    "i_fa_0p1": "lower",
    "v_brier": "lower",
    "i_brier": "lower",
    "max_v_err_p95": "lower",
    "max_i_err_p95": "lower",
    "runtime_per_snapshot_s": "lower",
}
R2_METRICS = ("crps_q_kvar", "vm_rmse_pu", "branch_err_pct")
R3_METRICS = ("crps_kw", "crps_q_kvar", "coverage90")


def load(cell_dir):
    with open(os.path.join(cell_dir, "metrics.json")) as f:
        info = json.load(f)
    return info, pd.read_csv(os.path.join(cell_dir, "per_snapshot.csv"))


def need(d, *keys, where=""):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            raise SystemExit(f"{where}: metrics.json has no {'/'.join(keys)}")
        d = d[k]
    return d


def summary_row(label, info, table):
    row = {"method": label, "name": DISPLAY[label]}
    s_all = need(info, "summary", "all", where=label)
    for c in SUMMARY_MEANS:
        row[c] = need(s_all, c, where=label)
    d_all = need(info, "dso", "all", where=label)
    row["attainable"] = need(d_all, "coverage90_attainable", where=label)
    for c, keys in DSO_KEYS.items():
        row[c] = need(d_all, *keys, where=label)
    for c in TIMES:
        row[c] = need(info, c, where=label)
    row["num_snapshots"] = len(table)
    return row


def contrast(a: pd.DataFrame, b: pd.DataFrame, metric: str, subset: str) -> dict:
    """Paired day-block bootstrap of mean(a) - mean(b) for one metric."""
    a = a.set_index("snapshot")
    b = b.set_index("snapshot")
    if set(a.index) != set(b.index) or len(a) != len(b):
        raise SystemExit("cells to contrast hold different snapshots")
    a = a.loc[b.index]
    m = np.ones(len(a), bool) if subset == "all" else (a["hour_bin"] == subset).values
    r = paired_block_bootstrap(
        a[metric].to_numpy()[m], b[metric].to_numpy()[m], a["day_of_year"].to_numpy()[m]
    )
    return {
        "mean": r["mean_diff"],
        "lo": r["ci_low"],
        "hi": r["ci_high"],
        "comparator_mean": float(b[metric].to_numpy()[m].mean()),
        "num_snapshots": r["num_snapshots"],
        "num_blocks": r["num_blocks"],
    }


def verdict(r):
    if r["hi"] < 0:
        return "below 0"
    if r["lo"] > 0:
        return "above 0"
    return "includes 0"


def fmt(r, spec=".3f"):
    return f"{r['mean']:+{spec}} [{r['lo']:+{spec}}, {r['hi']:+{spec}}]"


def win_counts(summary: pd.DataFrame) -> tuple[dict, list]:
    """Best method per summary metric, and every metric where a baseline is
    strictly better than PoSE-fast."""
    best, baseline_better = {}, []
    s = summary.set_index("method")
    for c, rule in BETTER.items():
        v = s[c].astype(float)
        if v.isna().all():
            continue
        if rule == "lower":
            score = v
        elif rule == "higher":
            score = -v
        elif rule == "attainable":
            score = (v - s["attainable"].astype(float)).abs()
        else:
            score = (v - 1.0).abs()
        best[c] = str(score.idxmin())
        for m in BASELINES:
            if score[m] < score[POSE]:
                baseline_better.append(
                    {
                        "metric": c,
                        "rule": rule,
                        "baseline": m,
                        "baseline_value": float(v[m]),
                        "pose_value": float(v[POSE]),
                    }
                )
    return best, baseline_better


def physics_facts(args, infos, balanced_infos, data) -> dict:
    with open(args.physics) as f:
        feeder3 = ThreePhaseFeeder.from_dict(json.load(f))
    mset3 = feeder3.measurement_set("S2-3ph")
    feeder1 = Feeder.from_meta(data.meta)
    mset1 = feeder1.measurement_set("S2")
    noise3 = mset3.noise_std(NOISE_LEVELS["nominal"], feeder3)
    noise1 = mset1.noise_std(NOISE_LEVELS["nominal"], feeder1)
    lin3 = np.asarray(need(infos[POSE], "linearization_error", "std"))
    lin1 = np.asarray(need(balanced_infos[POSE], "linearization_error", "std"))
    pf3 = ThreePhasePowerFlow(feeder3)
    pf1 = RadialPowerFlow(feeder1)
    snaps = pd.read_csv(
        os.path.join(
            args.root, "cells", f"iowa_S2-3ph_nominal_{POSE}", "per_snapshot.csv"
        )
    )["snapshot"].to_numpy()
    truth = torch.as_tensor(data.p_kw[snaps], dtype=torch.float64)
    with torch.no_grad():
        r3, r1 = pf3(truth), pf1(truth)
    _, table = load(os.path.join(args.root, "cells", f"iowa_S2-3ph_nominal_{POSE}"))
    derived = feeder3.to_dict().get("derived", {})
    mask = np.asarray(derived.get("branch_phase_mask", []), dtype=bool)
    return {
        "num_channels": mset3.num_channels,
        "channel_names": list(mset3.channel_names),
        "num_node_phases": int(pf3.num_node_phases),
        "num_branch_phases_kept": int(pf3.num_branch_phases),
        "num_branch_phases_excluded": int((~mask).sum()) if mask.size else None,
        "min_kept_rating_a": float(np.min(ratings_3ph(pf3))),
        "linearization_error": {
            "three_phase": {
                "channels": list(mset3.channel_names),
                "lin_std": lin3.tolist(),
                "noise_std": np.asarray(noise3).tolist(),
                "ratio": (lin3 / np.asarray(noise3)).tolist(),
            },
            "balanced": {
                "channels": list(mset1.channel_names),
                "lin_std": lin1.tolist(),
                "noise_std": np.asarray(noise1).tolist(),
                "ratio": (lin1 / np.asarray(noise1)).tolist(),
            },
        },
        "true_voltage_range_pu": {
            "three_phase_node_phases": [float(r3.vm.min()), float(r3.vm.max())],
            "balanced_buses": [float(r1.vm.min()), float(r1.vm.max())],
        },
        "true_violations_three_phase": {
            "voltage_operational_node_phase_snapshots": float(table["v_true"].sum()),
            "voltage_statutory_node_phase_snapshots": float(table["vstat_true"].sum())
            if "vstat_true" in table
            else None,
            "overload_branch_phase_snapshots": float(table["i_true"].sum()),
        },
        "max_pf_mismatch_truth": float(r3.mismatch.max()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", required=True, help="study output root ($OUT)")
    p.add_argument("--data", required=True)
    p.add_argument("--physics", required=True)
    p.add_argument("--balanced", nargs="+", required=True, help="LABEL=DIR")
    p.add_argument("--stage1", default=None, help="probe_summary.json")
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    infos, tables = {}, {}
    for m in METHODS:
        d = os.path.join(args.root, "cells", f"iowa_S2-3ph_nominal_{m}")
        if not os.path.exists(os.path.join(d, "metrics.json")):
            raise SystemExit(f"missing cell {d}")
        infos[m], tables[m] = load(d)
    balanced = dict(s.split("=", 1) for s in args.balanced)
    if set(balanced) != set(METHODS):
        raise SystemExit(f"--balanced needs {METHODS}, got {sorted(balanced)}")
    b_infos, b_tables = {}, {}
    for m, d in balanced.items():
        b_infos[m], b_tables[m] = load(d)

    summary = pd.DataFrame([summary_row(m, infos[m], tables[m]) for m in METHODS])
    rows = []
    for subset in ("all", "midday"):
        for m in BASELINES:
            r = contrast(tables[POSE], tables[m], "crps_kw", subset)
            rows.append(
                {
                    "contrast": "R1",
                    "metric": "crps_kw",
                    "a": POSE,
                    "b": m,
                    "subset": subset,
                    **r,
                }
            )
            for metric in R2_METRICS:
                r = contrast(tables[POSE], tables[m], metric, subset)
                rows.append(
                    {
                        "contrast": "R2",
                        "metric": metric,
                        "a": POSE,
                        "b": m,
                        "subset": subset,
                        **r,
                    }
                )
        for m in METHODS:
            for metric in R3_METRICS:
                r = contrast(tables[m], b_tables[m], metric, subset)
                rows.append(
                    {
                        "contrast": "R3",
                        "metric": metric,
                        "a": f"{m} 3ph",
                        "b": f"{m} balanced",
                        "subset": subset,
                        **r,
                    }
                )
    contrasts = pd.DataFrame(rows)
    contrasts["verdict"] = [verdict(r) for r in contrasts.to_dict("records")]
    best, baseline_better = win_counts(summary)
    data = SnapshotData.load(args.data)
    facts = physics_facts(args, infos, b_infos, data)
    stage1 = None
    if args.stage1 and os.path.exists(args.stage1):
        with open(args.stage1) as f:
            stage1 = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    summary.to_csv(os.path.join(args.output_dir, "iowa_3ph_summary.csv"), index=False)
    contrasts.to_csv(
        os.path.join(args.output_dir, "iowa_3ph_contrasts.csv"), index=False
    )
    wins = {m: sum(v == m for v in best.values()) for m in METHODS}
    report = {
        "summary": summary.to_dict("records"),
        "contrasts": contrasts.to_dict("records"),
        "best_per_metric": best,
        "win_count": wins,
        "baseline_better": baseline_better,
        "better_rule": BETTER,
        "physics": facts,
        "stage1": stage1,
        "cells": {
            m: {
                "per_snapshot_sha256": file_sha256(
                    os.path.join(
                        args.root,
                        "cells",
                        f"iowa_S2-3ph_nominal_{m}",
                        "per_snapshot.csv",
                    )
                ),
                "git": infos[m].get("git"),
                "physics": infos[m].get("physics"),
            }
            for m in METHODS
        },
        "balanced_cells": balanced,
        "git": git_state(),
    }
    write_json(os.path.join(args.output_dir, "report.json"), report)

    lines = [
        "# Three-phase Iowa test: report",
        "",
        (
            "Iowa 240-bus "
            "feeder as an unbalanced three-phase model (S2-3ph, nominal noise), "
            f"{len(tables[POSE])} extended-design test snapshots. PoSE-fast = "
            "M2-euler-meas. Contrasts: paired day-block bootstrap 95% CIs (2,000 "
            "resamples, seed 0). The per-snapshot vm_* and branch_* columns run "
            "over node-phases and the kept branch-phases."
        ),
        "",
        "## Summary (three-phase, all snapshots)",
        "",
        (
            "| Method | CRPS P (kW) | CRPS Q (kvar) | V RMSE (p.u.) | Branch err (%) "
            "| Cov90 (attainable) | Cov90 Q | chi2 | s/snapshot |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in summary.to_dict("records"):
        lines.append(
            f"| {r['name']} | {r['crps_kw']:.3f} | {r['crps_q_kvar']:.3f} | "
            f"{r['vm_rmse_pu']:.5f} | {r['branch_err_pct']:.3f} | "
            f"{r['coverage90']:.3f} ({r['attainable']:.3f}) | {r['coverage90_q']:.3f} | "
            f"{r['chi2_norm']:.3f} | {r['runtime_per_snapshot_s']:.3f} |"
        )
    lines += [
        "",
        "## DSO metrics (three-phase, all snapshots)",
        "",
        "| Method | " + " | ".join(DSO_KEYS) + " |",
        "| --- " * (len(DSO_KEYS) + 1) + "|",
    ]
    for r in summary.to_dict("records"):
        lines.append(
            f"| {r['name']} | "
            + " | ".join("n/a" if pd.isna(r[c]) else f"{r[c]:.4g}" for c in DSO_KEYS)
            + " |"
        )
    for tag, title in (
        ("R1", "R1 (primary): CRPS(PoSE-fast) - CRPS(X), kW"),
        ("R2", "R2: PoSE-fast - X on other metrics"),
        ("R3", "R3 (descriptive): three-phase - balanced, per method"),
    ):
        lines += [
            "",
            f"## {title}",
            "",
            "| Metric | A | B | Subset | Mean [95% CI] | B mean | Verdict |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for r in contrasts[contrasts["contrast"] == tag].to_dict("records"):
            spec = ".5f" if r["metric"] == "vm_rmse_pu" else ".4f"
            lines.append(
                f"| {r['metric']} | {DISPLAY.get(r['a'], r['a'])} | "
                f"{DISPLAY.get(r['b'], r['b'])} | {r['subset']} | {fmt(r, spec)} | "
                f"{r['comparator_mean']:{spec}} | {r['verdict']} |"
            )
    lines += [
        "",
        (
            "R3 note: the three-phase set measures per-phase head flows, so "
            "it carries more information than the balanced set; R3 is not a "
            "physics-only effect."
        ),
        "",
        "## Win count",
        "",
        (
            "Rule: lower is better except hit rates (higher), coverage (closest "
            "to attainable) and chi2_norm (closest to 1)."
        ),
        "",
    ]
    lines += [f"- {DISPLAY[m]}: {wins[m]} of {len(best)}" for m in METHODS]
    lines += ["", "Metrics where a baseline is better than PoSE-fast:", ""]
    lines += [
        f"- {b['metric']}: {DISPLAY[b['baseline']]} {b['baseline_value']:.5g} vs "
        f"PoSE-fast {b['pose_value']:.5g}"
        for b in baseline_better
    ] or ["- none"]
    lin = facts["linearization_error"]
    lines += [
        "",
        "## Physics facts",
        "",
        f"- channels m = {facts['num_channels']}: " + ", ".join(facts["channel_names"]),
        (
            f"- node-phases {facts['num_node_phases']}, kept branch-phases "
            f"{facts['num_branch_phases_kept']}, excluded "
            f"{facts['num_branch_phases_excluded']}"
        ),
        (
            f"- true voltage range: three-phase "
            f"{facts['true_voltage_range_pu']['three_phase_node_phases']}, balanced "
            f"{facts['true_voltage_range_pu']['balanced_buses']} p.u."
        ),
        f"- true violations (three-phase): {facts['true_violations_three_phase']}",
        "",
        "| Model | Channel | lin. error std | noise std | ratio |",
        "| --- | --- | --- | --- | --- |",
    ]
    for model in ("three_phase", "balanced"):
        e = lin[model]
        for c, a, b, r in zip(
            e["channels"], e["lin_std"], e["noise_std"], e["ratio"], strict=False
        ):
            lines.append(f"| {model} | {c} | {a:.4g} | {b:.4g} | {r:.3f} |")
    if stage1 is not None:
        lines += [
            "",
            "## Stage 1 probe (quoted from probe_summary.json)",
            "",
            "```json",
            json.dumps(stage1, indent=1, default=str)[:20000],
            "```",
        ]
    with open(os.path.join(args.output_dir, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:40]), flush=True)


if __name__ == "__main__":
    main()

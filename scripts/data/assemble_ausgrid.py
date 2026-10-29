"""Assemble IEEE 33-bus snapshot data from the Ausgrid Solar Home
Electricity Data (300 homes with rooftop PV, 1 July 2010 to 30 June 2013,
half-hourly; CC BY 3.0 AU, Ausgrid).

1. Read the three yearly CSVs: per customer and day, 48 half-hour energies in
   kWh (interval ending at the column label, local Eastern time with daylight
   saving) for GC (general consumption), CL (controlled load, optional) and
   GG (gross PV generation). Net power drawn in kW = 2 (GC + CL - GG).
2. Keep 288 homes (32 nodes x 9): drop the 12 with the most missing GC or GG
   days. The 54 homes with the largest PV systems go to buses 13-18, the
   others are shuffled over the remaining nodes (seed 20261014).
3. A step is valid when every home of every node reports GC and GG; missing
   CL counts as zero.
4. Scale node i by k_i = P_nom_i / q99_i, the 99th percentile of the node's
   gross offtake (GC + CL) over valid training steps; net power is scaled by
   the same factor, as for the Fluvius data.
5. Calendar: 1,096 days x 48 steps; quarter_hour = 2 x half-hour index;
   day_of_year = day index 0-1095; split by day of month (days 8-14 test,
   22-28 validation).
--pv_scale multiplies gross generation (high-PV scenario: 3, about 5 kWp
per home); the scale factors k_i come from gross offtake and do not change.
--scenario names the data. The archive is
https://pierreh.eu/downloads/Ausgrid_solar_home_data.zip (SHA-256
5a766f52b6c8b3b72730380f4422e478934bc94640a4b089dd0e0e3c055c5d82); the
three CSV files and their SHA-256 are printed and checked against
EXPECTED_SHA256 below.
Checks against the Ausgrid notes (August 2014): mean annual consumption
(GC + CL) and gross generation per customer and year, and mean system size.

    python scripts/dsse/assemble_ausgrid.py --raw_dir data/ausgrid/raw/extracted \\
        --output data/dsse/ausgrid/snapshots.npz
"""

import argparse
import datetime as dt
import os

import numpy as np
import pandas as pd

from pose.data import SPLITS, TEST_DAYS, VAL_DAYS, SnapshotData
from pose.experiment import file_sha256, git_state
from pose.feeder import NUM_NODES, P_NOMINAL_KW, Q_NOMINAL_KVAR, node_of_bus


FILES = (
    "Solar home 2010-2011.csv",
    "Solar home 2011-2012.csv",
    "Solar home 2012-2013.csv",
)
START, NUM_DAYS, STEPS = dt.date(2010, 7, 1), 1096, 48
HOMES_PER_NODE = 9
PV_BUSES = (13, 14, 15, 16, 17, 18)
EXPECTED_SHA256 = {
    "Solar home 2010-2011.csv": "2fc1f615e22fbd8ec350cdb19f7465d3a038453fc2ffd19473ef1bd38fd28289",
    "Solar home 2011-2012.csv": "a7d6abd1369311a5566e00d249141fd8aec28e9faba0bc928788607d961fc8c8",
    "Solar home 2012-2013.csv": "e4be46b3c9991c65735a9545319a2f1c33be8e639c75a17a88e0323a8e5098be",
}
NOTES = {  # Ausgrid notes, August 2014: mean per customer
    "2010-11": {"consumption_kwh": 6980, "generation_kwh": 2119},
    "2011-12": {"consumption_kwh": 6596, "generation_kwh": 2083},
    "2012-13": {"consumption_kwh": 6387, "generation_kwh": 2181},
}


def parse_dates(col: pd.Series) -> pd.Series:
    out = pd.to_datetime(col, format="%d-%b-%y", errors="coerce")
    alt = pd.to_datetime(col, format="%d/%m/%Y", errors="coerce")
    out = out.fillna(alt)
    if out.isna().any():
        raise SystemExit(f"unparsed dates: {col[out.isna()].unique()[:5]}")
    return out.dt.date


def calendar(num_days: int) -> dict[str, np.ndarray]:
    days = [START + dt.timedelta(days=d) for d in range(num_days)]
    rep = lambda v: np.repeat(np.asarray(v), STEPS)
    day = rep([d.day for d in days])
    split = np.full(num_days * STEPS, SPLITS["train"])
    split[np.isin(day, TEST_DAYS)] = SPLITS["test"]
    split[np.isin(day, VAL_DAYS)] = SPLITS["val"]
    return {
        "month": rep([d.month - 1 for d in days]).astype(np.int64),
        "quarter_hour": np.tile(2 * np.arange(STEPS), num_days).astype(np.int64),
        "weekend": rep([int(d.weekday() >= 5) for d in days]).astype(np.int64),
        "day_of_month": day.astype(np.int64),
        "day_of_year": rep(np.arange(num_days)).astype(np.int64),
        "split": split.astype(np.int64),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--raw_dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=20261014)
    p.add_argument("--pv_scale", type=float, default=1.0)
    p.add_argument("--scenario", default="ausgrid")
    p.add_argument("--state", default="p", choices=["p", "pq"])
    args = p.parse_args()

    frames, sha = [], {}
    for name in FILES:
        path = os.path.join(args.raw_dir, name)
        sha[name] = file_sha256(path)
        if sha[name] != EXPECTED_SHA256[name]:
            raise SystemExit(
                f"{name}: SHA-256 {sha[name]} does not match {EXPECTED_SHA256[name]}"
            )
        d = pd.read_csv(path, skiprows=1, low_memory=False)
        d["day"] = parse_dates(d["date"])
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    value_cols = list(frames[0].columns[5:53])
    day_index = {START + dt.timedelta(days=i): i for i in range(NUM_DAYS)}
    d = d[d["day"].isin(day_index)]
    d["di"] = d["day"].map(day_index).astype(int)
    customers = np.sort(d["Customer"].unique())
    cap = d.groupby("Customer")["Generator Capacity"].first().reindex(customers)

    n_c = len(customers)
    cidx = {c: i for i, c in enumerate(customers)}
    arr = {cat: np.full((n_c, NUM_DAYS, STEPS), np.nan) for cat in ("GC", "CL", "GG")}
    for cat, sub in d.groupby("Consumption Category"):
        if cat not in arr:
            continue
        ci = sub["Customer"].map(cidx).to_numpy()
        arr[cat][ci, sub["di"].to_numpy()] = sub[value_cols].to_numpy(dtype=np.float64)
    missing = np.isnan(arr["GC"]).any(-1).sum(1) + np.isnan(arr["GG"]).any(-1).sum(1)

    # checks against the Ausgrid notes (all 300 customers, actual rows)
    checks = {}
    for k, label in enumerate(NOTES):
        lo, hi = (
            (dt.date(2010 + k, 7, 1) - START).days,
            (dt.date(2011 + k, 7, 1) - START).days,
        )
        cons = np.nansum(arr["GC"][:, lo:hi], (1, 2)) + np.nansum(
            arr["CL"][:, lo:hi], (1, 2)
        )
        gen = np.nansum(arr["GG"][:, lo:hi], (1, 2))
        checks[label] = {
            "mean_consumption_kwh": float(cons.mean()),
            "mean_generation_kwh": float(gen.mean()),
            "notes": NOTES[label],
        }
    checks["mean_capacity_kwp"] = float(cap.mean())

    keep = np.sort(np.argsort(missing, kind="stable")[: NUM_NODES * HOMES_PER_NODE])
    rng = np.random.default_rng(args.seed)
    by_cap = keep[np.argsort(-cap.to_numpy()[keep], kind="stable")]
    heavy_nodes = [node_of_bus(b) for b in PV_BUSES]
    homes = {}
    for j, node in enumerate(heavy_nodes):
        homes[node] = by_cap[j * HOMES_PER_NODE : (j + 1) * HOMES_PER_NODE]
    rest = rng.permutation(by_cap[len(heavy_nodes) * HOMES_PER_NODE :])
    others = [i for i in range(NUM_NODES) if i not in heavy_nodes]
    for j, node in enumerate(others):
        homes[node] = rest[j * HOMES_PER_NODE : (j + 1) * HOMES_PER_NODE]

    gc, cl, gg = (
        np.nan_to_num(arr["CL"]) if c == "CL" else arr[c] for c in ("GC", "CL", "GG")
    )
    T = NUM_DAYS * STEPS
    gross = np.zeros((T, NUM_NODES))
    net = np.zeros((T, NUM_NODES))
    valid = np.ones(T, dtype=bool)
    for node in range(NUM_NODES):
        h = homes[node]
        g = 2.0 * (gc[h] + cl[h]).sum(0).reshape(T)
        n = 2.0 * (gc[h] + cl[h] - args.pv_scale * gg[h]).sum(0).reshape(T)
        valid &= ~(np.isnan(g) | np.isnan(n))
        gross[:, node], net[:, node] = g, n
    cal = calendar(NUM_DAYS)
    train = valid & (cal["split"] == SPLITS["train"])
    q99 = np.quantile(gross[train], 0.99, axis=0)
    scale = np.asarray(P_NOMINAL_KW) / q99
    p_kw = np.where(valid[:, None], net * scale, np.nan)
    if args.state == "pq":
        # Joint (P, Q) state: load Q at the nominal case33bw power factor of
        # the node applied to its gross consumption, PV at unity power factor.
        tphi = np.asarray(Q_NOMINAL_KVAR) / np.asarray(P_NOMINAL_KW)
        p_kw = np.concatenate(
            [p_kw, np.where(valid[:, None], gross * scale * tphi, np.nan)], 1
        )

    meta = {
        "scenario": args.scenario,
        "pv_scale": args.pv_scale,
        "state": args.state,
        "source": "Ausgrid Solar Home Electricity Data (CC BY 3.0 AU), mirror "
        "https://pierreh.eu/downloads/Ausgrid_solar_home_data.zip",
        "raw_sha256": sha,
        "steps_per_day": STEPS,
        "num_days": NUM_DAYS,
        "valid_steps": int(valid.sum()),
        "invalid_steps": int((~valid).sum()),
        "steps_per_split": {
            s: int((valid & (cal["split"] == i)).sum()) for s, i in SPLITS.items()
        },
        "homes_per_node": HOMES_PER_NODE,
        "node_customers": [customers[homes[i]].tolist() for i in range(NUM_NODES)],
        "node_pv_kwp": [
            float(cap.to_numpy()[homes[i]].sum()) for i in range(NUM_NODES)
        ],
        "excluded_customers": sorted(
            set(customers.tolist()) - set(customers[keep].tolist())
        ),
        "missing_gc_gg_days_kept": int(missing[keep].sum()),
        "scale": scale.tolist(),
        "checks_vs_notes": checks,
        "node_mean_kw": np.nanmean(p_kw, 0).tolist(),
        "node_min_kw": np.nanmin(p_kw, 0).tolist(),
        "node_max_kw": np.nanmax(p_kw, 0).tolist(),
        "git": git_state(),
    }
    SnapshotData(p_kw, valid, cal, meta).save(args.output)
    print(
        {
            k: meta[k]
            for k in (
                "valid_steps",
                "invalid_steps",
                "steps_per_split",
                "missing_gc_gg_days_kept",
            )
        }
    )
    print("excluded", meta["excluded_customers"])
    print("checks", checks)
    print("node PV kWp", np.round(meta["node_pv_kwp"], 1).tolist())


if __name__ == "__main__":
    main()

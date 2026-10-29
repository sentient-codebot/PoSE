"""Assemble IEEE 33-bus snapshot data from the Fluvius 2024 archives.

Steps:
1. Read every meter's offtake and injection (kWh per 15 min) from the raw zip
   archives and map timestamps to the fixed UTC+1 clock (366 x 96 steps).
2. Assign the 2,400 homes to the 32 load nodes, 75 each, with a fixed seed.
   base: buses 13-18 get 60 PV and 15 non-PV homes each; the other homes are
   shuffled over the remaining 26 nodes. high_pv: all PV homes go to buses
   11-18 and 26-33, the non-PV homes to the other 16 nodes.
3. Node net power = sum over its homes of 4 * (offtake - injection) in kW.
   Missing readings: if at least 90% of a node's homes report, the sum is the
   mean of the reporting homes times the number of homes; otherwise the step
   is invalid for every node.
4. Scale node i by k_i = P_nom_i / q99_i, where q99_i is the 99th percentile
   of the node's gross offtake sum over valid training steps.

Outputs (--output_dir): {scenario}/snapshots.npz with _meta.json, and a cache
of the per-meter arrays (meters.npz) so a second scenario does not re-read the
archives.

The expected raw archives and their SHA-256 are printed by
scripts/data/download_fluvius.sh, which also downloads them.
"""

import argparse
import hashlib
import io
import os
import zipfile

import numpy as np
import pandas as pd

from pose.data import (
    NUM_STEPS,
    SPLITS,
    SnapshotData,
    calendar_arrays,
    split_hash,
)
from pose.feeder import (
    NUM_NODES,
    P_NOMINAL_KW,
    PV_BUSES_HIGH,
    PV_HEAVY_BUSES_BASE,
    Q_NOMINAL_KVAR,
    node_of_bus,
)


HOMES_PER_NODE = 75
PV_PER_HEAVY_NODE_BASE = 60
EXPECTED_SHA256 = {
    "P6269_1_50_DMK_Sample_Elek_2024_Deel_1.zip": "0f8addd57bd9f7bc0fd0c28b019a4b6c8753a19e3fd301de758f5a5858c9d86c",
    "P6269_1_50_DMK_Sample_Elek_2024_Deel_2.zip": "cfef37973c40e503fe53afa4966bbdcd7a12525fb8b20881f365d0ee8616a3d6",
}
MIN_REPORTING = 0.9
SCALE_QUANTILE = 0.99
YEAR = 2024
STEPS_PER_DAY = 96
COLUMNS = {
    "EAN_ID": "meter",
    "Datum_Startuur": "start",
    "Volume_Afname_kWh": "offtake_kwh",
    "Volume_Injectie_kWh": "injection_kwh",
    "Warmtepomp_Indicator": "heat_pump",
    "Elektrisch_Voertuig_Indicator": "ev",
    "PV-Installatie_Indicator": "pv",
}


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_member(zf: zipfile.ZipFile, name: str) -> pd.DataFrame:
    raw = zf.read(name)
    lower = name.lower()
    if lower.endswith(".parquet"):
        df = pd.read_parquet(io.BytesIO(raw))
    elif lower.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(raw))
    else:
        head = raw[:4096].decode("utf-8-sig", errors="replace")
        sep = ";" if head.count(";") > head.count(",") else ","
        decimal = "," if sep == ";" and "," in head.split("\n", 2)[-1] else "."
        df = pd.read_csv(
            io.BytesIO(raw), sep=sep, decimal=decimal, encoding="utf-8-sig"
        )
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(
        columns={
            "Volume_Afname_KWh": "Volume_Afname_kWh",
            "Volume_Injectie_KWh": "Volume_Injectie_kWh",
            "PV_Installatie_Indicator": "PV-Installatie_Indicator",
        }
    )
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"{name}: missing columns {missing}; found {list(df.columns)}")
    df = df[list(COLUMNS)].rename(columns=COLUMNS)
    if lower.endswith(".csv"):
        # The CSV exports append Z to local CET/CEST readings. The legend
        # specifies local time; retain those clock values before DST handling.
        df["start"] = pd.to_datetime(df["start"], errors="raise").dt.tz_localize(None)
    return df


def to_fixed_utc_plus_one(start: pd.Series) -> pd.Series:
    """Map CET/CEST timestamps to a fixed UTC+1 clock."""
    if isinstance(start.dtype, pd.DatetimeTZDtype):
        ts = start
    else:
        parsed = pd.to_datetime(start, errors="raise")
        if parsed.dt.tz is not None:
            ts = parsed
        else:
            # The repeated October hour appears twice; the first occurrence of
            # each repeated timestamp is summer time (CEST), the second CET.
            occurrence = parsed.groupby(parsed).cumcount().to_numpy()
            ts = parsed.dt.tz_localize(
                "Europe/Brussels", ambiguous=occurrence == 0, nonexistent="raise"
            )
    return ts.dt.tz_convert("UTC").dt.tz_localize(None) + pd.Timedelta(hours=1)


def read_meters(raw_dir: str, max_meters: int | None = None) -> dict:
    zips = sorted(
        os.path.join(raw_dir, f)
        for f in os.listdir(raw_dir)
        if f.lower().endswith(".zip")
    )
    if not zips:
        raise FileNotFoundError(f"no zip archives in {raw_dir}")
    ids, attrs, offtake, injection, members = [], [], [], [], []
    start = pd.Timestamp(YEAR, 1, 1)
    for path in zips:
        digest = sha256(path)
        expected = EXPECTED_SHA256.get(os.path.basename(path))
        if expected and digest != expected:
            raise SystemExit(
                f"{os.path.basename(path)}: SHA-256 {digest} does not match {expected}"
            )
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                if name.endswith("/") or name.startswith("__MACOSX"):
                    continue
                df = read_member(zf, name)
                members.append({"archive": os.path.basename(path), "member": name})
                for meter, sub in df.groupby("meter", sort=False):
                    if max_meters and len(ids) >= max_meters:
                        break
                    if not isinstance(sub["start"].dtype, pd.DatetimeTZDtype):
                        sub = sub.assign(start=pd.to_datetime(sub["start"]))
                    sub = sub.sort_values("start", kind="stable")
                    sub = sub.reset_index(drop=True)
                    t = to_fixed_utc_plus_one(sub["start"])
                    idx = ((t - start) / pd.Timedelta(minutes=15)).to_numpy()
                    idx = idx.astype(np.int64)
                    ok = (idx >= 0) & (idx < NUM_STEPS)
                    if len(np.unique(idx[ok])) != ok.sum():
                        raise ValueError(f"meter {meter}: duplicate timestamps")
                    off = np.full(NUM_STEPS, np.nan, dtype=np.float32)
                    inj = np.full(NUM_STEPS, np.nan, dtype=np.float32)
                    off[idx[ok]] = sub["offtake_kwh"].to_numpy()[ok]
                    inj[idx[ok]] = sub["injection_kwh"].to_numpy()[ok]
                    ids.append(str(meter))
                    attrs.append(
                        [int(sub[c].iloc[0]) for c in ("pv", "heat_pump", "ev")]
                    )
                    offtake.append(off)
                    injection.append(inj)
                print(f"{name}: {len(ids)} meters", flush=True)
    return {
        "meter_id": np.asarray(ids),
        "attrs": np.asarray(attrs, dtype=np.int64),
        "offtake_kwh": np.stack(offtake),
        "injection_kwh": np.stack(injection),
        "source_archives": {os.path.basename(p): sha256(p) for p in zips},
        "members": members,
    }


def compose(pv: np.ndarray, scenario: str, seed: int) -> list[np.ndarray]:
    """Meter indices per node (75 each)."""
    rng = np.random.default_rng(seed)
    pv_idx = rng.permutation(np.flatnonzero(pv == 1))
    other_idx = rng.permutation(np.flatnonzero(pv == 0))
    need = NUM_NODES * HOMES_PER_NODE
    if len(pv_idx) + len(other_idx) < need:
        raise ValueError(f"need {need} meters, have {len(pv_idx) + len(other_idx)}")
    homes: list[np.ndarray | None] = [None] * NUM_NODES
    if scenario == "base":
        heavy = [node_of_bus(b) for b in PV_HEAVY_BUSES_BASE]
        n_pv, n_other = PV_PER_HEAVY_NODE_BASE, HOMES_PER_NODE - PV_PER_HEAVY_NODE_BASE
        for k, node in enumerate(heavy):
            homes[node] = np.concatenate(
                [
                    pv_idx[k * n_pv : (k + 1) * n_pv],
                    other_idx[k * n_other : (k + 1) * n_other],
                ]
            )
        rest = rng.permutation(
            np.concatenate(
                [pv_idx[len(heavy) * n_pv :], other_idx[len(heavy) * n_other :]]
            )
        )
        others = [i for i in range(NUM_NODES) if i not in heavy]
    elif scenario == "high_pv":
        pv_nodes = [node_of_bus(b) for b in PV_BUSES_HIGH]
        pool = np.concatenate([pv_idx, other_idx])  # PV homes first
        for k, node in enumerate(pv_nodes):
            homes[node] = pool[k * HOMES_PER_NODE : (k + 1) * HOMES_PER_NODE]
        rest = rng.permutation(pool[len(pv_nodes) * HOMES_PER_NODE :])
        others = [i for i in range(NUM_NODES) if i not in pv_nodes]
    else:
        raise ValueError(scenario)
    for k, node in enumerate(others):
        homes[node] = rest[k * HOMES_PER_NODE : (k + 1) * HOMES_PER_NODE]
    out = [np.sort(h) for h in homes]
    used = np.concatenate(out)
    assert len(np.unique(used)) == need and all(len(h) == HOMES_PER_NODE for h in out)
    return out


def aggregate(meters: dict, homes: list[np.ndarray]):
    """Node net and gross power (kW) and the per-step validity."""
    net = np.empty((NUM_STEPS, NUM_NODES))
    gross = np.empty((NUM_STEPS, NUM_NODES))
    reporting = np.empty((NUM_STEPS, NUM_NODES))
    for i, h in enumerate(homes):
        off = 4.0 * meters["offtake_kwh"][h].astype(np.float64)
        inj = 4.0 * meters["injection_kwh"][h].astype(np.float64)
        p = off - inj
        ok = np.isfinite(p)
        count = ok.sum(0)
        reporting[:, i] = count / len(h)
        with np.errstate(invalid="ignore", divide="ignore"):
            net[:, i] = np.where(ok, p, 0.0).sum(0) / count * len(h)
            gross[:, i] = np.where(ok, off, 0.0).sum(0) / count * len(h)
    valid = (reporting >= MIN_REPORTING).all(1)
    return net, gross, valid, reporting


def build(
    meters: dict,
    scenario: str,
    seed: int,
    state: str = "p",
    homes: list[np.ndarray] | None = None,
    valid: np.ndarray | None = None,
) -> SnapshotData:
    """Snapshot data of one composition. homes: precomputed meter indices
    per node (default compose()); valid: validity override, a subset of the
    composition's own valid steps (default the 90% reporting rule)."""
    if homes is None:
        homes = compose(meters["attrs"][:, 0], scenario, seed)
    net, gross, own_valid, reporting = aggregate(meters, homes)
    if valid is None:
        valid = own_valid
    else:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != own_valid.shape or (valid & ~own_valid).any():
            raise ValueError("validity override must be a subset of the valid steps")
    cal = calendar_arrays()
    train = valid & (cal["split"] == SPLITS["train"])
    q = np.quantile(gross[train], SCALE_QUANTILE, axis=0)
    scale = np.asarray(P_NOMINAL_KW) / q
    p_kw = np.where(valid[:, None], net * scale, np.nan)
    if state == "pq":
        # Joint (P, Q) state: load Q at the nominal case33bw power factor of
        # the node applied to the metered offtake (gross consumption is not
        # metered), PV export at unity power factor.
        tphi = np.asarray(Q_NOMINAL_KVAR) / np.asarray(P_NOMINAL_KW)
        q_kvar = np.where(valid[:, None], gross * scale * tphi, np.nan)
        p_kw = np.concatenate([p_kw, q_kvar], axis=1)
    partial = valid[:, None] & (reporting < 1.0)
    meta = {
        "dataset": "Fluvius 2024 15-min digital meter profiles (2,400 homes)",
        "scenario": scenario,
        "seed": seed,
        "homes_per_node": HOMES_PER_NODE,
        "min_reporting_fraction": MIN_REPORTING,
        "scale_quantile": SCALE_QUANTILE,
        "scale_factor": scale.tolist(),
        "gross_q99_kw_unscaled": q.tolist(),
        "pv_homes_per_node": [int(meters["attrs"][h, 0].sum()) for h in homes],
        "heat_pump_homes_per_node": [int(meters["attrs"][h, 1].sum()) for h in homes],
        "ev_homes_per_node": [int(meters["attrs"][h, 2].sum()) for h in homes],
        "node_meter_ids": [meters["meter_id"][h].tolist() for h in homes],
        "valid_steps": int(valid.sum()),
        "invalid_steps": int((~valid).sum()),
        "rescaled_node_steps": int(partial.sum()),
        "steps_per_split": {
            k: int((valid & (cal["split"] == v)).sum()) for k, v in SPLITS.items()
        },
        "state": state,
        "node_mean_kw": np.nanmean(p_kw, 0).tolist(),
        "node_min_kw": np.nanmin(p_kw, 0).tolist(),
        "node_max_kw": np.nanmax(p_kw, 0).tolist(),
        "fraction_negative": np.nanmean(p_kw < 0, 0).tolist(),
        "time_axis": "fixed UTC+1 (CET without daylight saving), 2024",
        "split_protocol": "test days 8-14, validation days 22-28, else train",
        "split_sha256": split_hash(),
        "source_archives": meters["source_archives"],
    }
    data = SnapshotData(p_kw, valid, cal, meta)
    data.meta["p_kw_sha256"] = hashlib.sha256(np.ascontiguousarray(p_kw)).hexdigest()
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw_dir", default="data/fluvius/raw")
    parser.add_argument("--output_dir", default="data/fluvius")
    parser.add_argument("--scenarios", nargs="+", default=["base", "high_pv"])
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--max_meters", type=int, default=None, help="debug")
    parser.add_argument("--state", default="p", choices=["p", "pq"])
    parser.add_argument(
        "--cache", default=None, help="meters cache (default output_dir/meters.npz)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="single-scenario output file (default output_dir/<scenario>/snapshots.npz)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cache = args.cache or os.path.join(args.output_dir, "meters.npz")
    if os.path.exists(cache) and args.max_meters is None:
        loaded = np.load(cache, allow_pickle=True)
        meters = {k: loaded[k] for k in loaded.files}
        meters["source_archives"] = meters["source_archives"].item()
    else:
        meters = read_meters(args.raw_dir, args.max_meters)
        if args.max_meters is None:
            np.savez(
                cache,
                **{k: v for k, v in meters.items() if k != "members"},
            )
    print(
        f"meters {len(meters['meter_id'])}, PV {int(meters['attrs'][:, 0].sum())}, "
        f"missing readings {np.isnan(meters['offtake_kwh']).mean():.5f}",
        flush=True,
    )
    if args.output and len(args.scenarios) != 1:
        raise SystemExit("--output needs exactly one scenario")
    for scenario in args.scenarios:
        data = build(meters, scenario, args.seed, args.state)
        path = args.output or os.path.join(args.output_dir, scenario, "snapshots.npz")
        data.save(path)
        m = data.meta
        print(
            scenario,
            {
                k: m[k]
                for k in ("valid_steps", "rescaled_node_steps", "steps_per_split")
            },
            flush=True,
        )


if __name__ == "__main__":
    main()

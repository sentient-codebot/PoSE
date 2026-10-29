"""Iowa 240-bus distribution test system (Bu et al., NAPS 2019) as a balanced
radial feeder with hourly P and Q at the primary buses.

1. Topology from the OpenDSS model: lines and closed breakers from bus1 (the
   13.8 kV bus after the substation regulators); breakers to the .4.5.6
   dummy nodes are normally open and left out. Dead ends without load or
   capacitor are pruned (they never carry current).
2. Positive-sequence impedance per line code (self minus mutual for three
   phases; self impedance for one phase), times 3 / n_phases for one- and
   two-phase laterals, so the balanced equivalent keeps the voltage drop of
   the served phase. Breakers: 1e-4 ohm.
3. Slack voltage 1.025 p.u. (regulator set point 123 V on a 120 V base); the
   two 50 kvar capacitors are fixed reactive injections.
4. State nodes: buses with load. P and Q per hour from "Calculated Nodal
   P&Q.xlsx" (2017; hour-ending timestamps, the interval starts one hour
   earlier); joint (P, Q) state. Nominal P per node: 99th percentile of its
   training P; nominal Q: P_nom times the node's median Q / P.
5. Measurement set S2: head P and Q and the voltage magnitude at the
   electrically farthest leaf of each of the three feeders.
6. Calendar: 365 days x 24 hours, quarter_hour = 4 x hour; split by day of
   month (days 8-14 test, 22-28 validation).

Download OpenDSS Model.zip (SHA-256
146c9a5278439422e3ee7bcea51e86b3a78de97484e6ff2d9c353bd212d44da5) and
Smart Meter Data.zip (SHA-256
11d6b19fc049c113dcf7c21324504848e99e8c871622e311a86cb2b3a59ad9a0) from
https://wzy.ece.iastate.edu/Testsystem.html and extract both into --raw_dir.

    python scripts/data/assemble_iowa.py --raw_dir data/iowa/raw/extracted \
        --output data/dsse/iowa/snapshots.npz
"""

import argparse
import datetime as dt
import os
import re
from collections import defaultdict, deque

import numpy as np

from pose.data import SPLITS, TEST_DAYS, VAL_DAYS, SnapshotData
from pose.experiment import file_sha256, git_state
from pose.feeder import Feeder


MODEL = "OpenDSS Model (08.13.2020)"
LOADS = "Smart Meter Data (06.17.2019)/Calculated Nodal P&Q.xlsx"
V_BASE_KV, S_BASE_KVA, V_SLACK = 13.8, 1000.0, 1.025
CAPACITORS = {"bus2038": 50.0, "bus3079": 50.0}
FT_PER_MI = 5280.0
EXPECTED_SHA256 = {
    LOADS: "c950f0801294bc3b5ca9e843d77843f7e5e19a7f55064196f499b5d8594e8ca6",
    os.path.join(
        MODEL, "Line.dss"
    ): "6cf001c67dba345cc1d60d5f0d85cfd1e14b42c88c0a31447783338b9060965c",
    os.path.join(
        MODEL, "Linecode.dss"
    ): "f40aeeead8c17fc8efc2938afb1651505afc848ce628338b28743453315817fb",
}


def statements(path: str) -> list[str]:
    """OpenDSS statements with '~' continuations joined and comments removed."""
    out = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    for raw in lines:
        line = re.split(r"!|//", raw, maxsplit=1)[0].strip()
        if not line:
            continue
        if line.startswith("~") and out:
            out[-1] += " " + line[1:]
        else:
            out.append(line)
    return out


def props(stmt: str) -> dict[str, str]:
    stmt = re.sub(r"\(\s*", "(", re.sub(r"\s*\)", ")", stmt))
    d = {}
    for m in re.finditer(r"(\w+)\s*=\s*(\([^)]*\)|\S+)", stmt):
        d[m.group(1).lower()] = m.group(2)
    return d


def matrix(text: str) -> np.ndarray:
    rows = [list(map(float, r.split())) for r in text.strip("()").split("|")]
    n = len(rows)
    m = np.zeros((n, n))
    for i, r in enumerate(rows):
        m[i, : len(r)] = r
        m[: len(r), i] = r
    return m


def z1_per_mile(rm: np.ndarray, xm: np.ndarray) -> complex:
    z = rm + 1j * xm
    n = len(z)
    if n == 1:
        return complex(z[0, 0])
    off = (z.sum() - np.trace(z)) / (n * (n - 1))
    return complex(np.trace(z) / n - off)


def bus_name(spec: str) -> tuple[str, str]:
    name, *phases = spec.lower().split(".")
    return name, ".".join(phases)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--raw_dir", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    model = os.path.join(args.raw_dir, MODEL)
    for rel, expected in EXPECTED_SHA256.items():
        path = os.path.join(args.raw_dir, rel)
        if file_sha256(path) != expected:
            raise SystemExit(f"{rel}: SHA-256 does not match {expected}")

    codes = {}
    for st in statements(os.path.join(model, "Linecode.dss")):
        if st.lower().startswith("new linecode."):
            name = st.split()[1].split(".", 1)[1].lower()
            d = props(st)
            codes[name] = (
                z1_per_mile(matrix(d["rmatrix"]), matrix(d["xmatrix"])),
                int(d["nphases"]),
            )

    edges = []
    for fname in ("Line.dss", "CircuitBreaker.dss"):
        for st in statements(os.path.join(model, fname)):
            if not st.lower().startswith("new line."):
                continue
            d = props(st)
            b1, _ = bus_name(d["bus1"])
            b2, ph2 = bus_name(d["bus2"])
            if "4.5.6" in ph2:
                continue  # normally open breaker to a dummy node
            if "linecode" in d:
                z, nph = codes[d["linecode"].lower()]
                length = float(d["length"]) / (
                    FT_PER_MI if d.get("units", "ft").lower() == "ft" else 1.0
                )
                z = z * length * 3.0 / nph
            else:
                z = complex(float(d.get("r1", 1e-4)), float(d.get("x1", 0.0)))
            edges.append((b1, b2, z))

    adj = defaultdict(list)
    for a, b, z in edges:
        adj[a].append((b, z))
        adj[b].append((a, z))
    order, parent_of, z_of = ["bus1"], {}, {}
    queue, seen = deque(["bus1"]), {"bus1"}
    while queue:
        a = queue.popleft()
        for b, z in adj[a]:
            if b not in seen:
                seen.add(b)
                parent_of[b], z_of[b] = a, z
                order.append(b)
                queue.append(b)
    if len(seen) != len(adj):
        raise SystemExit(f"{len(adj) - len(seen)} buses are not connected to bus1")

    import openpyxl

    wb = openpyxl.load_workbook(
        os.path.join(args.raw_dir, LOADS), read_only=True, data_only=True
    )
    series, stamps = {}, None
    for feeder in ("A", "B", "C"):
        for kind in ("P", "Q"):
            rows = list(wb[f"Feeder{feeder}_{kind}"].iter_rows(values_only=True))
            head = rows[0]
            times = [r[0] for r in rows[1:] if r[0] is not None]
            data = np.array(
                [
                    [float(v or 0.0) for v in r[1 : len(head)]]
                    for r in rows[1 : 1 + len(times)]
                ]
            )
            stamps = stamps or times
            if times != stamps:
                raise SystemExit("time axes of the sheets differ")
            for j, label in enumerate(head[1:]):
                bus = "bus" + str(label).split()[-1]
                series[(bus, kind)] = data[:, j]
    hours = len(stamps)
    start = stamps[0] - dt.timedelta(hours=1)  # hour-ending stamps
    if start != dt.datetime(2017, 1, 1, 0, 0) or hours != 8760:
        raise SystemExit(f"unexpected time axis: first {stamps[0]}, {hours} hours")

    load_buses = [
        b for b in order[1:] if (b, "P") in series and series[(b, "P")].max() > 0
    ]
    # Dead ends without load or capacitor carry no current in any snapshot
    # (zero rating, 0/0 in the branch metrics); prune them, leaves first.
    keep = set(load_buses) | set(CAPACITORS)
    kids = defaultdict(set)
    for b in order[1:]:
        kids[parent_of[b]].add(b)
    pruned = []
    for b in reversed(order[1:]):
        if not kids[b] and b not in keep:
            pruned.append(b)
            kids[parent_of[b]].discard(b)
    order = [b for b in order if b not in set(pruned)]
    index = {b: k for k, b in enumerate(order)}
    p_kw = np.stack([series[(b, "P")] for b in load_buses], 1)
    q_kvar = np.stack([series[(b, "Q")] for b in load_buses], 1)

    days = [dt.date(2017, 1, 1) + dt.timedelta(days=d) for d in range(365)]
    rep = lambda v: np.repeat(np.asarray(v), 24)
    day = rep([d.day for d in days])
    split = np.full(hours, SPLITS["train"])
    split[np.isin(day, TEST_DAYS)] = SPLITS["test"]
    split[np.isin(day, VAL_DAYS)] = SPLITS["val"]
    cal = {
        "month": rep([d.month - 1 for d in days]).astype(np.int64),
        "quarter_hour": np.tile(4 * np.arange(24), 365).astype(np.int64),
        "weekend": rep([int(d.weekday() >= 5) for d in days]).astype(np.int64),
        "day_of_month": day.astype(np.int64),
        "day_of_year": rep(np.arange(365)).astype(np.int64),
        "split": split.astype(np.int64),
    }
    train = split == SPLITS["train"]
    p_nom = np.quantile(p_kw[train], 0.99, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(p_kw > 0, q_kvar / p_kw, np.nan)
    tphi = np.nanmedian(ratio, axis=0)

    path_len = {}
    for b in order[1:]:
        path_len[b] = path_len.get(parent_of[b], 0.0) + abs(z_of[b])
    children = {parent_of[b] for b in order[1:]}
    ends = []
    for prefix in ("bus1", "bus2", "bus3"):
        leaves = [
            b
            for b in order[1:]
            if b not in children and b.startswith(prefix) and len(b) == 7
        ]
        ends.append(max(leaves, key=path_len.get))

    feeder = Feeder(
        name="iowa240",
        parent=tuple(index[parent_of[b]] for b in order[1:]),
        r_ohm=tuple(z_of[b].real for b in order[1:]),
        x_ohm=tuple(z_of[b].imag for b in order[1:]),
        load_buses=tuple(index[b] for b in load_buses),
        p_nominal=tuple(p_nom.tolist()),
        q_nominal=tuple((p_nom * tphi).tolist()),
        bus_labels=tuple(order),
        v_base_kv=V_BASE_KV,
        s_base_kva=S_BASE_KVA,
        v_slack=V_SLACK,
        measurement_sets=(("S2", tuple(ends), ()),),
        shunt_q_kvar=tuple(-CAPACITORS.get(b, 0.0) for b in order[1:]),
    )
    x = np.concatenate([p_kw, q_kvar], axis=1)
    meta = {
        "scenario": "iowa",
        "state": "pq",
        "feeder": feeder.to_dict(),
        "source": "Iowa 240-bus distribution test system (Bu et al., NAPS 2019), "
        "http://wzy.ece.iastate.edu/Testsystem.html",
        "raw_sha256": {
            f: file_sha256(os.path.join(args.raw_dir, f))
            for f in (
                LOADS,
                os.path.join(MODEL, "Line.dss"),
                os.path.join(MODEL, "Linecode.dss"),
            )
        },
        "num_buses": len(order) - 1,
        "num_load_nodes": len(load_buses),
        "zero_injection_buses": [b for b in order[1:] if b not in set(load_buses)],
        "pruned_dead_end_buses": sorted(pruned),
        "feeder_ends": ends,
        "steps_per_day": 24,
        "time_axis": "2017, hourly, hour-ending stamps shifted to interval start; local clock of the source",
        "steps_per_split": {k: int((split == v).sum()) for k, v in SPLITS.items()},
        "node_p_nominal_kw": p_nom.tolist(),
        "node_tan_phi": tphi.tolist(),
        "git": git_state(),
    }
    SnapshotData(x, np.ones(hours, dtype=bool), cal, meta).save(args.output)
    print(
        {
            k: meta[k]
            for k in ("num_buses", "num_load_nodes", "feeder_ends", "steps_per_split")
        }
    )
    print(
        "zero-injection buses:",
        len(meta["zero_injection_buses"]),
        "pruned dead ends:",
        meta["pruned_dead_end_buses"],
        "total P mean/max kW:",
        round(float(p_kw.sum(1).mean()), 1),
        round(float(p_kw.sum(1).max()), 1),
    )


if __name__ == "__main__":
    main()

"""Iowa 240-bus distribution test system (Bu et al., NAPS 2019) as an
unbalanced three-phase radial feeder.

Parses the OpenDSS text files (no OpenDSS engine) with the helpers of
assemble_iowa.py and writes a ThreePhaseFeeder (pose.threephase) whose buses,
tree, state nodes and nominal loading are those of the balanced model in
snapshots_meta.json. snapshots.npz is not touched.

1. Topology as in assemble_iowa.py: lines and closed breakers from bus1,
   normally open breakers to the .4.5.6 dummy nodes left out, dead ends
   without load or capacitor pruned. Bus set, parents, load buses and pruned
   buses are checked against the meta; any difference is an error.
2. Branch k: the phases of its bus2 spec and the n x n line-code matrices
   (ohm per mile) times length (ft unless ``units`` says mi), rows and
   columns ordered a, b, c. Breakers: diagonal 1e-4 ohm on their phases.
   The positive-sequence equivalent of each three-phase branch is checked
   against the meta's r_ohm / x_ohm.
3. Loads: each load bus T_<bus>_L hangs off a distribution transformer whose
   winding 1 sits on primary bus <bus>; that winding's phases and connection
   define the load's connections (wye: one connection per phase, share 1/n;
   delta: one ordered pair for two phases, ab, bc, ca for three). A load's
   share of its node is proportional to its nominal kW.
4. Capacitors: fixed reactive injection with the meta's total per bus, split
   equally over the connections of Capacitor.dss.
5. Measurement set S2-3ph: the S2 voltage buses of the meta, all phases.

    python scripts/data/assemble_iowa_3ph.py \
        --raw_dir "data/iowa/raw/extracted/OpenDSS Model (08.13.2020)" \
        --meta data/iowa/snapshots_meta.json --output feeder3ph.json"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict, deque

import numpy as np
import torch
from assemble_iowa import FT_PER_MI, bus_name, matrix, props, statements

from pose.experiment import file_sha256, git_state
from pose.threephase import (
    MIN_RATING_A,
    Capacitor,
    Connection,
    ThreePhaseFeeder,
    ThreePhasePowerFlow,
    ratings_3ph,
)


RAW_FILES = (
    "Master.dss",
    "Line.dss",
    "Linecode.dss",
    "Load.dss",
    "Capacitor.dss",
    "CircuitBreaker.dss",
    "DistriTransformer.dss",
)
BREAKER_R_OHM = 1e-4
SLACK = "bus1"


LOW_RATING_REPORT_A = 0.5  # kept ratings below this are listed (record only)


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def is_new(stmt: str, cls: str) -> bool:
    """True for a 'New <cls>.<name>' statement (any spacing and case)."""
    words = stmt.split(maxsplit=2)
    return (
        len(words) > 1
        and words[0].lower() == "new"
        and words[1].lower().startswith(cls + ".")
    )


def phase_list(spec: str, default: int) -> list[int]:
    """OpenDSS node spec ('1.2.3', '2.0', '') -> 0-based phases in spec
    order, ground (0) dropped; empty spec -> the first ``default`` phases."""
    nodes = [int(t) for t in spec.split(".") if t]
    if not nodes:
        return list(range(default))
    if any(t > 3 for t in nodes):
        raise ValueError(f"node spec {spec} has non-phase nodes")
    return [t - 1 for t in nodes if t != 0]


def wdg_specs(stmt: str) -> list[tuple[str, str, str]]:
    """(bus spec, conn, kv) per winding of a Transformer statement."""
    out = []
    for part in stmt.split("wdg=")[1:]:
        d = props("wdg=" + part)
        out.append((d["bus"], d.get("conn", "wye").lower(), d.get("kv", "")))
    return out


def connections_of(kind: str, phases: list[int], share: float) -> list[tuple]:
    """(kind, phases, share) of a wye or delta element on the given phases."""
    if kind == "wye":
        return [("wye", (p,), share / len(phases)) for p in phases]
    if len(phases) == 2:
        return [("delta", tuple(phases), share)]
    if sorted(phases) == [0, 1, 2]:
        return [("delta", pq, share / 3.0) for pq in ((0, 1), (1, 2), (2, 0))]
    raise ValueError(f"delta element on phases {phases}")


def parse_lines(model: str) -> list[dict]:
    codes = {}
    for st in statements(os.path.join(model, "Linecode.dss")):
        if is_new(st, "linecode"):
            d = props(st)
            if d.get("units", "mi").lower() != "mi":
                fail(f"line code {st.split()[1]} not per mile")
            codes[st.split()[1].split(".", 1)[1].lower()] = (
                matrix(d["rmatrix"]),
                matrix(d["xmatrix"]),
                int(d["nphases"]),
            )
    lines = []
    for fname in ("Line.dss", "CircuitBreaker.dss"):
        for st in statements(os.path.join(model, fname)):
            if not is_new(st, "line"):
                continue
            d = props(st)
            name = st.split()[1].split(".", 1)[1].lower()
            b1, ph1 = bus_name(d["bus1"])
            b2, ph2 = bus_name(d["bus2"])
            if "4.5.6" in ph2:
                continue  # normally open breaker to a dummy node
            if "linecode" in d:
                rm, xm, n = codes[d["linecode"].lower()]
                units = d.get("units", "ft").lower()
                if units not in ("ft", "mi"):
                    fail(f"line {name}: unsupported units {units}")
                length = float(d["length"]) / (FT_PER_MI if units == "ft" else 1.0)
                rm, xm = rm * length, xm * length
                declared = int(d.get("phases", n))
            else:  # breaker: diagonal r1 on its phases
                n = int(d.get("phases", 3))
                rm = np.eye(n) * float(d.get("r1", BREAKER_R_OHM))
                xm = np.eye(n) * float(d.get("x1", 0.0))
                declared = n
            p1, p2 = phase_list(ph1, n), phase_list(ph2, n)
            if p1 != p2 or len(p2) != n:
                fail(f"line {name}: phases {ph1} / {ph2} vs {n} conductors")
            order = np.argsort(p2)  # rows of the matrix follow the spec order
            lines.append(
                {
                    "name": name,
                    "bus1": b1,
                    "bus2": b2,
                    "phases": [p2[i] for i in order],
                    "r": rm[np.ix_(order, order)],
                    "x": xm[np.ix_(order, order)],
                    "declared_phases": declared,
                    "linecode": d.get("linecode", "").lower(),
                }
            )
    return lines


def parse_loads(model: str) -> tuple[list[dict], Counter]:
    """Loads with their primary bus and the primary-side phases and
    connection of their distribution transformer."""
    primary = {}
    for st in statements(os.path.join(model, "DistriTransformer.dss")):
        if not is_new(st, "transformer"):
            continue
        wdgs = wdg_specs(st)
        pbus, pspec = bus_name(wdgs[0][0])
        for spec, _, _ in wdgs[1:]:
            primary[bus_name(spec)[0]] = (pbus, pspec, wdgs[0][1], st.split()[1])
    loads, agree = [], Counter()
    for st in statements(os.path.join(model, "Load.dss")):
        if not is_new(st, "load"):
            continue
        d = props(st)
        lbus, lspec = bus_name(d["bus1"])
        if lbus not in primary:
            fail(f"load bus {lbus} has no distribution transformer")
        pbus, pspec, pconn, xfmr = primary[lbus]
        if pbus != lbus.replace("t_", "").replace("_l", ""):
            fail(f"load bus {lbus}: transformer primary {pbus} differs from name")
        nph = int(d.get("phases", 3))
        phases = phase_list(pspec, 3)
        own = phase_list(lspec, nph)
        agree["primary==load spec" if own == phases else "primary!=load spec"] += 1
        loads.append(
            {
                "name": st.split()[1].split(".", 1)[1].lower(),
                "bus": pbus,
                "kw": float(d.get("kw", 0.0)),
                "conn": pconn,
                "phases": phases,
                "service": f"{len(phases)}ph-{pconn}",
                "secondary": f"{nph}ph-{d.get('conn', 'wye').lower()}",
                "transformer": xfmr.lower(),
            }
        )
    return loads, agree


def build_tree(lines: list[dict]) -> tuple[list[str], dict, dict]:
    adj = defaultdict(list)
    for ln in lines:
        adj[ln["bus1"]].append((ln["bus2"], ln))
        adj[ln["bus2"]].append((ln["bus1"], ln))
    order, parent_of, line_of = [SLACK], {}, {}
    queue, seen = deque([SLACK]), {SLACK}
    while queue:
        a = queue.popleft()
        for b, ln in adj[a]:
            if b not in seen:
                seen.add(b)
                if ln["bus2"] != b:
                    fail(f"line {ln['name']} points towards the source")
                parent_of[b], line_of[b] = a, ln
                order.append(b)
                queue.append(b)
    if len(seen) != len(adj):
        fail(f"{len(adj) - len(seen)} buses are not connected to {SLACK}")
    return order, parent_of, line_of


def z1(r: np.ndarray, x: np.ndarray) -> complex:
    z = r + 1j * x
    n = len(z)
    if n == 1:
        return complex(z[0, 0])
    off = (z.sum() - np.trace(z)) / (n * (n - 1))
    return complex(np.trace(z) / n - off)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--raw_dir", required=True, help="OpenDSS model folder")
    p.add_argument("--meta", required=True, help="snapshots_meta.json")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    model = args.raw_dir
    with open(args.meta) as fh:
        meta = json.load(fh)
    mf = meta["feeder"]
    labels = list(mf["bus_labels"])
    if labels[0] != SLACK:
        fail(f"meta slack is {labels[0]}, expected {SLACK}")
    load_labels = [labels[k] for k in mf["load_buses"]]
    shunt = {labels[k + 1]: -q for k, q in enumerate(mf["shunt_q_kvar"]) if q != 0.0}

    # ---- topology ------------------------------------------------------
    lines = parse_lines(model)
    order, parent_of, line_of = build_tree(lines)
    keep = set(load_labels) | set(shunt)
    kids = defaultdict(set)
    for b in order[1:]:
        kids[parent_of[b]].add(b)
    pruned = []
    for b in reversed(order[1:]):
        if not kids[b] and b not in keep:
            pruned.append(b)
            kids[parent_of[b]].discard(b)
    kept = [b for b in order if b not in set(pruned)]
    errors = []
    if set(kept) != set(labels):
        errors.append(
            f"bus sets differ: only here {sorted(set(kept) - set(labels))[:10]}, "
            f"only in meta {sorted(set(labels) - set(kept))[:10]}"
        )
    else:
        bad = [
            b
            for k, b in enumerate(labels[1:])
            if parent_of[b] != labels[mf["parent"][k]]
        ]
        if bad:
            errors.append(f"parents differ at {len(bad)} buses: {bad[:10]}")
    if sorted(pruned) != sorted(meta.get("pruned_dead_end_buses", [])):
        errors.append(
            f"pruned buses {sorted(pruned)} vs meta {meta.get('pruned_dead_end_buses')}"
        )
    if errors:
        fail("; ".join(errors))
    index = {b: k for k, b in enumerate(labels)}
    branch = [line_of[b] for b in labels[1:]]
    phases = [tuple(ln["phases"]) for ln in branch]
    for k, b in enumerate(labels[1:]):
        par = mf["parent"][k]
        if par > 0 and not set(phases[k]) <= set(phases[par - 1]):
            fail(f"bus {b}: phases {phases[k]} not within its parent's")

    # positive-sequence check against the balanced meta
    dev = defaultdict(list)
    for k, ln in enumerate(branch):
        n = len(ln["phases"])
        zm = complex(mf["r_ohm"][k], mf["x_ohm"][k])
        z = z1(ln["r"], ln["x"])
        dev[f"{n}ph"].append(abs(z - zm))
        if n < 3:  # balanced convention: z1 times 3 / n
            dev[f"{n}ph x3/n"].append(abs(z * 3.0 / n - zm))
    zcheck = {k: float(max(v)) for k, v in sorted(dev.items())}
    print("positive-sequence |z1 - z_meta| max (ohm):", zcheck)

    # ---- loads ---------------------------------------------------------
    loads, agree = parse_loads(model)
    by_bus = defaultdict(list)
    for ld in loads:
        by_bus[ld["bus"]].append(ld)
    off_node = [
        (ld["name"], ld["bus"], ld["kw"]) for ld in loads if ld["bus"] not in keep
    ]
    missing = [b for b in load_labels if b not in by_bus]
    if missing:
        fail(f"meta load buses without a load object: {missing}")
    conns = []
    for i, b in enumerate(load_labels):
        group = by_bus[b]
        total = sum(ld["kw"] for ld in group)
        for ld in group:
            share = ld["kw"] / total if total > 0 else 1.0 / len(group)
            if not set(ld["phases"]) <= set(phases[index[b] - 1]):
                fail(f"load {ld['name']} phases {ld['phases']} not at bus {b}")
            for kind, ph, s in connections_of(ld["conn"], ld["phases"], share):
                conns.append(Connection(i, kind, ph, s))
    shares = np.zeros(len(load_labels))
    for c in conns:
        shares[c.node] += c.share
    if np.abs(shares - 1.0).max() > 1e-12:
        fail("connection shares do not sum to 1")

    # ---- capacitors ----------------------------------------------------
    caps, cap_files = [], {}
    for st in statements(os.path.join(model, "Capacitor.dss")):
        if not is_new(st, "capacitor"):
            continue
        d = props(st)
        if d.get("enabled", "yes").lower() not in ("yes", "true", "y"):
            continue
        b, spec = bus_name(d["bus1"])
        n = int(d.get("phases", 3))
        cap_files[b] = (float(d["kvar"]), d.get("conn", "wye").lower(), spec, n)
    if set(cap_files) != set(shunt):
        fail(f"capacitor buses {sorted(cap_files)} vs meta {sorted(shunt)}")
    cap_kvar_diff = {}
    for b in sorted(shunt):
        kvar, conn, spec, n = cap_files[b]
        cap_kvar_diff[b] = kvar - shunt[b]
        for kind, ph, s in connections_of(conn, phase_list(spec, n), 1.0):
            caps.append(Capacitor(index[b] - 1, kind, ph, shunt[b] * s))

    # ---- feeder --------------------------------------------------------
    s2 = [m for m in mf["measurement_sets"] if m[0] == "S2"]
    if len(s2) != 1:
        fail("meta has no unique measurement set S2")
    feeder = ThreePhaseFeeder(
        name=mf["name"] + "_3ph",
        bus_labels=tuple(labels),
        parent=tuple(mf["parent"]),
        phases=tuple(phases),
        r_ohm=tuple(tuple(map(tuple, ln["r"].tolist())) for ln in branch),
        x_ohm=tuple(tuple(map(tuple, ln["x"].tolist())) for ln in branch),
        load_buses=tuple(mf["load_buses"]),
        p_nominal=tuple(mf["p_nominal"]),
        q_nominal=tuple(mf["q_nominal"]),
        connections=tuple(conns),
        capacitors=tuple(caps),
        voltage_buses={"S2-3ph": tuple(s2[0][1])},
        v_slack=mf["v_slack"],
        v_base_kv=mf["v_base_kv"],
        s_base_kva=mf["s_base_kva"],
    )
    pf = ThreePhasePowerFlow(feeder)
    nom = np.concatenate([feeder.p_nominal_kw, feeder.q_nominal_kvar])
    with torch.no_grad():
        res = pf(torch.as_tensor(nom, dtype=torch.float64).unsqueeze(0))
    mask = feeder.branch_phase_mask
    kept_bp = [
        bp for bp, keep_it in zip(feeder.branch_phases, mask, strict=True) if keep_it
    ]
    ratings = ratings_3ph(pf)  # raises below MIN_RATING_A (stop rule)
    low = [
        [labels[kept_bp[i][0] + 1], "abc"[kept_bp[i][1]], float(ratings[i])]
        for i in np.argsort(ratings)
        if ratings[i] < LOW_RATING_REPORT_A
    ]
    counts = {
        "num_buses": feeder.num_buses,
        "num_nodes": feeder.num_nodes,
        "num_node_phases": feeder.num_node_phases,
        "num_branch_phases": len(mask),
        "num_branch_phases_kept": int(sum(mask)),
        "num_branch_phases_excluded": int(len(mask) - sum(mask)),
        "buses_by_num_phases": dict(
            sorted(Counter(str(len(ph)) for ph in phases).items())
        ),
        "connections_by_kind": dict(
            sorted(Counter(f"{c.kind}-{len(c.phases)}" for c in conns).items())
        ),
        "loads_by_primary_service": dict(
            sorted(Counter(by_bus[b][0]["service"] for b in load_labels).items())
        ),
        "loads_by_secondary": dict(
            sorted(Counter(ld["secondary"] for ld in loads).items())
        ),
        "load_objects_per_node": dict(
            sorted(Counter(str(len(by_bus[b])) for b in load_labels).items())
        ),
        "load_phase_spec_agreement": dict(agree),
        "loads_off_state_nodes": off_node,
        "num_capacitor_connections": len(caps),
        "capacitor_kvar_file_minus_meta": cap_kvar_diff,
        "min_kept_rating_a": float(ratings.min()),
        "min_rating_floor_a": MIN_RATING_A,
        "kept_branch_phases_below_0p5_a": low,
        "nominal_mismatch_pu": float(res.mismatch.item()),
        "nominal_head_p_kw": float(res.p_head_kw.item()),
        "nominal_head_q_kvar": float(res.q_head_kvar.item()),
        "z1_check_max_abs_ohm": zcheck,
        "lines_declared_phases_differ": [
            ln["name"]
            for ln in branch
            if ln["linecode"] and ln["declared_phases"] != len(ln["phases"])
        ],
    }
    out = feeder.to_dict()
    out["source"] = {
        "dataset": "Iowa 240-bus distribution test system (Bu et al., NAPS 2019)",
        "raw_sha256": {f: file_sha256(os.path.join(model, f)) for f in RAW_FILES},
        "meta_sha256": file_sha256(args.meta),
        "git": git_state(),
    }
    out["counts"] = counts
    if ThreePhaseFeeder.from_dict(json.loads(json.dumps(out))) != feeder:
        fail("from_dict(to_dict()) does not reproduce the feeder")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(out, fh)
    for k, v in counts.items():
        print(f"{k}: {v}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

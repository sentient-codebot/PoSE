"""Probe of the three-phase Iowa 240-bus feeder.

1. Inventory (inventory.json): loads, lines and line codes, capacitors,
   transformers and regulators of the OpenDSS model, and the mapping of the
   balanced meta's buses to phases and load connections. Stop conditions: a
   transformer or regulator between bus 1 and a load's primary bus, a line
   code of order 4 (neutral not Kron-reduced), or a missing input. When one
   fires, inventory.json records it and the script exits 1.
2. Reduced reference model, built in memory after ``compile`` (the source
   files are never edited): stiff balanced 1.025 p.u. source at bus 1
   (substation transformer and regulators disabled), every load replaced by
   a constant-power load at its primary bus on the primary-side phases and
   connection of its distribution transformer (transformers disabled), loads
   of a bus share the bus's nodal P and Q in proportion to their nominal kW,
   line capacitance zero, each capacitor a constant -50 kvar load, open
   .4.5.6 breakers and lines to pruned dead ends disabled (the balanced
   meta's tree). Checked with a no-load solve.
3. Probe (probe_summary.json): the reduced model and the balanced
   ``RadialPowerFlow`` for the 1,000 extended-design test snapshots;
   unbalance, phase spread, deviation from the balanced model, head and
   branch imbalance, voltage band and a balanced-model check.

OpenDSS and torch may live in different environments, so the work is split
in two steps: ``--step opendss`` (inventory and OpenDSS solves, needs only
numpy and OpenDSSDirect; writes reference_3ph.npz) and ``--step summary``
(balanced power flow and statistics, needs torch). ``--step all`` (default)
runs both in one environment.

    python scripts/validate/iowa_3ph_probe.py \
        --raw_dir "data/iowa/raw/extracted/OpenDSS Model (08.13.2020)" \
        --data data/iowa/snapshots.npz --output_dir results/iowa_3ph/probe

Units: kW, kvar, p.u. on V_b = 13.8/sqrt(3) kV phase-to-neutral and
S_b = 1000/3 kVA per phase (I_b = 41.84 A), A for branch currents. Data
p_kw > 0 is consumption. Phases 1, 2, 3 are a, b, c."""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from collections import defaultdict, deque
from typing import TYPE_CHECKING

import numpy as np


REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")

if TYPE_CHECKING:
    from pose.data import SnapshotData

# torch and pandas are imported inside functions: the opendss step runs in
# an environment with only numpy and OpenDSSDirect (pose.data needs only
# numpy).


V_BASE_KV = 13.8
V_SLACK = 1.025
V_PHASE_V = V_BASE_KV * 1000.0 / np.sqrt(3.0)
S_PHASE_KVA = 1000.0 / 3.0
I_BASE_A = S_PHASE_KVA / (V_PHASE_V / 1000.0)
CAPACITOR_KVAR = 50.0  # assemble_iowa.CAPACITORS: fixed -50 kvar per capacitor
SIGMA_V = 0.01 / 3.0
SOURCE_Z_OHM = 1e-6
SLACK_TOL = 1e-6
NO_LOAD_V_TOL = 1e-9
NO_LOAD_PQ_TOL = 1e-6  # kW, kvar
BALANCE_TOL_KW = 1e-3
A = np.exp(2j * np.pi / 3)
SLACK_PHASORS = V_SLACK * np.array([1.0, A**2, A])  # 0, -120, +120 degrees
INPUT_FILES = (
    "Master.dss",
    "Vsource.dss",
    "SubTransformer.dss",
    "RegControl.dss",
    "DistriTransformer.dss",
    "Linecode.dss",
    "Line.dss",
    "CircuitBreaker.dss",
    "Load.dss",
    "Capacitor.dss",
)


def _dss():
    import opendssdirect as dss

    return dss


def split_bus(spec: str) -> tuple[str, list[int]]:
    """'bus1011.1.0' -> ('bus1011', [1, 0])."""
    name, *nodes = spec.lower().split(".")
    return name, [int(n) for n in nodes]


def primary_of(load_bus: str) -> str:
    """Primary bus of a load bus, as in iowa_opendss_export.py."""
    return load_bus.replace("t_", "").replace("_l", "")


def compile_model(raw_dir: str) -> None:
    dss = _dss()
    cwd = os.getcwd()  # compile changes the directory
    try:
        master = os.path.join(os.path.abspath(raw_dir), "Master.dss")
        dss.Text.Command(f'compile "{master}"')
    finally:
        os.chdir(cwd)


def dist(x) -> dict:
    """Median, p95 and max over snapshots."""
    x = np.asarray(x, dtype=float)
    return {
        "median": float(np.median(x)),
        "p95": float(np.quantile(x, 0.95)),
        "max": float(np.max(x)),
    }


def dist_signed(x) -> dict:
    x = np.asarray(x, dtype=float)
    return {**dist(x), "min": float(np.min(x)), "mean": float(np.mean(x))}


# --------------------------------------------------------------------------
# 3.1 Inventory
# --------------------------------------------------------------------------


def inventory(raw_dir: str, meta: dict) -> dict:
    """Inventory of the compiled OpenDSS model (plan 3.1) with stop checks."""
    dss = _dss()
    compile_model(raw_dir)
    feeder = meta["feeder"]
    labels = feeder["bus_labels"]
    load_labels = [labels[k] for k in feeder["load_buses"]]

    # transformers (winding buses, connection of winding 1)
    xfmrs = {}
    for name in dss.Transformers.AllNames():
        dss.Transformers.Name(name)
        buses = [b.lower() for b in dss.CktElement.BusNames()]
        dss.Transformers.Wdg(1)
        xfmrs[name] = {
            "name": name,
            "buses": buses,
            "phases": dss.CktElement.NumPhases(),
            "windings": dss.Transformers.NumWindings(),
            "wdg1_conn": "delta" if dss.Transformers.IsDelta() else "wye",
            "wdg1_kv": dss.Transformers.kV(),
            "kva": dss.Transformers.kVA(),
        }
    regs = {}
    for name in dss.RegControls.AllNames():
        dss.RegControls.Name(name)
        regs[name] = dss.RegControls.Transformer().lower()
    for x in xfmrs.values():
        x["regulator"] = x["name"] in set(regs.values())

    serving = {}  # secondary (load) bus -> transformer
    for x in xfmrs.values():
        for b in x["buses"][1:]:
            serving.setdefault(split_bus(b)[0], x["name"])

    # loads
    loads = []
    for name in dss.Loads.AllNames():
        dss.Loads.Name(name)
        bus_spec = dss.CktElement.BusNames()[0].lower()
        bus, nodes = split_bus(bus_spec)
        prim = primary_of(bus)
        t = serving.get(bus)
        rec = {
            "name": name,
            "bus": bus_spec,
            "primary_bus": prim,
            "phases": [n for n in nodes if n > 0],
            "conn": "delta" if dss.Loads.IsDelta() else "wye",
            "num_phases": dss.CktElement.NumPhases(),
            "kv": dss.Loads.kV(),
            "model": dss.Loads.Model(),
            "kw": dss.Loads.kW(),
            "kvar": dss.Loads.kvar(),
            "transformer": t,
        }
        if t is not None:
            x = xfmrs[t]
            pbus, pnodes = split_bus(x["buses"][0])
            rec["primary_service"] = {
                "bus": pbus,
                "phases": [n for n in pnodes if n > 0],
                "conn": x["wdg1_conn"],
                "kv": x["wdg1_kv"],
            }
        loads.append(rec)
    per_bus = defaultdict(list)
    for rec in loads:
        per_bus[rec["primary_bus"]].append(rec["name"])

    # line codes and lines
    codes = {}
    for name in dss.LineCodes.AllNames():
        dss.LineCodes.Name(name)
        order = round(np.sqrt(len(dss.LineCodes.Rmatrix())))
        codes[name] = {
            "nphases": dss.LineCodes.Phases(),
            "matrix_order": order,
            "cmatrix_nonzero": bool(np.any(np.asarray(dss.LineCodes.Cmatrix()))),
        }
    lines = []
    for name in dss.Lines.AllNames():
        dss.Lines.Name(name)
        b1, n1 = split_bus(dss.Lines.Bus1())
        b2, n2 = split_bus(dss.Lines.Bus2())
        code = dss.Lines.LineCode().lower()
        lines.append(
            {
                "name": name,
                "bus1": b1,
                "bus1_phases": n1,
                "bus2": b2,
                "bus2_phases": n2,
                "phases": dss.Lines.Phases(),
                "linecode": code or None,
                "linecode_nphases": codes[code]["nphases"] if code else None,
                "linecode_matrix_order": codes[code]["matrix_order"] if code else None,
                "switch": name.startswith("cb_"),
                "normally_open": any(n >= 4 for n in n2),
            }
        )

    caps = []
    for name in dss.Capacitors.AllNames():
        dss.Capacitors.Name(name)
        bus, nodes = split_bus(dss.CktElement.BusNames()[0])
        caps.append(
            {
                "name": name,
                "bus": bus,
                "phases": [n for n in nodes if n > 0] or [1, 2, 3],
                "conn": "delta" if dss.Capacitors.IsDelta() else "wye",
                "kvar": dss.Capacitors.kvar(),
                "kv": dss.Capacitors.kV(),
            }
        )

    # graph of all closed PD elements; BFS tree from bus1
    adj = defaultdict(list)
    for ln in lines:
        if not ln["normally_open"]:
            adj[ln["bus1"]].append((ln["bus2"], "Line." + ln["name"]))
            adj[ln["bus2"]].append((ln["bus1"], "Line." + ln["name"]))
    for x in xfmrs.values():
        bs = [split_bus(b)[0] for b in x["buses"]]
        for b in bs[1:]:
            adj[bs[0]].append((b, "Transformer." + x["name"]))
            adj[b].append((bs[0], "Transformer." + x["name"]))
    up = {"bus1": None}
    queue = deque(["bus1"])
    while queue:
        a = queue.popleft()
        for b, el in adj[a]:
            if b not in up:
                up[b] = (a, el)
                queue.append(b)

    def path(bus: str) -> list[str]:
        out = []
        while bus in up and up[bus] is not None:
            bus, el = up[bus][0], up[bus][1]
            out.append(el)
        return out

    on_primary, on_load = set(), set()
    for rec in loads:
        on_primary.update(e for e in path(rec["primary_bus"]) if e.startswith("T"))
        on_load.update(e for e in path(split_bus(rec["bus"])[0]) if e.startswith("T"))
    for x in xfmrs.values():
        key = "Transformer." + x["name"]
        x["on_path_bus1_to_load_bus"] = key in on_load
        x["on_path_bus1_to_primary_bus"] = key in on_primary
        x["serves_loads"] = sorted(
            r["name"] for r in loads if r["transformer"] == x["name"]
        )

    stop = []
    for x in xfmrs.values():
        if x["on_path_bus1_to_primary_bus"]:
            stop.append(f"transformer {x['name']} lies between bus1 and a primary bus")
        elif x["on_path_bus1_to_load_bus"] and not x["serves_loads"]:
            stop.append(f"transformer {x['name']} lies between bus1 and a load bus")
    for r in loads:
        p = [e for e in path(split_bus(r["bus"])[0]) if e.startswith("T")]
        if p != ["Transformer." + str(r["transformer"])]:
            stop.append(f"load {r['name']}: transformers on its path {p}")
    for c, d in codes.items():
        if d["matrix_order"] != d["nphases"] or d["matrix_order"] >= 4:
            stop.append(
                f"line code {c}: order {d['matrix_order']}, nphases {d['nphases']}"
            )
    unreached = sorted(b for b in labels if b not in up)
    if unreached:
        stop.append(f"meta buses not connected to bus1: {unreached}")

    # mapping to the balanced meta
    bus_phases = {}
    dummy_nodes = {}  # .4.5.6 nodes of the normally open breakers
    for b in labels:
        dss.Circuit.SetActiveBus(b)
        bus_phases[b] = sorted(n for n in dss.Bus.Nodes() if 1 <= n <= 3)
        if any(n >= 4 for n in dss.Bus.Nodes()):
            dummy_nodes[b] = sorted(n for n in dss.Bus.Nodes() if n >= 4)
    load_conn = {}
    for b in load_labels:
        load_conn[b] = [
            {
                "load": r["name"],
                "transformer": r["transformer"],
                "nominal_kw": r["kw"],
                **{k: r.get("primary_service", {}).get(k) for k in ("phases", "conn")},
            }
            for r in loads
            if r["primary_bus"] == b
        ]
    unmapped_loads = sorted(
        r["name"] for r in loads if r["primary_bus"] not in load_conn
    )
    meta_without_load = sorted(b for b in load_labels if not load_conn[b])

    def kind(r):
        ps = r.get("primary_service") or {}
        return f"{len(ps.get('phases', []))}ph-{ps.get('conn')}"

    counts = {
        "loads": len(loads),
        "loads_by_num_phases": _count(r["num_phases"] for r in loads),
        "loads_by_conn": _count(r["conn"] for r in loads),
        "loads_by_secondary_phases_conn": _count(
            f"{r['num_phases']}ph-{r['conn']}" for r in loads
        ),
        "loads_by_primary_service": _count(kind(r) for r in loads),
        "load_objects_per_primary_bus": _count(len(v) for v in per_bus.values()),
        "primary_buses_by_service_phases": _count(
            len({p for c in load_conn[b] for p in (c["phases"] or [])})
            for b in load_labels
            if load_conn[b]
        ),
        "meta_buses_by_phase_count": _count(len(v) for v in bus_phases.values()),
        "lines": len(lines),
        "lines_by_phases": _count(ln["phases"] for ln in lines),
        "linecodes": len(codes),
        "linecode_matrix_orders": _count(d["matrix_order"] for d in codes.values()),
        "capacitors": len(caps),
        "transformers": len(xfmrs),
        "regulators": len(regs),
        "transformers_inside_feeder": sum(
            x["on_path_bus1_to_primary_bus"] for x in xfmrs.values()
        ),
    }
    return {
        "stop": bool(stop),
        "stop_reasons": stop,
        "counts": counts,
        "loads": loads,
        "load_objects_per_primary_bus": dict(sorted(per_bus.items())),
        "linecodes": codes,
        "lines": lines,
        "capacitors": caps,
        "transformers": list(xfmrs.values()),
        "regcontrols": regs,
        "meta_bus_phases": bus_phases,
        "meta_bus_open_breaker_dummy_nodes": dummy_nodes,
        "meta_load_bus_connections": load_conn,
        "loads_not_on_meta_load_bus": unmapped_loads,
        "meta_load_buses_without_load_object": meta_without_load,
        "opendss_version": dss.Basic.Version(),
    }


def _count(it) -> dict:
    out = defaultdict(int)
    for v in it:
        out[str(v)] += 1
    return dict(sorted(out.items()))


# --------------------------------------------------------------------------
# 3.2 Reduced reference model
# --------------------------------------------------------------------------


class ReducedModel:
    """Reduced three-phase reference model of the Iowa feeder in OpenDSS.

    Build once with ``build_reduced_model(raw_dir, meta)`` (``raw_dir`` is the
    folder holding Master.dss, ``meta`` the balanced snapshots_meta.json);
    OpenDSS holds one circuit per process, so only one instance is live.

    Attributes:
        load_labels: the 193 state-node bus labels (balanced meta order).
        node_phases: list of (bus label, phase) for every phase of every meta
            bus (bus1 included), phases 1, 2, 3 = a, b, c.
        branch_phases: list of (downstream bus label, phase) for every phase
            of every branch; branch k of the balanced tree feeds non-slack
            bus k (bus_labels[k + 1]).
        branch_nphases: number of phases of the branch of each entry.
        loads: replacement loads, list of dicts (name, bus, phases, conn,
            node index, share).
    """

    def __init__(self, raw_dir: str, meta: dict):
        dss = _dss()
        self.dss = dss
        feeder = meta["feeder"]
        self.labels = list(feeder["bus_labels"])
        self.label_set = set(self.labels)
        self.load_labels = [self.labels[k] for k in feeder["load_buses"]]
        parent = {
            self.labels[k + 1]: self.labels[p] for k, p in enumerate(feeder["parent"])
        }
        node_of = {b: k for k, b in enumerate(self.load_labels)}
        compile_model(raw_dir)
        cmd = dss.Text.Command

        # original loads and their serving transformers
        serving = {}
        for name in dss.Transformers.AllNames():
            dss.Transformers.Name(name)
            buses = [b.lower() for b in dss.CktElement.BusNames()]
            dss.Transformers.Wdg(1)
            delta = dss.Transformers.IsDelta()
            for b in buses[1:]:
                serving.setdefault(split_bus(b)[0], (buses[0], delta))
        orig = []
        for name in dss.Loads.AllNames():
            dss.Loads.Name(name)
            bus = split_bus(dss.CktElement.BusNames()[0])[0]
            if bus not in serving:
                raise SystemExit(f"load {name}: no serving transformer")
            orig.append((name, primary_of(bus), serving[bus], dss.Loads.kW()))
        caps = []
        for name in dss.Capacitors.AllNames():
            dss.Capacitors.Name(name)
            bus, nodes = split_bus(dss.CktElement.BusNames()[0])
            caps.append(
                (
                    name,
                    bus,
                    [n for n in nodes if n > 0] or [1, 2, 3],
                    dss.Capacitors.IsDelta(),
                )
            )

        # slack: stiff balanced source at bus1; substation equipment off
        cmd(
            "Edit Vsource.source bus1=bus1.1.2.3 phases=3 basekv=13.8 "
            f"pu={V_SLACK} angle=0 R1={SOURCE_Z_OHM} X1={SOURCE_Z_OHM} "
            f"R0={SOURCE_Z_OHM} X0={SOURCE_Z_OHM}"
        )
        for name in dss.RegControls.AllNames():
            cmd(f"Edit RegControl.{name} enabled=no")
        for name in dss.Transformers.AllNames():
            cmd(f"Edit Transformer.{name} enabled=no")
        for name, *_ in orig:
            cmd(f"Edit Load.{name} enabled=no")
        for name, *_ in caps:
            cmd(f"Edit Capacitor.{name} enabled=no")

        # lines: tree of the balanced meta, capacitance off
        self.disabled_lines = []
        self.lines = []  # (name, upstream terminal 0/1, downstream bus, nphases)
        seen = set()
        for name in list(dss.Lines.AllNames()):
            dss.Lines.Name(name)
            b1, _n1 = split_bus(dss.Lines.Bus1())
            b2, n2 = split_bus(dss.Lines.Bus2())
            if any(n >= 4 for n in n2) or not {b1, b2} <= self.label_set:
                cmd(f"Edit Line.{name} enabled=no")
                self.disabled_lines.append(name)
                continue
            nph = dss.Lines.Phases()
            dss.Lines.CMatrix([0.0] * (nph * nph))
            if name.startswith("cb_"):
                # Switch=y sets length 0.001, so r1=1e-4 per unit length gives
                # 1e-7 ohm; use 1e-4 ohm as the balanced model (assemble_iowa)
                cmd(f"Edit Line.{name} length=1 units=none")
            if parent.get(b2) == b1:
                down, term = b2, 0
            elif parent.get(b1) == b2:
                down, term = b1, 1
            else:
                raise SystemExit(f"line {name} ({b1}-{b2}) is not a meta tree branch")
            seen.add(down)
            self.lines.append((name, term, down, nph))
        missing = sorted(set(self.labels[1:]) - seen)
        if missing or len(self.lines) != len(self.labels) - 1:
            raise SystemExit(f"meta branches without an OpenDSS line: {missing}")

        # constant-power replacement loads on the primary-side connection
        nominal = defaultdict(float)
        count = defaultdict(int)
        for _, prim, _, kw in orig:
            nominal[prim] += kw
            count[prim] += 1
        # A load on a bus outside the meta tree (a pruned dead end, zero P in
        # every hour) is left out: a load on an isolated bus makes DSS C-API
        # 0.14.5 return stale solutions reported as converged.
        self.skipped_loads = []
        self.loads = []
        for name, prim, (wdg1, delta), kw in orig:
            if prim not in self.label_set:
                self.skipped_loads.append(name)
                continue
            pbus, pnodes = split_bus(wdg1)
            ph = [n for n in pnodes if n > 0]
            if pbus != prim:
                raise SystemExit(f"load {name}: transformer primary {pbus} != {prim}")
            share = kw / nominal[prim] if nominal[prim] > 0 else 1.0 / count[prim]
            if delta and len(ph) == 2:
                spec = {"phases": 1, "conn": "delta", "kv": V_BASE_KV}
            elif len(ph) == 1 and not delta:
                spec = {"phases": 1, "conn": "wye", "kv": V_BASE_KV / np.sqrt(3.0)}
            elif len(ph) == 3 and not delta:
                spec = {"phases": 3, "conn": "wye", "kv": V_BASE_KV}
            else:
                raise SystemExit(
                    f"load {name}: unsupported service {wdg1} delta={delta}"
                )
            rname = f"r_{name}"
            cmd(
                f"New Load.{rname} bus1={prim}.{'.'.join(map(str, ph))} "
                f"phases={spec['phases']} conn={spec['conn']} kV={spec['kv']:.6f} "
                "kW=0 kvar=0 model=1 vminpu=0.5 vmaxpu=1.5"
            )
            self.loads.append(
                {
                    "name": rname,
                    "original": name,
                    "bus": prim,
                    "phases": ph,
                    "conn": spec["conn"],
                    "node": node_of.get(prim),
                    "share": share,
                }
            )
        self.capacitor_loads = []
        for name, bus, ph, delta in caps:
            conn = "delta" if delta else "wye"
            kv = V_BASE_KV if (len(ph) == 3 or delta) else V_BASE_KV / np.sqrt(3.0)
            cname = f"c_{name}"
            cmd(
                f"New Load.{cname} bus1={bus}.{'.'.join(map(str, ph))} "
                f"phases={len(ph) if len(ph) == 3 else 1} conn={conn} kV={kv:.6f} "
                f"kW=0 kvar={-CAPACITOR_KVAR} model=1 vminpu=0.5 vmaxpu=1.5"
            )
            self.capacitor_loads.append(
                {
                    "name": cname,
                    "bus": bus,
                    "phases": ph,
                    "conn": conn,
                    "kvar": -CAPACITOR_KVAR,
                }
            )
        cmd("Set controlmode=off")
        cmd("Set tolerance=1e-10")
        cmd("Set maxiterations=200")
        cmd("Set VoltageBases=[69.0 13.8 0.208]")
        cmd("CalcVoltageBases")

        # line capacitance must be zero
        for name, *_ in self.lines:
            dss.Lines.Name(name)
            if np.any(np.asarray(dss.Lines.CMatrix())):
                raise SystemExit(f"line {name}: capacitance not removed")

        # index of node-phases and branch-phases
        dss.Solution.Solve()
        names = [n.lower() for n in dss.Circuit.AllNodeNames()]
        pos = {n: k for k, n in enumerate(names)}
        self.node_phases, self._node_pos = [], []
        for b in self.labels:
            for ph in (1, 2, 3):
                if f"{b}.{ph}" in pos:
                    self.node_phases.append((b, ph))
                    self._node_pos.append(pos[f"{b}.{ph}"])
        self._node_pos = np.asarray(self._node_pos)
        self._slack_pos = np.asarray([pos[f"bus1.{ph}"] for ph in (1, 2, 3)])
        self.branch_phases, self.branch_nphases, self._branch_idx = [], [], []
        for k, (name, term, down, nph) in enumerate(self.lines):
            dss.Circuit.SetActiveElement(f"Line.{name}")
            order = dss.CktElement.NodeOrder()
            ncond = len(order) // 2
            for c in range(ncond):
                ph = order[term * ncond + c]
                self.branch_phases.append((down, ph))
                self.branch_nphases.append(nph)
                self._branch_idx.append((k, term * ncond + c, ph))
        self.head_lines = [
            (k, ln)
            for k, ln in enumerate(self.lines)
            if parent[ln[2]] == self.labels[0]
        ]
        self.load_names = [ld["name"] for ld in self.loads]
        self.all_load_names = self.load_names + [
            c["name"] for c in self.capacitor_loads
        ]
        self.load_node = np.asarray(
            [-1 if ld["node"] is None else ld["node"] for ld in self.loads]
        )
        self.load_share = np.asarray([ld["share"] for ld in self.loads])

    def set_injections(self, p_kw: np.ndarray, q_kvar: np.ndarray) -> None:
        dss = self.dss
        for name, k, s in zip(
            self.load_names, self.load_node, self.load_share, strict=False
        ):
            dss.Loads.Name(name)
            dss.Loads.kW(float(p_kw[k] * s) if k >= 0 else 0.0)
            dss.Loads.kvar(float(q_kvar[k] * s) if k >= 0 else 0.0)

    def set_capacitors(self, kvar: float) -> None:
        for c in self.capacitor_loads:
            self.dss.Loads.Name(c["name"])
            self.dss.Loads.kW(0.0)
            self.dss.Loads.kvar(kvar)

    def solve(self, p_kw, q_kvar) -> dict:
        """Solve one hour.

        Args:
            p_kw, q_kvar: nodal P (kW) and Q (kvar) of the 193 state nodes in
                the balanced meta order (positive = consumption).

        Returns a dict:
            converged (bool), slack_deviation_pu (max |V_bus1 - set point|,
            complex, p.u.), v_slack (complex [3], bus-1 phases a, b, c, p.u.),
            v_node (complex [len(node_phases)], phase-to-neutral p.u.,
            aligned with ``node_phases``), v (dict (bus, phase) -> complex),
            p_head_kw, q_head_kvar (float [3], per phase, from bus 1 into the
            feeder), i_head_a (complex [3], per-phase head current, A),
            i_branch_a (float [len(branch_phases)], current magnitude at the
            upstream terminal, A, aligned with ``branch_phases``), i_branch
            (dict (downstream bus, phase) -> A), load_p_kw (total P drawn by the
            OpenDSS loads, kW), power_balance_kw (source P minus load P minus
            losses, kW; nonzero flags a stale solution).
        """
        dss = self.dss
        self.set_injections(np.asarray(p_kw, float), np.asarray(q_kvar, float))
        dss.Solution.Solve()
        return self._read()

    def _read(self) -> dict:
        dss = self.dss
        raw = np.asarray(dss.Circuit.AllBusVolts())
        v = (raw[0::2] + 1j * raw[1::2]) / V_PHASE_V
        v_node = v[self._node_pos]
        v_slack = v[self._slack_pos]
        currents = []
        for name, *_ in self.lines:
            dss.Circuit.SetActiveElement(f"Line.{name}")
            c = np.asarray(dss.CktElement.Currents())
            currents.append(c[0::2] + 1j * c[1::2])
        i_branch = np.asarray([abs(currents[k][c]) for k, c, _ in self._branch_idx])
        p_head, q_head, i_head = np.zeros(3), np.zeros(3), np.zeros(3, complex)
        for k, (name, term, _, _) in self.head_lines:
            dss.Circuit.SetActiveElement(f"Line.{name}")
            order = dss.CktElement.NodeOrder()
            pw = np.asarray(dss.CktElement.Powers())
            ncond = len(order) // 2
            for c in range(ncond):
                j = term * ncond + c
                ph = order[j]
                p_head[ph - 1] += pw[2 * j]
                q_head[ph - 1] += pw[2 * j + 1]
                i_head[ph - 1] += currents[k][j]
        # power balance: source P = load P + losses (a stale solution fails)
        p_load = 0.0
        for name in self.all_load_names:
            dss.Circuit.SetActiveElement(f"Load.{name}")
            p_load += float(np.sum(dss.CktElement.Powers()[0::2]))
        p_src = -float(dss.Circuit.TotalPower()[0])
        balance = p_src - p_load - float(dss.Circuit.Losses()[0]) / 1000.0
        return {
            "converged": bool(dss.Solution.Converged()),
            "power_balance_kw": balance,
            "load_p_kw": p_load,
            "iterations": int(dss.Solution.Iterations()),
            "slack_deviation_pu": float(np.max(np.abs(v_slack - SLACK_PHASORS))),
            "v_slack": v_slack,
            "v_node": v_node,
            "v": dict(zip(self.node_phases, v_node, strict=False)),
            "p_head_kw": p_head,
            "q_head_kvar": q_head,
            "i_head_a": i_head,
            "i_branch_a": i_branch,
            "i_branch": dict(zip(self.branch_phases, i_branch, strict=False)),
        }

    def no_load_check(self) -> dict:
        """Loads and capacitors at zero: every node-phase at 1.025 p.u. and
        zero head power, which holds only without line charging."""
        n = len(self.load_labels)
        self.set_capacitors(0.0)
        res = self.solve(np.zeros(n), np.zeros(n))
        self.set_capacitors(-CAPACITOR_KVAR)
        dv = float(np.max(np.abs(np.abs(res["v_node"]) - V_SLACK)))
        out = {
            "converged": res["converged"],
            "max_abs_vm_minus_1p025_pu": dv,
            "max_abs_head_p_kw": float(np.max(np.abs(res["p_head_kw"]))),
            "max_abs_head_q_kvar": float(np.max(np.abs(res["q_head_kvar"]))),
            "slack_deviation_pu": res["slack_deviation_pu"],
        }
        out["pass"] = bool(
            res["converged"]
            and dv <= NO_LOAD_V_TOL
            and out["max_abs_head_p_kw"] <= NO_LOAD_PQ_TOL
            and out["max_abs_head_q_kvar"] <= NO_LOAD_PQ_TOL
        )
        return out


def build_reduced_model(raw_dir: str, meta: dict) -> ReducedModel:
    """Compile Master.dss in ``raw_dir`` and build the reduced model once."""
    return ReducedModel(raw_dir, meta)


# --------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------


def select_ids(data: "SnapshotData", max_snapshots: int | None) -> np.ndarray:
    """Same IDs as run_cell.select_snapshots(data, "extended", max_snapshots)."""
    from pose.data import select_test_snapshots

    idx = select_test_snapshots(data, "extended")
    if max_snapshots:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(idx, size=min(max_snapshots, len(idx)), replace=False))
    return idx


def git_commit() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
        dirty = subprocess.check_output(
            [
                "git",
                "--no-optional-locks",
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            cwd=REPO,
            text=True,
        ).strip()
        return {"commit": commit, "state": "dirty" if dirty else "clean"}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "state": "unknown"}


def step_opendss(args, data: "SnapshotData") -> int:
    dss = _dss()
    missing = [
        p
        for p in [args.data, os.path.splitext(args.data)[0] + "_meta.json"]
        + [os.path.join(args.raw_dir, f) for f in INPUT_FILES]
        if not os.path.exists(p)
    ]
    inv_path = os.path.join(args.output_dir, "inventory.json")
    if missing:
        reasons = [f"missing input {p}" for p in missing]
        _write(inv_path, {"stop": True, "stop_reasons": reasons})
        print("STOP: missing inputs", missing)
        return 1
    inv = inventory(args.raw_dir, data.meta)
    t0 = time.time()
    model = build_reduced_model(args.raw_dir, data.meta)
    inv["reduced_model"] = {
        "disabled_lines": model.disabled_lines,
        "skipped_loads_outside_meta_tree": model.skipped_loads,
        "num_lines": len(model.lines),
        "replacement_loads": model.loads,
        "capacitor_loads": model.capacitor_loads,
        "matches_meta_tree": True,  # build raises otherwise
        "source_impedance_ohm": SOURCE_Z_OHM,
    }
    no_load = model.no_load_check()
    inv["reduced_model"]["no_load_check"] = no_load
    _write(inv_path, inv)
    print(json.dumps({"counts": inv["counts"], "no_load": no_load}, indent=1))
    if inv["stop"]:
        print("STOP:", "; ".join(inv["stop_reasons"]))
        return 1
    if not no_load["pass"]:
        print("STOP: no-load check failed")
        return 1

    ids = select_ids(data, args.max_snapshots)
    n = len(model.load_labels)
    x = data.p_kw[ids]
    out = defaultdict(list)
    t1 = time.time()
    for row in x:
        res = model.solve(row[:n], row[n:])
        keys = (
            "v_slack", "v_node", "p_head_kw", "q_head_kvar", "i_head_a",
            "i_branch_a", "converged", "iterations", "slack_deviation_pu",
            "power_balance_kw", "load_p_kw",
        )  # fmt: skip
        for k in keys:
            out[k].append(res[k])
    t2 = time.time()
    np.savez(
        os.path.join(args.output_dir, "reference_3ph.npz"),
        ids=ids,
        node_bus=np.array([b for b, _ in model.node_phases]),
        node_phase=np.array([p for _, p in model.node_phases]),
        branch_bus=np.array([b for b, _ in model.branch_phases]),
        branch_phase=np.array([p for _, p in model.branch_phases]),
        branch_nphases=np.array(model.branch_nphases),
        total_load_p_kw=x[:, :n].sum(1),
        build_seconds=t1 - t0,
        solve_seconds=t2 - t1,
        opendss_version=dss.Basic.Version(),
        **{k: np.asarray(v) for k, v in out.items()},
    )
    print(
        f"solved {len(ids)} snapshots in {t2 - t1:.1f} s; converged "
        f"{int(np.sum(out['converged']))}; max slack deviation "
        f"{max(out['slack_deviation_pu']):.2e} p.u.; max |power balance| "
        f"{np.max(np.abs(out['power_balance_kw'])):.2e} kW"
    )
    return 0


def step_summary(args, data: "SnapshotData") -> int:
    import torch

    from pose.feeder import Feeder
    from pose.powerflow import RadialPowerFlow

    spec = importlib.util.spec_from_file_location(
        "run_cell", os.path.join(REPO, "scripts", "run_cell.py")
    )
    run_cell = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_cell)
    ref = np.load(os.path.join(args.output_dir, "reference_3ph.npz"))
    ids = ref["ids"]
    expected = run_cell.select_snapshots(data, "extended", args.max_snapshots)
    assert np.array_equal(ids, expected), "snapshot IDs differ from run_cell"
    assert np.array_equal(
        select_ids(data, None), run_cell.select_snapshots(data, "extended", None)
    ), "1,000 extended IDs differ from run_cell"

    feeder = Feeder.from_meta(data.meta)
    labels = list(feeder.bus_labels)
    bidx = {b: k for k, b in enumerate(labels)}
    with torch.no_grad():
        bal = RadialPowerFlow(feeder)(torch.as_tensor(data.p_kw[ids]))
    vm_bal = np.concatenate([np.full((len(ids), 1), feeder.v_slack), bal.vm.numpy()], 1)
    ib_bal = bal.branch_current_a.numpy()  # branch k feeds bus k + 1
    p_bal, q_bal = bal.p_head_kw.numpy(), bal.q_head_kvar.numpy()

    node_bus, node_phase = ref["node_bus"], ref["node_phase"]
    v = ref["v_node"]
    vmag = np.abs(v)
    s = len(ids)
    nonslack = node_bus != labels[0]

    # buses by phase set
    phases_of = defaultdict(list)
    for j, (b, ph) in enumerate(zip(node_bus, node_phase, strict=True)):
        phases_of[b].append((ph, j))
    three = [b for b in labels if len(phases_of[b]) == 3]
    multi = [b for b in labels if len(phases_of[b]) >= 2]

    # VUF at three-phase buses
    cols = np.array([[j for _, j in sorted(phases_of[b])] for b in three])
    va, vb, vc = v[:, cols[:, 0]], v[:, cols[:, 1]], v[:, cols[:, 2]]
    v1 = (va + A * vb + A**2 * vc) / 3
    v2 = (va + A**2 * vb + A * vc) / 3
    vuf = 100 * np.abs(v2) / np.abs(v1)
    vuf_max = vuf.max(1)
    peak = np.unravel_index(np.argmax(vuf), vuf.shape)
    argmax_bus = _count(three[k] for k in vuf.argmax(1))

    # phase spread
    spread = np.stack(
        [np.ptp(vmag[:, [j for _, j in phases_of[b]]], 1) for b in multi], 1
    )
    sp_peak = np.unravel_index(np.argmax(spread), spread.shape)

    # deviation from the balanced model
    d = vmag - vm_bal[:, [bidx[b] for b in node_bus]]
    d_ns = d[:, nonslack]
    s2 = next(m for m in feeder.measurement_sets if m[0] == "S2")[1]
    s2_cols = [j for j, b in enumerate(node_bus) if b in s2]
    d_s2 = np.abs(d[:, s2_cols]).max(1)
    worst_np = int(np.argmax(np.abs(d_ns).max(0)))
    ns_idx = np.flatnonzero(nonslack)

    # head imbalance
    ph_, qh_ = ref["p_head_kw"], ref["q_head_kvar"]
    ih = np.abs(ref["i_head_a"])
    ih_imb = (ih.max(1) - ih.min(1)) / ih.mean(1)
    names = ("a", "b", "c")

    # branch currents on three-phase branches
    br_bus, br_nph, ib = ref["branch_bus"], ref["branch_nphases"], ref["i_branch_a"]
    three_br = sorted({b for b, n in zip(br_bus, br_nph, strict=True) if n == 3})
    ratios, exceed, total = [], 0, 0
    for b in three_br:
        cols_b = np.flatnonzero(br_bus == b)
        i_bal = ib_bal[:, bidx[b] - 1]
        ok = i_bal > 1e-3
        r = ib[:, cols_b] / np.where(ok, i_bal, np.nan)[:, None]
        ratios.append(np.nanmax(r, 1))
        exceed += int(np.sum(r[ok] > 1.1))
        total += int(ok.sum() * len(cols_b))
    ratios = np.stack(ratios, 1)
    ratio_flat = ratios[np.isfinite(ratios)]

    # voltage band
    out_band = (vmag[:, nonslack] < 0.95) | (vmag[:, nonslack] > 1.05)
    bal_band = (vm_bal[:, 1:] < 0.95) | (vm_bal[:, 1:] > 1.05)

    # balanced-model check
    v1_all = np.abs(v1)
    v1_dev = np.abs(v1_all - vm_bal[:, [bidx[b] for b in three]]).max(1)
    phase_mean = np.stack(
        [vmag[:, [j for _, j in phases_of[b]]].mean(1) for b in labels[1:]], 1
    )
    pm_dev = np.abs(phase_mean - vm_bal[:, 1:]).max(1)
    load_p = ref["total_load_p_kw"]

    summary = {
        "num_snapshots": int(s),
        "snapshot_design": "extended"
        + (f", subset of {args.max_snapshots}" if args.max_snapshots else ""),
        "opendss_version": str(ref["opendss_version"]),
        "git": git_commit(),
        "convergence": {
            "all_converged": bool(ref["converged"].all()),
            "max_iterations": int(ref["iterations"].max()),
            "max_slack_deviation_pu": float(ref["slack_deviation_pu"].max()),
            "slack_within_1e-6": bool(ref["slack_deviation_pu"].max() <= SLACK_TOL),
            "balanced_max_mismatch": float(bal.mismatch.max()),
            "max_abs_power_balance_kw": float(np.abs(ref["power_balance_kw"]).max()),
            "max_abs_opendss_load_p_minus_data_kw": float(
                np.abs(ref["load_p_kw"] - ref["total_load_p_kw"]).max()
            ),
        },
        "runtime_seconds": {
            "build": float(ref["build_seconds"]),
            "opendss_solves": float(ref["solve_seconds"]),
        },
        "counts": {
            "node_phases": len(node_bus),
            "three_phase_buses": len(three),
            "buses_two_or_more_phases": len(multi),
            "three_phase_branches": len(three_br),
        },
        "vuf_percent": {
            "per_snapshot_max": dist(vuf_max),
            "peak_bus": three[peak[1]],
            "peak_snapshot": int(ids[peak[0]]),
            "argmax_bus_counts": argmax_bus,
        },
        "phase_spread_pu": {
            "per_snapshot_max": dist(spread.max(1)),
            "peak_bus": multi[sp_peak[1]],
        },
        "deviation_from_balanced": {
            "sigma_v_pu": SIGMA_V,
            "all_node_phases_max_abs_pu": dist(np.abs(d_ns).max(1)),
            "all_node_phases_max_abs_over_sigma": dist(np.abs(d_ns).max(1) / SIGMA_V),
            "all_node_phases_mean_signed_pu": dist_signed(d_ns.mean(1)),
            "worst_node_phase": [
                str(node_bus[ns_idx[worst_np]]),
                int(node_phase[ns_idx[worst_np]]),
            ],
            "s2_buses": list(s2),
            "s2_node_phases": [[str(node_bus[j]), int(node_phase[j])] for j in s2_cols],
            "s2_max_abs_pu": dist(d_s2),
            "s2_max_abs_over_sigma": dist(d_s2 / SIGMA_V),
            "s2_per_node_phase_max_abs_over_sigma": {
                f"{node_bus[j]}.{node_phase[j]}": float(np.abs(d[:, j]).max() / SIGMA_V)
                for j in s2_cols
            },
        },
        "head": {
            **{f"p_kw_{names[k]}": dist_signed(ph_[:, k]) for k in range(3)},
            **{f"q_kvar_{names[k]}": dist_signed(qh_[:, k]) for k in range(3)},
            **{
                f"p_share_{names[k]}": dist_signed(ph_[:, k] / ph_.sum(1))
                for k in range(3)
            },
            **{
                f"q_share_{names[k]}": dist_signed(qh_[:, k] / qh_.sum(1))
                for k in range(3)
            },
            "current_imbalance_max_minus_min_over_mean": dist(ih_imb),
        },
        "branch_current_3ph": {
            "max_phase_over_balanced_all_branch_snapshots": dist(ratio_flat),
            "max_phase_over_balanced_per_snapshot_max": dist(np.nanmax(ratios, 1)),
            "branch_phase_snapshots_over_1p1": exceed,
            "branch_phase_snapshots_total": total,
        },
        "voltage_band": {
            "node_phase_snapshots_outside_0p95_1p05": int(out_band.sum()),
            "node_phase_snapshots_below_0p95": int((vmag[:, nonslack] < 0.95).sum()),
            "node_phase_snapshots_above_1p05": int((vmag[:, nonslack] > 1.05).sum()),
            "node_phase_snapshots_total": int(out_band.size),
            "balanced_bus_snapshots_outside": int(bal_band.sum()),
            "balanced_bus_snapshots_total": int(bal_band.size),
            "min_vm_3ph_pu": float(vmag[:, nonslack].min()),
            "min_vm_balanced_pu": float(vm_bal[:, 1:].min()),
        },
        "balanced_check": {
            "head_p_3ph_minus_balanced_kw": dist_signed(ph_.sum(1) - p_bal),
            "head_q_3ph_minus_balanced_kvar": dist_signed(qh_.sum(1) - q_bal),
            "losses_3ph_kw": dist(ph_.sum(1) - load_p),
            "losses_balanced_kw": dist(p_bal - load_p),
            "mean_vm_3ph_minus_balanced_pu": dist_signed(d_ns.mean(1)),
            "max_abs_pos_seq_vm_minus_balanced_at_3ph_buses_pu": dist(v1_dev),
            "max_abs_phase_mean_vm_minus_balanced_pu": dist(pm_dev),
        },
    }
    _write(os.path.join(args.output_dir, "probe_summary.json"), summary)
    print(json.dumps(summary, indent=1))
    conv = summary["convergence"]
    ok = (
        conv["all_converged"]
        and conv["slack_within_1e-6"]
        and conv["max_abs_power_balance_kw"] <= BALANCE_TOL_KW
        and conv["max_abs_opendss_load_p_minus_data_kw"] <= BALANCE_TOL_KW
    )
    if not ok:
        print(
            "STOP: a solve did not converge, bus 1 left its set point "
            "or power did not balance"
        )
    return 0 if ok else 1


def _write(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        return str(o)

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=default)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--raw_dir", required=True, help="folder holding Master.dss")
    p.add_argument("--data", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_snapshots", type=int, default=None, help="local tests only")
    p.add_argument("--step", choices=("opendss", "summary", "all"), default="all")
    args = p.parse_args()
    args.raw_dir = os.path.abspath(args.raw_dir)
    args.data = os.path.abspath(args.data)
    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists(args.data):
        _write(
            os.path.join(args.output_dir, "inventory.json"),
            {"stop": True, "stop_reasons": [f"missing input {args.data}"]},
        )
        print(f"STOP: missing input {args.data}")
        sys.exit(1)
    from pose.data import SnapshotData

    data = SnapshotData.load(args.data)
    code = 0
    if args.step in ("opendss", "all"):
        code = step_opendss(args, data)
    if code == 0 and args.step in ("summary", "all"):
        code = step_summary(args, data)
    sys.exit(code)


if __name__ == "__main__":
    main()

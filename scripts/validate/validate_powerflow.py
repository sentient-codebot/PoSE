"""Validate the torch AC power flow against pandapower on test snapshots.

Reads a SnapshotData file, selects test snapshots, solves each one with
pandapower case33bw (Newton-Raphson, tolerance 1e-10 MVA) and writes JSON with
error statistics. Exit code 1 on failure. Tolerances: 1e-6 p.u. voltage,
1e-3 kW / kvar head flow, 1e-3 A current. With --topology, branches 1..37
(ties 33..37) are taken out of service and branch k is compared with the
in-service line between parent[k] and bus k + 1.
"""

import argparse
import json
import os

import numpy as np
import pandapower as pp
import pandapower.networks as pn
import torch

from pose.data import SnapshotData
from pose.feeder import RECONFIGURATIONS, Feeder, reconfigured_ieee33
from pose.powerflow import RadialPowerFlow


def branch_lines(net, parent) -> list[int]:
    """Row of net.line feeding bus k + 1 from parent[k], k = 0..31 (pandapower
    bus index = label - 1 = Feeder bus index); in-service rows only."""
    live = net.line[net.line.in_service]
    idx = []
    for k, par in enumerate(parent):
        ends = {int(par), k + 1}
        rows = [
            i
            for i, u, v in zip(live.index, live.from_bus, live.to_bus, strict=False)
            if {int(u), int(v)} == ends
        ]
        if len(rows) != 1:
            raise ValueError(f"branch {k}: {len(rows)} in-service lines {ends}")
        idx.append(int(rows[0]))
    return idx


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", required=True, help="SnapshotData file")
    p.add_argument("--output", required=True)
    p.add_argument("--num_snapshots", type=int, default=20)
    p.add_argument("--seed", type=int, default=20261014)
    p.add_argument("--topology", default=None, choices=list(RECONFIGURATIONS))
    args = p.parse_args()

    data = SnapshotData.load(args.data)
    feeder = Feeder.from_meta(data.meta)
    if args.topology and args.topology != "base":
        feeder = reconfigured_ieee33(
            RECONFIGURATIONS[args.topology], name=args.topology, base=feeder
        )
    test = data.indices("test")
    if not len(test):
        raise SystemExit("no test snapshots in the data")
    rng = np.random.default_rng(args.seed)
    snapshots = np.sort(
        rng.choice(test, size=min(args.num_snapshots, len(test)), replace=False)
    )
    state = torch.as_tensor(data.p_kw[snapshots], dtype=torch.float64)
    with torch.no_grad():
        res = RadialPowerFlow(feeder)(state)
    p_kw = state[:, : feeder.num_nodes].numpy()
    q_kvar = (
        state[:, feeder.num_nodes :].numpy()
        if state.shape[1] == 2 * feeder.num_nodes
        else p_kw * feeder.tan_phi
    )

    open_branches = RECONFIGURATIONS[args.topology] if args.topology else None
    line_idx = None
    errs = {"vm_pu": [], "p_head_kw": [], "q_head_kvar": [], "branch_current_a": []}
    for k in range(len(snapshots)):
        net = pn.case33bw()
        net.load["p_mw"] = p_kw[k] / 1000.0
        net.load["q_mvar"] = q_kvar[k] / 1000.0
        if open_branches is not None:
            net.line["in_service"] = True
            for b in open_branches:
                net.line.loc[int(b) - 1, "in_service"] = False
            if line_idx is None:
                line_idx = branch_lines(net, feeder.parent)
        pp.runpp(net, tolerance_mva=1e-10, numba=False)
        if line_idx is None:
            i_ka = net.res_line[net.line.in_service.to_numpy()].i_ka.to_numpy()
        else:
            i_ka = net.res_line.loc[line_idx, "i_ka"].to_numpy()
        errs["vm_pu"].append(
            np.abs(net.res_bus.vm_pu.to_numpy()[1:] - res.vm[k].numpy()).max()
        )
        errs["p_head_kw"].append(
            abs(net.res_ext_grid.p_mw.iloc[0] * 1000 - res.p_head_kw[k].item())
        )
        errs["q_head_kvar"].append(
            abs(net.res_ext_grid.q_mvar.iloc[0] * 1000 - res.q_head_kvar[k].item())
        )
        errs["branch_current_a"].append(
            np.abs(i_ka * 1000 - res.branch_current_a[k].numpy()).max()
        )
    tol = {
        "vm_pu": 1e-6,
        "p_head_kw": 1e-3,
        "q_head_kvar": 1e-3,
        "branch_current_a": 1e-3,
    }
    result = {
        "num_snapshots": len(snapshots),
        "snapshots": snapshots.tolist(),
        "max_abs_error": {k: float(np.max(v)) for k, v in errs.items()},
        "tolerance": tol,
        "pandapower_version": pp.__version__,
    }
    if args.topology:
        result["topology"] = args.topology
        result["open_branches"] = [int(b) for b in open_branches]
    result["pass"] = all(result["max_abs_error"][k] <= tol[k] for k in tol)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()

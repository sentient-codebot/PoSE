"""Solve the original (unbalanced) Iowa 240-bus OpenDSS model for test hours.

Requires OpenDSSDirect.py. For each hour, every load is set to the nodal P
and Q of its primary bus (data/iowa/snapshots.npz), the circuit is solved with
the substation regulators active, and the per-unit voltage magnitude of every
primary bus (mean over its phases), the bus-1 voltage and the power through
the three feeder breakers are written to --output for validate_iowa.py."""

import argparse
import json
import os

import numpy as np
import opendssdirect as dss


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", required=True)
    p.add_argument("--master", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--num_hours", type=int, default=100)
    p.add_argument("--seed", type=int, default=20261014)
    args = p.parse_args()
    args.output = os.path.abspath(args.output)  # compile changes the directory

    arrays = np.load(args.data)
    with open(os.path.splitext(args.data)[0] + "_meta.json") as fh:
        meta = json.load(fh)
    feeder = meta["feeder"]
    labels = feeder["bus_labels"]
    load_labels = [labels[k] for k in feeder["load_buses"]]
    n = len(load_labels)
    x = arrays["p_kw"]
    test = np.flatnonzero(arrays["split"] == 2)
    hours = np.sort(
        np.random.default_rng(args.seed).choice(test, args.num_hours, replace=False)
    )

    dss.Text.Command(f'compile "{os.path.abspath(args.master)}"')
    loads = {}
    for name in dss.Loads.AllNames():
        dss.Loads.Name(name)
        bus = dss.CktElement.BusNames()[0].split(".")[0].lower()
        loads[name] = bus.replace("t_", "").replace("_l", "")
    col = {b: k for k, b in enumerate(load_labels)}
    unmapped = sorted(b for b in set(loads.values()) if b not in col)

    vm, v1, ph, qh, conv = [], [], [], [], []
    for h in hours:
        for name, bus in loads.items():
            dss.Loads.Name(name)
            k = col.get(bus)
            dss.Loads.kW(float(x[h, k]) if k is not None else 0.0)
            dss.Loads.kvar(float(x[h, n + k]) if k is not None else 0.0)
        dss.Solution.Solve()
        conv.append(bool(dss.Solution.Converged()))
        row = []
        for b in labels:
            dss.Circuit.SetActiveBus(b)
            mags = dss.Bus.puVmagAngle()[0::2]
            row.append(float(np.mean(mags)) if mags else np.nan)
        v1.append(row[0])
        vm.append(row[1:])
        p_tot = q_tot = 0.0
        for cb in ("CB_101", "CB_201", "CB_301"):
            dss.Circuit.SetActiveElement(f"Line.{cb}")
            pw = dss.CktElement.Powers()
            k = len(pw) // 2
            p_tot += sum(pw[0:k:2])
            q_tot += sum(pw[1:k:2])
        ph.append(p_tot)
        qh.append(q_tot)
    np.savez(
        args.output,
        hours=hours,
        vm=np.array(vm),
        v_bus1=np.array(v1),
        p_head_kw=np.array(ph),
        q_head_kvar=np.array(qh),
        converged=np.array(conv),
    )
    print(
        f"solved {len(hours)} hours, converged {sum(conv)}; loads {len(loads)}, unmapped load buses {unmapped}"
    )


if __name__ == "__main__":
    main()

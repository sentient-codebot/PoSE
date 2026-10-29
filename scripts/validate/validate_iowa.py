"""Compare the balanced Iowa equivalent with the original OpenDSS solution.

The voltage magnitudes of the primary buses must agree with OpenDSS to within
0.01 p.u. For every exported hour the equivalent is solved with the slack at
the OpenDSS bus-1 voltage. Writes --output (JSON) with error statistics over
all buses, by phase count of the bus, and of the feeder-head power, and exits
1 if the tolerance fails."""

import argparse
import dataclasses
import json
import sys

import numpy as np
import torch

from pose.data import SnapshotData
from pose.experiment import write_json
from pose.feeder import Feeder
from pose.powerflow import RadialPowerFlow


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", required=True)
    p.add_argument("--opendss", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--tolerance", type=float, default=0.01)
    args = p.parse_args()
    data = SnapshotData.load(args.data)
    base = Feeder.from_meta(data.meta)
    ref = np.load(args.opendss)
    errs, dp, dq = [], [], []
    for k, h in enumerate(ref["hours"]):
        f = dataclasses.replace(base, v_slack=float(ref["v_bus1"][k]))
        with torch.no_grad():
            res = RadialPowerFlow(f)(torch.as_tensor(data.p_kw[h : h + 1]))
        errs.append(res.vm.numpy()[0] - ref["vm"][k])
        dp.append(float(res.p_head_kw[0]) - float(ref["p_head_kw"][k]))
        dq.append(float(res.q_head_kvar[0]) - float(ref["q_head_kvar"][k]))
    e = np.abs(np.array(errs))
    worst_bus = int(np.nanargmax(np.nanmax(e, 0)))
    report = {
        "num_hours": len(ref["hours"]),
        "all_converged": bool(ref["converged"].all()),
        "max_abs_vm_error_pu": float(np.nanmax(e)),
        "p99_abs_vm_error_pu": float(np.nanquantile(e, 0.99)),
        "median_abs_vm_error_pu": float(np.nanmedian(e)),
        "mean_vm_error_pu": float(np.nanmean(np.array(errs))),
        "worst_bus": base.bus_labels[worst_bus + 1],
        "head_p_error_kw": {
            "mean": float(np.mean(dp)),
            "max_abs": float(np.max(np.abs(dp))),
        },
        "head_q_error_kvar": {
            "mean": float(np.mean(dq)),
            "max_abs": float(np.max(np.abs(dq))),
        },
        "opendss_v_bus1_range": [
            float(ref["v_bus1"].min()),
            float(ref["v_bus1"].max()),
        ],
        "tolerance_pu": args.tolerance,
    }
    report["pass"] = bool(
        report["all_converged"] and report["max_abs_vm_error_pu"] <= args.tolerance
    )
    write_json(args.output, report)
    print(json.dumps(report, indent=1))
    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()

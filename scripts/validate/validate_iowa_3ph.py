"""OpenDSS validation of the three-phase Iowa torch model.

Two steps, because the OpenDSS environment has no torch:

    opendss   solves the reduced reference model of the probe
              (iowa_3ph_probe.build_reduced_model) on 100 test hours (seed
              20261014, as iowa_opendss_export.py) and writes an npz
    compare   runs ThreePhasePowerFlow on the same hours with the slack set
              to the solved OpenDSS bus-1 phase voltages and compares

Pass when max |d|V|| <= 1e-6 p.u. over all node-phases, per-phase head P and Q
within 0.01 kW and 0.01 kvar, and every branch-phase current within 0.01 A,
and every OpenDSS solve converged with a closed power balance. Exit 1 on
failure.

    python scripts/validate/validate_iowa_3ph.py opendss --raw_dir RAW --data DATA \
        --output ref.npz --num_hours 100
    python scripts/validate/validate_iowa_3ph.py compare --data DATA --physics F.json \
        --opendss ref.npz --output opendss_validation.json"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np


TOL_V_PU = 1e-6
TOL_PQ = 0.01  # kW, kvar
TOL_I_A = 0.01
TOL_BALANCE_KW = 1e-3
TOL_MISMATCH_PU = 1e-10
TOL_SLACK_PU = 1e-6  # bus 1 within 1e-6 p.u. of the set point
PHASE_LABELS = "abc"


def load_probe():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "iowa_3ph_probe.py")
    spec = importlib.util.spec_from_file_location("iowa_3ph_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_hours(data_path: str, num_hours: int, seed: int) -> np.ndarray:
    """The test hours of iowa_opendss_export.py."""
    split = np.load(data_path)["split"]
    test = np.flatnonzero(split == 2)
    return np.sort(np.random.default_rng(seed).choice(test, num_hours, replace=False))


def step_opendss(args) -> int:
    probe = load_probe()
    with open(os.path.splitext(args.data)[0] + "_meta.json") as f:
        meta = json.load(f)
    x = np.load(args.data)["p_kw"]
    n = len(meta["feeder"]["load_buses"])
    hours = select_hours(args.data, args.num_hours, args.seed)
    model = probe.build_reduced_model(args.raw_dir, meta)
    out = {k: [] for k in ("v_slack", "v_node", "p_head", "q_head", "i_branch")}
    conv, dev, bal = [], [], []
    for h in hours:
        r = model.solve(x[h, :n], x[h, n:])
        out["v_slack"].append(r["v_slack"])
        out["v_node"].append(r["v_node"])
        out["p_head"].append(r["p_head_kw"])
        out["q_head"].append(r["q_head_kvar"])
        out["i_branch"].append(r["i_branch_a"])
        conv.append(r["converged"])
        dev.append(r["slack_deviation_pu"])
        bal.append(r["power_balance_kw"])
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez(
        args.output,
        hours=hours,
        node_phases=np.array([f"{b}.{p}" for b, p in model.node_phases]),
        branch_phases=np.array([f"{b}.{p}" for b, p in model.branch_phases]),
        converged=np.asarray(conv),
        slack_deviation_pu=np.asarray(dev),
        power_balance_kw=np.asarray(bal),
        **{k: np.asarray(v) for k, v in out.items()},
    )
    print(f"opendss: {len(hours)} hours, all converged {all(conv)}", flush=True)
    return 0


def step_compare(args) -> int:
    import torch

    from pose.experiment import file_sha256, git_state, write_json
    from pose.threephase import ThreePhaseFeeder, ThreePhasePowerFlow

    with open(args.physics) as f:
        feeder = ThreePhaseFeeder.from_dict(json.load(f))
    ref = np.load(args.opendss)
    x = np.load(args.data)["p_kw"]
    labels = feeder.bus_labels
    tags = [f"{labels[k + 1]}.{p + 1}" for k, p in feeder.node_phases]
    pos_v = {t: i for i, t in enumerate(ref["node_phases"].tolist())}
    pos_i = {t: i for i, t in enumerate(ref["branch_phases"].tolist())}
    missing = [t for t in tags if t not in pos_v or t not in pos_i]
    if missing:
        raise SystemExit(f"node- or branch-phases missing in OpenDSS: {missing[:10]}")
    iv = np.array([pos_v[t] for t in tags])
    ib = np.array([pos_i[t] for t in tags])
    worst = {"v_pu": 0.0, "p_kw": 0.0, "q_kvar": 0.0, "i_a": 0.0, "mismatch": 0.0}
    where = {}
    for j, h in enumerate(ref["hours"]):
        pf = ThreePhasePowerFlow(feeder, v_slack_phasors=ref["v_slack"][j])
        with torch.no_grad():
            r = pf(torch.as_tensor(x[h], dtype=torch.float64).unsqueeze(0))
        diffs = {
            "v_pu": np.abs(r.vm[0].numpy() - np.abs(ref["v_node"][j][iv])),
            "p_kw": np.abs(r.p_head_phase_kw[0].numpy() - ref["p_head"][j]),
            "q_kvar": np.abs(r.q_head_phase_kvar[0].numpy() - ref["q_head"][j]),
            "i_a": np.abs(r.branch_current_all_a[0].numpy() - ref["i_branch"][j][ib]),
        }
        for k, d in diffs.items():
            if d.max() > worst[k]:
                worst[k] = float(d.max())
                i = int(d.argmax())
                where[k] = {
                    "hour": int(h),
                    "element": tags[i] if k in ("v_pu", "i_a") else PHASE_LABELS[i],
                }
        worst["mismatch"] = max(worst["mismatch"], float(r.mismatch.max()))
    checks = {
        "v_pu": worst["v_pu"] <= TOL_V_PU,
        "p_kw": worst["p_kw"] <= TOL_PQ,
        "q_kvar": worst["q_kvar"] <= TOL_PQ,
        "i_a": worst["i_a"] <= TOL_I_A,
        "torch_mismatch": worst["mismatch"] <= TOL_MISMATCH_PU,
        "opendss_slack": bool(ref["slack_deviation_pu"].max() <= TOL_SLACK_PU),
        "opendss_converged": bool(ref["converged"].all()),
        "opendss_power_balance": bool(
            np.abs(ref["power_balance_kw"]).max() <= TOL_BALANCE_KW
        ),
    }
    passed = all(checks.values())
    write_json(
        args.output,
        {
            "pass": passed,
            "checks": checks,
            "max_abs_diff": worst,
            "where": where,
            "tolerances": {
                "v_pu": TOL_V_PU,
                "p_kw": TOL_PQ,
                "q_kvar": TOL_PQ,
                "i_a": TOL_I_A,
                "power_balance_kw": TOL_BALANCE_KW,
            },
            "num_hours": len(ref["hours"]),
            "num_node_phases": len(tags),
            "max_slack_deviation_pu": float(ref["slack_deviation_pu"].max()),
            "max_abs_power_balance_kw": float(np.abs(ref["power_balance_kw"]).max()),
            "physics_sha256": file_sha256(args.physics),
            "opendss_sha256": file_sha256(args.opendss),
            "git": git_state(),
        },
    )
    print(json.dumps({"pass": passed, **worst}), flush=True)
    return 0 if passed else 1


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="step", required=True)
    a = sub.add_parser("opendss")
    a.add_argument("--raw_dir", required=True, help="folder with Master.dss")
    a.add_argument("--data", required=True)
    a.add_argument("--output", required=True)
    a.add_argument("--num_hours", type=int, default=100)
    a.add_argument("--seed", type=int, default=20261014)
    b = sub.add_parser("compare")
    b.add_argument("--data", required=True)
    b.add_argument("--physics", required=True)
    b.add_argument("--opendss", required=True)
    b.add_argument("--output", required=True)
    args = p.parse_args()
    sys.exit(step_opendss(args) if args.step == "opendss" else step_compare(args))


if __name__ == "__main__":
    main()

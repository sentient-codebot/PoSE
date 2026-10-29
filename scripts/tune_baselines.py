"""Choose baseline hyperparameters once on the validation weeks (then fixed).

B3: number of mixture components K in {1, 2, 3, 4} by the mean validation
    log-likelihood of the injections under the bin mixture.
B4: DPS step size zeta by the CRPS on 100 validation snapshots, 50 samples,
    for the measurement set and noise level given (default S2, nominal);
    grid --dps_grid (default {0.001, ..., 1}; per-setting retuning uses
    {1e-4, ..., 1}).
B7: kernel bandwidths (h_hours, h_months) and shrinkage alpha by the mean
    validation log-likelihood of the injections.
--tasks selects what to tune; --base_hyper copies B3 and B7 from an earlier
file instead of refitting them.

Writes --output (hyper.json), read by run_cell.py --hyper."""

import argparse
import json
import os

import numpy as np
import torch

from pose import baselines as bl
from pose.data import SnapshotData
from pose.experiment import (
    Setting,
    generators,
    git_state,
    write_json,
)
from pose.guidance import sample_dps
from pose.metrics import crps_samples
from pose.prior import context_tensors, load_prior


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", required=True)
    p.add_argument("--prior", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--num_val", type=int, default=100)
    p.add_argument("--num_steps", type=int, default=200)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--mset", default="S2")
    p.add_argument("--noise", default="nominal")
    p.add_argument(
        "--tasks", nargs="+", default=["b3", "b4"], choices=["b3", "b4", "b7"]
    )
    p.add_argument("--base_hyper", default=None, help="copy B3 and B7 from this file")
    p.add_argument(
        "--dps_grid",
        type=float,
        nargs="+",
        default=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
    )
    args = p.parse_args()
    device = torch.device(args.device)
    data = SnapshotData.load(args.data)
    scenario = data.meta.get("scenario", "synthetic")
    rng = np.random.default_rng(20261015)
    val = np.sort(rng.choice(data.indices("val"), size=args.num_val, replace=False))
    ctx_np = data.context_of(val)
    out = {"validation_snapshots": val.tolist(), "git": git_state()}

    out["mset"], out["noise"] = args.mset, args.noise
    if args.base_hyper:
        with open(args.base_hyper) as f:
            base = json.load(f)
        for key in ("B3", "B7"):
            if key in base:
                out[key] = base[key]
    if "b3" in args.tasks:
        bins = bl.ContextBins(data)
        scores = {}
        for k in (1, 2, 3, 4):
            mix = bl.MixtureBaseline(bins, k)
            scores[k] = mix.log_likelihood(data.p_kw[val], ctx_np)
            print(f"B3 K={k}: val log-lik {scores[k]:.4f}", flush=True)
        best_k = max(scores, key=scores.get)
        out["B3"] = {"num_components": best_k, "val_loglik": scores}
    if "b7" in args.tasks:
        grid7 = {}
        for hh in (0.5, 1.0, 2.0, 4.0):
            for hm in (0.5, 1.0, 2.0):
                for alpha in (0.0, 0.05, 0.1, 0.2):
                    kb = bl.KernelBins(data, hh, hm, alpha)
                    ll = kb.log_likelihood(data.p_kw[val], ctx_np)
                    grid7[f"{hh}_{hm}_{alpha}"] = ll
                    print(
                        f"B7 h={hh} h_m={hm} alpha={alpha}: val log-lik {ll:.4f}",
                        flush=True,
                    )
        best = max(grid7, key=grid7.get)
        hh, hm, alpha = (float(v) for v in best.split("_"))
        out["B7"] = {"h_hours": hh, "h_months": hm, "alpha": alpha, "val_loglik": grid7}
    if "b4" not in args.tasks:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        write_json(args.output, out)
        print({k: out[k] for k in ("B3", "B7") if k in out}, flush=True)
        return

    setting = Setting.from_names(scenario, args.mset, args.noise, device, data.meta)
    truth, y = setting.observe(data, val)
    ctx = context_tensors(ctx_np, device)
    prior, _ = load_prior(args.prior, device)
    crps = {}
    for zeta in args.dps_grid:
        vals = []
        for start in range(0, len(val), 25):
            sl = slice(start, start + 25)
            samples = sample_dps(
                prior,
                setting.measurement,
                setting.noise_cov,
                {k: v[sl] for k, v in ctx.items()},
                y[sl],
                50,
                args.num_steps,
                generators("init", scenario, val[sl]),
                zeta,
            )
            n = setting.power_flow.num_nodes  # active-power CRPS (kW)
            vals.append(crps_samples(samples[..., :n], truth[sl][..., :n]).mean(-1))
        c = torch.cat(vals)
        crps[zeta] = float(c.mean()) if torch.isfinite(c).all() else float("inf")
        print(f"B4 zeta={zeta}: val CRPS {crps[zeta]:.4f} kW", flush=True)
    out["B4"] = {"step_size": min(crps, key=crps.get), "val_crps_kw": crps}
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    write_json(args.output, out)
    print({k: out[k] for k in ("B3", "B4", "B7") if k in out}, flush=True)


if __name__ == "__main__":
    main()

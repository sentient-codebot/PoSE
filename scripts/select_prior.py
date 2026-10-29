"""Select the prior configuration of the sweep on the validation weeks.

Recomputes the flow-matching validation loss of every sweep run and of B2's
prior (full-covariance Ledoit-Wolf Gaussian per context bin, in closed form)
on identical noise draws. Selection (pose.prior.select_run): the lowest
validation loss; runs within --tie of it are tied, and the one with the
fewest parameters, then the fewest steps, wins.

The selected prior must beat the Gaussian by more than two standard errors of
the paired difference, with validation days as blocks. Otherwise the report is
still written and the script exits with code 1, which stops a dependency
chain.

Writes --output: the selected PriorConfig overrides (for train_prior.py
--config_json), all runs, references and the comparison. With
--link_selected PATH, a symlink at PATH (created or replaced) points to the
selected run's prior.pt, and the report records it.

    python scripts/select_prior.py --data data/ausgrid/snapshots.npz \
        --sweep_dir results/priors/sweep \
        --output results/priors/sweep/selected.json"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

from pose import baselines as bl
from pose.data import SnapshotData
from pose.experiment import file_sha256, git_state, hardware, write_json
from pose.prior import context_tensors, load_prior, select_run


SWEPT = ("context", "width", "dropout", "weight_decay", "num_steps")


def gaussian_endpoint(x, t, mu, cov):
    """E[z | x_t] for z ~ N(mu, cov) on the linear path; rows [B, d]."""
    eye = torch.eye(x.shape[-1], dtype=x.dtype, device=x.device)
    a = t[:, None, None] ** 2 * cov + (1 - t)[:, None, None] ** 2 * eye
    sol = torch.linalg.solve(a, (x - t[:, None] * mu).unsqueeze(-1))
    return mu + t[:, None] * (cov @ sol).squeeze(-1)


def per_snapshot_loss(velocity, z, draws, batch=8192):
    """Flow-matching loss per snapshot [n], averaged over the noise draws."""
    total = torch.zeros(len(z), dtype=torch.float64, device=z.device)
    for t_all, eps_all in draws:
        for start in range(0, len(z), batch):
            sl = slice(start, start + batch)
            t, eps, zz = t_all[sl], eps_all[sl], z[sl]
            x = t[:, None] * zz + (1 - t[:, None]) * eps
            v = velocity(x, t, sl)
            total[sl] += ((v.double() - (zz - eps).double()) ** 2).mean(-1)
    return total / len(draws)


def link_prior(link: str, target: str) -> None:
    """Create or replace the symlink link -> target (atomic rename)."""
    os.makedirs(os.path.dirname(os.path.abspath(link)), exist_ok=True)
    tmp = f"{link}.tmp{os.getpid()}"
    if os.path.lexists(tmp):
        os.remove(tmp)
    os.symlink(target, tmp)
    os.replace(tmp, link)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", required=True)
    p.add_argument("--sweep_dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--reference", nargs="*", default=[], help="priors to report only")
    p.add_argument("--tie", type=float, default=0.002)
    p.add_argument("--num_draws", type=int, default=16)
    p.add_argument("--seed", type=int, default=20261013)
    p.add_argument("--report_only", action="store_true", help="never exit 1")
    p.add_argument(
        "--link_selected", default=None, help="symlink to the selected prior.pt"
    )
    p.add_argument("--bin_min_members", type=int, default=0, help="see run_cell.py")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)

    data = SnapshotData.load(args.data)
    tr, va = data.indices("train"), data.indices("val")
    mean, std = data.p_kw[tr].mean(0), data.p_kw[tr].std(0)
    z = torch.as_tensor(
        (data.p_kw[va] - mean) / std, dtype=torch.float64, device=device
    )
    ctx_np = data.context_of(va)
    ctx = context_tensors(ctx_np, device)
    g = torch.Generator().manual_seed(args.seed)
    draws = [
        (
            torch.rand(len(va), generator=g, dtype=torch.float64).to(device),
            torch.randn(len(va), z.shape[1], generator=g, dtype=torch.float64).to(
                device
            ),
        )
        for _ in range(args.num_draws)
    ]

    bins = bl.ContextBins(data, min_members=args.bin_min_members)
    mus, covs = [], []
    for m, q, w in zip(
        ctx_np["month"], ctx_np["quarter_hour"], ctx_np["weekend"], strict=True
    ):
        mk, ck, _ = bins.gaussian(int(m), int(q), int(w), True)
        mus.append((mk - mean) / std)
        covs.append(ck / np.outer(std, std))
    mu = torch.as_tensor(np.stack(mus), dtype=torch.float64, device=device)
    cov = torch.as_tensor(np.stack(covs), dtype=torch.float64, device=device)

    def gauss_velocity(x, t, sl):
        return (gaussian_endpoint(x, t, mu[sl], cov[sl]) - x) / (1 - t[:, None])

    with torch.no_grad():
        loss_gauss = per_snapshot_loss(gauss_velocity, z, draws).cpu().numpy()

    def evaluate_prior(path):
        prior, _ = load_prior(path, device)
        if not np.allclose(prior.mean_kw.cpu().numpy(), mean, rtol=1e-4, atol=1e-3):
            raise ValueError(f"{path}: standardization differs from the data")

        def velocity(x, t, sl):
            c = {k: v[sl] for k, v in ctx.items()}
            return prior.velocity(x.float(), t.float(), c)

        with torch.no_grad():
            return per_snapshot_loss(velocity, z, draws).cpu().numpy()

    days = data.context["day_of_year"][va]

    def paired(loss):
        diff = loss - loss_gauss
        day_means = np.array([diff[days == d].mean() for d in np.unique(days)])
        se = day_means.std(ddof=1) / np.sqrt(len(day_means))
        return {
            "mean_diff": float(day_means.mean()),
            "se": float(se),
            "num_days": len(day_means),
        }

    rows = []
    for meta_path in sorted(
        glob.glob(os.path.join(args.sweep_dir, "*", "metadata.json"))
    ):
        with open(meta_path) as f:
            meta = json.load(f)
        path = os.path.join(os.path.dirname(meta_path), "prior.pt")
        loss = evaluate_prior(path)
        rows.append(
            {
                "run_name": meta["run_name"],
                "path": path,
                "sha256": file_sha256(path),
                "val_loss": float(loss.mean()),
                "num_parameter": meta["num_parameter"],
                "num_steps": meta["config"]["num_steps"],
                "best_step": meta["best_step"],
                "best_val_loss_ema_training": meta["best_val_loss_ema"],
                "config": {k: meta["config"][k] for k in SWEPT},
                "vs_gaussian": paired(loss),
            }
        )
        print(f"{meta['run_name']}: val {rows[-1]['val_loss']:.5f}", flush=True)
    if not rows:
        raise SystemExit(f"no sweep runs in {args.sweep_dir}")
    refs = []
    for path in args.reference:
        loss = evaluate_prior(path)
        refs.append(
            {"path": path, "val_loss": float(loss.mean()), "vs_gaussian": paired(loss)}
        )

    chosen = select_run(rows, args.tie)
    cmp = chosen["vs_gaussian"]
    passed = cmp["mean_diff"] < -2.0 * cmp["se"]
    report = {
        "config": chosen["config"],
        "selected_run": chosen["run_name"],
        "stop_rule": {
            "rule": "mean_diff(selected - gaussian) < -2 se, validation days as blocks",
            **cmp,
            "pass": bool(passed),
        },
        "gaussian_val_loss": float(loss_gauss.mean()),
        "tie": args.tie,
        "noise": {"seed": args.seed, "num_draws": args.num_draws},
        "runs": sorted(rows, key=lambda r: r["val_loss"]),
        "references": refs,
        "data": {"path": args.data, "sha256": file_sha256(args.data)},
        "git": git_state(),
        "hardware": hardware(),
    }
    write_json(args.output, report)
    if args.link_selected:
        target = os.path.abspath(chosen["path"])
        link_prior(args.link_selected, target)
        report["linked_prior"] = args.link_selected
        report["linked_prior_target"] = target
        write_json(args.output, report)
    print(
        f"gaussian {report['gaussian_val_loss']:.5f}; selected {chosen['run_name']} "
        f"val {chosen['val_loss']:.5f}; diff {cmp['mean_diff']:+.5f} "
        f"(se {cmp['se']:.5f}); stop rule pass: {passed}",
        flush=True,
    )
    if not passed and not args.report_only:
        sys.exit(1)


if __name__ == "__main__":
    main()

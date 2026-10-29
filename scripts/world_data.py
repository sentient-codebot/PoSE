"""Draw the model world from a teacher prior and write the training-size data.

The teacher (train_prior.py --train_splits train,val,test --checkpoint last)
is sampled without measurements at every 15-minute step of --years virtual
copies of the 2024 calendar. One SnapshotData file per size is written to
--output_dir/<size>/snapshots.npz, each a mask over the same draws
(pose.world), with scenario "world-<scenario>" so observations and initial
noise are paired across sizes. A realism check compares the teacher's first
year with the real data: per-node mean and standard deviation by hour bin,
reverse head flow, voltage-band and overload rates.

    python scripts/world_data.py --teacher results/world/teacher/base/prior.pt \
        --real_data data/fluvius/base/snapshots.npz --scenario base \
        --output_dir data/world/base"""

import argparse
import os
import time

import numpy as np
import torch

from pose.data import HOUR_BINS, SnapshotData, hour_bin
from pose.experiment import (
    file_sha256,
    git_state,
    hardware,
    stable_seed,
    write_json,
)
from pose.powerflow import RadialPowerFlow, branch_ratings_a
from pose.prior import context_tensors, load_prior, sample_prior
from pose.world import world_calendar, world_valid_mask


SIZES = {"n2k": (1, 0.1), "n19k": (1, 1.0), "n190k": (10, 1.0)}  # years, fraction


def realism(p_kw: np.ndarray, qh: np.ndarray, pf: RadialPowerFlow, ratings) -> dict:
    out = {}
    with torch.no_grad():
        res = pf(torch.as_tensor(p_kw, dtype=torch.float64))
    vm = res.vm.numpy()
    loading = res.branch_current_a.numpy() / ratings * 100.0
    bins = hour_bin(qh)
    for name in list(HOUR_BINS) + ["all"]:
        m = np.ones(len(p_kw), bool) if name == "all" else bins == name
        out[name] = {
            "count": int(m.sum()),
            "node_mean_kw": p_kw[m].mean(0).round(3).tolist(),
            "node_sd_kw": p_kw[m].std(0).round(3).tolist(),
            "total_mean_kw": float(p_kw[m].sum(1).mean()),
            "total_sd_kw": float(p_kw[m].sum(1).std()),
            "frac_reverse_head_flow": float((p_kw[m].sum(1) < 0).mean()),
            "frac_v_outside_095_105": float(
                ((vm[m] < 0.95) | (vm[m] > 1.05)).any(1).mean()
            ),
            "frac_overload": float((loading[m] > 100).any(1).mean()),
            "vm_min_q01": float(np.quantile(vm[m].min(1), 0.01)),
            "vm_max_q99": float(np.quantile(vm[m].max(1), 0.99)),
        }
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--teacher", required=True)
    p.add_argument("--real_data", required=True)
    p.add_argument("--scenario", required=True, help="base or high_pv")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--years", type=int, default=10)
    p.add_argument("--num_steps", type=int, default=200)
    p.add_argument("--batch", type=int, default=65536)
    p.add_argument("--sizes", nargs="+", default=list(SIZES))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)
    t0 = time.time()

    teacher, _ = load_prior(args.teacher, device)
    cal = world_calendar(args.years)
    n = len(cal["split"])
    p_kw = np.empty((n, teacher.mean_kw.shape[0]), dtype=np.float64)
    ctx_all = context_tensors(
        {k: cal[k] for k in ("month", "quarter_hour", "weekend")}, device
    )
    for start in range(0, n, args.batch):
        sl = slice(start, min(start + args.batch, n))
        g = torch.Generator().manual_seed(stable_seed("world", args.scenario, start))
        x0 = torch.randn(sl.stop - sl.start, p_kw.shape[1], generator=g).to(device)
        ctx = {k: v[sl] for k, v in ctx_all.items()}
        p_kw[sl] = sample_prior(teacher, ctx, x0, args.num_steps).cpu().numpy()
        print(f"drawn {sl.stop}/{n}", flush=True)

    real = SnapshotData.load(args.real_data)
    pf = RadialPowerFlow()
    ratings = branch_ratings_a(pf)
    year0 = cal["year"] == 0
    rv = real.valid
    check = {
        "teacher_year0": realism(p_kw[year0], cal["quarter_hour"][year0], pf, ratings),
        "real": realism(real.p_kw[rv], real.context["quarter_hour"][rv], pf, ratings),
    }
    context = {k: v for k, v in cal.items() if k != "year"}
    sizes = {}
    for name in args.sizes:
        years_used, fraction = SIZES[name]
        if years_used > args.years:
            raise SystemExit(f"{name} needs {years_used} years, have {args.years}")
        valid = world_valid_mask(cal, years_used, fraction)
        data = SnapshotData(
            np.where(valid[:, None], p_kw, np.nan),
            valid,
            context,
            {
                "scenario": f"world-{args.scenario}",
                "size": name,
                "teacher": args.teacher,
                "teacher_sha256": file_sha256(args.teacher),
                "years": args.years,
                "train_years": years_used,
                "train_fraction": fraction,
                "steps_per_split": {
                    s: int((valid & (context["split"] == i)).sum())
                    for s, i in (("train", 0), ("val", 1), ("test", 2))
                },
                "git": git_state(),
            },
        )
        path = os.path.join(args.output_dir, name, "snapshots.npz")
        data.save(path)
        sizes[name] = {
            "path": path,
            "sha256": file_sha256(path),
            **data.meta["steps_per_split"],
        }
        print(name, data.meta["steps_per_split"], flush=True)
    write_json(
        os.path.join(args.output_dir, "world_report.json"),
        {
            "scenario": args.scenario,
            "teacher": args.teacher,
            "sizes": sizes,
            "realism": check,
            "git": git_state(),
            "hardware": hardware(),
            "wall_time_s": time.time() - t0,
        },
    )
    for name in list(HOUR_BINS) + ["all"]:
        a, b = check["teacher_year0"][name], check["real"][name]
        print(
            f"{name:8s} total mean {a['total_mean_kw']:8.1f} / real {b['total_mean_kw']:8.1f} kW; "
            f"sd {a['total_sd_kw']:7.1f} / {b['total_sd_kw']:7.1f}; reverse {a['frac_reverse_head_flow']:.3f} / "
            f"{b['frac_reverse_head_flow']:.3f}; V outside {a['frac_v_outside_095_105']:.3f} / "
            f"{b['frac_v_outside_095_105']:.3f}; overload {a['frac_overload']:.3f} / {b['frac_overload']:.3f}"
        )


if __name__ == "__main__":
    main()

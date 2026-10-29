"""Run one (setting, method) cell: posterior samples and per-snapshot metrics.

Methods: M0 projection, M1 isotropic, M2 PoSE (endpoint correction with the
Euler or ancestral sampler); B1 WLS with a diagonal Gaussian prior, B2
full-covariance Gaussian MAP + Laplace, B3 Gaussian mixture + Kalman, B4 DPS,
B5 MLP regression with MC dropout, B6 conditional flow matching (trained here
for this measurement set and noise level), B7 pooled Gaussian.

--topology runs the cell on a radial reconfiguration of the IEEE 33-bus
feeder (feeder.RECONFIGURATIONS); the prior and the baseline statistics are
reused, and h, H, the ratings and the linearization error are rebuilt.
--supervised_train_feeder base trains B5 / B6 on the base topology and applies
them to the reconfigured feeder (label suffix -frozen).

--physics FEEDER.json with --mset S2-3ph runs the cell on the unbalanced
three-phase model of the feeder (threephase.py): the data, prior and baseline
statistics are unchanged, and h, H, the ratings, the noise and the
linearization error come from the three-phase physics. Voltage and branch
metrics then run over node-phases and branch-phases.

Outputs in --output_dir: metrics.json (aggregates by hour bin, runtime,
provenance), per_snapshot.csv, and samples.npz when --save_samples.

Example:
    python scripts/run_cell.py --data data/ausgrid/snapshots.npz \
        --prior checkpoints/prior_ausgrid_base.pt --mset S2 --noise nominal \
        --method M2 --design extended --output_dir results/ausgrid_S2_nominal_M2
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from pose import baselines as bl
from pose.data import (
    HOUR_BINS,
    SnapshotData,
    hour_bin,
    select_test_snapshots,
)
from pose.experiment import (
    Setting,
    file_sha256,
    generators,
    git_state,
    hardware,
    stable_seed,
    write_json,
)
from pose.feeder import (
    RECONFIGURATIONS,
    Feeder,
    reconfigured_ieee33,
    state_mode,
)
from pose.guidance import (
    DEFAULT_GRID_POWER,
    SIGMA_MODES,
    VARIANTS,
    EndpointCorrector,
    sample_corrected,
    sample_dps,
)
from pose.metrics import dso_summary, evaluate
from pose.prior import PriorConfig, context_tensors, load_prior
from pose.threephase import ThreePhaseFeeder


SAMPLING_METHODS = ("M0", "M1", "M2", "B4", "B6")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", required=True)
    p.add_argument("--prior", default=None, help="prior checkpoint (M0-M2, B4)")
    p.add_argument("--scenario", default=None, help="default: from data meta")
    p.add_argument("--mset", default="S2", choices=["S1", "S2", "S3", "S2-3ph"])
    p.add_argument(
        "--physics",
        default=None,
        help="three-phase feeder JSON (threephase.py); requires --mset S2-3ph",
    )
    p.add_argument("--noise", default="nominal", choices=["nominal", "high"])
    p.add_argument("--method", required=True)
    p.add_argument("--sampler", default="euler", choices=["euler", "ancestral"])
    p.add_argument(
        "--design",
        default="extended",
        choices=["pilot", "extended", "test", "val"],
        help="test snapshots: 'extended' (1,000, paper design), 'pilot' (300), "
        "'test' (the whole test split) or 'val' (200 validation snapshots)",
    )
    p.add_argument("--max_snapshots", type=int, default=None, help="smoke tests")
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--num_steps", type=int, default=200)
    p.add_argument("--grid_power", type=float, default=DEFAULT_GRID_POWER)
    p.add_argument("--chunk", type=int, default=25, help="snapshots per batch")
    p.add_argument("--hyper", default=None, help="tuned hyperparameters JSON (B3, B4)")
    p.add_argument("--b3_components", type=int, default=None)
    p.add_argument("--dps_step", type=float, default=None)
    p.add_argument("--b5_steps", type=int, default=20_000)
    p.add_argument("--b6_steps", type=int, default=None, help="default: prior budget")
    p.add_argument(
        "--b6_config_json",
        default=None,
        help="PriorConfig overrides for B6 (a select_prior.py 'config'); applied "
        "before --b6_steps",
    )
    p.add_argument("--save_samples", action="store_true")
    p.add_argument(
        "--bin_min_members",
        type=int,
        default=0,
        help="widen B1-B3 context bins with fewer members (0 = off)",
    )
    p.add_argument(
        "--shard",
        type=int,
        default=0,
        help="run contiguous block SHARD of NUM_SHARDS of the snapshots; seeds are "
        "per snapshot, so merge_shards.py reproduces the unsharded cell",
    )
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument(
        "--sigma_mode",
        default="full",
        choices=SIGMA_MODES,
        help="Tweedie covariance mode (M2): 'full' (default) or 'meas' (PoSE-fast)",
    )
    p.add_argument(
        "--snapshot_ids",
        default=None,
        help="JSON file of snapshot IDs; runs only the IDs under --ids_key, "
        "which must belong to the design's test set",
    )
    p.add_argument("--ids_key", default="ids")
    p.add_argument(
        "--topology",
        default=None,
        choices=list(RECONFIGURATIONS),
        help="radial IEEE 33-bus topology (default: the data set's feeder)",
    )
    p.add_argument(
        "--supervised_train_feeder",
        default="same",
        choices=["same", "base"],
        help="B5, B6: train on the cell's feeder or on the base topology",
    )
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    check_args(args)
    return args


def check_args(args) -> None:
    if (args.mset == "S2-3ph") != (args.physics is not None):
        raise ValueError("--mset S2-3ph and --physics require each other")
    if args.physics is not None and args.topology not in (None, "base"):
        raise ValueError("--physics does not combine with --topology")
    if args.physics is not None and args.supervised_train_feeder == "base":
        raise ValueError("--physics does not combine with --supervised_train_feeder")
    if args.supervised_train_feeder == "base" and args.method not in ("B5", "B6"):
        raise ValueError(
            f"--supervised_train_feeder base applies to B5 and B6, not {args.method}"
        )


def select_snapshots(data: SnapshotData, design: str, max_snapshots: int | None):
    if design == "val":
        idx = data.indices("val")
        rng = np.random.default_rng(20261015)
        idx = np.sort(rng.choice(idx, size=min(200, len(idx)), replace=False))
    elif design == "test":
        # The whole test split, in order, for runs that need a contiguous
        # stretch of test days rather than one of the stratified designs.
        idx = data.indices("test")
    else:
        idx = select_test_snapshots(data, design)
    if max_snapshots:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(idx, size=min(max_snapshots, len(idx)), replace=False))
    return idx


def shard_slice(num: int, shard: int, num_shards: int) -> slice:
    if not 0 <= shard < num_shards:
        raise ValueError(f"shard {shard} outside 0..{num_shards - 1}")
    bounds = np.linspace(0, num, num_shards + 1).round().astype(int)
    return slice(bounds[shard], bounds[shard + 1])


def summarize(table: pd.DataFrame, metric_cols: list[str], k_samples: int):
    """Metric means and DSO summaries, overall and by hour bin."""
    summary = {"all": table[metric_cols].mean().to_dict()}
    dso = {"all": dso_summary(table, k_samples)}
    for name in list(HOUR_BINS) + ["other"]:
        sub = table[table["hour_bin"] == name]
        if len(sub):
            summary[name] = {**sub[metric_cols].mean().to_dict(), "count": len(sub)}
            dso[name] = dso_summary(sub, k_samples)
    return summary, dso


def main():
    args = parse_args()
    device = torch.device(args.device)
    data = SnapshotData.load(args.data)
    scenario = args.scenario or data.meta.get("scenario", "synthetic")
    t_topo = time.time()
    base_feeder = Feeder.from_meta(data.meta)
    feeder = (
        reconfigured_ieee33(
            RECONFIGURATIONS[args.topology], name=args.topology, base=base_feeder
        )
        if args.topology and args.topology != "base"
        else base_feeder
    )
    physics = None
    if args.physics:
        with open(args.physics) as f:
            physics = ThreePhaseFeeder.from_dict(json.load(f))
        mset = physics.measurement_set(args.mset)
    else:
        mset = feeder.measurement_set(args.mset)
    setting = Setting(
        scenario, mset, args.noise, device, feeder, state_mode(data.meta), physics
    )
    lin = setting.linearization(data)
    lin_cov = torch.as_tensor(lin.cov, dtype=torch.float64, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    topology_update_s = time.time() - t_topo
    snapshots = select_snapshots(data, args.design, args.max_snapshots)
    if args.snapshot_ids:
        with open(args.snapshot_ids) as f:
            frozen = np.asarray(json.load(f)["ids"][args.ids_key], dtype=np.int64)
        if not np.isin(frozen, snapshots).all():
            raise ValueError("frozen IDs outside the design's test set")
        snapshots = np.sort(frozen)
    truth, y = setting.observe(data, snapshots)
    if args.num_shards > 1:
        keep = shard_slice(len(snapshots), args.shard, args.num_shards)
        snapshots, truth, y = snapshots[keep], truth[keep], y[keep]
    ctx_np = data.context_of(snapshots)
    ctx = context_tensors(ctx_np, device)
    method = args.method
    label = f"{method}-{args.sampler}" if method in VARIANTS else method
    if method in SAMPLING_METHODS and args.num_steps != 200:
        label += f"-s{args.num_steps}"
    if method in VARIANTS and args.sigma_mode != "full":
        label += f"-{args.sigma_mode}"
    frozen = args.supervised_train_feeder == "base"
    if frozen:
        label += "-frozen"
    hyper = {}
    if args.hyper and os.path.exists(args.hyper):
        with open(args.hyper) as f:
            hyper = json.load(f)
    info: dict = {"method": method, "label": label, "setting": setting.name}
    info["topology"] = args.topology or "base"
    info["open_branches"] = list(RECONFIGURATIONS[args.topology or "base"])
    info["topology_update_s"] = topology_update_s
    if physics is not None:
        info["physics"] = {
            "path": args.physics,
            "sha256": file_sha256(args.physics),
            "num_node_phases": setting.power_flow.num_node_phases,
            "num_branch_phases": setting.power_flow.num_branch_phases,
            "num_channels": mset.num_channels,
        }
    if frozen:
        info["supervised_train_feeder"] = "base"
    if args.sigma_mode != "full":
        info["sigma_mode"] = args.sigma_mode
    if args.snapshot_ids:
        info["snapshot_ids"] = {
            "path": args.snapshot_ids,
            "sha256": file_sha256(args.snapshot_ids),
            "key": args.ids_key,
        }
    if args.num_shards > 1:
        info["shard"] = {"shard": args.shard, "num_shards": args.num_shards}

    prior = None
    if method in ("M0", "M1", "M2", "B4"):
        if not args.prior:
            raise ValueError(f"{method} needs --prior")
        prior, ckpt = load_prior(args.prior, device)
        info["prior"] = {
            "path": args.prior,
            "sha256": file_sha256(args.prior),
            "seed": ckpt.get("seed"),
            "best_step": ckpt.get("history", {}).get("best_step"),
        }
    info["linearization_error"] = {"mean": lin.mean, "std": np.sqrt(np.diag(lin.cov))}

    t_setup = time.time()
    extra_state: dict = {}
    if method in ("B1", "B2"):
        extra_state["bins"] = bl.ContextBins(data, min_members=args.bin_min_members)
    elif method == "B7":
        cfg7 = hyper["B7"]
        extra_state["bins"] = bl.KernelBins(
            data, cfg7["h_hours"], cfg7["h_months"], cfg7["alpha"]
        )
        info["b7"] = cfg7
    elif method == "B3":
        k = args.b3_components or hyper.get("B3", {}).get("num_components", 2)
        extra_state["mix"] = bl.MixtureBaseline(
            bl.ContextBins(data, min_members=args.bin_min_members), k
        )
        info["b3_components"] = k
    elif method in ("B5", "B6"):
        idx = data.indices("train")
        train_setting = (
            Setting(
                scenario,
                base_feeder.measurement_set(args.mset),
                args.noise,
                device,
                base_feeder,
                state_mode(data.meta),
            )
            if frozen
            else setting
        )
        with torch.no_grad():
            h_clean = (
                train_setting.measurement(
                    torch.as_tensor(data.p_kw[idx], dtype=torch.float64, device=device)
                )
                .cpu()
                .numpy()
            )
        trainer = bl.SupervisedTrainer(
            data.p_kw[idx],
            data.context_of(idx),
            h_clean,
            train_setting.noise_std.cpu().numpy(),
            device,
        )
        extra_state["trainer"] = trainer
        seed = stable_seed("train", method, setting.name)
        if method == "B5":
            extra_state["model"] = bl.train_regression(trainer, args.b5_steps, seed)
        else:
            cfg = PriorConfig()
            if args.b6_config_json:
                with open(args.b6_config_json) as f:
                    for name, value in json.load(f)["config"].items():
                        setattr(cfg, name, value)
                cfg.warmup_steps = min(cfg.warmup_steps, max(1, cfg.num_steps // 10))
            if args.b6_steps:
                cfg.num_steps = args.b6_steps
                cfg.warmup_steps = min(cfg.warmup_steps, max(1, args.b6_steps // 10))
            extra_state["model"] = bl.train_conditional_flow(trainer, cfg, seed)
            info["b6_config"] = vars(cfg)
    elif method == "B4":
        info["dps_step"] = args.dps_step or hyper.get("B4", {}).get("step_size", 1.0)
    setup_time = time.time() - t_setup

    all_samples, run_time = [], 0.0
    for start in range(0, len(snapshots), args.chunk):
        sl = slice(start, start + args.chunk)
        snaps = snapshots[sl]
        c_chunk = {k: v[sl] for k, v in ctx.items()}
        c_np = {k: v[sl] for k, v in ctx_np.items()}
        y_chunk = y[sl]
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        init = generators("init", scenario, snaps)
        if method in VARIANTS:
            corrector = EndpointCorrector(
                prior,
                setting.measurement,
                VARIANTS[method],
                setting.h_lin,
                setting.noise_cov,
                lin_cov,
                sigma_mode=args.sigma_mode,
            )
            samples = sample_corrected(
                prior,
                corrector,
                c_chunk,
                y_chunk,
                args.num_samples,
                args.num_steps,
                init,
                args.sampler,
                generators("ancestral", scenario, snaps),
                args.grid_power,
            )
        elif method == "B4":
            samples = sample_dps(
                prior,
                setting.measurement,
                setting.noise_cov,
                c_chunk,
                y_chunk,
                args.num_samples,
                args.num_steps,
                init,
                info["dps_step"],
                args.grid_power,
            )
        elif method in ("B1", "B2", "B7"):
            _, samples = bl.gaussian_baseline(
                extra_state["bins"],
                setting.measurement,
                c_np,
                y_chunk,
                setting.noise_cov,
                method != "B1",
                args.num_samples,
                generators("draw", scenario, snaps),
            )
        elif method == "B3":
            r_eff = (setting.noise_cov + lin_cov).cpu().numpy()
            offset = setting.c0.cpu().numpy() + lin.mean
            rngs = [
                np.random.default_rng(stable_seed("draw", scenario, int(i)))
                for i in snaps
            ]
            _, draws = extra_state["mix"].posterior(
                c_np,
                y_chunk.cpu().numpy(),
                setting.h_lin.cpu().numpy(),
                offset,
                r_eff,
                args.num_samples,
                rngs,
            )
            samples = torch.as_tensor(draws, dtype=torch.float64, device=device)
        elif method == "B5":
            _, samples = bl.regression_predict(
                extra_state["model"],
                extra_state["trainer"],
                y_chunk,
                c_chunk,
                args.num_samples,
                stable_seed("dropout", scenario, int(snaps[0])),
            )
        elif method == "B6":
            samples = bl.sample_conditional_flow(
                extra_state["model"],
                extra_state["trainer"],
                y_chunk,
                c_chunk,
                args.num_samples,
                args.num_steps,
                init,
                args.grid_power,
            )
        else:
            raise ValueError(f"unknown method {method}")
        if device.type == "cuda":
            torch.cuda.synchronize()
        run_time += time.time() - t0
        all_samples.append(samples.double())
        print(f"{label}: {start + len(snaps)}/{len(snapshots)} snapshots", flush=True)
    samples = torch.cat(all_samples)

    per = {}
    for start in range(0, len(snapshots), args.chunk):
        sl = slice(start, start + args.chunk)
        m = evaluate(
            samples[sl],
            truth[sl],
            y[sl],
            setting.measurement,
            setting.noise_std,
            setting.ratings,
            setting.power_flow,
        )
        for k, v in m.items():
            per.setdefault(k, []).append(v)
    per = {k: np.concatenate(v) for k, v in per.items()}
    table = pd.DataFrame(per)
    table.insert(0, "snapshot", snapshots)
    table.insert(1, "day_of_year", data.context["day_of_year"][snapshots])
    table.insert(2, "quarter_hour", data.context["quarter_hour"][snapshots])
    table.insert(3, "hour_bin", hour_bin(data.context["quarter_hour"][snapshots]))
    os.makedirs(args.output_dir, exist_ok=True)
    table.to_csv(os.path.join(args.output_dir, "per_snapshot.csv"), index=False)

    metric_cols = list(per)
    summary, dso = summarize(table, metric_cols, samples.shape[1])
    info.update(
        {
            "summary": summary,
            "dso": dso,
            "metric_cols": metric_cols,
            "num_snapshots": len(snapshots),
            "num_samples": args.num_samples,
            "num_steps": args.num_steps if method in SAMPLING_METHODS else None,
            "grid_power": args.grid_power if method in SAMPLING_METHODS else None,
            "sampler": args.sampler if method in VARIANTS else None,
            "design": args.design,
            "runtime_per_snapshot_s": run_time / len(snapshots),
            "setup_time_s": setup_time,
            "data": {
                "path": args.data,
                "sha256": file_sha256(args.data),
                "p_kw_sha256": data.meta.get("p_kw_sha256"),
                "split_sha256": data.meta.get("split_sha256"),
            },
            "git": git_state(),
            "hardware": hardware(),
            "args": vars(args),
        }
    )
    write_json(os.path.join(args.output_dir, "metrics.json"), info)
    if args.save_samples:
        np.savez_compressed(
            os.path.join(args.output_dir, "samples.npz"),
            samples=samples.float().cpu().numpy(),
            truth=truth.cpu().numpy(),
            y=y.cpu().numpy(),
            snapshots=snapshots,
        )
    print(json.dumps({k: round(v, 5) for k, v in summary["all"].items()}), flush=True)


if __name__ == "__main__":
    main()

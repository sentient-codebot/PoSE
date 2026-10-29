"""Train the conditional flow matching prior p(s | c) for one scenario and seed.

Outputs (--output_dir/--run_name): prior.pt (EMA weights at the lowest
validation loss, standardization, config, history) and metadata.json.

Example:
    python scripts/train_prior.py --data data/ausgrid/snapshots.npz \
        --run_name ausgrid-s0 --seed 0"""

import argparse
import json
import os
from dataclasses import asdict

import numpy as np
import torch

from pose.data import SnapshotData
from pose.experiment import (
    file_sha256,
    git_state,
    hardware,
    write_json,
)
from pose.prior import PriorConfig, save_prior, train_prior


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", required=True)
    p.add_argument("--output_dir", default="results/priors")
    p.add_argument("--run_name", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_steps", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--num_blocks", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--val_every", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--context", default=None, choices=["full", "cyclic"])
    p.add_argument(
        "--train_splits",
        default="train",
        help="comma-separated splits to train on (a model-world teacher uses "
        "train,val,test)",
    )
    p.add_argument("--checkpoint", default="best", choices=["best", "last"])
    p.add_argument(
        "--config_json",
        default=None,
        help="JSON with a 'config' dict of PriorConfig overrides (select_prior.py); "
        "explicit flags take precedence",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    cfg = PriorConfig()
    if args.config_json:
        with open(args.config_json) as f:
            for name, value in json.load(f)["config"].items():
                if not hasattr(cfg, name):
                    raise ValueError(f"unknown PriorConfig field {name}")
                setattr(cfg, name, value)
    names = ("num_steps", "width", "num_blocks", "batch_size", "val_every")
    names += ("dropout", "weight_decay", "context")
    for name in names:
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)
    cfg.warmup_steps = min(cfg.warmup_steps, max(1, cfg.num_steps // 10))
    cfg.val_every = min(cfg.val_every, cfg.num_steps)

    data = SnapshotData.load(args.data)
    tr = np.sort(
        np.concatenate([data.indices(s) for s in args.train_splits.split(",")])
    )
    va = data.indices("val")
    out = os.path.join(args.output_dir, args.run_name)
    os.makedirs(out, exist_ok=True)
    log_path = os.path.join(out, "train_log.txt")

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    log(f"train {len(tr)} val {len(va)} snapshots; config {asdict(cfg)}")
    prior, history = train_prior(
        cfg,
        data.p_kw[tr],
        data.context_of(tr),
        data.p_kw[va],
        data.context_of(va),
        args.seed,
        torch.device(args.device),
        log,
        args.checkpoint,
    )
    num_param = sum(p.numel() for p in prior.net.parameters())
    meta = {
        "run_name": args.run_name,
        "seed": args.seed,
        "num_parameter": num_param,
        "num_train_snapshots": len(tr),
        "num_val_snapshots": len(va),
        "config": asdict(cfg),
        "train_splits": args.train_splits,
        "checkpoint": args.checkpoint,
        "data": {
            "path": args.data,
            "sha256": file_sha256(args.data),
            "p_kw_sha256": data.meta.get("p_kw_sha256"),
            "split_sha256": data.meta.get("split_sha256"),
            "scenario": data.meta.get("scenario"),
        },
        "git": git_state(),
        "hardware": hardware(),
        "best_step": history["best_step"],
        "best_val_loss_ema": history["best_val_loss_ema"],
        "wall_time_s": history["wall_time_s"],
    }
    save_prior(
        os.path.join(out, "prior.pt"),
        prior,
        {"seed": args.seed, "history": history, "meta": meta},
    )
    write_json(os.path.join(out, "metadata.json"), {**meta, "history": history})
    log(
        f"done: best step {history['best_step']} val {history['best_val_loss_ema']:.5f}"
    )


if __name__ == "__main__":
    main()

"""Unconditional samples of the learned prior p(s | c) for the example week.

Uses SnapshotData.load, load_prior, context_tensors, generators and
sample_prior. The initial noise of each snapshot is the run_cell.py "init"
draw, so the prior samples start from the same noise as the posterior samples
of the same snapshot.

Usage:
    python prior_samples.py --data .../snapshots.npz --prior .../prior.pt \
        --snapshot_ids selected_ids.json --ids_key midday --output prior.npz"""

import argparse
import json
import os

import numpy as np
import torch

import pose as pose_pkg
from pose.data import SnapshotData
from pose.experiment import file_sha256, generators, hardware
from pose.prior import context_tensors, load_prior, sample_prior


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--prior", required=True)
    p.add_argument("--snapshot_ids", required=True)
    p.add_argument("--ids_key", required=True)
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--num_steps", type=int, default=200)
    p.add_argument("--chunk", type=int, default=25)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    data = SnapshotData.load(args.data)
    scenario = data.meta.get("scenario")
    with open(args.snapshot_ids) as fh:
        snaps = np.array(json.load(fh)["ids"][args.ids_key], dtype=np.int64)
    prior, ckpt = load_prior(args.prior, device)
    dim = prior.mean_kw.shape[0]
    out = []
    for c0 in range(0, len(snaps), args.chunk):
        cs = snaps[c0 : c0 + args.chunk]
        ctx = context_tensors(data.context_of(cs), device)
        x0 = torch.cat(
            [
                torch.randn(args.num_samples, dim, generator=g)
                for g in generators("init", scenario, cs)
            ]
        ).to(device)
        rows = {k: v.repeat_interleave(args.num_samples, 0) for k, v in ctx.items()}
        s = sample_prior(prior, rows, x0, args.num_steps)
        out.append(s.view(len(cs), args.num_samples, dim).float().cpu().numpy())
        print(f"prior: {c0 + len(cs)}/{len(snaps)}", flush=True)
    samples = np.concatenate(out)
    np.savez_compressed(args.output, samples=samples, snapshots=snaps)
    meta = {
        "args": vars(args),
        "package": pose_pkg.__file__,
        "scenario": scenario,
        "data_sha256": file_sha256(args.data),
        "prior_sha256": file_sha256(args.prior),
        "prior_seed": ckpt.get("seed"),
        "hardware": hardware(),
        "shape": list(samples.shape),
    }
    with open(os.path.splitext(args.output)[0] + ".json", "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    print(f"wrote {args.output} {samples.shape}", flush=True)


if __name__ == "__main__":
    main()

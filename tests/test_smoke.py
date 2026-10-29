"""End-to-end CPU smoke test: train a tiny prior and run one PoSE cell."""

import json
import os
import subprocess
import sys

import numpy as np

from pose.data import synthetic_snapshots
from pose.prior import PriorConfig, save_prior, train_prior


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_train_prior_and_run_one_cell(tmp_path):
    import torch

    data = synthetic_snapshots(0)
    data_path = str(tmp_path / "data" / "snapshots.npz")
    data.save(data_path)
    tr, va = data.indices("train"), data.indices("val")
    cfg = PriorConfig(
        width=32,
        num_blocks=1,
        emb_dim=16,
        batch_size=16,
        num_steps=2,
        warmup_steps=1,
        val_every=1,
        val_noise_draws=1,
        context="cyclic",
    )
    prior, _ = train_prior(
        cfg,
        data.p_kw[tr],
        data.context_of(tr),
        data.p_kw[va],
        data.context_of(va),
        seed=0,
        device=torch.device("cpu"),
        log=lambda _msg: None,
        checkpoint="last",
    )
    prior_path = str(tmp_path / "prior.pt")
    save_prior(prior_path, prior, {"seed": 0})

    out = tmp_path / "cell"
    cmd = [
        sys.executable,
        os.path.join(ROOT, "scripts", "run_cell.py"),
        "--data",
        data_path,
        "--prior",
        prior_path,
        "--mset",
        "S2",
        "--noise",
        "nominal",
        "--method",
        "M2",
        "--design",
        "extended",
        "--max_snapshots",
        "2",
        "--num_samples",
        "4",
        "--num_steps",
        "4",
        "--chunk",
        "2",
        "--device",
        "cpu",
        "--output_dir",
        str(out),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        os.path.join(ROOT, "src") + os.pathsep + env.get("PYTHONPATH", "")
    )
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)

    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["num_snapshots"] == 2
    assert metrics["num_samples"] == 4
    assert np.isfinite(metrics["summary"]["all"]["crps_kw"])
    table = np.genfromtxt(out / "per_snapshot.csv", delimiter=",", names=True)
    assert len(table) == 2

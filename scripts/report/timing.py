"""Runtime of M2 per covariance mode.

Matched to run_cell.py: one batch of snapshots (25 on Ausgrid, 10 on Iowa),
100 samples, the GPU synchronized before starting and before stopping each
timer. Per mode and step count: a warm-up (one snapshot, 5 steps), then the
throughput batch. Optionally the latency (three single-snapshot runs at 200
steps, median) and an instrumented breakdown (5 snapshots, 200 steps, a GPU
synchronization at every section boundary), kept apart from the uninstrumented
numbers.

    python scripts/report/timing.py --data DATA --prior PRIOR --mset S2 \\
        --ids_key t1_ausgrid --modes full,meas --steps 200,50 --latency \\
        --breakdown --output t1.json
"""

import argparse
import time

import numpy as np
import torch

from pose.data import SnapshotData, select_test_snapshots
from pose.experiment import (
    Setting,
    file_sha256,
    generators,
    git_state,
    hardware,
    write_json,
)
from pose.guidance import (
    DEFAULT_GRID_POWER,
    SIGMA_MODES,
    VARIANTS,
    EndpointCorrector,
    StepTimer,
    sample_corrected,
)
from pose.prior import context_tensors, load_prior


NUM_SAMPLES = 100
SUBSET_STRIDE = 5


def frozen_ids(data: SnapshotData, design: str = "extended") -> dict[str, list[int]]:
    """Stratified snapshot IDs for the timing runs.

    t3: within each hour-of-day stratum of the stored test set, the snapshots
    in date order, every fifth starting with the first. t1_ausgrid and
    t1_iowa take every eighth and twentieth T3 ID (first 25 and 10).
    """
    test = select_test_snapshots(data, design)
    hour = data.context["quarter_hour"][test] // 4
    t3 = []
    for h in range(24):
        members = np.sort(test[hour == h])
        t3.extend(members[::SUBSET_STRIDE].tolist())
    t3 = sorted(int(i) for i in t3)
    return {
        "t3": t3,
        "t1_ausgrid": t3[::8][:25],
        "t1_iowa": t3[::20][:10],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", required=True)
    p.add_argument("--prior", required=True)
    p.add_argument("--mset", default="S2")
    p.add_argument("--noise", default="nominal")
    p.add_argument("--design", default="extended")
    p.add_argument("--ids_key", default="t1_ausgrid")
    p.add_argument("--modes", default="full,meas")
    p.add_argument("--steps", default="200")
    p.add_argument("--latency", action="store_true")
    p.add_argument("--breakdown", action="store_true")
    p.add_argument("--profile_steps", type=int, default=200, help="latency, breakdown")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    modes = [m for m in args.modes.split(",") if m]
    unknown = set(modes) - set(SIGMA_MODES)
    if unknown:
        raise SystemExit(f"unknown sigma mode(s): {sorted(unknown)}")

    device = torch.device(args.device)
    data = SnapshotData.load(args.data)
    scenario = data.meta.get("scenario", "synthetic")
    setting = Setting.from_names(scenario, args.mset, args.noise, device, data.meta)
    snaps = np.asarray(frozen_ids(data, args.design)[args.ids_key], dtype=np.int64)
    _, y = setting.observe(data, snaps)
    ctx = context_tensors(data.context_of(snaps), device)
    prior, _ = load_prior(args.prior, device)
    lin_cov = torch.as_tensor(
        setting.linearization(data).cov, dtype=torch.float64, device=device
    )

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def run(mode, rows, steps, timer=None):
        corr = EndpointCorrector(
            prior,
            setting.measurement,
            VARIANTS["M2"],
            setting.h_lin,
            setting.noise_cov,
            lin_cov,
            sigma_mode=mode,
            timer=timer,
        )
        sub = {k: v[rows] for k, v in ctx.items()}
        sync()
        t0 = time.perf_counter()
        sample_corrected(
            prior,
            corr,
            sub,
            y[rows],
            NUM_SAMPLES,
            steps,
            generators("init", scenario, snaps[rows]),
            "euler",
            None,
            DEFAULT_GRID_POWER,
        )
        sync()
        return time.perf_counter() - t0

    results = {}
    all_rows = np.arange(len(snaps))
    for mode in modes:
        res = {"throughput_s_per_snapshot": {}}
        for steps in (int(s) for s in args.steps.split(",")):
            run(mode, all_rows[:1], 5)  # warm-up
            secs = run(mode, all_rows, steps)
            res["throughput_s_per_snapshot"][str(steps)] = secs / len(snaps)
            print(mode, steps, secs / len(snaps), flush=True)
        if args.latency:
            run(mode, all_rows[:1], 5)
            lat = [run(mode, all_rows[i : i + 1], args.profile_steps) for i in range(3)]
            res["latency_s"] = lat
            res["latency_median_s"] = float(np.median(lat))
        if args.breakdown:
            timer = StepTimer(device)
            run(mode, all_rows[:5], args.profile_steps, timer)
            res["breakdown_s_per_snapshot"] = {
                k: v / 5 for k, v in timer.totals.items()
            }
        results[mode] = res

    out = {
        "setting": setting.name,
        "state_dim": int(prior.mean_kw.shape[0]),
        "num_channels": int(setting.mset.num_channels),
        "batch_snapshots": len(snaps),
        "num_samples": NUM_SAMPLES,
        "profile_steps": args.profile_steps,
        "snapshots": snaps.tolist(),
        "results": results,
        "data": {"path": args.data, "sha256": file_sha256(args.data)},
        "prior": {"path": args.prior, "sha256": file_sha256(args.prior)},
        "ids": {"key": args.ids_key, "design": args.design},
        "git": git_state(),
        "hardware": hardware(),
        "torch_threads": torch.get_num_threads(),
    }
    write_json(args.output, out)


if __name__ == "__main__":
    main()

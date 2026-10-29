"""Exactness test on the real 32-node data.

A Gaussian is fitted (in the prior's standardized units) to the training
snapshots of one hour bin, 12:00-13:00 on the UTC+1 clock, all months and
day types. Its analytic E[z | x_t] replaces the network, the measurement
model is LinDistFlow for S2 with the nominal noise, so the posterior is
known in closed form. Criteria for M2 with the Euler and the ancestral
sampler, 200 steps on the default time grid (power 2), 4,000 samples:

    mean:       max_i |mean_i - exact_i| / (sd_i / sqrt(n)) <= 4
    spread:     every per-node sd ratio (sample / exact) within
                1 +- (4 / sqrt(2 n) + 0.025), i.e. [0.93, 1.07] at n = 4,000
    covariance: relative Frobenius error <= 2 x that of n exact draws + 0.02

Reported alongside: M0 and M1 on the same problem, and with exact
observations (R -> 1e-8 R) the mean error of M0 and M2 in Monte Carlo
standard errors and the shift ratio <m - mu, m* - mu> / |m* - mu|^2. The
measurements are aggregates rather than a subset of coordinates, so M0's
expected failure shows as a mean error far outside Monte Carlo error."""

import argparse
import time

import numpy as np
import torch

from pose.data import SnapshotData, select_test_snapshots
from pose.experiment import (
    file_sha256,
    git_state,
    hardware,
    write_json,
)
from pose.feeder import MEASUREMENT_SETS
from pose.guidance import (
    DEFAULT_GRID_POWER,
    SIGMA_MODES,
    VARIANTS,
    AnalyticGaussianPrior,
    EndpointCorrector,
    LinearMeasurement,
    gaussian_posterior,
    sample_corrected,
)
from pose.lindistflow import lindistflow_operator


NUM_SAMPLES = 4000
NUM_STEPS = 200
QH_RANGE = (48, 52)


def compare(samples, mean, cov, ref_frob):
    n = samples.shape[0]
    m, c = samples.mean(0), torch.cov(samples.T)
    sd = cov.diag().sqrt()
    mean_z = ((m - mean).abs() / (sd / np.sqrt(n))).max().item()
    ratio = c.diag().sqrt() / sd
    band = 4.0 / np.sqrt(2 * n) + 0.025
    frob = ((c - cov).norm() / cov.norm()).item()
    out = {
        "max_mean_error_mc_se": mean_z,
        "sd_ratio_min": ratio.min().item(),
        "sd_ratio_max": ratio.max().item(),
        "rel_frobenius": frob,
        "rel_frobenius_exact_draws": ref_frob,
        "sd_ratio_band": band,
    }
    out["pass"] = bool(
        mean_z <= 4.0
        and 1 - band <= out["sd_ratio_min"]
        and out["sd_ratio_max"] <= 1 + band
        and frob <= 2 * ref_frob + 0.02
    )
    return out


def shift_ratio(samples, mu, exact_mean):
    d = exact_mean - mu
    return ((samples.mean(0) - mu) @ d / (d @ d)).item()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--num_samples", type=int, default=NUM_SAMPLES)
    p.add_argument("--num_steps", type=int, default=NUM_STEPS)
    p.add_argument(
        "--sigma_mode",
        default="full",
        choices=SIGMA_MODES,
        help="covariance mode of M2 (M0 and M1 are isotropic)",
    )
    args = p.parse_args()
    torch.manual_seed(0)

    data = SnapshotData.load(args.data)
    train = data.indices("train")
    mean_kw = torch.as_tensor(data.p_kw[train].mean(0))
    std_kw = torch.as_tensor(data.p_kw[train].std(0))
    qh = data.context["quarter_hour"][train]
    in_bin = train[(qh >= QH_RANGE[0]) & (qh < QH_RANGE[1])]
    z = (torch.as_tensor(data.p_kw[in_bin]) - mean_kw) / std_kw
    mu_z, cov_z = z.mean(0), torch.cov(z.T)
    prior = AnalyticGaussianPrior(mu_z, cov_z, mean_kw, std_kw)
    mu_s = mean_kw + std_kw * mu_z
    cov_s = std_kw[:, None] * cov_z * std_kw[None, :]

    mset = MEASUREMENT_SETS["S2"]
    h_mat, c0 = (torch.as_tensor(a) for a in lindistflow_operator(mset))
    meas = LinearMeasurement(h_mat, c0)
    noise_cov = torch.diag(torch.as_tensor(mset.noise_std(1.0)) ** 2)
    test = select_test_snapshots(data, "pilot")
    test_qh = data.context["quarter_hour"][test]
    truth_idx = int(test[(test_qh >= QH_RANGE[0]) & (test_qh < QH_RANGE[1])][0])
    s_true = torch.as_tensor(data.p_kw[truth_idx])
    g = torch.Generator().manual_seed(20261015)
    noise = torch.randn(mset.num_channels, generator=g, dtype=torch.float64)
    y = c0 + h_mat @ s_true + noise * noise_cov.diag().sqrt()
    ctx = {
        k: torch.zeros(1, dtype=torch.long)
        for k in ("month", "quarter_hour", "weekend")
    }

    exact_mean, exact_cov = gaussian_posterior(mu_s, cov_s, h_mat, c0, noise_cov, y)
    root = torch.linalg.cholesky(exact_cov + 1e-12 * torch.eye(32, dtype=torch.float64))
    exact_draws = (
        exact_mean
        + torch.randn(args.num_samples, 32, generator=g, dtype=torch.float64) @ root.T
    )
    ref_frob = ((torch.cov(exact_draws.T) - exact_cov).norm() / exact_cov.norm()).item()

    def run(name, sampler, r, y_obs):
        mode = args.sigma_mode if name == "M2" else "full"
        corr = EndpointCorrector(
            prior, meas, VARIANTS[name], h_mat, r, None, sigma_mode=mode
        )
        t0 = time.time()
        out = sample_corrected(
            prior,
            corr,
            ctx,
            y_obs[None],
            args.num_samples,
            args.num_steps,
            [torch.Generator().manual_seed(1)],
            sampler,
            [torch.Generator().manual_seed(2)],
        )[0]
        return out, time.time() - t0

    report = {
        "bin": {"quarter_hours": QH_RANGE, "num_snapshots": len(in_bin)},
        "truth_snapshot": truth_idx,
        "num_samples": args.num_samples,
        "num_steps": args.num_steps,
        "grid_power": DEFAULT_GRID_POWER,
        "measurement_set": "S2",
        "noise": "nominal",
        "sigma_mode_m2": args.sigma_mode,
        "noisy": {},
        "exact_observations": {},
    }
    y_exact = c0 + h_mat @ s_true
    for name, sampler in (
        ("M2", "euler"),
        ("M2", "ancestral"),
        ("M1", "euler"),
        ("M0", "euler"),
    ):
        samples, secs = run(name, sampler, noise_cov, y)
        res = compare(samples, exact_mean, exact_cov, ref_frob)
        res["shift_ratio"] = shift_ratio(samples, mu_s, exact_mean)
        res["seconds"] = secs
        report["noisy"][f"{name}-{sampler}"] = res
        print(name, sampler, res, flush=True)
    tiny = noise_cov * 1e-8
    exact_mean0, exact_cov0 = gaussian_posterior(mu_s, cov_s, h_mat, c0, tiny, y_exact)
    for name in ("M0", "M2"):
        samples, _ = run(name, "euler", tiny, y_exact)
        sd0 = exact_cov0.diag().clamp(min=1e-12).sqrt()
        err = (
            (samples.mean(0) - exact_mean0).abs() / (sd0 / np.sqrt(len(samples)))
        ).max()
        report["exact_observations"][name] = {
            "shift_ratio": shift_ratio(samples, mu_s, exact_mean0),
            "max_mean_error_mc_se": err.item(),
        }
    report["pass"] = bool(
        report["noisy"]["M2-euler"]["pass"] and report["noisy"]["M2-ancestral"]["pass"]
    )
    report["data"] = {"path": args.data, "sha256": file_sha256(args.data)}
    report["git"] = git_state()
    report["hardware"] = hardware()
    write_json(args.output, report)
    print("pass:", report["pass"], report["exact_observations"], flush=True)


if __name__ == "__main__":
    main()

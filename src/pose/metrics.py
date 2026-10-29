"""Metrics and the paired block bootstrap.

Per snapshot, from samples s^(k) [K, 32] of the injection posterior:

    injection CRPS per node (kW), averaged over nodes;
    energy score of the 32-node vector (kW);
    90% central interval coverage of the injections (fraction of nodes);
    voltage magnitude: RMSE and maximum absolute error over the 32 buses of
        the posterior mean of the AC voltages of the samples (p.u.);
    branch current: mean and maximum absolute error of the posterior mean of
        the AC branch currents, in percent of the branch rating;
    normalized residual chi-squared: mean of ((y - h(s_bar)) / sigma)^2 over
        channels, at the posterior-mean injection s_bar.

A point estimator is evaluated as a single sample (its CRPS equals the MAE).

DSO-relevant metrics (reported, not gated): per bus and snapshot,
violation of the operational voltage band
(0.95-1.05 p.u.) and the statutory band (0.90-1.10 p.u.), and per branch,
loading above 100% of the rating. The predicted violation probability is the
fraction of AC-consistent samples that violate. Per snapshot: the number of
true violations, hits and false alarms at alarm levels p >= 0.5 and p >= 0.1,
the Brier score, and feeder-level flags for any violation (dso_summary turns
them into rates).
"""

import numpy as np
import torch
from torch import Tensor

from .powerflow import MeasurementModel, RadialPowerFlow


V_BAND_OPERATIONAL = (0.95, 1.05)  # p.u.
V_BAND_STATUTORY = (0.90, 1.10)  # p.u.
LOADING_LIMIT_PCT = 100.0  # percent of the branch rating
ALARM_LEVELS = (0.5, 0.1)
DSO_PREFIXES = ("v", "vstat", "i")


def attainable_coverage(num_samples: int, level: float = 0.9) -> float | None:
    """Coverage of the central sample interval (linear interpolation) of K
    exact samples for a new draw: level (K - 1) / (K + 1); None for K = 1."""
    if num_samples < 2:
        return None
    return level * (num_samples - 1) / (num_samples + 1)


def exceedance_counts(prefix: str, prob: Tensor, viol: Tensor) -> dict[str, Tensor]:
    """Per-snapshot violation scores; prob [S, n] in [0, 1], viol [S, n] bool."""
    truth = viol.double()
    out = {
        f"{prefix}_n": torch.full_like(truth[:, 0], viol.shape[-1]),
        f"{prefix}_true": truth.sum(-1),
        f"{prefix}_brier": ((prob.double() - truth) ** 2).mean(-1),
        f"{prefix}_any_true": viol.any(-1).double(),
    }
    for level in ALARM_LEVELS:
        alarm = prob >= level
        tag = f"{round(level * 100):d}"
        out[f"{prefix}_hit{tag}"] = (alarm & viol).double().sum(-1)
        out[f"{prefix}_fa{tag}"] = (alarm & ~viol).double().sum(-1)
        out[f"{prefix}_any_hit{tag}"] = (alarm.any(-1) & viol.any(-1)).double()
        out[f"{prefix}_any_fa{tag}"] = (alarm.any(-1) & ~viol.any(-1)).double()
    return out


def dso_summary(table, num_samples: int) -> dict:
    """Rates from the per-snapshot DSO columns of a table (DataFrame or dict
    of arrays): hit and false-alarm rates per bus or branch and per snapshot,
    mean Brier score, error tails and attainable coverage."""

    def col(name):
        return np.asarray(table[name], dtype=np.float64)

    def ratio(a, b):
        return float(a / b) if b > 0 else float("nan")

    out = {"num_snapshots": len(col("crps_kw"))}
    for prefix in DSO_PREFIXES:
        if f"{prefix}_true" not in table:
            continue
        true, n = col(f"{prefix}_true").sum(), col(f"{prefix}_n").sum()
        any_true = col(f"{prefix}_any_true").sum()
        row = {
            "violations": true,
            "elements": n,
            "snapshots_with_violation": any_true,
            "brier": float(col(f"{prefix}_brier").mean()),
        }
        for level in ALARM_LEVELS:
            tag = f"{round(level * 100):d}"
            row[f"hit_rate{tag}"] = ratio(col(f"{prefix}_hit{tag}").sum(), true)
            row[f"false_alarm_rate{tag}"] = ratio(
                col(f"{prefix}_fa{tag}").sum(), n - true
            )
            row[f"snapshot_detection{tag}"] = ratio(
                col(f"{prefix}_any_hit{tag}").sum(), any_true
            )
            row[f"snapshot_false_alarm{tag}"] = ratio(
                col(f"{prefix}_any_fa{tag}").sum(), out["num_snapshots"] - any_true
            )
        out[prefix] = row
    out["vm_max_abs_pu_p95"] = float(np.quantile(col("vm_max_abs_pu"), 0.95))
    out["branch_max_err_pct_p95"] = float(np.quantile(col("branch_max_err_pct"), 0.95))
    out["coverage90"] = float(col("coverage90").mean()) if num_samples > 1 else None
    out["coverage90_attainable"] = attainable_coverage(num_samples)
    return out


def dso_table_lines(summaries: dict[str, dict]) -> list[str]:
    """Markdown table of dso_summary outputs, one row per method."""

    def f(x, fmt):
        return "n/a" if x is None or np.isnan(x) else format(x, fmt)

    head = (
        "| method | V viol. | V hit (p>=0.5) | V false alarm | V hit (p>=0.1) | V Brier "
        "| I overload | I hit (p>=0.5) | I false alarm | I hit (p>=0.1) | I Brier "
        "| snap. V detect | snap. I detect | max V err p95 (p.u.) | max I err p95 (%) "
        "| cov90 (attainable) |"
    )
    lines = [head, "|---" * 16 + "|"]
    for label, d in summaries.items():
        v, i = d.get("v", {}), d.get("i", {})
        cov = (
            f(d["coverage90"], ".3f")
            + " ("
            + f(d["coverage90_attainable"], ".3f")
            + ")"
        )
        lines.append(
            f"| {label} | {v.get('violations', 0):.0f} | {f(v.get('hit_rate50'), '.3f')} "
            f"| {f(v.get('false_alarm_rate50'), '.4f')} | {f(v.get('hit_rate10'), '.3f')} "
            f"| {f(v.get('brier'), '.4f')} | {i.get('violations', 0):.0f} "
            f"| {f(i.get('hit_rate50'), '.3f')} | {f(i.get('false_alarm_rate50'), '.4f')} "
            f"| {f(i.get('hit_rate10'), '.3f')} | {f(i.get('brier'), '.4f')} "
            f"| {f(v.get('snapshot_detection50'), '.3f')} | {f(i.get('snapshot_detection50'), '.3f')} "
            f"| {d['vm_max_abs_pu_p95']:.5f} | {d['branch_max_err_pct_p95']:.2f} | {cov} |"
        )
    return lines


def crps_samples(samples: Tensor, truth: Tensor) -> Tensor:
    """Sample CRPS [S, d] for samples [S, K, d] and truth [S, d]."""
    k = samples.shape[1]
    term1 = (samples - truth[:, None, :]).abs().mean(1)
    srt, _ = torch.sort(samples, dim=1)
    weights = (
        2 * torch.arange(1, k + 1, device=samples.device, dtype=samples.dtype) - k - 1
    )
    term2 = (weights[None, :, None] * srt).sum(1) / (k * k)
    return term1 - term2


def energy_score(samples: Tensor, truth: Tensor) -> Tensor:
    """Energy score [S] for samples [S, K, d] and truth [S, d]."""
    term1 = (samples - truth[:, None, :]).norm(dim=-1).mean(1)
    pair = torch.cdist(samples, samples)
    term2 = pair.mean(dim=(1, 2))
    return term1 - 0.5 * term2


def interval_coverage(samples: Tensor, truth: Tensor, level: float = 0.9) -> Tensor:
    """Fraction of nodes whose truth lies in the central interval, [S]."""
    lo = torch.quantile(samples, (1 - level) / 2, dim=1)
    hi = torch.quantile(samples, 1 - (1 - level) / 2, dim=1)
    return ((truth >= lo) & (truth <= hi)).double().mean(-1)


@torch.no_grad()
def evaluate(
    samples_kw: Tensor,
    truth_kw: Tensor,
    y: Tensor,
    measurement: MeasurementModel,
    noise_std: Tensor,
    ratings_a: Tensor,
    power_flow: RadialPowerFlow,
) -> dict[str, np.ndarray]:
    """Per-snapshot metrics for samples [S, K, d]; returns {name: [S]}.

    The state is P per node (d = N) or P then Q (d = 2N). Injection scores
    (CRPS, energy score, MAE, coverage) use the active power in kW; with the
    joint state the reactive CRPS and coverage are added in kvar."""
    s, k, d = samples_kw.shape
    n = power_flow.num_nodes
    samples_all, truth_all = samples_kw.double(), truth_kw.double()
    samples_kw, truth_kw = samples_all[..., :n], truth_all[..., :n]
    pf_samples = power_flow(samples_all.reshape(s * k, d))
    pf_true = power_flow(truth_all)
    vm = pf_samples.vm.view(s, k, -1)
    cur = pf_samples.branch_current_a.view(s, k, -1)
    v_err = vm.mean(1) - pf_true.vm
    loading = cur / ratings_a * 100.0
    loading_true = pf_true.branch_current_a / ratings_a * 100.0
    i_err = (cur.mean(1) - pf_true.branch_current_a).abs() / ratings_a * 100.0
    s_bar = samples_kw.mean(1)
    resid = (y.double() - measurement(samples_all.mean(1))) / noise_std.double()
    out = {
        "crps_kw": crps_samples(samples_kw, truth_kw).mean(-1),
        "energy_score_kw": energy_score(samples_kw, truth_kw),
        "mae_kw": (s_bar - truth_kw).abs().mean(-1),
        "coverage90": interval_coverage(samples_kw, truth_kw)
        if k > 1
        else torch.full((s,), float("nan"), dtype=torch.float64),
        "vm_rmse_pu": v_err.pow(2).mean(-1).sqrt(),
        "vm_max_abs_pu": v_err.abs().amax(-1),
        "vm_crps_pu": crps_samples(vm, pf_true.vm).mean(-1),
        "branch_err_pct": i_err.mean(-1),
        "branch_max_err_pct": i_err.amax(-1),
        "chi2_norm": resid.pow(2).mean(-1),
        "max_pf_mismatch": pf_samples.mismatch.view(s, k).amax(-1),
    }
    if d == 2 * n:
        q_s, q_t = samples_all[..., n:], truth_all[..., n:]
        out["crps_q_kvar"] = crps_samples(q_s, q_t).mean(-1)
        out["coverage90_q"] = (
            interval_coverage(q_s, q_t)
            if k > 1
            else torch.full((s,), float("nan"), dtype=torch.float64)
        )
    for prefix, (lo, hi) in (("v", V_BAND_OPERATIONAL), ("vstat", V_BAND_STATUTORY)):
        prob = ((vm < lo) | (vm > hi)).double().mean(1)
        out.update(
            exceedance_counts(prefix, prob, (pf_true.vm < lo) | (pf_true.vm > hi))
        )
    prob = (loading > LOADING_LIMIT_PCT).double().mean(1)
    out.update(exceedance_counts("i", prob, loading_true > LOADING_LIMIT_PCT))
    return {name: v.cpu().numpy() for name, v in out.items()}


def paired_block_bootstrap(
    values_a: np.ndarray,
    values_b: np.ndarray,
    blocks: np.ndarray,
    num_resamples: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, float]:
    """Mean of (a - b) with a percentile CI from resampling blocks (test days)
    with replacement; a and b are aligned per snapshot."""
    diff = np.asarray(values_a, dtype=np.float64) - np.asarray(
        values_b, dtype=np.float64
    )
    if diff.size == 0:
        nan = float("nan")
        return {
            "mean_diff": nan,
            "ci_low": nan,
            "ci_high": nan,
            "num_blocks": 0,
            "num_snapshots": 0,
        }
    keys, inverse = np.unique(blocks, return_inverse=True)
    sums = np.bincount(inverse, weights=diff, minlength=len(keys))
    counts = np.bincount(inverse, minlength=len(keys)).astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(keys), size=(num_resamples, len(keys)))
    means = sums[draws].sum(1) / counts[draws].sum(1)
    alpha = (1 - level) / 2
    return {
        "mean_diff": float(diff.mean()),
        "ci_low": float(np.quantile(means, alpha)),
        "ci_high": float(np.quantile(means, 1 - alpha)),
        "num_blocks": len(keys),
        "num_snapshots": len(diff),
    }

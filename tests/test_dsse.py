"""Unit tests for the DSSE module (PoSE): power flow, operators, guidance,
metrics. The pandapower comparison runs separately (validate_powerflow.py)."""

import numpy as np
import pytest
import torch

from pose.data import (
    calendar_arrays,
    select_test_snapshots,
    synthetic_snapshots,
)
from pose.feeder import MEASUREMENT_SETS, Feeder
from pose.guidance import (
    VARIANTS,
    AnalyticGaussianPrior,
    EndpointCorrector,
    LinearMeasurement,
    gaussian_posterior,
    sample_corrected,
    tweedie_covariance,
)
from pose.lindistflow import lindistflow_operator
from pose.metrics import crps_samples, paired_block_bootstrap
from pose.powerflow import MeasurementModel, RadialPowerFlow


CTX_KEYS = ("month", "quarter_hour", "weekend")


def ctx(n):
    return {k: torch.zeros(n, dtype=torch.long) for k in CTX_KEYS}


def gaussian_problem(mset="S2", seed=0):
    g = torch.Generator().manual_seed(seed)
    d = 32
    a = torch.randn(d, d, generator=g, dtype=torch.float64)
    cov_z = a @ a.T / d + 0.1 * torch.eye(d, dtype=torch.float64)
    mu_z = 0.3 * torch.randn(d, generator=g, dtype=torch.float64)
    mean_kw = torch.linspace(20, 200, d, dtype=torch.float64)
    std_kw = torch.linspace(5, 60, d, dtype=torch.float64)
    prior = AnalyticGaussianPrior(mu_z, cov_z, mean_kw, std_kw)
    h_mat, c0 = (
        torch.as_tensor(x) for x in lindistflow_operator(MEASUREMENT_SETS[mset])
    )
    noise_cov = torch.diag(torch.as_tensor(MEASUREMENT_SETS[mset].noise_std(1.0)) ** 2)
    mu_s = mean_kw + std_kw * mu_z
    cov_s = std_kw[:, None] * cov_z * std_kw[None]
    s_true = mu_s + torch.linalg.cholesky(cov_s) @ torch.randn(
        d, generator=g, dtype=torch.float64
    )
    y = (
        c0
        + h_mat @ s_true
        + noise_cov.diag().sqrt()
        * torch.randn(len(c0), generator=g, dtype=torch.float64)
    )
    return prior, LinearMeasurement(h_mat, c0), h_mat, c0, noise_cov, mu_s, cov_s, y


def test_power_flow_nominal_matches_known_case33bw_values():
    pf = RadialPowerFlow()
    res = pf(torch.as_tensor(Feeder().p_nominal_kw)[None])
    assert res.vm.min().item() == pytest.approx(0.913090, abs=1e-6)  # bus 18
    assert res.p_head_kw.item() - 3715.0 == pytest.approx(202.68, abs=0.05)  # losses
    assert res.mismatch.item() < 1e-12


def test_ac_jacobian_matches_finite_differences():
    model = MeasurementModel(MEASUREMENT_SETS["S3"], RadialPowerFlow())
    s = torch.as_tensor(Feeder().p_nominal_kw)[None] * 0.7
    jac = model.jacobian(s)[0]
    eps = 1e-4
    fd = torch.stack(
        [
            (model(s + eps * torch.eye(32)[i]) - model(s - eps * torch.eye(32)[i]))[0]
            / (2 * eps)
            for i in range(32)
        ],
        dim=1,
    )
    assert torch.allclose(jac, fd, rtol=1e-5, atol=1e-9)


def test_lindistflow_close_to_ac_jacobian_at_light_load():
    mset = MEASUREMENT_SETS["S2"]
    h_mat, _ = lindistflow_operator(mset)
    model = MeasurementModel(mset, RadialPowerFlow())
    jac = model.jacobian(torch.zeros(1, 32, dtype=torch.float64))[0].numpy()
    assert np.allclose(jac, h_mat, rtol=0.05, atol=1e-7)


def test_tweedie_covariance_exact_for_gaussian_prior():
    prior, *_ = gaussian_problem()
    x = torch.randn(50, 32)
    for t in (0.2, 0.7):
        tt = torch.tensor(t, dtype=torch.float64)
        cov = tweedie_covariance(prior, x, tt, ctx(50))
        eye = torch.eye(32, dtype=torch.float64)
        exact = (
            (1 - tt) ** 2
            * prior.sigma_z
            @ torch.linalg.inv(tt**2 * prior.sigma_z + (1 - tt) ** 2 * eye)
        )
        assert (cov - exact).abs().max() < 1e-8


def test_m0_reduces_to_projection():
    """Isotropic covariance and R = 0: K = H_z^T (H_z H_z^T)^-1, the
    SmartMeterFM projection; the corrected endpoint fits y exactly."""
    prior, meas, h_mat, _c0, noise_cov, *_, y = gaussian_problem()
    corr = EndpointCorrector(prior, meas, VARIANTS["M0"], h_mat, noise_cov)
    x = torch.randn(8, 32, dtype=torch.float64)
    t = torch.tensor(0.4, dtype=torch.float64)
    z_tilde, z_hat = corr(x, t, ctx(8), y[None].expand(8, -1))
    h_z = h_mat * prior.std_kw
    resid = y - meas(prior.to_kw(z_hat))
    proj = z_hat + (h_z.T @ torch.linalg.solve(h_z @ h_z.T, resid.T)).T
    assert torch.allclose(z_tilde, proj, atol=1e-8)
    assert torch.allclose(meas(prior.to_kw(z_tilde)), y.expand(8, -1), atol=1e-6)


@pytest.mark.parametrize("sampler", ["euler", "ancestral"])
def test_m2_samples_exact_gaussian_posterior(sampler):
    prior, meas, h_mat, c0, noise_cov, mu_s, cov_s, y = gaussian_problem()
    mean, cov = gaussian_posterior(mu_s, cov_s, h_mat, c0, noise_cov, y)
    corr = EndpointCorrector(prior, meas, VARIANTS["M2"], h_mat, noise_cov)
    n = 1500
    s = sample_corrected(
        prior,
        corr,
        ctx(1),
        y[None],
        n,
        100,
        [torch.Generator().manual_seed(1)],
        sampler,
        [torch.Generator().manual_seed(2)],
    )[0]
    sd = cov.diag().sqrt()
    assert ((s.mean(0) - mean).abs() / (sd / n**0.5)).max() < 4.5
    ratio = torch.cov(s.T).diag().sqrt() / sd
    assert 0.88 < ratio.min() and ratio.max() < 1.12


def test_crps_of_point_mass_is_absolute_error():
    s = torch.full((2, 1, 3), 2.0, dtype=torch.float64)
    truth = torch.tensor([[1.0, 2.0, 4.0], [2.0, 2.0, 2.0]], dtype=torch.float64)
    assert torch.allclose(crps_samples(s, truth), (s[:, 0] - truth).abs())


def test_crps_of_normal_samples_matches_closed_form():
    g = torch.Generator().manual_seed(0)
    s = torch.randn(1, 20000, 1, generator=g, dtype=torch.float64)
    crps = crps_samples(s, torch.zeros(1, 1, dtype=torch.float64)).item()
    exact = 2 / np.sqrt(2 * np.pi) - 1 / np.sqrt(np.pi)  # CRPS(N(0,1), 0)
    assert crps == pytest.approx(exact, abs=0.01)


def test_bootstrap_detects_shift_and_contains_zero_without_one():
    rng = np.random.default_rng(0)
    blocks = np.repeat(np.arange(40), 5)
    b = rng.normal(size=200)
    shifted = paired_block_bootstrap(b - 0.5 + 0.1 * rng.normal(size=200), b, blocks)
    assert shifted["ci_high"] < 0
    null = paired_block_bootstrap(b + 0.1 * rng.normal(size=200), b, blocks)
    assert null["ci_low"] < 0 < null["ci_high"]


def test_split_and_stored_designs():
    cal = calendar_arrays()
    assert cal["split"].shape == (366 * 96,)
    assert (cal["split"] == 0).sum() == (366 - 7 * 24) * 96
    data = synthetic_snapshots(0)
    idx = select_test_snapshots(data, "pilot")
    assert len(idx) == 300 and np.all(data.context["split"][idx] == 2)
    assert len(np.unique(data.context["month"][idx])) == 12
    ext = select_test_snapshots(data, "extended")
    assert len(ext) == 1000


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_setting_and_linearization_on_cuda():
    """Branch ratings and the linearization follow the power-flow device."""
    from pose.experiment import Setting

    data = synthetic_snapshots(0)
    on_gpu = Setting.from_names("synthetic", "S2", "nominal", torch.device("cuda"))
    on_cpu = Setting.from_names("synthetic", "S2", "nominal", torch.device("cpu"))
    assert on_gpu.ratings.device.type == "cuda"
    torch.testing.assert_close(on_gpu.ratings.cpu(), on_cpu.ratings)
    lin_gpu, lin_cpu = on_gpu.linearization(data), on_cpu.linearization(data)
    np.testing.assert_allclose(lin_gpu.cov, lin_cpu.cov, rtol=1e-8, atol=1e-12)


def test_cyclic_context_has_no_lookup_tables_and_old_configs_load():
    from dataclasses import asdict

    from pose.prior import FlowPrior, PriorConfig

    mean, std = np.zeros(32), np.ones(32)
    small = {"width": 32, "num_blocks": 1, "emb_dim": 16}
    cyc = FlowPrior(PriorConfig(**small, context="cyclic"), mean, std)
    children = dict(cyc.net.embed.named_children())
    assert "month" not in children and "quarter_hour" not in children
    c = {
        "month": torch.tensor([0, 11]),
        "quarter_hour": torch.tensor([0, 95]),
        "weekend": torch.tensor([0, 1]),
    }
    assert cyc.velocity(torch.zeros(2, 32), torch.tensor(0.5), c).shape == (2, 32)
    old = asdict(PriorConfig(**small))
    del old["context"]  # a checkpoint without the context field
    full = FlowPrior(PriorConfig(**old), mean, std)
    assert full.cfg.context == "full" and hasattr(full.net.embed, "quarter_hour")


def test_prior_checkpoint_round_trip(tmp_path):
    from pose.prior import FlowPrior, PriorConfig, load_prior, save_prior

    mean, std = np.zeros(32), np.ones(32)
    cfg = PriorConfig(width=32, num_blocks=1, emb_dim=16, context="cyclic")
    prior = FlowPrior(cfg, mean, std)
    path = tmp_path / "prior.pt"
    save_prior(str(path), prior, {"seed": 0})
    loaded, ckpt = load_prior(str(path), torch.device("cpu"))
    assert ckpt["standardization"]["mean_kw"].shape == (32,)
    assert set(loaded.state_dict()) == set(prior.state_dict())
    for a, b in zip(
        loaded.state_dict().values(), prior.state_dict().values(), strict=False
    ):
        assert torch.equal(a, b)


def test_select_run_breaks_ties_by_size_then_steps():
    from pose.prior import select_run

    rows = [
        {"run_name": "a", "val_loss": 0.4500, "num_parameter": 5, "num_steps": 100},
        {"run_name": "b", "val_loss": 0.4515, "num_parameter": 2, "num_steps": 100},
        {"run_name": "c", "val_loss": 0.4515, "num_parameter": 2, "num_steps": 10},
        {"run_name": "d", "val_loss": 0.4700, "num_parameter": 1, "num_steps": 10},
    ]
    assert select_run(rows, 0.002)["run_name"] == "c"
    assert select_run(rows, 0.0)["run_name"] == "a"


def test_select_prior_gaussian_endpoint_matches_analytic_prior():
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).parents[1] / "scripts" / "select_prior.py"
    spec = importlib.util.spec_from_file_location("select_prior", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    g = torch.Generator().manual_seed(3)
    a = torch.randn(32, 32, generator=g, dtype=torch.float64)
    cov = a @ a.T / 32 + 0.1 * torch.eye(32, dtype=torch.float64)
    mu = 0.3 * torch.randn(32, generator=g, dtype=torch.float64)
    ones = torch.ones(32, dtype=torch.float64)
    exact = AnalyticGaussianPrior(mu, cov, 0 * ones, ones)
    x = torch.randn(5, 32, generator=g, dtype=torch.float64)
    t = torch.tensor(0.37, dtype=torch.float64)
    got = mod.gaussian_endpoint(x, t.expand(5), mu.expand(5, 32), cov.expand(5, 32, 32))
    torch.testing.assert_close(got, exact.endpoint(x, t))


def test_attainable_coverage_of_sample_intervals():
    from pose.metrics import attainable_coverage

    assert abs(attainable_coverage(100) - 0.9 * 99 / 101) < 1e-12
    assert attainable_coverage(1) is None


def test_dso_exceedance_counts_and_rates():
    from pose.metrics import dso_summary, exceedance_counts

    prob = torch.tensor([[0.9, 0.2, 0.0], [0.6, 0.05, 0.3]])
    viol = torch.tensor([[True, True, False], [False, False, False]])
    out = exceedance_counts("v", prob, viol)
    assert out["v_true"].tolist() == [2.0, 0.0]
    assert out["v_hit50"].tolist() == [1.0, 0.0]
    assert out["v_fa50"].tolist() == [0.0, 1.0]
    assert out["v_hit10"].tolist() == [2.0, 0.0]
    assert out["v_fa10"].tolist() == [0.0, 2.0]
    assert out["v_any_hit50"].tolist() == [1.0, 0.0]
    assert out["v_any_fa50"].tolist() == [0.0, 1.0]
    brier = ((prob - viol.float()) ** 2).mean(-1).double()
    torch.testing.assert_close(out["v_brier"], brier)
    table = {k: v.numpy() for k, v in out.items()}
    table.update(
        crps_kw=np.zeros(2),
        coverage90=np.ones(2),
        vm_max_abs_pu=np.zeros(2),
        branch_max_err_pct=np.zeros(2),
    )
    s = dso_summary(table, 100)["v"]
    assert s["hit_rate50"] == 0.5 and s["hit_rate10"] == 1.0
    assert s["false_alarm_rate50"] == 0.25 and s["false_alarm_rate10"] == 0.5
    assert s["snapshot_detection50"] == 1.0 and s["snapshot_false_alarm50"] == 1.0


def test_evaluate_flags_violations_of_exact_samples():
    from pose.metrics import evaluate
    from pose.powerflow import branch_ratings_a

    pf = RadialPowerFlow()
    meas = MeasurementModel(MEASUREMENT_SETS["S2"], pf)
    p_nom = torch.as_tensor(Feeder().p_nominal_kw, dtype=torch.float64)
    truth = torch.stack([0.5 * p_nom, 2.0 * p_nom])
    samples = truth[:, None].repeat(1, 4, 1)
    ratings = torch.as_tensor(branch_ratings_a(pf), dtype=torch.float64)
    out = evaluate(samples, truth, meas(truth), meas, torch.ones(5), ratings, pf)
    assert out["v_true"][0] == 0 and out["v_true"][1] > 0
    assert (out["v_hit50"] == out["v_true"]).all() and (out["v_fa50"] == 0).all()
    assert (out["v_brier"] == 0).all() and (out["i_brier"] == 0).all()
    assert out["i_true"][0] == 0 and out["i_true"][1] == 32


def test_world_masks_share_validation_and_test_across_sizes():
    from pose.data import SPLITS
    from pose.world import world_calendar, world_valid_mask

    cal = world_calendar(3)
    split, year = cal["split"], cal["year"]
    train = split == SPLITS["train"]
    full = world_valid_mask(cal, 3)
    one = world_valid_mask(cal, 1)
    tenth = world_valid_mask(cal, 1, 0.1)
    per_year = int((train & (year == 0)).sum())
    assert (full & train).sum() == 3 * per_year
    assert (one & train).sum() == per_year
    assert (tenth & train).sum() == round(0.1 * per_year)
    for mask in (full, one, tenth):
        assert ((mask & ~train) == ((year == 0) & ~train)).all()
    assert not (tenth & train & (year > 0)).any()


def test_context_bins_widen_only_when_asked():
    from pose import baselines as bl

    data = synthetic_snapshots(0)
    keep = np.zeros(len(data.valid), dtype=bool)
    keep[np.random.default_rng(0).choice(data.indices("train"), 300, replace=False)] = (
        True
    )
    keep |= data.context["split"] != 0
    data.valid = data.valid & keep
    plain = bl.ContextBins(data)
    wide = bl.ContextBins(data, min_members=33)
    x_plain = plain.members(0, 48, 1)
    x_wide = wide.members(0, 48, 1)
    assert len(x_plain) < 33 <= len(x_wide)
    assert (0, 48, 1) in wide.widened and not plain.widened


def test_feeder_as_data_reproduces_ieee33():
    from pose.feeder import path_matrix

    f = Feeder()
    assert f.num_buses == 32 and f.num_nodes == 32
    np.testing.assert_array_equal(f.path, path_matrix())
    g = Feeder.from_dict(f.to_dict())
    np.testing.assert_array_equal(g.z_path[0], f.z_path[0])
    assert f.node_index(18) == 16 and f.bus_index(33) == 31
    assert (
        Feeder.from_meta(None) == f and Feeder.from_meta({"feeder": f.to_dict()}) == f
    )


def test_joint_state_matches_fixed_power_factor():
    from pose.lindistflow import lindistflow_operator

    pf = RadialPowerFlow()
    p = torch.as_tensor(Feeder().p_nominal_kw, dtype=torch.float64)[
        None
    ] * torch.tensor([[0.5], [1.3], [-0.8]], dtype=torch.float64)
    q = p * torch.as_tensor(Feeder().tan_phi)
    a, b = pf(p), pf(torch.cat([p, q], dim=-1))
    for name in ("vm", "branch_current_a", "p_head_kw", "q_head_kvar"):
        torch.testing.assert_close(
            getattr(a, name), getattr(b, name), rtol=0, atol=1e-12
        )
    mset = MEASUREMENT_SETS["S3"]
    h_p, c_p = lindistflow_operator(mset, state="p")
    h_pq, c_pq = lindistflow_operator(mset, state="pq")
    n = Feeder().num_nodes
    np.testing.assert_allclose(
        h_p, h_pq[:, :n] + h_pq[:, n:] * Feeder().tan_phi, atol=1e-15
    )
    np.testing.assert_array_equal(c_p, c_pq)


def test_zero_injection_bus_equals_zero_load():
    import dataclasses

    full = Feeder()
    reduced = dataclasses.replace(
        full,
        name="ieee33-no33",
        load_buses=full.load_buses[:-1],
        p_nominal=full.p_nominal[:-1],
        q_nominal=full.q_nominal[:-1],
    )
    p = torch.as_tensor(full.p_nominal_kw, dtype=torch.float64)[None]
    p_zero = p.clone()
    p_zero[0, -1] = 0.0
    a = RadialPowerFlow(full)(p_zero)
    b = RadialPowerFlow(reduced)(p[:, :-1])
    torch.testing.assert_close(a.vm, b.vm, rtol=0, atol=1e-12)
    torch.testing.assert_close(a.p_head_kw, b.p_head_kw, rtol=0, atol=1e-9)


def test_kernel_bins_gaussian_is_positive_definite():
    from pose import baselines as bl

    data = synthetic_snapshots(0)
    kb = bl.KernelBins(data, 1.0, 1.0, 0.0)
    mean, cov, n_eff = kb.gaussian(5, 48, 0)
    assert mean.shape == (32,) and n_eff > 10
    assert np.linalg.eigvalsh(cov).min() > 0
    val = data.indices("val")[:20]
    assert np.isfinite(kb.log_likelihood(data.p_kw[val], data.context_of(val)))


def test_evaluate_reports_reactive_scores_for_the_joint_state():
    from pose.metrics import evaluate
    from pose.powerflow import branch_ratings_a

    pf = RadialPowerFlow()
    meas = MeasurementModel(MEASUREMENT_SETS["S2"], pf)
    p = torch.as_tensor(Feeder().p_nominal_kw, dtype=torch.float64)[None]
    x = torch.cat([p, p * torch.as_tensor(Feeder().tan_phi)], dim=-1)
    samples = x[:, None].repeat(1, 4, 1)
    ratings = torch.as_tensor(branch_ratings_a(pf), dtype=torch.float64)
    out = evaluate(samples, x, meas(x), meas, torch.ones(5), ratings, pf)
    assert out["crps_q_kvar"][0] == 0 and out["crps_kw"][0] == 0

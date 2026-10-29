"""Unit tests for the unbalanced three-phase feeder model. The Iowa tests need
the converted feeder file named by the environment variable IOWA_3PH_FEEDER
and are skipped without it."""

import json
import os
from dataclasses import replace

import numpy as np
import pytest
import torch

from pose.feeder import Feeder
from pose.metrics import evaluate
from pose.powerflow import RadialPowerFlow
from pose.threephase import (
    MIN_RATING_A,
    PHASORS,
    Capacitor,
    Connection,
    ThreePhaseFeeder,
    ThreePhaseMeasurementModel,
    ThreePhasePowerFlow,
    lindistflow_3ph,
    ratings_3ph,
)


def line_matrix(phases, length_km, transposed=False):
    """R, X (ohm) of a line with the given phases (a rows/cols subset)."""
    if transposed:
        r = np.full((3, 3), 0.06) + np.eye(3) * 0.24
        x = np.full((3, 3), 0.30) + np.eye(3) * 0.50
    else:
        r = np.array([[0.35, 0.16, 0.15], [0.16, 0.36, 0.14], [0.15, 0.14, 0.34]])
        x = np.array([[1.02, 0.50, 0.38], [0.50, 1.05, 0.42], [0.38, 0.42, 1.04]])
    sub = np.ix_(phases, phases)
    return (
        tuple(map(tuple, r[sub] * length_km)),
        tuple(map(tuple, x[sub] * length_km)),
    )


def balanced_pair():
    """Fully three-phase transposed feeder with balanced wye loads, and the
    balanced feeder with the positive-sequence impedance."""
    parent = (0, 1, 2, 1, 4)
    lengths = (2.0, 1.5, 1.0, 2.5, 1.2)
    p_nom = (300.0, 200.0, 250.0, 150.0, 400.0)
    q_nom = (100.0, 80.0, 60.0, 50.0, 150.0)
    labels = ("s",) + tuple(f"b{k + 1}" for k in range(5))
    load_buses = (1, 2, 3, 4, 5)
    mats = [line_matrix([0, 1, 2], ln, transposed=True) for ln in lengths]
    conns = tuple(
        Connection(i, "wye", (p,), 1.0 / 3.0) for i in range(5) for p in range(3)
    )
    # shares of 1/3 sum to 1 - 1.1e-16: fine for the 1e-12 check
    f3 = ThreePhaseFeeder(
        name="syn_bal",
        bus_labels=labels,
        parent=parent,
        phases=((0, 1, 2),) * 5,
        r_ohm=tuple(m[0] for m in mats),
        x_ohm=tuple(m[1] for m in mats),
        load_buses=load_buses,
        p_nominal=p_nom,
        q_nominal=q_nom,
        connections=conns,
        voltage_buses={"S2-3ph": ("b3", "b5")},
    )
    z1 = [
        (
            np.asarray(r)[0, 0] - np.asarray(r)[0, 1],
            np.asarray(x)[0, 0] - np.asarray(x)[0, 1],
        )
        for r, x in mats
    ]
    f1 = Feeder(
        name="syn_bal_1ph",
        parent=parent,
        r_ohm=tuple(z[0] for z in z1),
        x_ohm=tuple(z[1] for z in z1),
        load_buses=load_buses,
        p_nominal=p_nom,
        q_nominal=q_nom,
        bus_labels=labels,
        v_base_kv=13.8,
        s_base_kva=1000.0,
        v_slack=1.025,
    )
    return f3, f1


def unbalanced_feeder():
    """Three-phase trunk with a two-phase and a single-phase lateral, wye and
    delta loads, capacitors, and an unloaded three-phase spur."""
    #        k: 0    1    2     3    4      5    6
    parent = (0, 1, 2, 2, 2, 3, 5)
    phases = ((0, 1, 2), (0, 1, 2), (0, 1), (2,), (0, 1, 2), (1,), (0, 1, 2))
    lengths = (1.5, 1.0, 0.8, 0.6, 1.2, 0.5, 0.4)
    mats = [line_matrix(list(ph), ln) for ph, ln in zip(phases, lengths, strict=False)]
    labels = ("s",) + tuple(f"b{k + 1}" for k in range(7))
    load_buses = (1, 3, 4, 5, 6, 2)
    p_nom = (300.0, 120.0, 80.0, 450.0, 60.0, 200.0)
    q_nom = (100.0, 40.0, 30.0, 150.0, 20.0, 70.0)
    third = 1.0 / 3.0
    conns = (
        Connection(0, "wye", (0,), third),
        Connection(0, "wye", (1,), third),
        Connection(0, "wye", (2,), third),
        Connection(1, "delta", (0, 1), 1.0),
        Connection(2, "wye", (2,), 1.0),
        Connection(3, "delta", (0, 1), third),
        Connection(3, "delta", (1, 2), third),
        Connection(3, "delta", (2, 0), third),
        Connection(4, "wye", (1,), 1.0),
        Connection(5, "wye", (0,), 0.5),
        Connection(5, "wye", (1,), 0.25),
        Connection(5, "delta", (1, 2), 0.25),
    )
    caps = (
        Capacitor(4, "wye", (0,), 50.0),
        Capacitor(4, "wye", (1,), 50.0),
        Capacitor(4, "wye", (2,), 50.0),
        Capacitor(2, "delta", (0, 1), 30.0),
    )
    return ThreePhaseFeeder(
        name="syn_unbal",
        bus_labels=labels,
        parent=parent,
        phases=phases,
        r_ohm=tuple(m[0] for m in mats),
        x_ohm=tuple(m[1] for m in mats),
        load_buses=load_buses,
        p_nominal=p_nom,
        q_nominal=q_nom,
        connections=conns,
        capacitors=caps,
        voltage_buses={"S2-3ph": ("b3", "b6", "b5")},
    )


def nominal_state(feeder, scale=1.0):
    return torch.as_tensor(
        np.concatenate([feeder.p_nominal_kw, feeder.q_nominal_kvar]) * scale
    )[None]


def iowa_feeder():
    path = os.environ.get("IOWA_3PH_FEEDER")
    if not path or not os.path.exists(path):
        pytest.skip("IOWA_3PH_FEEDER not set or missing")
    with open(path) as fh:
        d = json.load(fh)
    d = d.get("feeder", d)
    return ThreePhaseFeeder.from_dict(d), d


# ---- 1. balanced reduction ---------------------------------------------
def test_balanced_reduction():
    f3, f1 = balanced_pair()
    s = nominal_state(f3)
    r3 = ThreePhasePowerFlow(f3)(s)
    r1 = RadialPowerFlow(f1)(s)
    v1 = (r1.e + 1j * r1.f)[0].numpy()
    v3 = (r3.e + 1j * r3.f)[0].numpy()
    expect = np.array([PHASORS[p] * v1[k] for k, p in f3.node_phases])
    assert np.abs(v3 - expect).max() < 1e-10
    assert r1.vm.min() < 1.02  # a non-trivial voltage drop
    assert np.allclose(
        r3.p_head_phase_kw[0].numpy(), r1.p_head_kw.item() / 3, atol=1e-8
    )
    assert np.allclose(
        r3.q_head_phase_kvar[0].numpy(), r1.q_head_kvar.item() / 3, atol=1e-8
    )
    i1 = r1.branch_current_a[0].numpy()
    assert np.allclose(r3.branch_current_a[0].numpy(), np.repeat(i1, 3), atol=1e-8)


# ---- 2. delta ----------------------------------------------------------
def test_single_phase_delta_load():
    r, x = line_matrix([0, 1, 2], 1.0)
    f = ThreePhaseFeeder(
        name="delta",
        bus_labels=("s", "b1"),
        parent=(0,),
        phases=((0, 1, 2),),
        r_ohm=(r,),
        x_ohm=(x,),
        load_buses=(1,),
        p_nominal=(200.0,),
        q_nominal=(60.0,),
        connections=(Connection(0, "delta", (0, 1), 1.0),),
    )
    pf = ThreePhasePowerFlow(f)
    s = nominal_state(f)
    ir, ii = pf.node_phase_currents(s)
    i = (ir + 1j * ii)[0].numpy()
    assert abs(i[0] + i[1]) < 1e-14 and abs(i[2]) < 1e-14 and abs(i[0]) > 0.1
    res = pf(s)
    cur = res.branch_current_all_a[0].numpy()
    assert cur[0] == pytest.approx(cur[1], rel=1e-14) and cur[2] == 0.0
    assert f.branch_phase_mask == (True, True, False)
    assert res.branch_current_a.shape[-1] == 2
    e_mat, _ = f.wye_equivalent()
    assert abs(e_mat[:, 0].sum() - 1.0) < 1e-14


def test_wye_equivalent_sums_to_share_and_shares_sum_to_one():
    f = unbalanced_feeder()
    e_mat, s_cap = f.wye_equivalent()
    assert np.abs(e_mat.sum(0) - 1.0).max() < 1e-12
    q_cap = sum(c.q_kvar for c in f.capacitors) / f.s_base_phase_kva
    assert abs(s_cap.sum() - (-1j * q_cap)) < 1e-12
    shares = np.zeros(f.num_nodes)
    for c in f.connections:
        shares[c.node] += c.share
    assert np.abs(shares - 1).max() < 1e-12
    bad = replace(f.connections[0], share=0.3)
    with pytest.raises(ValueError):
        replace(f, connections=(bad,) + f.connections[1:])
    with pytest.raises(ValueError):  # phase c does not exist at bus b3
        replace(f, connections=f.connections[:3] + (Connection(1, "delta", (0, 2), 1.0),)
                + f.connections[4:])  # fmt: skip


# ---- 3. LinDistFlow equals the AC Jacobian at state 0 --------------------
def check_lindistflow_exact(feeder):
    f0 = replace(feeder, capacitors=())
    mset = f0.measurement_set("S2-3ph")
    h_mat, c0 = lindistflow_3ph(mset, f0)
    model = ThreePhaseMeasurementModel(mset, ThreePhasePowerFlow(f0))
    zero = torch.zeros(1, 2 * f0.num_nodes, dtype=torch.float64)
    jac = model.jacobian(zero)[0].numpy()
    val = model(zero)[0].numpy()
    assert h_mat.shape == (mset.num_channels, 2 * f0.num_nodes)
    assert np.abs(h_mat - jac).max() < 1e-9
    assert np.abs(c0 - val).max() < 1e-9
    v = slice(6, None)  # voltage rows are small per kW: check relatively too
    assert np.abs(h_mat[v] - jac[v]).max() <= 1e-9 * np.abs(jac[v]).max()


def test_lindistflow_equals_ac_jacobian_synthetic():
    check_lindistflow_exact(unbalanced_feeder())
    check_lindistflow_exact(balanced_pair()[0])


def test_lindistflow_capacitor_offset_close_to_ac():
    f = unbalanced_feeder()
    mset = f.measurement_set("S2-3ph")
    _, c0 = lindistflow_3ph(mset, f)
    model = ThreePhaseMeasurementModel(mset, ThreePhasePowerFlow(f))
    val = model(torch.zeros(1, 2 * f.num_nodes, dtype=torch.float64))[0].numpy()
    assert np.allclose(c0[:6], val[:6], atol=0.5)  # kW, kvar (losses only)
    assert np.allclose(c0[6:], val[6:], atol=1e-4)


def test_lindistflow_equals_ac_jacobian_iowa():
    feeder, _ = iowa_feeder()
    check_lindistflow_exact(feeder)


# ---- 4. interface ------------------------------------------------------
def test_jacobian_and_evaluate_interface():
    f = unbalanced_feeder()
    pf = ThreePhasePowerFlow(f)
    mset = f.measurement_set("S2-3ph")
    model = ThreePhaseMeasurementModel(mset, pf)
    assert mset.num_channels == 6 + 2 + 1 + 3
    assert len(mset.channel_names) == mset.num_channels
    assert mset.channel_names[6] == "V_b3_a"
    g = torch.Generator().manual_seed(0)
    s = nominal_state(f) * (0.5 + torch.rand(3, 2 * f.num_nodes, generator=g,
                                             dtype=torch.float64))  # fmt: skip
    jac = model.jacobian(s)
    assert jac.shape == (3, mset.num_channels, 2 * f.num_nodes)
    eps = 1e-3
    i = 3  # P of the three-phase delta node
    d = torch.zeros_like(s)
    d[:, i] = eps
    fd = (model(s + d) - model(s - d)) / (2 * eps)
    assert torch.allclose(jac[..., i], fd, rtol=1e-6, atol=1e-10)

    ratings = torch.as_tensor(ratings_3ph(pf))
    assert ratings.shape == (pf.num_branch_phases,) and ratings.min() >= MIN_RATING_A
    noise_std = torch.as_tensor(mset.noise_std(1.0, f))
    assert noise_std.shape == (mset.num_channels,) and (noise_std > 0).all()
    truth = s[:2]
    y = model(truth) + noise_std * torch.randn(2, mset.num_channels, generator=g,
                                               dtype=torch.float64)  # fmt: skip
    samples = truth[:, None] * (1 + 0.05 * torch.randn(2, 3, 2 * f.num_nodes,
                                                       generator=g, dtype=torch.float64))  # fmt: skip
    out = evaluate(samples, truth, y, model, noise_std, ratings, pf)
    assert out["vm_rmse_pu"].shape == (2,)
    assert all(np.isfinite(v).all() for k, v in out.items() if "coverage" not in k)


def test_round_trip_and_mask():
    f = unbalanced_feeder()
    d = json.loads(json.dumps(f.to_dict()))
    g = ThreePhaseFeeder.from_dict(d)
    assert g == f
    assert g.to_dict() == d
    mask = f.branch_phase_mask
    assert d["derived"]["branch_phase_mask"] == list(mask)
    assert len(mask) - sum(mask) == 3  # the unloaded spur b7
    pf = ThreePhasePowerFlow(f)
    assert pf.num_node_phases == 16 and pf.num_branch_phases == 13
    res = pf(nominal_state(f))
    assert res.mismatch.item() < 1e-12
    assert res.branch_current_all_a[0, ~torch.as_tensor(mask)].abs().max() < 1e-12
    with pytest.raises(ValueError):
        pf(torch.zeros(1, f.num_nodes, dtype=torch.float64))


# ---- 5. Iowa structure ----------------------------------------------------
def test_iowa_structure():
    feeder, d = iowa_feeder()
    shares = np.zeros(feeder.num_nodes)
    for c in feeder.connections:
        shares[c.node] += c.share
    assert np.abs(shares - 1).max() < 1e-12
    assert feeder.num_nodes == 193 and len(feeder.bus_labels) == 239
    mine = feeder.derived()
    for key, value in d.get("derived", {}).items():
        if key in mine:
            if isinstance(value, float):
                assert mine[key] == pytest.approx(value, rel=1e-12), key
            else:
                assert mine[key] == value, key
    pf = ThreePhasePowerFlow(feeder)
    assert pf.num_branch_phases == sum(feeder.branch_phase_mask)
    res = pf(nominal_state(feeder))
    assert res.mismatch.item() < 1e-10
    ratings = ratings_3ph(pf)
    assert ratings.min() >= MIN_RATING_A
    # values recorded by assemble_iowa_3ph.py (counts block) and at the build
    # of October 21, 2026: 455 node-phases, 432 kept, minimum rating 0.25176 A
    counts = d.get("counts", {})
    assert feeder.num_node_phases == counts.get("num_node_phases", 455) == 455
    assert pf.num_branch_phases == counts.get("num_branch_phases_kept", 432) == 432
    rec_min = counts.get("min_kept_rating_a", 0.2517582673313279)
    assert ratings.min() == pytest.approx(rec_min, rel=1e-8)
    assert ratings.min() == pytest.approx(0.2517582673313279, rel=1e-6)
    mset = feeder.measurement_set("S2-3ph")
    assert mset.voltage_buses == ("bus1017", "bus2048", "bus3091")

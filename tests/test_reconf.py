"""Radial reconfigurations of the IEEE 33-bus feeder."""

import os
import sys

import numpy as np
import pytest
import torch

from pose.feeder import (
    MEASUREMENT_SETS,
    RECONFIGURATIONS,
    Feeder,
    reconfigured_ieee33,
)
from pose.powerflow import RadialPowerFlow, branch_ratings_a


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

# Nominal case33bw loads: V_min (p.u.), loss (kW),
# max |V - V_base| (p.u.), V at buses 18, 22, 33 (p.u.)
REFERENCE = {
    "base": (0.9131, 202.68, 0.0000, (0.9131, 0.9916, 0.9166)),
    "reconf_bw": (0.9378, 139.55, 0.0412, (0.9475, 0.9702, 0.9472)),
    "reconf_all5": (0.9413, 139.98, 0.0458, (0.9475, 0.9702, 0.9472)),
    "reconf_t33": (0.9299, 158.39, 0.0192, (0.9299, 0.9740, 0.9336)),
    "reconf_t34": (0.9167, 196.42, 0.0134, (0.9265, 0.9916, 0.9167)),
    "reconf_t35": (0.9267, 156.79, 0.0416, (0.9547, 0.9777, 0.9267)),
    "reconf_t36": (0.9067, 203.95, 0.0098, (0.9071, 0.9916, 0.9067)),
    "reconf_t37": (0.9285, 175.13, 0.0297, (0.9285, 0.9916, 0.9340)),
}


def nominal(feeder: Feeder):
    pf = RadialPowerFlow(feeder)
    with torch.no_grad():
        res = pf(torch.as_tensor(feeder.p_nominal_kw).unsqueeze(0))
    return pf, res


def test_base_preset_equals_default_feeder():
    base, ref = reconfigured_ieee33(RECONFIGURATIONS["base"], name="ieee33"), Feeder()
    assert base.parent == ref.parent
    assert base.r_ohm == ref.r_ohm
    assert base.x_ohm == ref.x_ohm
    np.testing.assert_array_equal(base.path, ref.path)
    for a, b in zip(base.z_path, ref.z_path, strict=True):
        np.testing.assert_array_equal(a, b)
    for name, mset in MEASUREMENT_SETS.items():
        assert base.measurement_set(name) == mset


@pytest.mark.parametrize("topology", list(RECONFIGURATIONS))
def test_presets_are_radial_and_solve(topology):
    feeder = reconfigured_ieee33(RECONFIGURATIONS[topology], name=topology)
    a = feeder.path
    assert a.shape == (32, 32)
    assert set(np.unique(a)) <= {0.0, 1.0}
    pf, res = nominal(feeder)
    assert float(res.mismatch.max()) < 1e-10
    assert (branch_ratings_a(pf) > 2.5).all()


@pytest.mark.parametrize(
    "open_branches",
    [
        (1, 34, 35, 36, 37),  # slack isolated; tie 33 closes a cycle
        (32, 33, 34, 35, 36),  # bus 33 isolated; tie 37 closes a cycle
        (33, 34, 35, 36),  # only four branches open
    ],
)
def test_invalid_configurations_raise(open_branches):
    with pytest.raises(ValueError):
        reconfigured_ieee33(open_branches)


def test_reference_table():
    _, base = nominal(Feeder())
    v_base = base.vm[0].numpy()
    s2 = [Feeder().bus_index(b) for b in MEASUREMENT_SETS["S2"].voltage_buses]
    for topology, (v_min, loss, shift, v_s2) in REFERENCE.items():
        feeder = reconfigured_ieee33(RECONFIGURATIONS[topology], name=topology)
        _, res = nominal(feeder)
        vm = res.vm[0].numpy()
        cur = res.branch_current_a[0].numpy()
        loss_kw = float(np.sum(cur**2 * np.asarray(feeder.r_ohm))) * 3 / 1000.0
        assert abs(vm.min() - v_min) < 1e-4, topology
        assert abs(loss_kw - loss) < 0.05, (topology, loss_kw)
        assert abs(np.abs(vm - v_base).max() - shift) < 1e-4, topology
        np.testing.assert_allclose(vm[s2], v_s2, atol=1e-4, err_msg=topology)


def test_supervised_train_feeder_needs_b5_or_b6():
    from run_cell import parse_args

    common = ["--data", "x.npz", "--output_dir", "out", "--device", "cpu"]
    with pytest.raises(ValueError):
        parse_args([*common, "--method", "B2", "--supervised_train_feeder", "base"])
    args = parse_args([*common, "--method", "B5", "--supervised_train_feeder", "base"])
    assert args.supervised_train_feeder == "base"
    assert parse_args([*common, "--method", "B2"]).topology is None

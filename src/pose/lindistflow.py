"""LinDistFlow operator h_lin(s) = c0 + H s and its linearization error.

LinDistFlow (Baran and Wu, 1989) without losses and with V0 = 1 p.u.:
feeder-head P = sum_i P_i, Q = sum_i Q_i, and
|V_j| = 1 - sum_i (R_ji P_i + X_ji Q_i), with R and X the common-path
resistance and reactance in p.u. and Q_i = P_i tan(phi_i). H is expressed
per kW of node injection, in the units of each measured channel.

The LinDistFlow variants use the gain from H and add the empirical
covariance of the AC-minus-LinDistFlow error on training snapshots to the
measurement noise covariance R.
"""

from dataclasses import dataclass

import numpy as np
import torch

from .feeder import Feeder, MeasurementSet
from .powerflow import MeasurementModel


def lindistflow_operator(
    mset: MeasurementSet, feeder: Feeder | None = None, state: str = "p"
) -> tuple[np.ndarray, np.ndarray]:
    """Return (H [m, d], c0 [m]) for the measurement set: d = N for the
    fixed-power-factor state (Q = P tan(phi)), d = 2N for the joint (P, Q)
    state. With slack voltage V0, |V_j| = V0 - sum_i (R_ji P_i + X_ji Q_i) / V0."""
    feeder = feeder or Feeder()
    r_path, x_path = feeder.z_path
    n = feeder.num_nodes
    loads = feeder.load_index
    tphi = feeder.tan_phi
    v0 = feeder.v_slack
    s_base = feeder.s_base_kva
    if state == "p":
        rows = [np.ones(n), tphi.copy()]
    elif state == "pq":
        rows = [
            np.concatenate([np.ones(n), np.zeros(n)]),
            np.concatenate([np.zeros(n), np.ones(n)]),
        ]
    else:
        raise ValueError(f"unknown state {state}")
    offset = [0.0, 0.0]
    for bus in mset.voltage_buses:
        j = feeder.bus_index(bus)
        r_j, x_j = r_path[j, loads], x_path[j, loads]
        if state == "p":
            rows.append(-(r_j + x_j * tphi) / s_base / v0)
        else:
            rows.append(np.concatenate([-r_j, -x_j]) / s_base / v0)
        offset.append(v0)
    if feeder.shunt_q_kvar and any(feeder.shunt_q_kvar):
        # fixed shunt injections (capacitors) shift the head Q and the voltages
        q_sh = np.asarray(feeder.shunt_q_kvar)
        offset[1] += q_sh.sum()
        for k, bus in enumerate(mset.voltage_buses):
            offset[2 + k] -= x_path[feeder.bus_index(bus)] @ q_sh / s_base / v0
    for bus in mset.monitored_buses:
        row = np.zeros(n if state == "p" else 2 * n)
        row[feeder.node_index(bus)] = 1.0
        rows.append(row)
        offset.append(0.0)
    return np.stack(rows), np.asarray(offset)


@dataclass
class LinearizationError:
    mean: np.ndarray  # [m]
    cov: np.ndarray  # [m, m]
    num_snapshots: int


def linearization_error(
    model: MeasurementModel,
    p_kw: np.ndarray,
    batch_size: int = 4096,
    operator: tuple[np.ndarray, np.ndarray] | None = None,
) -> LinearizationError:
    """Mean and covariance of h_AC(s) - h_lin(s) over snapshots p_kw [N, 32].
    ``operator`` is (H, c0) of h_lin; default: lindistflow_operator of the
    model's measurement set and feeder."""
    pf = model.power_flow
    if operator is None:
        state = "p" if p_kw.shape[1] == pf.num_nodes else "pq"
        h_mat, c0 = lindistflow_operator(model.mset, pf.feeder, state)
    else:
        h_mat, c0 = operator
    device = pf.r_path.device
    errors = []
    with torch.no_grad():
        for start in range(0, len(p_kw), batch_size):
            chunk = torch.as_tensor(p_kw[start : start + batch_size])
            h_ac = model(chunk.to(device)).cpu().numpy()
            errors.append(h_ac - (c0 + chunk.numpy() @ h_mat.T))
    err = np.concatenate(errors)
    return LinearizationError(err.mean(0), np.cov(err, rowvar=False), len(err))

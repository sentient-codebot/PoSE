"""Batched AC power flow for the radial feeder, differentiable in torch.

Backward/forward sweep in current-summation form with constant-power loads:
with load currents I = conj(S / V) and the common-path impedance matrix
Z = A diag(z) A^T, the node voltages satisfy V = V0 - Z I(V). The sweep
iterates this map a fixed number of times, so the solution is a smooth
function of the injections and forward-mode Jacobians (torch.func.jacfwd)
are exact up to the iteration tolerance. Arithmetic is real (e + jf), in
float64 by default.

Sign convention: ``p_kw`` is the net active power drawn at each node
(positive = consumption, negative = net PV export). The state is either P per
state node (N columns, Q = P tan(phi)) or P followed by Q in kvar (2N
columns); buses without a state node carry no injection.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from .feeder import Feeder, MeasurementSet


DEFAULT_ITERATIONS = 30


@dataclass
class PowerFlowResult:
    vm: Tensor  # [..., B] voltage magnitude of the non-slack buses, p.u.
    e: Tensor  # [..., B] real part of the voltage, p.u.
    f: Tensor  # [..., B] imaginary part, p.u.
    branch_current_a: Tensor  # [..., B] branch current magnitude, A
    p_head_kw: Tensor  # [...] active power from the substation, kW
    q_head_kvar: Tensor  # [...] reactive power from the substation, kvar
    mismatch: Tensor  # [...] max |V - (V0 - Z I(V))| after the last sweep


class RadialPowerFlow(torch.nn.Module):
    def __init__(
        self,
        feeder: Feeder | None = None,
        num_iterations: int = DEFAULT_ITERATIONS,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        feeder = feeder or Feeder()
        self.feeder = feeder
        r_path, x_path = feeder.z_path
        self.num_iterations = num_iterations
        self.num_nodes = feeder.num_nodes
        self.num_buses = feeder.num_buses
        self.s_base_kva = feeder.s_base_kva
        self.i_base_a = feeder.i_base_a
        self.v_slack = feeder.v_slack
        index = feeder.load_index
        self.identity = bool(
            len(index) == feeder.num_buses and (index == np.arange(len(index))).all()
        )
        scatter = np.zeros((feeder.num_nodes, feeder.num_buses))
        scatter[np.arange(feeder.num_nodes), index] = 1.0
        self.register_buffer("scatter", torch.as_tensor(scatter, dtype=dtype))
        self.has_shunt = bool(feeder.shunt_q_kvar) and any(feeder.shunt_q_kvar)
        if self.has_shunt:
            shunt = np.asarray(feeder.shunt_q_kvar) / feeder.s_base_kva
            self.register_buffer("shunt_q", torch.as_tensor(shunt, dtype=dtype))
        self.register_buffer("r_path", torch.as_tensor(r_path, dtype=dtype))
        self.register_buffer("x_path", torch.as_tensor(x_path, dtype=dtype))
        self.register_buffer("path", torch.as_tensor(feeder.path, dtype=dtype))
        self.register_buffer("tan_phi", torch.as_tensor(feeder.tan_phi, dtype=dtype))

    def _currents(self, p: Tensor, q: Tensor, e: Tensor, f: Tensor):
        """Load currents conj(S / V) in p.u.: real and imaginary parts."""
        mag2 = e * e + f * f
        return (p * e + q * f) / mag2, (p * f - q * e) / mag2

    def _sweep(self, p: Tensor, q: Tensor, e: Tensor, f: Tensor):
        ir, ii = self._currents(p, q, e, f)
        dr = ir @ self.r_path.T - ii @ self.x_path.T
        di = ii @ self.r_path.T + ir @ self.x_path.T
        return self.v_slack - dr, -di, ir, ii

    def _to_buses(self, v: Tensor) -> Tensor:
        if self.identity:
            return v
        return v @ self.scatter  # one-hot, differentiable and vmap-safe

    def forward(self, p_kw: Tensor) -> PowerFlowResult:
        """State [..., N] (P in kW) or [..., 2N] (P in kW, then Q in kvar)."""
        p_kw = p_kw.to(self.r_path.dtype)
        n = self.num_nodes
        if p_kw.shape[-1] == n:
            p = p_kw / self.s_base_kva
            q = p * self.tan_phi
        elif p_kw.shape[-1] == 2 * n:
            p = p_kw[..., :n] / self.s_base_kva
            q = p_kw[..., n:] / self.s_base_kva
        else:
            raise ValueError(
                f"state has {p_kw.shape[-1]} columns, expected {n} or {2 * n}"
            )
        p, q = self._to_buses(p), self._to_buses(q)
        if self.has_shunt:
            q = q + self.shunt_q
        e = (
            torch.full_like(p, self.v_slack)
            if self.v_slack != 1.0
            else torch.ones_like(p)
        )
        f = torch.zeros_like(p)
        for _ in range(self.num_iterations):
            e, f, _, _ = self._sweep(p, q, e, f)
        e_next, f_next, ir, ii = self._sweep(p, q, e, f)
        mismatch = torch.maximum(
            (e_next - e).abs().amax(-1), (f_next - f).abs().amax(-1)
        )
        # branch currents: each branch carries the load currents downstream
        br = ir @ self.path
        bi = ii @ self.path
        branch_current = torch.sqrt(br * br + bi * bi) * self.i_base_a
        # substation power S = V0 conj(I_head) over the branches leaving the slack
        root = [k for k, par in enumerate(self.feeder.parent) if par == 0]
        p_head = br[..., root].sum(-1) * self.s_base_kva * self.v_slack
        q_head = -bi[..., root].sum(-1) * self.s_base_kva * self.v_slack
        vm = torch.sqrt(e * e + f * f)
        return PowerFlowResult(vm, e, f, branch_current, p_head, q_head, mismatch)


class MeasurementModel(torch.nn.Module):
    """h(s): state [..., N or 2N] -> measured channels [..., m]."""

    def __init__(self, mset: MeasurementSet, power_flow: RadialPowerFlow):
        super().__init__()
        self.mset = mset
        self.power_flow = power_flow
        feeder = power_flow.feeder
        self.voltage_nodes = [feeder.bus_index(b) for b in mset.voltage_buses]
        self.monitored_nodes = [feeder.node_index(b) for b in mset.monitored_buses]

    def from_result(self, res: PowerFlowResult, p_kw: Tensor) -> Tensor:
        parts = [res.p_head_kw.unsqueeze(-1), res.q_head_kvar.unsqueeze(-1)]
        if self.voltage_nodes:
            parts.append(res.vm[..., self.voltage_nodes])
        if self.monitored_nodes:
            parts.append(p_kw[..., self.monitored_nodes].to(res.vm.dtype))
        return torch.cat(parts, dim=-1)

    def forward(self, p_kw: Tensor) -> Tensor:
        return self.from_result(self.power_flow(p_kw), p_kw)

    def jacobian(self, p_kw: Tensor) -> Tensor:
        """dh/ds at each row of p_kw [B, d] -> [B, m, d]."""
        from torch.func import jacfwd, vmap

        return vmap(jacfwd(self.forward))(p_kw)


def branch_ratings_a(power_flow: RadialPowerFlow | None = None) -> np.ndarray:
    """Branch rating: current at the feeder's nominal loading (nominal P
    with the nominal power factor), A."""
    power_flow = power_flow or RadialPowerFlow()
    p_nom = torch.as_tensor(
        power_flow.feeder.p_nominal_kw,
        dtype=torch.float64,
        device=power_flow.r_path.device,
    )
    with torch.no_grad():
        res = power_flow(p_nom.unsqueeze(0))
    return res.branch_current_a[0].cpu().numpy()

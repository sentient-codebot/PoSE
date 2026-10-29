"""Unbalanced three-phase radial feeder: data, AC power flow, measurement
model, branch ratings and the unbalanced LinDistFlow operator.

Per-unit system (equal to the balanced model's): phase-to-neutral voltage
base V_b = v_base_kv / sqrt(3) kV, per-phase power base S_b = s_base_kva / 3
kVA, impedance base Z_b = V_b^2 / S_b and current base I_b = S_b / V_b. The
phase rotation is a = exp(-j 2 pi / 3), with phasors 1, a, a^2 for phases a,
b and c (indices 0, 1, 2).

The state is P (kW) of the N state nodes followed by Q (kvar), as in the
balanced 'pq' state; p_kw > 0 is consumption. Each state node's power is
allocated to wye (one phase) or delta (ordered phase pair) connections by
fixed shares. Capacitors are constant-power connections injecting reactive
power.

The AC power flow is the current-summation sweep V = V_s - Z_path I(V) over
the node-phases (each existing phase of each non-slack bus), with Z_path the
common-path impedance matrix of the branch impedance matrices (self and
mutual terms). Arithmetic is real (e + jf) in float64, with a fixed number of
iterations, so the map is smooth and compatible with torch.func.vmap and
jacfwd.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from .powerflow import DEFAULT_ITERATIONS, PowerFlowResult


PHASE_NAMES = ("a", "b", "c")
A_ROT = np.exp(-2j * np.pi / 3)
PHASORS = np.array([1.0, A_ROT, A_ROT**2])
# guard against structurally empty kept branch-phases; just below the smallest
# balanced Iowa rating (0.208 A)
MIN_RATING_A = 0.2
KINDS = ("wye", "delta")


@dataclass(frozen=True)
class Connection:
    """A share of state node ``node``'s P and Q on a wye (one phase) or
    delta (ordered phase pair) connection."""

    node: int
    kind: str
    phases: tuple[int, ...]
    share: float


@dataclass(frozen=True)
class Capacitor:
    """Fixed capacitor connection at non-slack bus ``bus`` (0-based), with
    ``q_kvar`` > 0 the reactive power it injects (constant power)."""

    bus: int
    kind: str
    phases: tuple[int, ...]
    q_kvar: float


def _tuplify(x):
    if isinstance(x, (list, tuple)):
        return tuple(_tuplify(v) for v in x)
    return x


def _listify(x):
    if isinstance(x, (list, tuple)):
        return [_listify(v) for v in x]
    return x


@dataclass(frozen=True)
class ThreePhaseMeasurementSet:
    """Channels, in order: head P_a, P_b, P_c (kW); head Q_a, Q_b, Q_c
    (kvar); then the phase-to-neutral voltage magnitude (p.u.) of each
    existing phase of each bus in ``voltage_buses`` (phase order a, b, c)."""

    name: str
    voltage_buses: tuple = ()
    voltage_phases: tuple = ()  # per voltage bus, its existing phases
    monitored_buses: tuple = ()  # none; kept so generic code does not break

    @property
    def num_channels(self) -> int:
        return 6 + sum(len(p) for p in self.voltage_phases)

    @property
    def channel_names(self) -> list[str]:
        names = [f"P_head_{p}" for p in PHASE_NAMES]
        names += [f"Q_head_{p}" for p in PHASE_NAMES]
        for bus, phases in zip(self.voltage_buses, self.voltage_phases, strict=False):
            names += [f"V_{bus}_{PHASE_NAMES[p]}" for p in phases]
        return names

    def nominal_magnitude(self, feeder: "ThreePhaseFeeder") -> np.ndarray:
        """Per-phase head nominal: the nominal P (Q) allocated to the phase,
        delta connections split equally between their two phases; 1 p.u.
        for the voltages."""
        p_ph, q_ph = feeder.nominal_head_per_phase()
        if min(np.min(p_ph), np.min(q_ph)) <= 0:
            raise ValueError("a head phase has no positive nominal P or Q")
        values = list(p_ph) + list(q_ph)
        values += [1.0] * sum(len(p) for p in self.voltage_phases)
        return np.asarray(values, dtype=np.float64)

    def noise_std(
        self, noise_level: float = 1.0, feeder: "ThreePhaseFeeder | None" = None
    ) -> np.ndarray:
        """One third of a 1% maximum error of the nominal magnitude, times
        ``noise_level`` (1 nominal, 3 high)."""
        if feeder is None:
            raise ValueError("a ThreePhaseFeeder is required")
        return noise_level * (0.01 / 3.0) * self.nominal_magnitude(feeder)


@dataclass(frozen=True)
class ThreePhaseFeeder:
    """Unbalanced radial feeder as data. Bus 0 (bus_labels[0]) is the slack;
    non-slack bus k (bus_labels[k + 1]) is fed by branch k from bus
    ``parent[k]`` (0 = slack). ``phases[k]`` are the existing phases of bus k
    and branch k; ``r_ohm[k]``, ``x_ohm[k]`` the branch series impedance
    matrices in ohm (rows and columns ordered as ``phases[k]``). State node
    i sits at non-slack bus load_buses[i] - 1."""

    name: str
    bus_labels: tuple
    parent: tuple
    phases: tuple
    r_ohm: tuple
    x_ohm: tuple
    load_buses: tuple
    p_nominal: tuple
    q_nominal: tuple
    connections: tuple
    capacitors: tuple = ()
    voltage_buses: dict = field(default_factory=dict)
    v_slack: float = 1.025
    v_base_kv: float = 13.8  # line-to-line
    s_base_kva: float = 1000.0  # three-phase

    def __post_init__(self):
        self.validate()

    # ---- validation -----------------------------------------------------
    def validate(self) -> None:
        b = len(self.parent)
        if len(self.bus_labels) != b + 1:
            raise ValueError("bus_labels must hold the slack and every non-slack bus")
        for name in ("phases", "r_ohm", "x_ohm"):
            if len(getattr(self, name)) != b:
                raise ValueError(f"{name} must have one entry per branch")
        for k in range(b):
            par = self.parent[k]
            if not 0 <= par <= b or par == k + 1:
                raise ValueError(f"bus {k}: invalid parent {par}")
            ph = tuple(self.phases[k])
            if not ph or list(ph) != sorted(set(ph)) or not set(ph) <= {0, 1, 2}:
                raise ValueError(f"bus {k}: invalid phases {ph}")
            if par > 0 and not set(ph) <= set(self.phases[par - 1]):
                raise ValueError(f"bus {k}: phases {ph} not in its parent's")
            n = len(ph)
            for mat in (self.r_ohm[k], self.x_ohm[k]):
                if np.asarray(mat, dtype=np.float64).shape != (n, n):
                    raise ValueError(f"branch {k}: impedance must be {n}x{n}")
        for k in range(b):  # no cycles: every bus reaches the slack
            bus, steps = k + 1, 0
            while bus != 0:
                bus = self.parent[bus - 1]
                steps += 1
                if steps > b:
                    raise ValueError(f"bus {k} does not reach the slack (cycle)")
        nn = len(self.load_buses)
        if len(self.p_nominal) != nn or len(self.q_nominal) != nn:
            raise ValueError("p_nominal and q_nominal must have one entry per node")
        if any(not 1 <= lb <= b for lb in self.load_buses):
            raise ValueError("load_buses must be 1-based non-slack bus indices")
        share = np.zeros(nn)
        for c in self.connections:
            if not 0 <= c.node < nn:
                raise ValueError(f"connection node {c.node} out of range")
            self._check_phases(c.kind, c.phases, self.load_buses[c.node] - 1)
            share[c.node] += c.share
        if np.any(np.abs(share - 1.0) > 1e-12):
            bad = np.flatnonzero(np.abs(share - 1.0) > 1e-12)
            raise ValueError(f"connection shares do not sum to 1 at nodes {bad[:10]}")
        for cap in self.capacitors:
            if not 0 <= cap.bus < b:
                raise ValueError(f"capacitor bus {cap.bus} out of range")
            self._check_phases(cap.kind, cap.phases, cap.bus)
        for labels in self.voltage_buses.values():
            for lab in labels:
                self.bus_index(lab)

    def _check_phases(self, kind, phases, bus):
        if kind not in KINDS:
            raise ValueError(f"unknown connection kind {kind}")
        need = 1 if kind == "wye" else 2
        if len(phases) != need or len(set(phases)) != need:
            raise ValueError(f"{kind} connection needs {need} distinct phases")
        if not set(phases) <= set(self.phases[bus]):
            raise ValueError(f"phases {phases} do not exist at bus {bus}")

    # ---- sizes and bases ------------------------------------------------
    @property
    def num_buses(self) -> int:
        return len(self.parent)

    @property
    def num_nodes(self) -> int:
        return len(self.load_buses)

    @property
    def s_base_phase_kva(self) -> float:
        return self.s_base_kva / 3.0

    @property
    def z_base_ohm(self) -> float:
        return self.v_base_kv**2 / (self.s_base_kva / 1000.0)

    @property
    def i_base_a(self) -> float:
        return self.s_base_kva / (np.sqrt(3.0) * self.v_base_kv)

    @property
    def node_phases(self) -> list[tuple[int, int]]:
        return [(k, p) for k in range(self.num_buses) for p in self.phases[k]]

    @property
    def branch_phases(self) -> list[tuple[int, int]]:
        return self.node_phases

    @property
    def num_node_phases(self) -> int:
        return sum(len(p) for p in self.phases)

    def node_phase_index(self) -> dict[tuple[int, int], int]:
        return {kp: n for n, kp in enumerate(self.node_phases)}

    @property
    def p_nominal_kw(self) -> np.ndarray:
        return np.asarray(self.p_nominal, dtype=np.float64)

    @property
    def q_nominal_kvar(self) -> np.ndarray:
        return np.asarray(self.q_nominal, dtype=np.float64)

    @property
    def tan_phi(self) -> np.ndarray:
        return self.q_nominal_kvar / self.p_nominal_kw

    @property
    def load_index(self) -> np.ndarray:
        return np.asarray(self.load_buses, dtype=np.int64) - 1

    def bus_index(self, label) -> int:
        """Non-slack bus index (0..B-1) of a label."""
        k = list(self.bus_labels).index(str(label))
        if k == 0:
            raise ValueError(f"bus {label} is the slack")
        return k - 1

    def node_index(self, label) -> int:
        k = list(self.bus_labels).index(str(label))
        return list(self.load_buses).index(k)

    # ---- topology -------------------------------------------------------
    def bus_path(self) -> np.ndarray:
        """A[k, b] = 1 if branch b lies on the path from the slack to bus k."""
        b = self.num_buses
        a = np.zeros((b, b))
        for k in range(b):
            bus = k + 1
            while bus != 0:
                a[k, bus - 1] = 1.0
                bus = self.parent[bus - 1]
        return a

    def node_phase_path(self) -> np.ndarray:
        """A[n, bp] = 1 if branch-phase bp lies on the path of node-phase n
        (same phase, branch on the path to the bus of n)."""
        nps = self.node_phases
        idx = self.node_phase_index()
        a = np.zeros((len(nps), len(nps)))
        for n, (k, p) in enumerate(nps):
            bus = k + 1
            while bus != 0:
                a[n, idx[(bus - 1, p)]] = 1.0
                bus = self.parent[bus - 1]
        return a

    def z_path_pu(self) -> tuple[np.ndarray, np.ndarray]:
        """Common-path R and X over node-phases (p.u.): Z_nm = sum over the
        branches common to both paths of Z_b[phase(n), phase(m)] / Z_base."""
        m = self.num_node_phases
        r_blk, x_blk = np.zeros((m, m)), np.zeros((m, m))
        start = 0
        for k in range(self.num_buses):
            n = len(self.phases[k])
            sl = slice(start, start + n)
            r_blk[sl, sl] = np.asarray(self.r_ohm[k], dtype=np.float64)
            x_blk[sl, sl] = np.asarray(self.x_ohm[k], dtype=np.float64)
            start += n
        a = self.node_phase_path()
        zb = self.z_base_ohm
        return a @ r_blk @ a.T / zb, a @ x_blk @ a.T / zb

    @property
    def branch_phase_mask(self) -> tuple[bool, ...]:
        """False for a branch-phase with no load connection and no capacitor
        using that phase anywhere downstream (its current is then zero)."""
        used = np.zeros((self.num_buses, 3), dtype=bool)
        for c in self.connections:
            used[self.load_buses[c.node] - 1, list(c.phases)] = True
        for cap in self.capacitors:
            used[cap.bus, list(cap.phases)] = True
        a = self.bus_path()  # a[k, b]: bus k is downstream of (or at) branch b
        down = (a.T @ used.astype(np.float64)) > 0  # [branch, phase]
        return tuple(bool(down[k, p]) for k, p in self.branch_phases)

    # ---- allocation -----------------------------------------------------
    def wye_equivalent(self) -> tuple[np.ndarray, np.ndarray]:
        """Equivalent wye allocation at flat balanced voltages.

        Returns (E [n_np, N] complex, s_cap [n_np] complex): node-phase power
        S^eq (p.u.) = E @ (P + jQ) / S_b + s_cap. A wye share goes to its
        phase; a delta (phi, psi) share S gives S / (1 - a_psi / a_phi) at
        phi and S / (1 - a_phi / a_psi) at psi, which sum to S."""
        idx = self.node_phase_index()
        e = np.zeros((self.num_node_phases, self.num_nodes), dtype=np.complex128)
        s_cap = np.zeros(self.num_node_phases, dtype=np.complex128)

        def add(target, col, bus, kind, phases, value):
            if kind == "wye":
                pairs = [(phases[0], 1.0)]
            else:
                p, q = phases
                pairs = [
                    (p, 1.0 / (1.0 - PHASORS[q] / PHASORS[p])),
                    (q, 1.0 / (1.0 - PHASORS[p] / PHASORS[q])),
                ]
            for ph, w in pairs:
                if col is None:
                    target[idx[(bus, ph)]] += w * value
                else:
                    target[idx[(bus, ph)], col] += w * value

        for c in self.connections:
            add(e, c.node, self.load_buses[c.node] - 1, c.kind, c.phases, c.share)
        for cap in self.capacitors:
            add(s_cap, None, cap.bus, cap.kind, cap.phases,
                -1j * cap.q_kvar / self.s_base_phase_kva)  # fmt: skip
        return e, s_cap

    def nominal_head_per_phase(self) -> tuple[np.ndarray, np.ndarray]:
        """Nominal P (kW) and Q (kvar) allocated to each phase (delta split
        equally between its two phases)."""
        p, q = np.zeros(3), np.zeros(3)
        for c in self.connections:
            w = c.share / len(c.phases)
            for ph in c.phases:
                p[ph] += w * self.p_nominal[c.node]
                q[ph] += w * self.q_nominal[c.node]
        return p, q

    def measurement_set(self, name: str) -> ThreePhaseMeasurementSet:
        if name not in self.voltage_buses:
            raise KeyError(f"feeder {self.name} has no measurement set {name}")
        buses = tuple(self.voltage_buses[name])
        return ThreePhaseMeasurementSet(
            name,
            voltage_buses=buses,
            voltage_phases=tuple(tuple(self.phases[self.bus_index(b)]) for b in buses),
        )

    # ---- serialisation --------------------------------------------------
    def derived(self) -> dict:
        mask = self.branch_phase_mask
        return {
            "num_nodes": self.num_nodes,
            "num_buses": self.num_buses,
            "num_node_phases": self.num_node_phases,
            "num_branch_phases_kept": int(sum(mask)),
            "num_branch_phases_excluded": int(len(mask) - sum(mask)),
            "node_phases": [list(kp) for kp in self.node_phases],
            "branch_phases": [list(kp) for kp in self.branch_phases],
            "branch_phase_mask": list(mask),
            "z_base_ohm": self.z_base_ohm,
            "i_base_a": self.i_base_a,
            "s_base_phase_kva": self.s_base_phase_kva,
        }

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "bus_labels": list(self.bus_labels),
            "parent": list(self.parent),
            "phases": _listify(self.phases),
            "r_ohm": _listify(self.r_ohm),
            "x_ohm": _listify(self.x_ohm),
            "load_buses": list(self.load_buses),
            "p_nominal": list(self.p_nominal),
            "q_nominal": list(self.q_nominal),
            "connections": [
                {
                    "node": c.node,
                    "kind": c.kind,
                    "phases": list(c.phases),
                    "share": c.share,
                }
                for c in self.connections
            ],
            "capacitors": [
                {
                    "bus": c.bus,
                    "kind": c.kind,
                    "phases": list(c.phases),
                    "q_kvar": c.q_kvar,
                }
                for c in self.capacitors
            ],
            "voltage_buses": {k: list(v) for k, v in self.voltage_buses.items()},
            "v_slack": self.v_slack,
            "v_base_kv": self.v_base_kv,
            "s_base_kva": self.s_base_kva,
            "derived": self.derived(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ThreePhaseFeeder":
        return cls(
            name=d["name"],
            bus_labels=tuple(d["bus_labels"]),
            parent=tuple(d["parent"]),
            phases=_tuplify(d["phases"]),
            r_ohm=_tuplify(d["r_ohm"]),
            x_ohm=_tuplify(d["x_ohm"]),
            load_buses=tuple(d["load_buses"]),
            p_nominal=tuple(d["p_nominal"]),
            q_nominal=tuple(d["q_nominal"]),
            connections=tuple(
                Connection(int(c["node"]), c["kind"], tuple(c["phases"]), c["share"])
                for c in d["connections"]
            ),
            capacitors=tuple(
                Capacitor(int(c["bus"]), c["kind"], tuple(c["phases"]), c["q_kvar"])
                for c in d.get("capacitors", [])
            ),
            voltage_buses={k: tuple(v) for k, v in d.get("voltage_buses", {}).items()},
            v_slack=d["v_slack"],
            v_base_kv=d["v_base_kv"],
            s_base_kva=d["s_base_kva"],
        )


@dataclass
class ThreePhaseResult(PowerFlowResult):
    """vm, e, f over node-phases (phase-to-neutral, p.u.); branch_current_a
    over the kept (masked) branch-phases; p_head_kw and q_head_kvar summed
    over phases."""

    p_head_phase_kw: Tensor  # [..., 3]
    q_head_phase_kvar: Tensor  # [..., 3]
    branch_current_all_a: Tensor  # [..., all branch-phases]


class ThreePhasePowerFlow(torch.nn.Module):
    """Batched unbalanced AC power flow (current-summation sweep) with
    constant-power wye and delta connections; differentiable in torch."""

    def __init__(
        self,
        feeder: ThreePhaseFeeder,
        num_iterations: int = DEFAULT_ITERATIONS,
        dtype: torch.dtype = torch.float64,
        v_slack_phasors=None,
    ):
        super().__init__()
        self.feeder = feeder
        self.num_iterations = num_iterations
        self.num_nodes = feeder.num_nodes
        self.num_buses = feeder.num_buses
        self.num_node_phases = feeder.num_node_phases
        self.s_base_kva = feeder.s_base_kva
        self.s_base_phase_kva = feeder.s_base_phase_kva
        self.i_base_a = feeder.i_base_a
        self.v_slack = feeder.v_slack
        if v_slack_phasors is None:
            vs = feeder.v_slack * PHASORS
        else:
            vs = np.asarray(v_slack_phasors, dtype=np.complex128).reshape(3)
        self.v_slack_phasors = vs
        nps = feeder.node_phases
        ph = np.array([p for _, p in nps], dtype=np.int64)
        mask = np.asarray(feeder.branch_phase_mask, dtype=bool)
        self.num_branch_phases = int(mask.sum())
        r_path, x_path = feeder.z_path_pu()
        a_np = feeder.node_phase_path()

        # connections (loads, then capacitors): incidence D [C, n_np] gives
        # the voltage across each connection (U = D V) and the node-phase
        # currents drawn (I = D^T J)
        idx = feeder.node_phase_index()
        conns = [
            (feeder.load_buses[c.node] - 1, c.kind, c.phases)
            for c in feeder.connections
        ] + [(c.bus, c.kind, c.phases) for c in feeder.capacitors]
        n_c = len(conns)
        inc = np.zeros((n_c, len(nps)))
        for j, (bus, _kind, phases) in enumerate(conns):
            inc[j, idx[(bus, phases[0])]] = 1.0
            if len(phases) == 2:
                inc[j, idx[(bus, phases[1])]] = -1.0
        alloc = np.zeros((self.num_nodes, n_c))  # state (p.u.) -> connection
        for j, c in enumerate(feeder.connections):
            alloc[c.node, j] = c.share / feeder.s_base_phase_kva
        q_fix = np.zeros(n_c)
        for j, c in enumerate(feeder.capacitors):
            q_fix[len(feeder.connections) + j] = -c.q_kvar / feeder.s_base_phase_kva
        root = np.zeros((len(nps), 3))
        for n, (k, p) in enumerate(nps):
            if feeder.parent[k] == 0:
                root[n, p] = 1.0

        def buf(name, x):
            self.register_buffer(name, torch.as_tensor(np.asarray(x), dtype=dtype))

        buf("r_path", r_path)
        buf("x_path", x_path)
        buf("path", a_np)
        buf("incidence", inc)
        buf("alloc", alloc)
        buf("q_fix", q_fix)
        buf("root", root)
        buf("vs_re", vs.real[ph])
        buf("vs_im", vs.imag[ph])
        buf("vs3_re", vs.real)
        buf("vs3_im", vs.imag)
        self.register_buffer("mask_index", torch.as_tensor(np.flatnonzero(mask)))

    def _injections(self, state: Tensor):
        state = state.to(self.r_path.dtype)
        n = self.num_nodes
        if state.shape[-1] != 2 * n:
            raise ValueError(f"state has {state.shape[-1]} columns, expected {2 * n}")
        p = state[..., :n] @ self.alloc
        q = state[..., n:] @ self.alloc + self.q_fix
        return p, q

    def _currents(self, p, q, e, f):
        """Node-phase currents drawn, I = D^T conj(S / (D V)), p.u."""
        ue = e @ self.incidence.T
        uf = f @ self.incidence.T
        mag2 = ue * ue + uf * uf
        jr = (p * ue + q * uf) / mag2
        ji = (p * uf - q * ue) / mag2
        return jr @ self.incidence, ji @ self.incidence

    def _sweep(self, p, q, e, f):
        ir, ii = self._currents(p, q, e, f)
        dr = ir @ self.r_path.T - ii @ self.x_path.T
        di = ii @ self.r_path.T + ir @ self.x_path.T
        return self.vs_re - dr, self.vs_im - di, ir, ii

    def _solve(self, state: Tensor):
        p, q = self._injections(state)
        zero = torch.zeros_like(p[..., :1])
        e = zero + self.vs_re
        f = zero + self.vs_im
        for _ in range(self.num_iterations):
            e, f, _, _ = self._sweep(p, q, e, f)
        e_next, f_next, ir, ii = self._sweep(p, q, e, f)
        mismatch = torch.maximum(
            (e_next - e).abs().amax(-1), (f_next - f).abs().amax(-1)
        )
        return e, f, ir, ii, mismatch

    def node_phase_currents(self, state: Tensor) -> tuple[Tensor, Tensor]:
        """Real and imaginary node-phase currents drawn at the solution, p.u."""
        _, _, ir, ii, _ = self._solve(state)
        return ir, ii

    def forward(self, state: Tensor) -> ThreePhaseResult:
        """State [..., 2N]: P (kW) of the N state nodes, then Q (kvar)."""
        e, f, ir, ii, mismatch = self._solve(state)
        br = ir @ self.path
        bi = ii @ self.path
        cur_all = torch.sqrt(br * br + bi * bi) * self.i_base_a
        cur = cur_all[..., self.mask_index]
        hr = br @ self.root  # head current per phase
        hi = bi @ self.root
        sb = self.s_base_phase_kva
        p_ph = (self.vs3_re * hr + self.vs3_im * hi) * sb
        q_ph = (self.vs3_im * hr - self.vs3_re * hi) * sb
        vm = torch.sqrt(e * e + f * f)
        return ThreePhaseResult(
            vm=vm,
            e=e,
            f=f,
            branch_current_a=cur,
            p_head_kw=p_ph.sum(-1),
            q_head_kvar=q_ph.sum(-1),
            mismatch=mismatch,
            p_head_phase_kw=p_ph,
            q_head_phase_kvar=q_ph,
            branch_current_all_a=cur_all,
        )


class ThreePhaseMeasurementModel(torch.nn.Module):
    """h(s): state [..., 2N] -> channels [..., m] of a ThreePhaseMeasurementSet."""

    def __init__(self, mset: ThreePhaseMeasurementSet, power_flow: ThreePhasePowerFlow):
        super().__init__()
        self.mset = mset
        self.power_flow = power_flow
        feeder = power_flow.feeder
        idx = feeder.node_phase_index()
        self.voltage_index = [
            idx[(feeder.bus_index(b), p)]
            for b, phases in zip(mset.voltage_buses, mset.voltage_phases, strict=False)
            for p in phases
        ]

    def from_result(self, res: ThreePhaseResult, state: Tensor) -> Tensor:
        parts = [res.p_head_phase_kw, res.q_head_phase_kvar]
        if self.voltage_index:
            parts.append(res.vm[..., self.voltage_index])
        return torch.cat(parts, dim=-1)

    def forward(self, state: Tensor) -> Tensor:
        return self.from_result(self.power_flow(state), state)

    def jacobian(self, state: Tensor) -> Tensor:
        """dh/ds at each row of state [B, d] -> [B, m, d]."""
        from torch.func import jacfwd, vmap

        return vmap(jacfwd(self.forward))(state)


def ratings_3ph(pf: ThreePhasePowerFlow) -> np.ndarray:
    """Rating per kept branch-phase: its current at nominal loading in the
    three-phase model, A. Raises if any is below MIN_RATING_A."""
    feeder = pf.feeder
    nom = np.concatenate([feeder.p_nominal_kw, feeder.q_nominal_kvar])
    s = torch.as_tensor(nom, dtype=pf.r_path.dtype, device=pf.r_path.device)
    with torch.no_grad():
        res = pf(s.unsqueeze(0))
    ratings = res.branch_current_a[0].cpu().numpy()
    if ratings.size and ratings.min() < MIN_RATING_A:
        raise ValueError(
            f"kept branch-phase rating {ratings.min():.4g} A is below {MIN_RATING_A} A"
        )
    return ratings


def lindistflow_3ph(
    mset: ThreePhaseMeasurementSet, feeder: ThreePhaseFeeder
) -> tuple[np.ndarray, np.ndarray]:
    """Unbalanced LinDistFlow operator (H [m, 2N], c0 [m]) in channel units.

    Linearization at flat balanced voltages V_0 [1, a, a^2] (Gan and Low,
    "Convex relaxations and linear approximation for optimal power flow in
    multiphase radial networks," PSCC 2014, arXiv:1406.3054; Arnold, Sankur,
    Dobbe, Brady, Callaway and von Meier, "Optimal dispatch of reactive power
    for voltage regulation and balancing in unbalanced distribution systems,"
    IEEE PES General Meeting 2016, pp. 1-5). Delta connections are converted to equivalent wye
    powers at these voltages; with c_nm = conj(a_n) a_m Z_path_nm,
    |V_n| = V_0 - (1/V_0) sum_m (Re(c_nm) P_m + Im(c_nm) Q_m) and the head
    power per phase is the sum of the node-phase powers on that phase.
    Capacitors enter c0 as fixed injections. Without capacitors this equals
    the AC Jacobian at state 0, whose voltages are flat and balanced. For a
    single phase c_nn = R + jX, the balanced LinDistFlow."""
    r_path, x_path = feeder.z_path_pu()
    z = r_path + 1j * x_path
    nps = feeder.node_phases
    ph = np.array([p for _, p in nps], dtype=np.int64)
    a_n = PHASORS[ph]
    c = np.conj(a_n)[:, None] * a_n[None, :] * z
    e_mat, s_cap = feeder.wye_equivalent()  # S^eq = E s / S_b + s_cap
    v0 = feeder.v_slack
    sb = feeder.s_base_phase_kva
    onehot = (ph[None, :] == np.arange(3)[:, None]).astype(np.float64)  # [3, n_np]
    e_head = onehot @ e_mat  # [3, N]
    rows = [np.concatenate([e_head.real, -e_head.imag], axis=1)]
    rows.append(np.concatenate([e_head.imag, e_head.real], axis=1))
    cap_head = onehot @ s_cap * sb
    offset = [cap_head.real, cap_head.imag]
    idx = feeder.node_phase_index()
    vidx = [
        idx[(feeder.bus_index(b), p)]
        for b, phases in zip(mset.voltage_buses, mset.voltage_phases, strict=False)
        for p in phases
    ]
    if vidx:
        g = c[vidx] @ np.conj(e_mat)  # [nv, N]
        rows.append(np.concatenate([-g.real, -g.imag], axis=1) / (v0 * sb))
        offset.append(v0 - (c[vidx] @ np.conj(s_cap)).real / v0)
    return np.concatenate(rows, axis=0), np.concatenate(offset)

"""IEEE 33-bus radial feeder (Baran and Wu, 1989), measurement sets and noise.

Data as in pandapower ``case33bw`` (tie lines out of service): 12.66 kV,
slack at bus 1 with 1.0 p.u., 32 branches, 32 load nodes. Bus numbers in
names and docstrings are the usual 1-indexed ones; arrays use node index
``i = bus - 2`` for the 32 non-slack buses, and branch ``b`` feeds node
``b`` (its to-bus is bus ``b + 2``).

Units: active power in kW, reactive power in kvar, voltage magnitude in
p.u., branch current in A and in percent of the branch rating.
"""

from dataclasses import dataclass, replace

import numpy as np


V_BASE_KV = 12.66
S_BASE_KVA = 1000.0
Z_BASE_OHM = V_BASE_KV**2 / (S_BASE_KVA / 1000.0)
I_BASE_A = S_BASE_KVA / (np.sqrt(3.0) * V_BASE_KV)
NUM_NODES = 32

# 1-indexed from-bus of the branch feeding bus k + 2 (k = 0..31)
FROM_BUS = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
    2, 19, 20, 21, 3, 23, 24, 6, 26, 27, 28, 29, 30, 31, 32,
)  # fmt: skip
R_OHM = (
    0.0922, 0.493, 0.366, 0.3811, 0.819, 0.1872, 0.7114, 1.03, 1.044,
    0.1966, 0.3744, 1.468, 0.5416, 0.591, 0.7463, 1.289, 0.732, 0.164,
    1.5042, 0.4095, 0.7089, 0.4512, 0.898, 0.896, 0.203, 0.2842, 1.059,
    0.8042, 0.5075, 0.9744, 0.3105, 0.341,
)  # fmt: skip
X_OHM = (
    0.047, 0.2511, 0.1864, 0.1941, 0.707, 0.6188, 0.2351, 0.74, 0.74,
    0.065, 0.1238, 1.155, 0.7129, 0.526, 0.545, 1.721, 0.574, 0.1565,
    1.3554, 0.4784, 0.9373, 0.3083, 0.7091, 0.7011, 0.1034, 0.1447,
    0.9337, 0.7006, 0.2585, 0.963, 0.3619, 0.5302,
)  # fmt: skip
P_NOMINAL_KW = (
    100.0, 90.0, 120.0, 60.0, 60.0, 200.0, 200.0, 60.0, 60.0, 45.0, 60.0,
    60.0, 120.0, 60.0, 60.0, 60.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0,
    420.0, 420.0, 60.0, 60.0, 60.0, 120.0, 200.0, 150.0, 210.0, 60.0,
)  # fmt: skip
Q_NOMINAL_KVAR = (
    60.0, 40.0, 80.0, 30.0, 20.0, 100.0, 100.0, 20.0, 20.0, 30.0, 35.0,
    35.0, 80.0, 10.0, 20.0, 20.0, 40.0, 40.0, 40.0, 40.0, 40.0, 50.0,
    200.0, 200.0, 25.0, 25.0, 20.0, 70.0, 600.0, 70.0, 100.0, 40.0,
)  # fmt: skip

# PV placement: 1-indexed buses
PV_HEAVY_BUSES_BASE = (13, 14, 15, 16, 17, 18)
PV_BUSES_HIGH = tuple(range(11, 19)) + tuple(range(26, 34))


def node_of_bus(bus: int) -> int:
    if not 2 <= bus <= 33:
        raise ValueError(f"bus {bus} is not a load bus")
    return bus - 2


def tan_phi() -> np.ndarray:
    """Fixed reactive-to-active ratio per node, Q_i = P_i tan(phi_i)."""
    return np.asarray(Q_NOMINAL_KVAR) / np.asarray(P_NOMINAL_KW)


def path_matrix() -> np.ndarray:
    """A[i, b] = 1 if branch b lies on the path from the slack to node i."""
    a = np.zeros((NUM_NODES, NUM_NODES))
    for i in range(NUM_NODES):
        bus = i + 2
        while bus != 1:
            b = bus - 2
            a[i, b] = 1.0
            bus = FROM_BUS[b]
    return a


def branch_impedance_pu() -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(R_OHM) / Z_BASE_OHM, np.asarray(X_OHM) / Z_BASE_OHM


@dataclass(frozen=True)
class MeasurementSet:
    """Measured quantities; the slack voltage is fixed and known.

    Channels, in order: feeder-head P (kW) and Q (kvar), voltage magnitudes
    at ``voltage_buses`` (p.u.), active power at ``monitored_buses`` (kW).
    Buses are labels of the feeder (bus numbers for IEEE 33-bus).
    """

    name: str
    voltage_buses: tuple = ()
    monitored_buses: tuple = ()

    @property
    def num_channels(self) -> int:
        return 2 + len(self.voltage_buses) + len(self.monitored_buses)

    @property
    def channel_names(self) -> list[str]:
        names = ["p_head_kw", "q_head_kvar"]
        names += [f"vm_bus{b}_pu" for b in self.voltage_buses]
        names += [f"p_bus{b}_kw" for b in self.monitored_buses]
        return names

    def nominal_magnitude(self, feeder: "Feeder | None" = None) -> np.ndarray:
        """Nominal magnitude per channel, the reference for the noise level."""
        feeder = feeder or Feeder()
        p_nom = feeder.p_nominal_kw
        values = [p_nom.sum(), np.sum(feeder.q_nominal_kvar)]
        values += [1.0] * len(self.voltage_buses)
        values += [p_nom[feeder.node_index(b)] for b in self.monitored_buses]
        return np.asarray(values)

    def noise_std(
        self, noise_level: float = 1.0, feeder: "Feeder | None" = None
    ) -> np.ndarray:
        """Standard deviation per channel: one third of a 1% maximum error of
        the nominal magnitude, times ``noise_level`` (1 nominal, 3 high)."""
        return noise_level * (0.01 / 3.0) * self.nominal_magnitude(feeder)


MEASUREMENT_SETS = {
    "S1": MeasurementSet("S1"),
    "S2": MeasurementSet("S2", voltage_buses=(18, 22, 33)),
    "S3": MeasurementSet("S3", voltage_buses=(18, 22, 33), monitored_buses=(16, 30)),
}
NOISE_LEVELS = {"nominal": 1.0, "high": 3.0}


@dataclass(frozen=True)
class Feeder:
    """A balanced radial feeder as data (default: IEEE 33-bus, case33bw).

    Buses are indexed 0..B with 0 the slack; branch k feeds bus k + 1 from
    bus ``parent[k]``. The state holds one node per bus in ``load_buses``
    (bus indices 1..B); other buses carry no injection. Labels name the buses
    (bus numbers as strings for IEEE 33-bus). Impedances in ohm, nominal
    loads (per state node) in kW and kvar; the slack voltage is
    ``v_slack`` p.u. and known.
    """

    name: str = "ieee33"
    parent: tuple = tuple(b - 1 for b in FROM_BUS)
    r_ohm: tuple = R_OHM
    x_ohm: tuple = X_OHM
    load_buses: tuple = tuple(range(1, NUM_NODES + 1))
    p_nominal: tuple = P_NOMINAL_KW
    q_nominal: tuple = Q_NOMINAL_KVAR
    bus_labels: tuple = tuple(str(b) for b in range(1, NUM_NODES + 2))
    v_base_kv: float = V_BASE_KV
    s_base_kva: float = S_BASE_KVA
    v_slack: float = 1.0
    measurement_sets: tuple = ()  # (name, voltage_buses, monitored_buses)
    shunt_q_kvar: tuple = ()  # fixed reactive injection per non-slack bus (capacitors < 0)

    @property
    def num_buses(self) -> int:
        """Non-slack buses B (branches)."""
        return len(self.parent)

    @property
    def num_nodes(self) -> int:
        """State nodes N (buses with an injection)."""
        return len(self.load_buses)

    @property
    def z_base_ohm(self) -> float:
        return self.v_base_kv**2 / (self.s_base_kva / 1000.0)

    @property
    def i_base_a(self) -> float:
        return self.s_base_kva / (np.sqrt(3.0) * self.v_base_kv)

    @property
    def path(self) -> np.ndarray:
        """A[i, b] = 1 if branch b lies on the path from the slack to bus i + 1."""
        b_count = self.num_buses
        a = np.zeros((b_count, b_count))
        for i in range(b_count):
            bus = i + 1
            while bus != 0:
                a[i, bus - 1] = 1.0
                bus = self.parent[bus - 1]
        return a

    @property
    def r_pu(self) -> np.ndarray:
        return np.asarray(self.r_ohm) / self.z_base_ohm

    @property
    def x_pu(self) -> np.ndarray:
        return np.asarray(self.x_ohm) / self.z_base_ohm

    @property
    def tan_phi(self) -> np.ndarray:
        """Fixed reactive-to-active ratio per state node (nominal loads)."""
        return np.asarray(self.q_nominal) / np.asarray(self.p_nominal)

    @property
    def p_nominal_kw(self) -> np.ndarray:
        return np.asarray(self.p_nominal)

    @property
    def q_nominal_kvar(self) -> np.ndarray:
        return np.asarray(self.q_nominal)

    @property
    def load_index(self) -> np.ndarray:
        """Position of each state node among the non-slack buses (0..B-1)."""
        return np.asarray(self.load_buses, dtype=np.int64) - 1

    def bus_index(self, label) -> int:
        """Position of a non-slack bus (0..B-1) from its label."""
        k = self.bus_labels.index(str(label))
        if k == 0:
            raise ValueError(f"bus {label} is the slack")
        return k - 1

    def node_index(self, label) -> int:
        """State node of a load bus from its label."""
        k = self.bus_labels.index(str(label))
        return self.load_buses.index(k)

    @property
    def z_path(self) -> tuple[np.ndarray, np.ndarray]:
        """Common-path resistance and reactance, R = A diag(r) A^T (p.u.)."""
        a = self.path
        return a @ np.diag(self.r_pu) @ a.T, a @ np.diag(self.x_pu) @ a.T

    def measurement_set(self, name: str) -> MeasurementSet:
        if self.name == "ieee33" and not self.measurement_sets:
            return MEASUREMENT_SETS[name]
        for n, v, m in self.measurement_sets:
            if n == name:
                return MeasurementSet(n, tuple(v), tuple(m))
        raise KeyError(f"feeder {self.name} has no measurement set {name}")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "parent": list(self.parent),
            "r_ohm": list(self.r_ohm),
            "x_ohm": list(self.x_ohm),
            "load_buses": list(self.load_buses),
            "p_nominal": list(self.p_nominal),
            "q_nominal": list(self.q_nominal),
            "bus_labels": list(self.bus_labels),
            "v_base_kv": self.v_base_kv,
            "s_base_kva": self.s_base_kva,
            "v_slack": self.v_slack,
            "measurement_sets": [list(m) for m in self.measurement_sets],
            "shunt_q_kvar": list(self.shunt_q_kvar),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Feeder":
        return cls(
            name=d["name"],
            parent=tuple(d["parent"]),
            r_ohm=tuple(d["r_ohm"]),
            x_ohm=tuple(d["x_ohm"]),
            load_buses=tuple(d["load_buses"]),
            p_nominal=tuple(d["p_nominal"]),
            q_nominal=tuple(d["q_nominal"]),
            bus_labels=tuple(d["bus_labels"]),
            v_base_kv=d["v_base_kv"],
            s_base_kva=d["s_base_kva"],
            v_slack=d["v_slack"],
            measurement_sets=tuple(
                (m[0], tuple(m[1]), tuple(m[2])) for m in d["measurement_sets"]
            ),
            shunt_q_kvar=tuple(d.get("shunt_q_kvar", ())),
        )

    @classmethod
    def from_meta(cls, meta: dict | None) -> "Feeder":
        """The feeder recorded in a data set's metadata (default IEEE 33-bus)."""
        if meta and isinstance(meta.get("feeder"), dict):
            return cls.from_dict(meta["feeder"])
        return cls()


# Tie switches 33..37 of Baran and Wu (1989) / pandapower case33bw:
# (from_bus, to_bus, r_ohm, x_ohm), 1-indexed buses
TIE_BRANCHES = (
    (8, 21, 2.0, 2.0),
    (9, 15, 2.0, 2.0),
    (12, 22, 2.0, 2.0),
    (18, 33, 0.5, 0.5),
    (25, 29, 0.5, 0.5),
)

# Open branches (1-indexed, 1..37) of each radial topology
RECONFIGURATIONS: dict[str, tuple[int, ...]] = {
    "base": (33, 34, 35, 36, 37),
    "reconf_bw": (7, 9, 14, 32, 37),
    "reconf_all5": (7, 9, 14, 28, 32),
    "reconf_t33": (7, 34, 35, 36, 37),
    "reconf_t34": (14, 33, 35, 36, 37),
    "reconf_t35": (11, 33, 34, 36, 37),
    "reconf_t36": (32, 33, 34, 35, 37),
    "reconf_t37": (28, 33, 34, 35, 36),
}


def ieee33_all_branches() -> list[tuple[int, int, float, float]]:
    """All 37 branches (1-indexed 1..37) as (from_bus, to_bus, r_ohm, x_ohm)."""
    branches = [
        (FROM_BUS[k], k + 2, float(R_OHM[k]), float(X_OHM[k])) for k in range(NUM_NODES)
    ]
    branches.extend((int(u), int(v), float(r), float(x)) for u, v, r, x in TIE_BRANCHES)
    return branches


def reconfigured_ieee33(
    open_branches,
    name: str = "ieee33_reconf",
    base: "Feeder | None" = None,
) -> "Feeder":
    """Radial IEEE 33-bus feeder with the 5 given branches (1..37) open.

    All fields other than name, parent, r_ohm, x_ohm and measurement_sets are
    taken from ``base`` (default Feeder()), which must be the IEEE 33-bus
    base topology. Bus labels, load buses and nominal loads keep their order,
    so the state coordinates are unchanged. Raises ValueError if the closed
    branches do not form a spanning tree.
    """
    base = base or Feeder()
    if tuple(base.parent) != tuple(b - 1 for b in FROM_BUS):
        raise ValueError("base must be the IEEE 33-bus base topology")
    open_set = {int(b) for b in open_branches}
    if len(open_set) != 5 or not open_set <= set(range(1, 38)):
        raise ValueError(
            f"open_branches must be 5 distinct branches in 1..37, got {open_branches}"
        )
    all_br = ieee33_all_branches()
    adj: dict[int, list[tuple[int, float, float]]] = {
        u: [] for u in range(1, NUM_NODES + 2)
    }
    for b in range(1, 38):
        if b in open_set:
            continue
        u, v, r, x = all_br[b - 1]
        adj[u].append((v, r, x))
        adj[v].append((u, r, x))
    visited = {1}
    parent = [0] * NUM_NODES
    r_ohm = [0.0] * NUM_NODES
    x_ohm = [0.0] * NUM_NODES
    queue = [1]
    while queue:
        u = queue.pop(0)
        for v, r, x in adj[u]:
            if v not in visited:
                visited.add(v)
                parent[v - 2] = u - 1  # Feeder bus index, 0 = slack
                r_ohm[v - 2] = r
                x_ohm[v - 2] = x
                queue.append(v)
    if len(visited) != NUM_NODES + 1:
        raise ValueError(
            f"open branches {sorted(open_set)} disconnect the feeder "
            f"({len(visited)} of {NUM_NODES + 1} buses reached)"
        )
    msets = base.measurement_sets or tuple(
        (m.name, m.voltage_buses, m.monitored_buses) for m in MEASUREMENT_SETS.values()
    )
    return replace(
        base,
        name=name,
        parent=tuple(parent),
        r_ohm=tuple(r_ohm),
        x_ohm=tuple(x_ohm),
        measurement_sets=tuple(msets),
    )


def state_mode(meta: dict | None) -> str:
    """'p' (active power per node, Q = P tan(phi)) or 'pq' (P then Q)."""
    return (meta or {}).get("state", "p")

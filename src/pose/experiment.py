"""Shared setup for the DSSE scripts: settings, paired randomness, provenance.

Paired comparisons: every method sees the same truths, the same observations
and the same initial noise. One random generator per role and snapshot,
seeded from strings:

    obs:       (scenario, measurement set, noise level, snapshot)
    init:      (scenario, snapshot)        initial noise of every sampler
    ancestral: (scenario, snapshot)        noise of the ancestral steps
    draw:      (scenario, snapshot)        Gaussian / mixture posterior draws
"""

import hashlib
import json
import os
import platform
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from .data import SnapshotData
from .feeder import NOISE_LEVELS, Feeder, MeasurementSet, state_mode
from .lindistflow import lindistflow_operator, linearization_error
from .powerflow import MeasurementModel, RadialPowerFlow, branch_ratings_a


if TYPE_CHECKING:
    from .threephase import ThreePhaseFeeder


def stable_seed(*parts) -> int:
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**63 - 1)


def generators(role: str, scenario: str, snapshots, extra: tuple = ()) -> list:
    return [
        torch.Generator().manual_seed(stable_seed(role, scenario, *extra, int(i)))
        for i in snapshots
    ]


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state(repo_dir: str | None = None) -> dict:
    repo_dir = repo_dir or os.path.dirname(os.path.abspath(__file__))
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo_dir,
            text=True,
        ).strip()
        return {"commit": commit, "state": "dirty" if dirty else "clean"}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "state": "unknown"}


def hardware() -> dict:
    info = {"platform": platform.platform(), "torch": torch.__version__}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    info["slurm_job_id"] = os.environ.get("SLURM_JOB_ID", "none")
    info["slurm_array_task_id"] = os.environ.get("SLURM_ARRAY_TASK_ID", "none")
    return info


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


@dataclass
class Setting:
    scenario: str
    mset: MeasurementSet
    noise_name: str
    device: torch.device
    feeder: Feeder | None = None
    state: str = "p"  # "p" (Q = P tan(phi)) or "pq" (joint P, Q state)
    # unbalanced three-phase model of the feeder (threephase.py); with it, h, H,
    # c0, the ratings and the noise come from the three-phase physics, and
    # ``mset`` is its three-phase measurement set
    physics: "ThreePhaseFeeder | None" = None

    def __post_init__(self):
        self.feeder = self.feeder or Feeder()
        if self.physics is not None:
            self._init_three_phase()
            return
        self.power_flow = RadialPowerFlow(self.feeder).to(self.device)
        self.measurement = MeasurementModel(self.mset, self.power_flow).to(self.device)
        self.noise_std = torch.as_tensor(
            self.mset.noise_std(NOISE_LEVELS[self.noise_name], self.feeder),
            dtype=torch.float64,
            device=self.device,
        )
        self.noise_cov = torch.diag(self.noise_std**2)
        h_mat, c0 = lindistflow_operator(self.mset, self.feeder, self.state)
        self.h_lin = torch.as_tensor(h_mat, dtype=torch.float64, device=self.device)
        self.c0 = torch.as_tensor(c0, dtype=torch.float64, device=self.device)
        self.ratings = torch.as_tensor(
            branch_ratings_a(self.power_flow), dtype=torch.float64, device=self.device
        )

    def _init_three_phase(self):
        from .threephase import (
            ThreePhaseMeasurementModel,
            ThreePhasePowerFlow,
            lindistflow_3ph,
            ratings_3ph,
        )

        if self.state != "pq":
            raise ValueError("three-phase physics needs the joint (P, Q) state")
        physics = self.physics
        f = self.feeder
        same = (
            tuple(physics.load_buses) == tuple(f.load_buses)
            and tuple(physics.bus_labels) == tuple(f.bus_labels)
            and np.allclose(physics.p_nominal, f.p_nominal, rtol=0, atol=1e-9)
            and np.allclose(physics.q_nominal, f.q_nominal, rtol=0, atol=1e-9)
        )
        if not same:
            raise ValueError(
                "three-phase feeder does not match the data set's feeder (load "
                "buses, bus labels or nominal loads differ)"
            )
        self.power_flow = ThreePhasePowerFlow(physics).to(self.device)
        self.measurement = ThreePhaseMeasurementModel(self.mset, self.power_flow).to(
            self.device
        )
        self.noise_std = torch.as_tensor(
            self.mset.noise_std(NOISE_LEVELS[self.noise_name], physics),
            dtype=torch.float64,
            device=self.device,
        )
        self.noise_cov = torch.diag(self.noise_std**2)
        h_mat, c0 = lindistflow_3ph(self.mset, physics)
        self.h_lin = torch.as_tensor(h_mat, dtype=torch.float64, device=self.device)
        self.c0 = torch.as_tensor(c0, dtype=torch.float64, device=self.device)
        self.ratings = torch.as_tensor(
            ratings_3ph(self.power_flow), dtype=torch.float64, device=self.device
        )

    @classmethod
    def from_names(
        cls, scenario: str, mset: str, noise: str, device, meta: dict | None = None
    ) -> "Setting":
        """Feeder and state mode come from the data set's metadata ``meta``
        (default: IEEE 33-bus, fixed power factor)."""
        feeder = Feeder.from_meta(meta)
        return cls(
            scenario,
            feeder.measurement_set(mset),
            noise,
            device,
            feeder,
            state_mode(meta),
        )

    @property
    def name(self) -> str:
        return f"{self.scenario}_{self.mset.name}_{self.noise_name}"

    def linearization(self, data: SnapshotData):
        idx = data.indices("train")
        if self.physics is not None:
            operator = (self.h_lin.cpu().numpy(), self.c0.cpu().numpy())
            return linearization_error(
                self.measurement, data.p_kw[idx], operator=operator
            )
        return linearization_error(self.measurement, data.p_kw[idx])

    def observe(self, data: SnapshotData, snapshots: np.ndarray):
        """Truth [S, d] and noisy observations y [S, m] (paired across methods)."""
        truth = torch.as_tensor(
            data.p_kw[snapshots], dtype=torch.float64, device=self.device
        )
        with torch.no_grad():
            clean = self.measurement(truth)
        gens = generators(
            "obs", self.scenario, snapshots, (self.mset.name, self.noise_name)
        )
        noise = torch.stack(
            [
                torch.randn(self.mset.num_channels, generator=g, dtype=torch.float64)
                for g in gens
            ]
        ).to(self.device)
        return truth, clean + noise * self.noise_std

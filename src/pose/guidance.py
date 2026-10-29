"""Endpoint-corrected posterior samplers (M0-M2) and DPS (baseline B4).

At each step the prior gives the endpoint estimate z_hat = E[z | x_t] in
standardized space. The correction is

    z_tilde = z_hat + K (y - h(s_hat)),
    K = Sigma_t H_z^T (H_z Sigma_t H_z^T + R_eff)^-1,

with s_hat = mu + sigma * z_hat in kW, the AC measurement function h for
the residual, and H_z = H diag(sigma). The variants fix three choices:

    M0: Sigma_t = r_t^2 I, H = LinDistFlow, R_eff = 0 (projection)
    M1: Sigma_t = r_t^2 I, H = LinDistFlow, R_eff = R + Sigma_lin
    M2: Sigma_t = Tweedie, H = LinDistFlow, R_eff = R + Sigma_lin

r_t^2 = (1 - t)^2 / (t^2 + (1 - t)^2); Tweedie Sigma_t = (1 - t)^2 / t * dz_hat/dx_t,
symmetrized with negative eigenvalues clipped to zero, evaluated at
max(t, T_MIN_COV) so that the first step (t = 0) is defined.

Samplers on the time grid t_i = 1 - (1 - i / N)^kappa, i = 0..N (kappa = 1
is the uniform grid; kappa > 1 puts more steps near the data end, where a
narrow posterior is resolved):
    euler: x <- x + (t_{i+1} - t_i) (z_tilde - x) / (1 - t_i)
    ancestral: the DDPM step of the linear path with alpha_t = t,
        sigma_t = 1 - t, around z_tilde.
Both return the corrected endpoint of the final step.

Covariance modes for the Tweedie variants (``sigma_mode``):
    full: Sigma_t projected onto the PSD cone with a batched eigh (the
        method of every reported result)
    meas: only G = Sym(Sigma_t) H_z^T is formed (m forward- and m
        reverse-mode products), and the negative directions of the whitened
        H_z G are removed from G consistently (LinDistFlow operator only)
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.func import jacfwd, jvp, vjp, vmap


T_MIN_COV = 1e-3
DEFAULT_GRID_POWER = 2.0


def time_grid(num_steps: int, power: float = DEFAULT_GRID_POWER) -> list[float]:
    """t_0 = 0 < t_1 < ... < t_N = 1, with t_i = 1 - (1 - i / N)^power."""
    return [1.0 - (1.0 - i / num_steps) ** power for i in range(num_steps + 1)]


@dataclass(frozen=True)
class Variant:
    name: str
    covariance: str  # "isotropic" or "tweedie"
    noise: str  # "zero" or "true"
    add_linearization_error: bool


VARIANTS = {
    "M0": Variant("M0", "isotropic", "zero", False),
    "M1": Variant("M1", "isotropic", "true", True),
    "M2": Variant("M2", "tweedie", "true", True),
}


def isotropic_r2(t: Tensor) -> Tensor:
    return (1 - t) ** 2 / (t**2 + (1 - t) ** 2)


SIGMA_MODES = ("full", "meas")


class StepTimer:
    """Wall-clock split of the correction into sections (instrumented runs
    only: every mark synchronizes the GPU)."""

    def __init__(self, device):
        self.device = torch.device(device)
        self.totals: dict[str, float] = {}
        self._last = 0.0

    def _now(self) -> float:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def start(self) -> None:
        self._last = self._now()

    def mark(self, name: str) -> None:
        now = self._now()
        self.totals[name] = self.totals.get(name, 0.0) + now - self._last
        self._last = now


def _endpoint_row_fn(prior, tc: Tensor):
    def endpoint_row(xr, month, qh, weekend):
        c = {"month": month[None], "quarter_hour": qh[None], "weekend": weekend[None]}
        return prior.endpoint(xr[None], tc, c)[0]

    return endpoint_row


def tweedie_symmetric(prior, x: Tensor, t: Tensor, ctx: dict[str, Tensor]) -> Tensor:
    """(1 - t)^2 / t * Sym(d E[z | x_t] / dx_t) per row, before the PSD
    projection. [B, d, d], float64."""
    tc = torch.clamp(t, min=T_MIN_COV)
    endpoint_row = _endpoint_row_fn(prior, tc)
    jac = vmap(jacfwd(endpoint_row))(
        x, ctx["month"], ctx["quarter_hour"], ctx["weekend"]
    )
    cov = ((1 - tc) ** 2 / tc) * jac.double()
    return 0.5 * (cov + cov.transpose(-1, -2))


def psd_project_eigh(cov: Tensor) -> Tensor:
    """Projection onto the PSD cone: negative eigenvalues set to zero."""
    evals, evecs = torch.linalg.eigh(cov)
    return (evecs * evals.clamp(min=0.0)[..., None, :]) @ evecs.transpose(-1, -2)


def tweedie_covariance(
    prior,
    x: Tensor,
    t: Tensor,
    ctx: dict[str, Tensor],
    timer: StepTimer | None = None,
) -> Tensor:
    """(1 - t)^2 / t * d E[z | x_t] / dx_t per row, PSD-projected. [B, d, d]."""
    cov = tweedie_symmetric(prior, x, t, ctx)
    if timer is not None:
        timer.mark("jacobian")
    out = psd_project_eigh(cov)
    if timer is not None:
        timer.mark("psd")
    return out


def tweedie_cross_covariance(
    prior, x: Tensor, t: Tensor, ctx: dict[str, Tensor], h_z: Tensor
) -> Tensor:
    """G = (1 - t)^2 / t * Sym(dz_hat/dx_t) H_z^T without the d x d Jacobian:
    m forward-mode products (J H_z^T) and m reverse-mode products
    (J^T H_z^T). h_z [m, d] is shared by all rows. Returns [B, d, m], float64."""
    tc = torch.clamp(t, min=T_MIN_COV)
    endpoint_row = _endpoint_row_fn(prior, tc)
    rows = h_z.to(x.dtype)

    def products(xr, month, qh, weekend):
        def f(xx):
            return endpoint_row(xx, month, qh, weekend)

        jh = vmap(lambda v: jvp(f, (xr,), (v,))[1])(rows)
        _, pull = vjp(f, xr)
        jth = vmap(lambda u: pull(u)[0])(rows)
        return jh, jth

    jh, jth = vmap(products)(x, ctx["month"], ctx["quarter_hour"], ctx["weekend"])
    g = 0.5 * (jh.double() + jth.double()).transpose(-1, -2)
    return ((1 - tc) ** 2 / tc) * g


def whiten(a: Tensor, r_chol: Tensor) -> Tensor:
    """L^-1 A L^-T for symmetric A [B, m, m] and R_eff = L L^T, symmetrized."""
    left = torch.linalg.solve_triangular(r_chol, a, upper=False)
    aw = torch.linalg.solve_triangular(r_chol, left.transpose(-1, -2), upper=False)
    return 0.5 * (aw + aw.transpose(-1, -2))


def measurement_space_gain_terms(
    g: Tensor, h_z: Tensor, r_eff: Tensor, r_chol: Tensor, consistent: bool = True
) -> tuple[Tensor, Tensor]:
    """Numerator G' and innovation matrix S of K = G' S^-1 from G [B, d, m].

    The negative eigen-directions of the whitened A~ = L^-1 H_z G L^-T are
    removed. consistent=True removes them from G as well,
    G' = G L^-T P+ L^T, so that H_z G' = L A~_+ L^T and the update is a
    Kalman update of some PSD covariance; consistent=False keeps G. Rows
    whose A~ has no negative eigenvalue keep G unchanged."""
    a = h_z @ g
    a = 0.5 * (a + a.transpose(-1, -2))
    aw = whiten(a, r_chol)
    evals, evecs = torch.linalg.eigh(aw)
    keep = (evals > 0).to(evals.dtype)
    all_pos = keep.min(dim=-1).values > 0
    a_plus_w = (evecs * (evals * keep)[..., None, :]) @ evecs.transpose(-1, -2)
    a_plus = r_chol @ a_plus_w @ r_chol.transpose(-1, -2)
    a_plus = torch.where(
        all_pos[:, None, None], a, 0.5 * (a_plus + a_plus.transpose(-1, -2))
    )
    if consistent:
        p_plus = (evecs * keep[..., None, :]) @ evecs.transpose(-1, -2)
        g_lt = torch.linalg.solve_triangular(
            r_chol, g.transpose(-1, -2), upper=False
        ).transpose(-1, -2)
        g_proj = g_lt @ p_plus @ r_chol.transpose(-1, -2)
        g = torch.where(all_pos[:, None, None], g, g_proj)
    return g, a_plus + r_eff


class EndpointCorrector:
    """Kalman-type endpoint correction for one variant and measurement set.

    measurement: callable s_kw [B, 32] -> h(s) [B, m] (AC).
    h_lin: LinDistFlow matrix [m, 32] (per kW).
    noise_cov: true measurement noise covariance R [m, m].
    lin_cov: LinDistFlow error covariance Sigma_lin [m, m].
    """

    def __init__(
        self,
        prior,
        measurement,
        variant: Variant,
        h_lin: Tensor,
        noise_cov: Tensor,
        lin_cov: Tensor | None = None,
        sigma_mode: str = "full",
        timer: StepTimer | None = None,
    ):
        if sigma_mode not in SIGMA_MODES:
            raise ValueError(f"unknown sigma_mode {sigma_mode}")
        if sigma_mode == "meas" and variant.covariance != "tweedie":
            raise ValueError("sigma_mode 'meas' needs the Tweedie variant")
        self.sigma_mode = sigma_mode
        self.timer = timer
        self.prior = prior
        self.measurement = measurement
        self.variant = variant
        self.h_lin = h_lin.double()
        m = noise_cov.shape[0]
        if variant.noise == "zero":
            r_eff = torch.zeros(m, m, dtype=torch.float64, device=noise_cov.device)
        else:
            r_eff = noise_cov.double().clone()
        if variant.add_linearization_error and lin_cov is not None:
            r_eff = r_eff + lin_cov.double()
        self.r_eff = r_eff
        self.std = prior.std_kw.double()

    def covariance(self, x: Tensor, t: Tensor, ctx) -> Tensor:
        d = x.shape[-1]
        eye = torch.eye(d, dtype=torch.float64, device=x.device)
        if self.variant.covariance == "isotropic":
            return (isotropic_r2(t.double()) * eye).expand(x.shape[0], d, d)
        return tweedie_covariance(self.prior, x, t, ctx, self.timer)

    @torch.no_grad()
    def __call__(self, x: Tensor, t: Tensor, ctx, y: Tensor) -> tuple[Tensor, Tensor]:
        if self.sigma_mode == "meas":
            return self._call_meas(x, t, ctx, y)
        if self.timer is not None:
            self.timer.start()
        z_hat = self.prior.endpoint(x, t, ctx)
        s_hat = self.prior.to_kw(z_hat.double())
        if self.timer is not None:
            self.timer.mark("other")
        sigma = self.covariance(x, t, ctx)
        h_z = self.h_lin.expand(s_hat.shape[0], *self.h_lin.shape) * self.std
        s_mat = h_z @ sigma @ h_z.transpose(-1, -2) + self.r_eff
        resid = y - self.measurement(s_hat)
        gain_resid = (
            sigma
            @ h_z.transpose(-1, -2)
            @ torch.linalg.solve(s_mat, resid.unsqueeze(-1))
        )
        z_tilde = z_hat.double() + gain_resid.squeeze(-1)
        if self.timer is not None:
            self.timer.mark("other")
        return z_tilde, z_hat

    @property
    def r_chol(self) -> Tensor:
        if not hasattr(self, "_r_chol"):
            self._r_chol = torch.linalg.cholesky(self.r_eff)
        return self._r_chol

    def _call_meas(self, x: Tensor, t: Tensor, ctx, y: Tensor) -> tuple[Tensor, Tensor]:
        timer = self.timer
        if timer is not None:
            timer.start()
        z_hat = self.prior.endpoint(x, t, ctx)
        s_hat = self.prior.to_kw(z_hat.double())
        h_z = self.h_lin * self.std  # [m, d], shared by all rows
        if timer is not None:
            timer.mark("other")
        g = tweedie_cross_covariance(self.prior, x, t, ctx, h_z)
        if timer is not None:
            timer.mark("jacobian")
        g, s_mat = measurement_space_gain_terms(g, h_z, self.r_eff, self.r_chol)
        if timer is not None:
            timer.mark("psd")
        resid = y - self.measurement(s_hat)
        gain_resid = g @ torch.linalg.solve(s_mat, resid.unsqueeze(-1))
        z_tilde = z_hat.double() + gain_resid.squeeze(-1)
        if timer is not None:
            timer.mark("other")
        return z_tilde, z_hat


def _initial_noise(
    generators: Sequence[torch.Generator], num_samples: int, dim: int, device
) -> Tensor:
    return torch.cat(
        [torch.randn(num_samples, dim, generator=g) for g in generators]
    ).to(device)


def _expand_rows(ctx: dict[str, Tensor], y: Tensor, num_samples: int):
    ctx_rows = {k: v.repeat_interleave(num_samples, 0) for k, v in ctx.items()}
    return ctx_rows, y.repeat_interleave(num_samples, 0)


@torch.no_grad()
def sample_corrected(
    prior,
    corrector: Callable,
    ctx: dict[str, Tensor],
    y: Tensor,
    num_samples: int,
    num_steps: int,
    init_generators: Sequence[torch.Generator],
    sampler: str = "euler",
    step_generators: Sequence[torch.Generator] | None = None,
    grid_power: float = DEFAULT_GRID_POWER,
) -> Tensor:
    """Posterior samples in kW, [S, num_samples, 32], for S snapshots.

    ctx: {name: [S]} context; y: [S, m] observations. One initial-noise
    generator per snapshot (and one ancestral-noise generator per snapshot
    for ``sampler="ancestral"``), so batching does not change the samples.
    """
    device = y.device
    dim = prior.mean_kw.shape[0]
    ctx_rows, y_rows = _expand_rows(ctx, y.double(), num_samples)
    x = _initial_noise(init_generators, num_samples, dim, device)
    grid = time_grid(num_steps, grid_power)
    z_tilde = None
    for i in range(num_steps):
        t = torch.tensor(grid[i], device=device)
        z_tilde, _ = corrector(x, t, ctx_rows, y_rows)
        zt = z_tilde.to(x.dtype)
        if sampler == "euler":
            x = x + (grid[i + 1] - grid[i]) * (zt - x) / (1 - t)
        elif sampler == "ancestral":
            s = torch.tensor(grid[i + 1], device=device)
            a_t, s_t, a_s, s_s = t, 1 - t, s, 1 - s
            eps = (x - a_t * zt) / s_t
            c2 = (s_t**2 - (a_t / a_s) ** 2 * s_s**2).clamp(min=0) * s_s**2 / s_t**2
            c = c2.sqrt()
            z = _initial_noise(step_generators, num_samples, dim, device)
            x = a_s * zt + (s_s**2 - c2).clamp(min=0).sqrt() * eps + c * z
        else:
            raise ValueError(f"unknown sampler {sampler}")
    s_kw = prior.to_kw(z_tilde)
    return s_kw.view(y.shape[0], num_samples, dim)


def sample_dps(
    prior,
    measurement,
    noise_cov: Tensor,
    ctx: dict[str, Tensor],
    y: Tensor,
    num_samples: int,
    num_steps: int,
    init_generators: Sequence[torch.Generator],
    step_size: float,
    grid_power: float = DEFAULT_GRID_POWER,
) -> Tensor:
    """DPS (Chung et al., 2023) on the flow: Euler step of the prior velocity
    plus -zeta grad_x ||y - h(s_hat(x))||_{R^-1} (the unsquared norm, as in
    DPS). Returns the final endpoint estimate in kW, [S, num_samples, 32]."""
    device = y.device
    dim = prior.mean_kw.shape[0]
    ctx_rows, y_rows = _expand_rows(ctx, y.double(), num_samples)
    r_chol = torch.linalg.cholesky(noise_cov.double())
    x = _initial_noise(init_generators, num_samples, dim, device)
    grid = time_grid(num_steps, grid_power)
    for i in range(num_steps):
        t = torch.tensor(grid[i], device=device)
        dt = grid[i + 1] - grid[i]
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            v = prior.velocity(x, t, ctx_rows)
            z_hat = x + (1 - t) * v
            resid = y_rows - measurement(prior.to_kw(z_hat.double()))
            white = torch.linalg.solve_triangular(
                r_chol, resid.unsqueeze(-1), upper=False
            ).squeeze(-1)
            norm = white.pow(2).sum(-1).clamp(min=1e-12).sqrt()
            grad = torch.autograd.grad(norm.sum(), x)[0]
        x = (x + dt * v - step_size * grad).detach()
    with torch.no_grad():
        t = torch.tensor(grid[-2], device=device)
        z_hat = prior.endpoint(x, t, ctx_rows)
        s_kw = prior.to_kw(z_hat.double())
    return s_kw.view(y.shape[0], num_samples, dim)


class AnalyticGaussianPrior(torch.nn.Module):
    """Exact flow prior for z ~ N(mu_z, Sigma_z) on the linear path.

    E[z | x_t] = mu + t Sigma (t^2 Sigma + (1 - t)^2 I)^-1 (x - t mu); the
    context is ignored. Same interface as prior.FlowPrior.
    """

    def __init__(self, mu_z: Tensor, sigma_z: Tensor, mean_kw: Tensor, std_kw: Tensor):
        super().__init__()
        self.register_buffer("mu_z", mu_z.double())
        self.register_buffer("sigma_z", sigma_z.double())
        self.register_buffer("mean_kw", mean_kw.double())
        self.register_buffer("std_kw", std_kw.double())

    def endpoint(self, x: Tensor, t: Tensor, ctx=None) -> Tensor:
        x = x.double()
        d = self.mu_z.shape[0]
        eye = torch.eye(d, dtype=torch.float64, device=x.device)
        a = t**2 * self.sigma_z + (1 - t) ** 2 * eye
        # explicit inverse: linalg.solve with a batched right-hand side is
        # mis-batched under vmap(jacfwd) in torch 2.8
        gain = t * self.sigma_z @ torch.linalg.inv(a)
        return self.mu_z + (x - t * self.mu_z) @ gain.T

    def velocity(self, x: Tensor, t: Tensor, ctx=None) -> Tensor:
        return (self.endpoint(x, t, ctx) - x.double()) / (1 - t)

    def to_kw(self, z: Tensor) -> Tensor:
        return self.mean_kw + self.std_kw * z

    def to_z(self, p_kw: Tensor) -> Tensor:
        return (p_kw - self.mean_kw) / self.std_kw


class LinearMeasurement(torch.nn.Module):
    """h(s) = c0 + H s with an exact Jacobian (exactness test and unit tests)."""

    def __init__(self, h_mat: Tensor, c0: Tensor):
        super().__init__()
        self.register_buffer("h_mat", h_mat.double())
        self.register_buffer("c0", c0.double())

    def forward(self, s_kw: Tensor) -> Tensor:
        return self.c0 + s_kw.double() @ self.h_mat.T

    def jacobian(self, s_kw: Tensor) -> Tensor:
        return self.h_mat.expand(s_kw.shape[0], *self.h_mat.shape)


def gaussian_posterior(
    mu: Tensor, cov: Tensor, h_mat: Tensor, c0: Tensor, noise_cov: Tensor, y: Tensor
) -> tuple[Tensor, Tensor]:
    """Exact posterior of s ~ N(mu, cov), y = c0 + H s + e, e ~ N(0, R)."""
    s_mat = h_mat @ cov @ h_mat.T + noise_cov
    gain = cov @ h_mat.T @ torch.linalg.inv(s_mat)
    mean = mu + gain @ (y - c0 - h_mat @ mu)
    post = cov - gain @ h_mat @ cov
    return mean, 0.5 * (post + post.T)

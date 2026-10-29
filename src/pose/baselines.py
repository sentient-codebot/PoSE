"""Baselines B1-B7.

(i)   B1 WLS with diagonal Gaussian pseudo-measurements, Gauss-Newton on AC h.
(ii)  B2 full-covariance Gaussian prior (Ledoit-Wolf), Gauss-Newton MAP,
          Laplace covariance; B3 Gaussian mixture prior with a Kalman update
          per component under LinDistFlow; B7 pooled Gaussian with smooth
          kernel pooling over calendar context.
(iii) B5 MLP regression y -> s with dropout; MC dropout for samples.
(iv)  B4 DPS (guidance.sample_dps); B6 conditional flow matching p(s | c, y).

Gaussian pseudo-measurements come from a context bin: the training snapshots
of the same month and day type within +-1 hour (+-4 quarter-hours, circular)
of the target quarter-hour.
"""

import copy
import math
from collections.abc import Sequence

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.mixture import GaussianMixture
from torch import Tensor, nn

from .data import SnapshotData
from .guidance import DEFAULT_GRID_POWER, time_grid
from .prior import (
    ContextEmbedding,
    FlowPrior,
    PriorConfig,
    VelocityMLP,
    context_tensors,
)


BIN_HALF_WIDTH_QH = 4


BIN_FALLBACK_HALF_WIDTHS_QH = (8, 16, 24, 48)  # +-2 h, +-4 h, +-6 h, whole day


class ContextBins:
    """Training snapshots grouped by (month, day type, quarter-hour window).

    min_members > 0 (off by default; used for small model-world training
    sets): a bin with fewer members is widened in quarter-hours
    (BIN_FALLBACK_HALF_WIDTHS_QH), then over neighbouring months (+-1, +-2,
    ... with the whole day), until it has enough.
    """

    def __init__(
        self,
        data: SnapshotData,
        half_width: int = BIN_HALF_WIDTH_QH,
        min_members: int = 0,
    ):
        idx = data.indices("train")
        self.p_kw = data.p_kw[idx]
        self.month = data.context["month"][idx]
        self.weekend = data.context["weekend"][idx]
        self.qh = data.context["quarter_hour"][idx]
        self.half_width = half_width
        self.min_members = min_members
        self.widened: dict = {}
        self._cache: dict = {}

    def members(self, month: int, quarter_hour: int, weekend: int) -> np.ndarray:
        dq = np.abs(self.qh - quarter_hour)
        dq = np.minimum(dq, 96 - dq)
        day = self.weekend == weekend
        mask = (self.month == month) & day & (dq <= self.half_width)
        if self.min_members <= 0 or mask.sum() >= self.min_members:
            return self.p_kw[mask]
        for width in BIN_FALLBACK_HALF_WIDTHS_QH:
            mask = (self.month == month) & day & (dq <= width)
            if mask.sum() >= self.min_members:
                self.widened[(month, quarter_hour, weekend)] = ("qh", width)
                return self.p_kw[mask]
        dm = np.abs(self.month - month)
        dm = np.minimum(dm, 12 - dm)
        for months in range(1, 7):
            mask = day & (dm <= months)
            if mask.sum() >= self.min_members:
                break
        self.widened[(month, quarter_hour, weekend)] = ("month", months)
        return self.p_kw[mask]

    def gaussian(self, month: int, quarter_hour: int, weekend: int, full: bool):
        key = ("gauss", month, quarter_hour, weekend, full)
        if key not in self._cache:
            x = self.members(month, quarter_hour, weekend)
            mean = x.mean(0)
            if full:
                cov = LedoitWolf().fit(x).covariance_
            else:
                cov = np.diag(x.var(0, ddof=1))
            self._cache[key] = (mean, cov, len(x))
        return self._cache[key]


class KernelBins:
    """B7: Gaussian prior with smooth context pooling. For a test context,
    the mean and covariance are
    weighted over all training snapshots of the same day type, with circular
    Gaussian kernels in time of day (bandwidth ``h_hours``) and month
    (``h_months``); the covariance is shrunk towards its diagonal by
    ``alpha`` and given a 1e-6 relative ridge so it stays positive definite.
    Same interface as ContextBins.gaussian, so B2's inference applies."""

    def __init__(
        self, data: SnapshotData, h_hours: float, h_months: float, alpha: float
    ):
        idx = data.indices("train")
        self.p_kw = data.p_kw[idx]
        self.month = data.context["month"][idx]
        self.weekend = data.context["weekend"][idx]
        self.hour = data.context["quarter_hour"][idx] / 4.0
        self.h_hours, self.h_months, self.alpha = h_hours, h_months, alpha
        self._cache: dict = {}

    def weights(self, month: int, quarter_hour: int, weekend: int) -> np.ndarray:
        dh = np.abs(self.hour - quarter_hour / 4.0)
        dh = np.minimum(dh, 24.0 - dh)
        dm = np.abs(self.month - month)
        dm = np.minimum(dm, 12 - dm)
        w = np.exp(-0.5 * (dh / self.h_hours) ** 2 - 0.5 * (dm / self.h_months) ** 2)
        return np.where(self.weekend == weekend, w, 0.0)

    def gaussian(self, month: int, quarter_hour: int, weekend: int, full: bool = True):
        key = (month, quarter_hour, weekend)
        if key not in self._cache:
            w = self.weights(month, quarter_hour, weekend)
            keep = w > 1e-6 * w.max()
            x, w = self.p_kw[keep], w[keep] / w[keep].sum()
            mean = w @ x
            xc = x - mean
            cov = (xc * w[:, None]).T @ xc
            diag = np.diag(np.diag(cov))
            cov = (1 - self.alpha) * cov + self.alpha * diag
            cov = cov + 1e-6 * np.mean(np.diag(cov)) * np.eye(len(mean))
            n_eff = 1.0 / np.sum(w**2)
            self._cache[key] = (mean, cov, n_eff)
        return self._cache[key]

    def log_likelihood(self, p_kw: np.ndarray, ctx: dict[str, np.ndarray]) -> float:
        """Mean Gaussian log density (kW units) of p_kw under its context."""
        total = 0.0
        for i in range(len(p_kw)):
            mean, cov, _ = self.gaussian(
                int(ctx["month"][i]),
                int(ctx["quarter_hour"][i]),
                int(ctx["weekend"][i]),
            )
            _sign, logdet = np.linalg.slogdet(cov)
            r = p_kw[i] - mean
            total += -0.5 * (
                r @ np.linalg.solve(cov, r) + logdet + len(r) * np.log(2 * np.pi)
            )
        return total / len(p_kw)


def gauss_newton_map(
    measurement,
    y: Tensor,
    noise_cov: Tensor,
    prior_mean: Tensor,
    prior_cov: Tensor,
    num_iterations: int = 10,
) -> tuple[Tensor, Tensor]:
    """MAP of s under N(prior_mean, prior_cov) and y = h(s) + N(0, R), with
    the Laplace covariance at the MAP. Batched: y [S, m], mean [S, 32],
    cov [S, 32, 32]. Returns (map [S, 32], cov [S, 32, 32]) in kW."""
    r_inv = torch.linalg.inv(noise_cov.double())
    p_inv = torch.linalg.inv(prior_cov.double())
    s = prior_mean.double().clone()
    for _ in range(num_iterations):
        h_mat = measurement.jacobian(s)
        resid = y.double() - measurement(s)
        ht_rinv = h_mat.transpose(-1, -2) @ r_inv
        info = ht_rinv @ h_mat + p_inv
        grad = (ht_rinv @ resid.unsqueeze(-1)).squeeze(-1)
        grad -= (p_inv @ (s - prior_mean.double()).unsqueeze(-1)).squeeze(-1)
        s = s + torch.linalg.solve(info, grad.unsqueeze(-1)).squeeze(-1)
    h_mat = measurement.jacobian(s)
    info = h_mat.transpose(-1, -2) @ r_inv @ h_mat + p_inv
    cov = torch.linalg.inv(info)
    return s, 0.5 * (cov + cov.transpose(-1, -2))


def gaussian_samples(
    mean: Tensor, cov: Tensor, num_samples: int, generators: Sequence[torch.Generator]
) -> Tensor:
    """Samples [S, K, 32] from N(mean_s, cov_s), one generator per snapshot."""
    evals, evecs = torch.linalg.eigh(cov)
    root = evecs * evals.clamp(min=0).sqrt()[..., None, :]
    eps = torch.stack(
        [
            torch.randn(num_samples, mean.shape[-1], generator=g, dtype=torch.float64)
            for g in generators
        ]
    ).to(mean.device)
    return mean[:, None, :] + eps @ root.transpose(-1, -2)


def gaussian_baseline(
    bins: ContextBins,
    measurement,
    ctx: dict[str, np.ndarray],
    y: Tensor,
    noise_cov: Tensor,
    full: bool,
    num_samples: int,
    generators: Sequence[torch.Generator],
) -> tuple[Tensor, Tensor]:
    """B1 (full=False) or B2 (full=True): (MAP [S, 32], samples [S, K, 32])."""
    device = y.device
    means, covs = [], []
    for m, q, w in zip(ctx["month"], ctx["quarter_hour"], ctx["weekend"], strict=True):
        mean, cov, _ = bins.gaussian(int(m), int(q), int(w), full)
        means.append(mean)
        covs.append(cov)
    mean = torch.as_tensor(np.stack(means), dtype=torch.float64, device=device)
    cov = torch.as_tensor(np.stack(covs), dtype=torch.float64, device=device)
    s_map, post_cov = gauss_newton_map(measurement, y, noise_cov, mean, cov)
    return s_map, gaussian_samples(s_map, post_cov, num_samples, generators)


class MixtureBaseline:
    """B3: Gaussian mixture prior per context bin (in standardized units),
    exact Gaussian-mixture posterior under the bias-corrected LinDistFlow
    model y - c0 - mu_lin = H s + e, e ~ N(0, R + Sigma_lin)."""

    def __init__(
        self,
        bins: ContextBins,
        num_components: int,
        reg_covar: float = 1e-3,
        seed: int = 0,
    ):
        self.bins = bins
        self.k = num_components
        self.reg = reg_covar
        self.seed = seed
        self.scale = bins.p_kw.std(0)
        self._cache: dict = {}

    def fit(self, month: int, quarter_hour: int, weekend: int) -> GaussianMixture:
        key = (month, quarter_hour, weekend)
        if key not in self._cache:
            x = self.bins.members(month, quarter_hour, weekend) / self.scale
            gmm = GaussianMixture(
                n_components=min(self.k, len(x) // 2),
                covariance_type="full",
                reg_covar=self.reg,
                random_state=self.seed,
                n_init=2,
            ).fit(x)
            self._cache[key] = gmm
        return self._cache[key]

    def log_likelihood(self, p_kw: np.ndarray, ctx: dict[str, np.ndarray]) -> float:
        """Mean log density (standardized units) of p_kw under its bin model."""
        total = 0.0
        for i in range(len(p_kw)):
            gmm = self.fit(
                int(ctx["month"][i]),
                int(ctx["quarter_hour"][i]),
                int(ctx["weekend"][i]),
            )
            total += gmm.score(p_kw[i : i + 1] / self.scale)
        return total / len(p_kw)

    def posterior(
        self,
        ctx: dict[str, np.ndarray],
        y: np.ndarray,
        h_mat: np.ndarray,
        offset: np.ndarray,
        r_eff: np.ndarray,
        num_samples: int,
        generators: Sequence[np.random.Generator],
    ) -> tuple[np.ndarray, np.ndarray]:
        """(posterior mean [S, 32], samples [S, K, 32]) in kW."""
        d = np.diag(self.scale)
        means, samples = [], []
        for i in range(len(y)):
            gmm = self.fit(
                int(ctx["month"][i]),
                int(ctx["quarter_hour"][i]),
                int(ctx["weekend"][i]),
            )
            comp_mean, comp_cov, log_w = [], [], []
            for k in range(gmm.n_components):
                mu = d @ gmm.means_[k]
                cov = d @ gmm.covariances_[k] @ d
                s_mat = h_mat @ cov @ h_mat.T + r_eff
                resid = y[i] - offset - h_mat @ mu
                gain = cov @ h_mat.T @ np.linalg.inv(s_mat)
                comp_mean.append(mu + gain @ resid)
                post = cov - gain @ h_mat @ cov
                comp_cov.append(0.5 * (post + post.T))
                _, logdet = np.linalg.slogdet(s_mat)
                quad = resid @ np.linalg.solve(s_mat, resid)
                log_w.append(np.log(gmm.weights_[k]) - 0.5 * (logdet + quad))
            log_w = np.asarray(log_w)
            w = np.exp(log_w - log_w.max())
            w /= w.sum()
            rng = generators[i]
            labels = rng.choice(len(w), size=num_samples, p=w)
            draws = np.empty((num_samples, len(self.scale)))
            for k in range(len(w)):
                sel = labels == k
                if sel.any():
                    draws[sel] = rng.multivariate_normal(
                        comp_mean[k], comp_cov[k], size=int(sel.sum()), method="eigh"
                    )
            means.append(np.sum(w[:, None] * np.asarray(comp_mean), 0))
            samples.append(draws)
        return np.stack(means), np.stack(samples)


class MeasurementEmbedding(nn.Module):
    def __init__(self, num_channels: int, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_channels, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

    def forward(self, y_norm: Tensor) -> Tensor:
        return self.net(y_norm)


class RegressionMLP(nn.Module):
    """B5: (y, context) -> standardized injections, with dropout."""

    def __init__(
        self,
        num_channels: int,
        width: int = 512,
        depth: int = 4,
        dropout: float = 0.1,
        dim: int = 32,
    ):
        super().__init__()
        self.ctx = ContextEmbedding(128)
        self.y = MeasurementEmbedding(num_channels, 128)
        layers: list[nn.Module] = [
            nn.Linear(256, width),
            nn.SiLU(),
            nn.Dropout(dropout),
        ]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.SiLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(width, dim))
        self.body = nn.Sequential(*layers)

    def forward(self, y_norm: Tensor, ctx: dict[str, Tensor]) -> Tensor:
        t0 = torch.zeros(y_norm.shape[0], device=y_norm.device)
        h = torch.cat([self.ctx(t0, ctx), self.y(y_norm)], dim=-1)
        return self.body(h)


class ConditionalVelocityMLP(VelocityMLP):
    """B6: velocity network of p(z | c, y), y enters the AdaLN embedding."""

    def __init__(self, cfg: PriorConfig, num_channels: int, dim: int = 32):
        super().__init__(cfg, dim)
        self.y_embed = MeasurementEmbedding(num_channels, cfg.emb_dim)
        self.y_norm: Tensor | None = None

    def forward(self, x: Tensor, t: Tensor, ctx: dict[str, Tensor]) -> Tensor:
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        emb = self.emb_act(self.embed(t, ctx) + self.y_embed(ctx["y_norm"]))
        h = self.inp(x)
        for block in self.blocks:
            h = block(h, emb)
        return self.out(self.out_norm(h))


class SupervisedTrainer:
    """Shared training data for B5 and B6: clean h(s) on training snapshots,
    fresh measurement noise each batch, y standardized by its training
    mean and the noise-free spread plus noise."""

    def __init__(
        self,
        train_p_kw: np.ndarray,
        train_ctx: dict[str, np.ndarray],
        h_clean: np.ndarray,
        noise_std: np.ndarray,
        device: torch.device,
    ):
        self.device = device
        self.mean_kw = train_p_kw.mean(0)
        self.std_kw = train_p_kw.std(0)
        self.z = torch.as_tensor(
            (train_p_kw - self.mean_kw) / self.std_kw,
            dtype=torch.float32,
            device=device,
        )
        self.ctx = context_tensors(train_ctx, device)
        self.h = torch.as_tensor(h_clean, dtype=torch.float32, device=device)
        self.noise_std = torch.as_tensor(noise_std, dtype=torch.float32, device=device)
        self.y_mean = self.h.mean(0)
        self.y_std = torch.sqrt(self.h.var(0) + self.noise_std**2)

    def normalize_y(self, y: Tensor) -> Tensor:
        return (y.float() - self.y_mean) / self.y_std

    def batch(self, batch_size: int, gen: torch.Generator):
        idx = torch.randint(
            0, len(self.z), (batch_size,), device=self.device, generator=gen
        )
        noise = torch.randn(
            batch_size, self.h.shape[1], device=self.device, generator=gen
        )
        y = self.h[idx] + noise * self.noise_std
        ctx = {k: v[idx] for k, v in self.ctx.items()}
        return self.z[idx], ctx, self.normalize_y(y)


def _cosine_lr(step: int, total: int, base: float, warmup: int = 500) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    return (
        base
        * 0.5
        * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))
    )


def train_regression(
    trainer: SupervisedTrainer,
    num_steps: int,
    seed: int,
    batch_size: int = 1024,
    lr: float = 1e-3,
    log=print,
) -> RegressionMLP:
    torch.manual_seed(seed)
    model = RegressionMLP(trainer.h.shape[1], dim=trainer.z.shape[1]).to(trainer.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    gen = torch.Generator(device=trainer.device).manual_seed(seed)
    for step in range(num_steps):
        z, ctx, y = trainer.batch(batch_size, gen)
        loss = ((model(y, ctx) - z) ** 2).mean()
        for g in opt.param_groups:
            g["lr"] = _cosine_lr(step, num_steps, lr)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % max(1, num_steps // 10) == 0:
            log(f"B5 step {step + 1} loss {loss.item():.5f}")
    return model.eval()


@torch.no_grad()
def regression_predict(
    model: RegressionMLP,
    trainer: SupervisedTrainer,
    y: Tensor,
    ctx: dict[str, Tensor],
    num_samples: int,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """B5: (deterministic point [S, 32], MC-dropout samples [S, K, 32]) in kW."""
    mean = torch.as_tensor(trainer.mean_kw, device=y.device)
    std = torch.as_tensor(trainer.std_kw, device=y.device)
    y_norm = trainer.normalize_y(y)
    model.eval()
    point = mean + std * model(y_norm, ctx).double()
    model.train()  # dropout on
    torch.manual_seed(seed)
    draws = torch.stack([model(y_norm, ctx) for _ in range(num_samples)], dim=1)
    model.eval()
    return point, mean + std * draws.double()


def train_conditional_flow(
    trainer: SupervisedTrainer, cfg: PriorConfig, seed: int, log=print
) -> FlowPrior:
    """B6: conditional flow matching with y in the embedding (EMA weights)."""
    torch.manual_seed(seed)
    prior = FlowPrior(cfg, trainer.mean_kw, trainer.std_kw).to(trainer.device)
    prior.net = ConditionalVelocityMLP(cfg, trainer.h.shape[1], trainer.z.shape[1]).to(
        trainer.device
    )
    ema = copy.deepcopy(prior.net).eval()
    opt = torch.optim.AdamW(
        prior.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    gen = torch.Generator(device=trainer.device).manual_seed(seed)
    for step in range(cfg.num_steps):
        z, ctx, y = trainer.batch(cfg.batch_size, gen)
        ctx["y_norm"] = y
        t = torch.rand(len(z), device=z.device, generator=gen)
        eps = torch.randn(z.shape, device=z.device, generator=gen)
        x_t = t[:, None] * z + (1 - t[:, None]) * eps
        loss = ((prior.net(x_t, t, ctx) - (z - eps)) ** 2).mean()
        for g in opt.param_groups:
            g["lr"] = _cosine_lr(step, cfg.num_steps, cfg.lr, cfg.warmup_steps)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            decay = min(cfg.ema_decay, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), prior.net.parameters(), strict=True):
                pe.lerp_(pm, 1 - decay)
        if (step + 1) % max(1, cfg.num_steps // 10) == 0:
            log(f"B6 step {step + 1} loss {loss.item():.5f}")
    prior.net = ema
    return prior.eval()


@torch.no_grad()
def sample_conditional_flow(
    prior: FlowPrior,
    trainer: SupervisedTrainer,
    y: Tensor,
    ctx: dict[str, Tensor],
    num_samples: int,
    num_steps: int,
    generators: Sequence[torch.Generator],
    grid_power: float = DEFAULT_GRID_POWER,
) -> Tensor:
    """B6 samples [S, K, 32] in kW: Euler on p(z | c, y), final endpoint."""
    rows = {k: v.repeat_interleave(num_samples, 0) for k, v in ctx.items()}
    rows["y_norm"] = trainer.normalize_y(y).repeat_interleave(num_samples, 0)
    dim = trainer.z.shape[1]
    x = torch.cat([torch.randn(num_samples, dim, generator=g) for g in generators]).to(
        y.device
    )
    grid = time_grid(num_steps, grid_power)
    z_hat = x
    for i in range(num_steps):
        t = torch.tensor(grid[i], device=y.device)
        v = prior.net(x, t, rows)
        z_hat = x + (1 - t) * v
        x = x + (grid[i + 1] - grid[i]) * v
    return prior.to_kw(z_hat.double()).view(len(y), num_samples, dim)

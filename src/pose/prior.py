"""Conditional flow matching prior p(s | c) over the 32 node injections.

Linear (CondOT) path in standardized space z = (s - mu) / sigma with
per-node training mean and standard deviation: x_t = t z + (1 - t) eps,
eps ~ N(0, I), t in [0, 1] with t = 1 at the data. The network predicts the
velocity u = z - eps; the endpoint estimate is z_hat = x_t + (1 - t) v.

The velocity network is a residual MLP with adaptive layer norm on the
embedding of (t, month, quarter-hour, day type). With ``context="cyclic"``
the month and quarter-hour enter only through low-order Fourier features, so
the network cannot keep a separate lookup entry per calendar cell.
"""

import copy
import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import Tensor, nn

from .feeder import NUM_NODES


@dataclass
class PriorConfig:
    width: int = 512
    num_blocks: int = 4
    emb_dim: int = 128
    dropout: float = 0.0
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 1024
    num_steps: int = 100_000
    warmup_steps: int = 1_000
    ema_decay: float = 0.999
    val_every: int = 2_000
    val_noise_draws: int = 4
    context: str = "full"  # "full" (lookup tables + cyclic) or "cyclic"


QH_HARMONICS = 4  # Fourier harmonics of the daily cycle (context="cyclic")
MONTH_HARMONICS = 2  # Fourier harmonics of the yearly cycle


def sinusoidal(t: Tensor, dim: int, max_period: float = 10_000.0) -> Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device) / half
    )
    args = 1000.0 * t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def fourier_features(angle: Tensor, harmonics: int) -> Tensor:
    k = torch.arange(1, harmonics + 1, device=angle.device, dtype=torch.float32)
    args = angle.float()[:, None] * k[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ContextEmbedding(nn.Module):
    """Embedding of (t, month, quarter-hour, weekend).

    context="full": learned lookup tables for month and quarter-hour plus one
    harmonic of each cycle. context="cyclic": an MLP on QH_HARMONICS and
    MONTH_HARMONICS Fourier harmonics, with no per-cell lookup table.
    """

    def __init__(self, dim: int, context: str = "full"):
        super().__init__()
        if context not in ("full", "cyclic"):
            raise ValueError(f"unknown context {context}")
        self.dim = dim
        self.context = context
        self.time = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        if context == "full":
            self.month = nn.Embedding(12, dim)
            self.quarter_hour = nn.Embedding(96, dim)
        self.weekend = nn.Embedding(2, dim)
        if context == "full":
            self.cyclic = nn.Linear(4, dim)
        else:
            n_feat = 2 * (QH_HARMONICS + MONTH_HARMONICS)
            self.cyclic = nn.Sequential(
                nn.Linear(n_feat, dim), nn.SiLU(), nn.Linear(dim, dim)
            )

    def forward(self, t: Tensor, ctx: dict[str, Tensor]) -> Tensor:
        month, qh = ctx["month"].long(), ctx["quarter_hour"].long()
        angle_q = 2 * math.pi * qh.float() / 96.0
        angle_m = 2 * math.pi * month.float() / 12.0
        if self.context == "cyclic":
            cyc = torch.cat(
                [
                    fourier_features(angle_q, QH_HARMONICS),
                    fourier_features(angle_m, MONTH_HARMONICS),
                ],
                dim=-1,
            )
            return (
                self.time(sinusoidal(t, self.dim))
                + self.weekend(ctx["weekend"].long())
                + self.cyclic(cyc)
            )
        cyc = torch.stack(
            [angle_q.cos(), angle_q.sin(), angle_m.cos(), angle_m.sin()], dim=-1
        )
        return (
            self.time(sinusoidal(t, self.dim))
            + self.month(month)
            + self.quarter_hour(qh)
            + self.weekend(ctx["weekend"].long())
            + self.cyclic(cyc)
        )


class AdaLNBlock(nn.Module):
    def __init__(self, width: int, emb_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.mod = nn.Linear(emb_dim, 3 * width)
        self.fc1 = nn.Linear(width, 2 * width)
        self.fc2 = nn.Linear(2 * width, width)
        self.drop = nn.Dropout(dropout)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, h: Tensor, emb: Tensor) -> Tensor:
        shift, scale, gain = self.mod(emb).chunk(3, dim=-1)
        x = self.norm(h) * (1 + scale) + shift
        x = self.fc2(self.drop(nn.functional.silu(self.fc1(x))))
        return h + gain.add(1.0) * x


class VelocityMLP(nn.Module):
    def __init__(self, cfg: PriorConfig, dim: int = NUM_NODES):
        super().__init__()
        self.embed = ContextEmbedding(cfg.emb_dim, cfg.context)
        self.emb_act = nn.SiLU()
        self.inp = nn.Linear(dim, cfg.width)
        self.blocks = nn.ModuleList(
            [
                AdaLNBlock(cfg.width, cfg.emb_dim, cfg.dropout)
                for _ in range(cfg.num_blocks)
            ]
        )
        self.out_norm = nn.LayerNorm(cfg.width)
        self.out = nn.Linear(cfg.width, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: Tensor, t: Tensor, ctx: dict[str, Tensor]) -> Tensor:
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        emb = self.emb_act(self.embed(t, ctx))
        h = self.inp(x)
        for block in self.blocks:
            h = block(h, emb)
        return self.out(self.out_norm(h))


class FlowPrior(nn.Module):
    """Velocity network plus the standardization of the injections."""

    def __init__(self, cfg: PriorConfig, mean_kw: np.ndarray, std_kw: np.ndarray):
        super().__init__()
        self.cfg = cfg
        self.net = VelocityMLP(cfg, len(mean_kw))
        self.register_buffer("mean_kw", torch.as_tensor(mean_kw, dtype=torch.float32))
        self.register_buffer("std_kw", torch.as_tensor(std_kw, dtype=torch.float32))

    def velocity(self, x: Tensor, t: Tensor, ctx: dict[str, Tensor]) -> Tensor:
        return self.net(x, t, ctx)

    def endpoint(self, x: Tensor, t: Tensor, ctx: dict[str, Tensor]) -> Tensor:
        """z_hat_1 = E[z | x_t] as predicted by the network."""
        return x + (1.0 - t) * self.net(x, t, ctx)

    def to_kw(self, z: Tensor) -> Tensor:
        return self.mean_kw.to(z.dtype) + self.std_kw.to(z.dtype) * z

    def to_z(self, p_kw: Tensor) -> Tensor:
        return (p_kw - self.mean_kw.to(p_kw.dtype)) / self.std_kw.to(p_kw.dtype)


def context_tensors(ctx: dict[str, np.ndarray], device) -> dict[str, Tensor]:
    return {
        k: torch.as_tensor(np.asarray(ctx[k]), dtype=torch.long, device=device)
        for k in ("month", "quarter_hour", "weekend")
    }


def flow_loss(model: nn.Module, z: Tensor, ctx: dict[str, Tensor], gen=None):
    t = torch.rand(z.shape[0], device=z.device, generator=gen)
    eps = torch.randn(z.shape, device=z.device, generator=gen)
    x_t = t[:, None] * z + (1 - t[:, None]) * eps
    v = model(x_t, t, ctx)
    return ((v - (z - eps)) ** 2).mean()


def train_prior(
    cfg: PriorConfig,
    train_p_kw: np.ndarray,
    train_ctx: dict[str, np.ndarray],
    val_p_kw: np.ndarray,
    val_ctx: dict[str, np.ndarray],
    seed: int,
    device: torch.device,
    log=print,
    checkpoint: str = "best",
) -> tuple[FlowPrior, dict]:
    """Train with AdamW, warmup + cosine schedule and EMA; return the EMA
    model at the step with the lowest validation loss (checkpoint="best") or
    at the last step (checkpoint="last"), and a history."""
    if checkpoint not in ("best", "last"):
        raise ValueError(f"unknown checkpoint {checkpoint}")
    torch.manual_seed(seed)
    mean, std = train_p_kw.mean(0), train_p_kw.std(0)
    prior = FlowPrior(cfg, mean, std).to(device)
    ema = copy.deepcopy(prior.net).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    z_train = prior.to_z(
        torch.as_tensor(train_p_kw, dtype=torch.float32, device=device)
    )
    z_val = prior.to_z(torch.as_tensor(val_p_kw, dtype=torch.float32, device=device))
    c_train = context_tensors(train_ctx, device)
    c_val = context_tensors(val_ctx, device)
    opt = torch.optim.AdamW(
        prior.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    def lr_at(step: int) -> float:
        if step < cfg.warmup_steps:
            return cfg.lr * (step + 1) / cfg.warmup_steps
        frac = (step - cfg.warmup_steps) / max(1, cfg.num_steps - cfg.warmup_steps)
        return cfg.lr * 0.5 * (1 + math.cos(math.pi * min(frac, 1.0)))

    gen = torch.Generator(device=device).manual_seed(seed)

    @torch.no_grad()
    def val_loss(net) -> float:
        g = torch.Generator(device=device).manual_seed(12345)
        losses = []
        for _ in range(cfg.val_noise_draws):
            for start in range(0, len(z_val), 8192):
                sl = slice(start, start + 8192)
                ctx = {k: v[sl] for k, v in c_val.items()}
                losses.append(flow_loss(net, z_val[sl], ctx, g).item() * len(z_val[sl]))
        return sum(losses) / (cfg.val_noise_draws * len(z_val))

    history = {"step": [], "train_loss": [], "val_loss": [], "val_loss_ema": []}
    best = (math.inf, -1, None)
    n = len(z_train)
    t0 = time.time()
    running, since = 0.0, 0
    for step in range(cfg.num_steps):
        prior.net.train()
        idx = torch.randint(0, n, (cfg.batch_size,), device=device, generator=gen)
        ctx = {k: v[idx] for k, v in c_train.items()}
        loss = flow_loss(prior.net, z_train[idx], ctx, gen)
        for group in opt.param_groups:
            group["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        running += loss.item()
        since += 1
        with torch.no_grad():
            decay = min(cfg.ema_decay, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), prior.net.parameters(), strict=True):
                pe.lerp_(pm, 1 - decay)
        if (step + 1) % cfg.val_every == 0 or step + 1 == cfg.num_steps:
            prior.net.eval()
            vl, vle = val_loss(prior.net), val_loss(ema)
            history["step"].append(step + 1)
            history["train_loss"].append(running / since)
            history["val_loss"].append(vl)
            history["val_loss_ema"].append(vle)
            running, since = 0.0, 0
            if vle < best[0]:
                best = (vle, step + 1, copy.deepcopy(ema.state_dict()))
            log(
                f"step {step + 1} train {history['train_loss'][-1]:.5f} val {vl:.5f} "
                f"val_ema {vle:.5f} lr {lr_at(step):.2e} {time.time() - t0:.0f}s"
            )
    if checkpoint == "last":
        best = (history["val_loss_ema"][-1], history["step"][-1], ema.state_dict())
    prior.net.load_state_dict(best[2])
    prior.eval()
    history.update(
        best_step=best[1], best_val_loss_ema=best[0], wall_time_s=time.time() - t0
    )
    return prior, history


@torch.no_grad()
def sample_prior(prior, ctx: dict[str, Tensor], x0: Tensor, num_steps: int) -> Tensor:
    """Unguided Euler samples of the prior from initial noise x0 [B, d] on
    the default time grid; ctx holds one row per sample. Returns kW."""
    from .guidance import time_grid

    x = x0
    grid = time_grid(num_steps)
    for i in range(num_steps):
        t = torch.tensor(grid[i], device=x.device)
        x = x + (grid[i + 1] - grid[i]) * prior.velocity(x, t, ctx)
    return prior.to_kw(x.double())


def select_run(rows: list[dict], tie: float) -> dict:
    """Sweep selection: the lowest ``val_loss``; rows within ``tie`` of it are
    tied, and the one with the fewest ``num_parameter``, then the fewest
    ``num_steps``, wins."""
    best = min(r["val_loss"] for r in rows)
    tied = [r for r in rows if r["val_loss"] <= best + tie]
    return min(tied, key=lambda r: (r["num_parameter"], r["num_steps"], r["val_loss"]))


def save_prior(path: str, prior: FlowPrior, extra: dict) -> None:
    """Write the weights, architecture and standardization in a portable dict."""
    torch.save(
        {
            "state_dict": prior.state_dict(),
            "config": asdict(prior.cfg),
            "standardization": {
                "mean_kw": prior.mean_kw.cpu().numpy(),
                "std_kw": prior.std_kw.cpu().numpy(),
            },
            **extra,
        },
        path,
    )


def load_prior(path: str, device: torch.device) -> tuple[FlowPrior, dict]:
    """Load a released checkpoint or a checkpoint from save_prior."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    std = ckpt.get("standardization", ckpt)
    prior = FlowPrior(PriorConfig(**ckpt["config"]), std["mean_kw"], std["std_kw"])
    prior.load_state_dict(ckpt["state_dict"])
    prior.to(device).eval()
    return prior, ckpt

"""Example estimates over one test week (high PV, S2, nominal noise).

Input: fig_week_data.npz from the week run (scripts/figures/pack_week.py).
Samples are joint (P, Q) injections in kW/kvar (state index i -> bus i + 2).
Voltages and branch loadings of every sample and of the truth come from the
AC power flow in pose.powerflow. Bus, branch and snapshot are chosen on the
truth only.

Usage: python3 fig_week.py fig_week_data.npz fig_week.pdf
"""

import datetime as dt
import json
import sys

import matplotlib
import numpy as np
import torch


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, MultipleLocator
from scipy.stats import gaussian_kde

from pose.powerflow import RadialPowerFlow, branch_ratings_a


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.4,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5,
        "axes.edgecolor": "#52514e",
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "axes.labelcolor": "#0b0b0b",
        "lines.solid_capstyle": "butt",
    }
)

_PF = RadialPowerFlow()
_RATINGS = branch_ratings_a(_PF)


def power_flow(x):
    """Voltage magnitudes [..., 33] and branch loading [..., 32] in percent."""
    shape = x.shape[:-1]
    flat = torch.as_tensor(x, dtype=torch.float64).reshape(-1, x.shape[-1])
    with torch.no_grad():
        res = _PF(flat)
    vm = torch.cat(
        [torch.ones(len(flat), 1, dtype=res.vm.dtype), res.vm], dim=1
    ).numpy()
    loading = res.branch_current_a.numpy() / _RATINGS * 100.0
    return vm.reshape(*shape, 33), loading.reshape(*shape, 32)


BLUE, BLUE50, BLUE90 = "#2a78d6", "#8db8ea", "#d3e4f7"
INK, AX, GRAY, RULE, BAND = "#0b0b0b", "#52514e", "#8a8883", "#e4e3df", "#f6f5f2"
VIOL = "#f3ddd3"
WLSC = "#7b5ea7"  # WLS, as on the project page
D0 = dt.date(2010, 7, 1)
V_MEASURED = (18, 22, 33)
VLO, VHI = 0.95, 1.05


def when(snap):
    d = D0 + dt.timedelta(days=int(snap) // 48)
    return d, (int(snap) % 48) / 2.0


def main(src, out, mock=False):
    z = np.load(src)
    week, midday = z["week_snapshots"], z["midday_snapshots"]
    truth = z["truth"]
    t_h = (week - week[0]) / 2.0  # hours since the first half-hour of the week
    vt, it = power_flow(truth)  # [336, 33], [336, 32]
    pf = {k: power_flow(z[f"{k}_S2_samples"]) for k in ("pose", "wls")}
    vs = {k: v[0] for k, v in pf.items()}  # [336, 100, 33]
    cs = {k: v[1] for k, v in pf.items()}  # [336, 100, 32]

    # Rule 2: unmeasured bus with the most true voltage violations (ties: larger deviation).
    viol_t = (vt < VLO) | (vt > VHI)
    cand = [b for b in range(2, 34) if b not in V_MEASURED]
    key = lambda b: (viol_t[:, b - 1].sum(), np.abs(vt[:, b - 1] - 1).max())
    bus = max(cand, key=key)
    bi = bus - 1
    # Rule 4: branch with the most true overloads (ties: larger peak loading).
    over_t = it > 100.0
    br = max(range(32), key=lambda k: (over_t[:, k].sum(), it[:, k].max()))
    branch = br + 1  # branch k feeds bus k + 1
    # Rule 3: midday half-hour (10:00-14:00) with the largest true |V - 1| at that bus.
    rows_mid = np.searchsorted(week, midday)
    r_star = rows_mid[np.argmax(np.abs(vt[rows_mid, bi] - 1))]
    snap_star = week[r_star]
    m_star = int(np.flatnonzero(midday == snap_star)[0])
    info = {
        "bus": bus,
        "violations_at_bus": int(viol_t[:, bi].sum()),
        "branch": branch,
        "overloads_at_branch": int(over_t[:, br].sum()),
        "snapshot": int(snap_star),
        "date": str(when(snap_star)[0]),
        "hour": when(snap_star)[1],
        "week_start": str(when(week[0])[0]),
    }

    W, H = 18.2 / 2.54, 5.3 / 2.54
    fig = plt.figure(figsize=(W, H))
    L, R0, R1 = 0.074, 0.735, 0.992
    gs = fig.add_gridspec(
        3,
        1,
        left=L,
        right=0.672,
        bottom=0.165,
        top=0.865,
        height_ratios=[1, 1, 1],
        hspace=0.12,
    )
    axs = [fig.add_subplot(gs[i]) for i in range(3)]
    for a in axs[1:]:
        a.sharex(axs[0])

    def bands(ax, S, color_med=BLUE):
        q = np.percentile(S, [5, 25, 50, 75, 95], axis=1)
        ax.fill_between(t_h, q[0], q[4], color=BLUE90, lw=0, step=None)
        ax.fill_between(t_h, q[1], q[3], color=BLUE50, lw=0)
        ax.plot(t_h, q[2], color=color_med, lw=0.7)
        return q

    def wls_band(ax, S):
        q = np.percentile(S, [5, 50, 95], axis=1)
        for qq in (q[0], q[2]):
            ax.plot(t_h, qq, color=WLSC, lw=0.5, ls=(0, (2.2, 1.2)))
        ax.plot(t_h, q[1], color=WLSC, lw=0.6)

    # (a) net active injection at bus 13
    j13 = 13 - 2
    pa = z["pose_S2_samples"][:, :, j13]
    wa = z["wls_S2_samples"][:, :, j13]
    bands(axs[0], pa)
    wls_band(axs[0], wa)
    axs[0].plot(t_h, truth[:, j13], color=INK, lw=0.6)
    axs[0].axhline(0, color=RULE, lw=0.5, zorder=0)
    axs[0].set_ylabel("$p_{13}$ (kW)")

    # (b) voltage magnitude at the selected bus
    pb = vs["pose"][:, :, bi]
    wb = vs["wls"][:, :, bi]
    bands(axs[1], pb)
    wls_band(axs[1], wb)
    axs[1].plot(t_h, vt[:, bi], color=INK, lw=0.6)
    for lim in (VLO, VHI):
        lo, hi = (
            np.nanmin([pb.min(), vt[:, bi].min()]),
            np.nanmax([pb.max(), vt[:, bi].max()]),
        )
        if lo - 0.01 < lim < hi + 0.01:
            axs[1].axhline(lim, color="#b5502a", lw=0.6, ls=(0, (4, 2)))
    axs[1].set_ylabel(f"$|V_{{{bus}}}|$ (p.u.)")

    # (c) loading of the selected branch
    pc = cs["pose"][:, :, br]
    wc = cs["wls"][:, :, br]
    bands(axs[2], pc)
    wls_band(axs[2], wc)
    axs[2].plot(t_h, it[:, br], color=INK, lw=0.6)
    axs[2].axhline(100, color="#b5502a", lw=0.6, ls=(0, (4, 2)))

    def shade(ax, mask):
        start = None
        for k in range(len(mask) + 1):
            on = k < len(mask) and mask[k]
            if on and start is None:
                start = k
            if not on and start is not None:
                ax.axvspan(
                    t_h[start] - 0.25, t_h[k - 1] + 0.25, color=VIOL, lw=0, zorder=0
                )
                start = None

    shade(axs[1], viol_t[:, bi])
    shade(axs[2], over_t[:, br])
    brier = {}
    for k in ("pose", "wls"):
        pv = ((vs[k][:, :, bi] < VLO) | (vs[k][:, :, bi] > VHI)).mean(axis=1)
        pi = (cs[k][:, :, br] > 100.0).mean(axis=1)
        brier[k] = (
            ((pv - viol_t[:, bi]) ** 2).mean(),
            ((pi - over_t[:, br]) ** 2).mean(),
        )
    info["brier_v_bus"] = {k: float(v[0]) for k, v in brier.items()}
    info["brier_overload_branch"] = {k: float(v[1]) for k, v in brier.items()}
    axs[2].set_ylabel(f"$I_{{{branch}}}$ (\\%)")

    # time axis: days
    days = [when(week[0] + 48 * d)[0] for d in range(7)]
    axs[2].xaxis.set_major_locator(FixedLocator(np.arange(0, 7 * 24 + 1, 24)))
    axs[2].xaxis.set_minor_locator(MultipleLocator(6))
    axs[2].set_xticks(np.arange(12, 7 * 24, 24), minor=False)
    axs[2].set_xticklabels([d.strftime("%a %-d %b") for d in days])
    axs[2].set_xticks(np.arange(0, 7 * 24 + 1, 24), minor=True)
    axs[2].tick_params(axis="x", which="major", length=0)
    axs[2].tick_params(axis="x", which="minor", length=3)
    axs[2].set_xlim(t_h[0] - 0.25, t_h[-1] + 0.25)
    for ax in axs:
        for d in range(1, 7):
            ax.axvline(24 * d, color=RULE, lw=0.4, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.set_label_coords(-0.085, 0.5)
    for ax in axs[:2]:
        ax.tick_params(axis="x", which="both", labelbottom=False, length=0)
    # marker of the right-panel snapshot
    for ax in axs:
        ax.axvline(t_h[r_star], color=INK, lw=0.5, ls=(0, (1, 1.2)), zorder=3)
    axs[0].annotate(
        "(d)",
        (t_h[r_star], 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(0, 1.5),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=7,
    )
    for ax, lab in zip(axs, ("(a)", "(b)", "(c)"), strict=False):
        ax.text(
            0.004,
            0.97,
            lab,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"fc": "white", "ec": "none", "pad": 0.6, "alpha": 0.85},
        )

    # legend for the left block
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    hand = [
        Line2D([], [], color=INK, lw=0.8),
        Line2D([], [], color=BLUE, lw=0.8),
        Patch(fc=BLUE50, ec="none"),
        Patch(fc=BLUE90, ec="none"),
        Line2D([], [], color=WLSC, lw=0.7),
        Line2D([], [], color=WLSC, lw=0.6, ls=(0, (2.2, 1.2))),
        Line2D([], [], color="#b5502a", lw=0.6, ls=(0, (4, 2))),
        Patch(fc=VIOL, ec="none"),
    ]
    labs = [
        "Truth",
        "PoSE median",
        "PoSE 50\\%",
        "PoSE 90\\%",
        "WLS median",
        "WLS 90\\%",
        "Limit",
        "True violation",
    ]
    fig.legend(
        hand,
        labs,
        loc="upper left",
        bbox_to_anchor=(L - 0.012, 1.005),
        ncol=8,
        frameon=False,
        handlelength=1.6,
        columnspacing=0.9,
        handletextpad=0.5,
        borderaxespad=0.1,
    )

    # (d) posterior tightening at the marked snapshot
    axr = fig.add_axes([R0, 0.165, R1 - R0, 0.70])
    rows = [
        ("Prior", z["prior_midday_samples"][m_star, :, j13], GRAY),
        ("S1", z["pose_S1_samples"][m_star, :, j13], BLUE),
        ("S2", z["pose_S2_samples"][r_star, :, j13], BLUE),
        ("S3", z["pose_S3_samples"][m_star, :, j13], BLUE),
    ]
    allv = np.concatenate([r[1] for r in rows] + [z["wls_S2_samples"][r_star, :, j13]])
    lo, hi = np.percentile(allv, [0.2, 99.8])
    pad = 0.12 * (hi - lo)
    xx = np.linspace(lo - pad, hi + pad, 500)
    dens = [gaussian_kde(r[1])(xx) for r in rows]
    scale = 0.95 / max(d.max() for d in dens[1:])
    step = 1.0
    nrow = len(rows)
    for i, ((name, smp, col), d) in enumerate(zip(rows, dens, strict=False)):
        base = (nrow - 1 - i) * step
        dd = np.minimum(d * scale, 1.25)
        axr.fill_between(
            xx,
            base,
            base + dd,
            color=col,
            alpha=0.22 if col == BLUE else 0.3,
            lw=0,
            zorder=2 + i,
        )
        axr.plot(xx, base + dd, color=col, lw=0.8, zorder=2 + i)
        axr.plot(xx, np.full_like(xx, base), color=RULE, lw=0.5, zorder=1)
        q5, q95 = np.percentile(smp, [5, 95])
        axr.plot(
            [q5, q95],
            [base + 0.06] * 2,
            color=col,
            lw=1.6,
            solid_capstyle="butt",
            zorder=10,
        )
        axr.text(
            xx[-1],
            base + 0.5,
            f"{q95 - q5:.0f} kW",
            ha="right",
            va="bottom",
            fontsize=7,
            color=AX,
        )
        if name == "S2":
            w = z["wls_S2_samples"][r_star, :, j13]
            dw = np.minimum(gaussian_kde(w)(xx) * scale, 1.25)
            axr.fill_between(
                xx, base, base + dw, color=WLSC, alpha=0.10, lw=0, zorder=19
            )
            axr.plot(xx, base + dw, color=WLSC, lw=0.7, ls=(0, (2.2, 1.2)), zorder=20)
            w5, w95 = np.percentile(w, [5, 95])
            axr.text(
                xx[-1],
                base + 0.2,
                f"WLS {w95 - w5:.0f} kW",
                ha="right",
                va="bottom",
                fontsize=7,
                color=WLSC,
            )
    axr.axvline(truth[r_star, j13], color=INK, lw=0.7, zorder=30)
    axr.set_xlim(xx[0], xx[-1])
    axr.set_ylim(-0.08, (nrow - 1) * step + 1.15)
    axr.set_yticks([(nrow - 1 - i) * step + 0.3 for i in range(nrow)])
    axr.set_yticklabels([r[0] for r in rows])
    axr.tick_params(axis="y", length=0, pad=2)
    for s in ("left", "right", "top"):
        axr.spines[s].set_visible(False)
    d_, h_ = when(snap_star)
    axr.set_xlabel("$p_{13}$ (kW)", labelpad=1.5)
    axr.text(
        0.5,
        1.0,
        f"{d_.strftime('%a %-d %b')}, {int(h_):02d}:{int(60 * (h_ % 1)):02d}",
        transform=axr.transAxes,
        va="bottom",
        ha="center",
        fontsize=7.5,
    )
    axr.text(
        -0.13, 1.0, "(d)", transform=axr.transAxes, va="bottom", ha="left", fontsize=8
    )
    axr.text(
        1.0,
        1.0,
        "90\\% width",
        transform=axr.transAxes,
        va="bottom",
        ha="right",
        fontsize=7,
        color=AX,
    )
    if mock:
        fig.text(
            0.5,
            0.5,
            "MOCK DATA, LAYOUT ONLY",
            ha="center",
            va="center",
            fontsize=22,
            color="red",
            alpha=0.25,
            rotation=12,
        )
    fig.savefig(out)
    with open(out.replace(".pdf", "_info.json"), "w") as fh:
        json.dump(info, fh, indent=1)
    print(info)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], "--mock" in sys.argv)

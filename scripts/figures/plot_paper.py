"""Paper figures from the result summaries.

fig_crps_coverage: (a) injection CRPS of each estimator relative to WLS (B1,
    %), the reference of the main table, and (b) 90% interval coverage against
    the attainable value; one dot per measurement set and noise level, base
    and high PV. Both panels show the alternative estimators and PoSE only;
    the guidance variants that share the learned prior (projection M0,
    isotropic M1, DPS B4) form the ablation and are left out. Methods carry
    their paper names (NAMES).
fig_data_scaling: data-scaling study, M2 gain over B2 and coverage against the
    number of training snapshots (S2, nominal noise).
fig_feeder: IEEE 33-bus feeder with the PV-heavy buses 13-18 and the
    measurement locations of S1-S3.
fig_posterior: (a) posteriors of the bus-13 injection (largest PV capacity)
    for the midday test snapshots at the 10th, 40th, 60th and 90th
    percentiles of its true injection (chosen on the truth only), PoSE
    (M2-Euler) against the correlated Gaussian (B2); (b) sample skewness over
    all midday node-snapshots at buses 13-18.

Figures are drawn at their printed IEEEtran width (one column, or the full text
width for fig_posterior) in Times, so font sizes are the printed sizes.
Writes PDF and PNG to --output_dir, and figure_values.csv with every plotted
value, read back from the drawn artists. Colours: categorical slots 1-2 of the
reference palette (blue, orange), validated for CVD separation; markers differ
in shape as well."""

import argparse
import os

import matplotlib
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt


BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SCEN = {"ausgrid": ("Base PV", BLUE, "o"), "ausgrid_hpv": ("High PV", ORANGE, "s")}
NAMES = {
    "B1": "WLS",
    "B2": "Correlated Gaussian",
    "B7": "Pooled Gaussian",
    "B3": "Gaussian mixture",
    "B5": "Supervised MLP",
    "B6": "Supervised flow",
    "B4": "DPS",
    "M0-euler": "Projection",
    "M1-euler": "Isotropic",
    "M2-euler": "PoSE",
}
# Alternative estimators and PoSE, in the order of the earlier figures.
ROWS = [(m, NAMES[m]) for m in ("M2-euler", "B2", "B7", "B3", "B6", "B5", "B1")]
VALUES: list[dict] = []  # every plotted value, for figure_values.csv
# Printed widths of IEEEtran (journal, 10 pt): column 21 pc, text 43 pc. Figures
# are drawn at these widths so the font sizes below are the printed sizes.
COLUMN, TEXT = 252 / 72.27, 516 / 72.27  # inches
ATTAINABLE = 0.9 * 99 / 101  # coverage of exact samples with K = 100


def style():
    # Times, as the IEEEtran body text, also for mathtext. Fallbacks for hosts
    # without Times New Roman.
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Nimbus Roman",
                "TeX Gyre Termes",
                "Times",
                "STIXGeneral",
            ],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "mathtext.cal": "Times New Roman:italic",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.major.pad": 2,
            "ytick.major.pad": 2,
            "axes.labelpad": 2,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK,
            "ytick.labelcolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.handletextpad": 0.3,
            "legend.borderaxespad": 0.2,
            "figure.constrained_layout.h_pad": 0.02,
            "figure.constrained_layout.w_pad": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig, out, name):
    # Fixed page size (no tight bounding box), so the PDF prints at 100%.
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=300)
    plt.close(fig)


def dot_rows(ax, table, value, rows, figure, offset=0.17):
    """One row per estimator, one marker per setting; values tagged with figure."""
    for k, (method, _) in enumerate(rows):
        for j, (sc, (label, color, marker)) in enumerate(SCEN.items()):
            sel = table[(table.method == method) & (table.scenario == sc)]
            if sel.empty:
                continue
            y = len(rows) - 1 - k + (offset if j == 0 else -offset)
            pts = ax.scatter(
                sel[value],
                [y] * len(sel),
                s=13,
                color=color,
                marker=marker,
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
                label=label if k == 0 else None,
            )
            for (mset, noise), xy in zip(
                zip(sel.mset, sel.noise, strict=True), pts.get_offsets(), strict=True
            ):
                VALUES.append(
                    {
                        "figure": figure,
                        "method": method,
                        "scenario": sc,
                        "key": f"{mset}_{noise}",
                        "value": float(xy[0]),
                    }
                )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([lab for _, lab in rows][::-1])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    for k in range(len(rows)):
        if k % 2 == 0:  # light banding guides the eye along the rows
            ax.axhspan(k - 0.5, k + 0.5, color="#f6f5f2", zorder=0, linewidth=0)
    ax.grid(axis="x", color=GRID, linewidth=0.5, zorder=0.5)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


def accuracy_calibration_figure(out, table):
    """fig_crps_coverage: (a) CRPS relative to WLS, (b) 90% coverage.

    The panels share the estimator rows; their plotted values keep the figure
    tags fig_crps_vs_wls and fig_coverage of the checker.
    """
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(COLUMN, 1.72),
        sharey=True,
        layout="constrained",
        gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.04},
    )
    rows = ROWS
    ax = axes[0]
    dot_rows(ax, table[table.method != "B1"], "rel", rows, "fig_crps_vs_wls")
    ax.axvline(0, color=MUTED, linewidth=0.8, zorder=1)
    ax.text(-1.0, 0, "reference", color=MUTED, fontsize=7, ha="right", va="center")
    ax.set_xlim(-40, 3)
    ax.set_xticks([-40, -30, -20, -10, 0])
    ax.set_xlabel("(a) CRPS vs WLS (%)")
    ax = axes[1]
    dot_rows(ax, table, "coverage90", rows, "fig_coverage")
    ax.axvline(ATTAINABLE, color=MUTED, linewidth=0.8, zorder=1)
    ax.text(
        ATTAINABLE - 0.015,
        0,
        "attainable",
        color=MUTED,
        fontsize=7,
        ha="right",
        va="center",
    )
    ax.set_xlim(0.3, 1.0)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("(b) 90% interval coverage")
    axes[1].tick_params(axis="y", labelleft=False)
    axes[1].legend(  # empty upper left of (b)
        *axes[0].get_legend_handles_labels(),
        loc="upper left",
        handlelength=1.0,
        labelspacing=0.15,
        borderaxespad=0.3,
    )
    save(fig, out, "fig_crps_coverage")


def feeder_figure(out):
    from matplotlib.lines import Line2D

    from pose.feeder import FROM_BUS

    pos = {b: (b - 1, 0.0) for b in range(1, 19)}
    pos.update({19: (1, 1.0), 20: (2, 1.0), 21: (3, 1.0), 22: (4, 1.0)})
    pos.update({23: (2, -1.0), 24: (3, -1.0), 25: (4, -1.0)})
    pos.update({b: (5 + (b - 26), -1.0) for b in range(26, 34)})
    fig, ax = plt.subplots(figsize=(COLUMN, 0.95))
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    for k, frm in enumerate(FROM_BUS):
        (x0, y0), (x1, y1) = pos[frm], pos[k + 2]
        ax.plot(
            [x0, x0, x1],
            [y0, y1, y1],
            color="#8a8883",
            linewidth=0.9,
            zorder=1,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
    for b, (x, y) in pos.items():
        heavy = 13 <= b <= 18
        ax.scatter(
            x,
            y,
            s=16 if heavy else 7,
            color=ORANGE if heavy else "#7d7b76",
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
        # Labels above the trunk and upper lateral, below the lower laterals;
        # nudged right where a lateral leaves the node.
        shift = 0.2 if b in (2, 3, 6) else 0.0
        ax.text(
            x + shift,
            y + (0.2 if y >= 0 else -0.22),
            str(b),
            fontsize=6.5,
            color=MUTED,
            ha="left" if shift else "center",
            va="bottom" if y >= 0 else "top",
        )
    meas = {"^": ((1,), 38), "D": ((18, 22, 33), 26), "s": ((16, 30), 40)}
    for marker, (buses, size) in meas.items():
        for b in buses:
            ax.scatter(
                *pos[b],
                s=size,
                marker=marker,
                facecolor="none",
                edgecolor=INK,
                linewidth=0.8,
                zorder=4,
            )
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="none",
            mfc=ORANGE,
            mec="white",
            ms=5,
            label="PV-heavy, buses 13\u201318",
        ),
        Line2D(
            [],
            [],
            marker="^",
            ls="none",
            mfc="none",
            mec=INK,
            mew=0.8,
            ms=5.5,
            label="S1: feeder-head $P$, $Q$",
        ),
        Line2D(
            [],
            [],
            marker="D",
            ls="none",
            mfc="none",
            mec=INK,
            mew=0.8,
            ms=4.5,
            label="S2: adds $|V|$",
        ),
        Line2D(
            [],
            [],
            marker="s",
            ls="none",
            mfc="none",
            mec=INK,
            mew=0.8,
            ms=5.5,
            label="S3: adds substation $P$",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        ncol=2,
        fontsize=7,
        labelspacing=0.25,
        columnspacing=0.8,
        handlelength=1.0,
        borderaxespad=0.0,
    )
    ax.set_xlim(-0.5, 17.5)
    ax.set_ylim(-1.62, 1.5)
    ax.axis("off")
    save(fig, out, "fig_feeder")


def posterior_figure(out, stem):
    import numpy as np

    data = {m: np.load(f"{stem}{m}/samples.npz") for m in ("M2-euler", "B2")}
    table = pd.read_csv(f"{stem}M2-euler/per_snapshot.csv")
    mid = np.flatnonzero((table.hour_bin == "midday").to_numpy())
    node = 11  # bus 13
    truth = data["M2-euler"]["truth"][mid, node]
    order = np.argsort(truth)
    picks = [mid[order[round(q * (len(mid) - 1))]] for q in (0.1, 0.4, 0.6, 0.9)]
    fig = plt.figure(figsize=(TEXT, 1.32))
    grid = fig.add_gridspec(
        1,
        6,
        width_ratios=[1, 1, 1, 1, 0.08, 1.15],
        wspace=0.18,
        left=0.012,
        right=0.988,
        bottom=0.27,
        top=0.86,
    )
    for k, i in enumerate(picks):
        ax = fig.add_subplot(grid[0, k])
        m2 = data["M2-euler"]["samples"][i, :, node]
        b2 = data["B2"]["samples"][i, :, node]
        lo, hi = min(m2.min(), b2.min()), max(m2.max(), b2.max())
        bins = np.linspace(lo, hi, 16)
        cb, _, _ = ax.hist(
            b2,
            bins=bins,
            histtype="step",
            color=ORANGE,
            linewidth=1.1,
            label=NAMES["B2"],
        )
        cm, _, _ = ax.hist(
            m2,
            bins=bins,
            color=BLUE,
            alpha=0.6,
            edgecolor="white",
            linewidth=0.3,
            label=NAMES["M2-euler"],
        )
        line = ax.axvline(
            data["M2-euler"]["truth"][i, node], color=INK, linewidth=1.0, label="Truth"
        )
        q = (10, 40, 60, 90)[k]
        for method, counts in (("B2", cb), ("M2-euler", cm)):
            for b, c in enumerate(counts):
                VALUES.append(
                    {
                        "figure": "fig_posterior",
                        "method": method,
                        "scenario": "ausgrid",
                        "key": f"p{q}_snap{int(i)}_bin{b}",
                        "value": float(c),
                    }
                )
        VALUES.append(
            {
                "figure": "fig_posterior",
                "method": "truth",
                "scenario": "ausgrid",
                "key": f"p{q}_snap{int(i)}",
                "value": float(line.get_xdata()[0]),
            }
        )
        ax.set_yticks([])
        ax.set_ylim(0, 1.3 * max(cb.max(), cm.max()))  # headroom for the label
        ax.spines["left"].set_visible(False)
        ax.text(
            0.02,
            0.98,
            f"{q}th pct.",
            transform=ax.transAxes,
            fontsize=7.5,
            color=INK,
            ha="left",
            va="top",
        )
    fig.text(
        0.012 + 0.5 * (0.988 - 0.012) * 4.0 / 5.23,
        0.015,
        "Net injection at bus 13 (kW), midday test snapshots at percentiles of the truth",
        ha="center",
        va="bottom",
        fontsize=8,
        color=INK,
    )
    ax = fig.add_subplot(grid[0, 5])
    nodes = list(range(11, 17))
    for name, color, kind in (("B2", ORANGE, "step"), ("M2-euler", BLUE, "bar")):
        x = data[name]["samples"][mid][:, :, nodes]
        mean, sd = x.mean(1, keepdims=True), x.std(1, keepdims=True)
        skew = (((x - mean) / sd) ** 3).mean(1).ravel()
        kw = (
            {"histtype": "step", "linewidth": 1.1}
            if kind == "step"
            else {"alpha": 0.6, "edgecolor": "white", "linewidth": 0.3}
        )
        counts, _, _ = ax.hist(
            skew,
            bins=np.linspace(-2, 2, 33),
            color=color,
            label=NAMES[name],
            **kw,
        )
        for b, c in enumerate(counts):
            VALUES.append(
                {
                    "figure": "fig_posterior",
                    "method": name,
                    "scenario": "ausgrid",
                    "key": f"skew_bin{b}",
                    "value": float(c),
                }
            )
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_xlabel("Skewness, buses 13\u201318")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=INK, linewidth=1.0))
    labels.append("Truth")
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        fontsize=7.5,
        handlelength=1.2,
        columnspacing=1.5,
        borderaxespad=0.1,
    )
    save(fig, out, "fig_posterior")


def data_scaling_figure(out, lc):
    lc = lc[lc.mset == "S2"]
    sizes = {"n2k": 1901, "n19k": 19008, "n190k": 190080}
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN, 1.42), layout="constrained")
    for sc, label, color, marker in (
        ("base", "Base PV", BLUE, "o"),
        ("high_pv", "High PV", ORANGE, "s"),
    ):
        d = lc[lc.scenario == sc].copy()
        d["n"] = d["size"].map(sizes)
        d = d.sort_values("n")
        b2c = d["B2:crps_kw"]
        gain = 100 * d["m2_minus_b2_all"] / b2c
        lo = 100 * d["m2_minus_b2_all_lo"] / b2c
        hi = 100 * d["m2_minus_b2_all_hi"] / b2c
        eb = axes[0].errorbar(
            d.n,
            gain,
            yerr=[gain - lo, hi - gain],
            color=color,
            marker=marker,
            ms=3.5,
            linewidth=1.1,
            elinewidth=0.8,
            capsize=1.5,
            label=label,
            mec="white",
            mew=0.4,
        )
        (cov_line,) = axes[1].plot(
            d.n,
            d["M2-euler:coverage90"],
            color=color,
            marker=marker,
            ms=3.5,
            linewidth=1.1,
            label=label,
            mec="white",
            mew=0.4,
        )
        # Read the plotted values back from the artists.
        segments = eb.lines[2][0].get_segments()
        drawn = zip(
            eb.lines[0].get_xdata(),
            eb.lines[0].get_ydata(),
            [seg[0][1] for seg in segments],
            [seg[1][1] for seg in segments],
            cov_line.get_ydata(),
            strict=True,
        )
        for n, g, lo_, hi_, cov in drawn:
            for key, v in (("gain", g), ("gain_lo", lo_), ("gain_hi", hi_)):
                VALUES.append(
                    {
                        "figure": "fig_data_scaling",
                        "method": "M2-euler",
                        "scenario": sc,
                        "key": f"{key}_n{int(n)}",
                        "value": float(v),
                    }
                )
            VALUES.append(
                {
                    "figure": "fig_data_scaling",
                    "method": "M2-euler",
                    "scenario": sc,
                    "key": f"coverage_n{int(n)}",
                    "value": float(cov),
                }
            )
    axes[0].axhline(0, color=MUTED, linewidth=0.8)
    axes[1].axhline(ATTAINABLE, color=MUTED, linewidth=0.8)
    axes[1].text(
        sizes["n2k"],
        ATTAINABLE + 0.003,
        "attainable",
        color=MUTED,
        fontsize=7,
        ha="left",
        va="bottom",
    )
    axes[0].set_ylabel("Relative CRPS (%)")
    axes[0].set_yticks([-10, -8, -6, -4, -2, 0])
    axes[1].set_ylabel("PoSE coverage")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(list(sizes.values()))
        ax.set_xticklabels(["1.9k", "19k", "190k"])
        ax.minorticks_off()
        ax.set_xlim(1300, 280000)
        ax.grid(axis="y", color=GRID, linewidth=0.5)
    axes[1].legend(loc="lower right", handlelength=1.4)
    fig.supxlabel("Training snapshots (model world)", fontsize=8)
    save(fig, out, "fig_data_scaling")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--extended", default="results/ausgrid/report/extended_summary.csv")
    p.add_argument("--world", default="results/world/report/learning_curve.csv")
    p.add_argument("--output_dir", default="results/figures")
    p.add_argument(
        "--posterior_stem",
        default="results/ausgrid_p/cells/ausgrid/real/ausgrid_S2_nominal_",
    )
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    style()

    ext = pd.read_csv(args.extended)
    main_m = ext[ext.method.isin([m for m, _ in ROWS])].copy()
    b1 = main_m[main_m.method == "B1"].set_index(["scenario", "mset", "noise"])[
        "crps_kw"
    ]
    main_m["rel"] = [
        100 * (r.crps_kw / b1[(r.scenario, r.mset, r.noise)] - 1)
        for r in main_m.itertuples()
    ]
    accuracy_calibration_figure(args.output_dir, main_m)
    data_scaling_figure(args.output_dir, pd.read_csv(args.world))
    feeder_figure(args.output_dir)
    posterior_figure(args.output_dir, args.posterior_stem)
    pd.DataFrame(VALUES).to_csv(
        os.path.join(args.output_dir, "figure_values.csv"), index=False
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Adapted from AxBench — https://github.com/stanfordnlp/axbench (Apache-2.0).
"""Mean-of-K vs best-of-K selection lift boxplots across repulsion arms."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from loaders import (
    load_single_arm_per_row,
    mean_over_prompts,
    per_particle_fstar_single,
)
from paper_style_axbench import apply_axbench_style, save_pub_fig, style_score_axis, SCORE_YLIM
from plot_behavioral_correlation import REPULSION_ARMS

POLICY_ORDER = ["Mean-of-$K$", "Best-of-$K$"]

# Light (mean) / saturated (best) pairs per repulsion arm.
ARM_BOX_COLORS: dict[str, dict[str, str]] = {
    "latent_corr": {"mean": "#A8BDD4", "best": "#4A6FA5"},
    "rbf": {"mean": "#8FC4B8", "best": "#2E7D6F"},
    "cosine": {"mean": "#C4A3B5", "best": "#7D4E6F"},
}


def load_policy_df(root: Path, *, particles: list[int] | None = None) -> pd.DataFrame:
    """Per-concept scores for mean-of-K and best-of-K policies at per-particle f*."""
    per_row, _ = load_single_arm_per_row(root)
    if per_row.empty:
        raise FileNotFoundError(f"No eval data under {root / 'evaluate'}")
    if particles is not None:
        per_row = per_row[per_row["particle_id"].isin(particles)]
    fstar = per_particle_fstar_single(mean_over_prompts(per_row))
    rows: list[dict[str, Any]] = []
    for cid, g in fstar.groupby("concept_id"):
        winner = g.loc[g["overall"].idxmax()]
        rows.append({"concept_id": int(cid), "policy": POLICY_ORDER[0], "overall": float(g["overall"].mean())})
        rows.append({"concept_id": int(cid), "policy": POLICY_ORDER[1], "overall": float(winner["overall"])})
    return pd.DataFrame(rows)


def _palette_for_arm(key: str) -> list[str]:
    c = ARM_BOX_COLORS[key]
    return [c["mean"], c["best"]]


def _upper_whisker(vals: pd.Series) -> float:
    s = np.sort(vals.to_numpy(dtype=float))
    q1, q3 = np.percentile(s, [25, 75])
    fence = q3 + 1.5 * (q3 - q1)
    within = s[s <= fence]
    return float(within.max()) if len(within) else float(s.max())


def _lift_y_top(dfs: dict[str, pd.DataFrame]) -> float:
    """Normal AxBench overall-score axis top (0–2.05); data whiskers reach ~1.5."""
    return float(SCORE_YLIM[1])


def _style_lift_axis(
    ax: plt.Axes,
    *,
    ylabel: str,
    y_top: float,
    show_ylabel: bool = True,
) -> None:
    ax.set_ylim(0.0, y_top)
    ax.set_ylabel(ylabel if show_ylabel else "")
    if not show_ylabel:
        ax.set_yticklabels([])
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.grid(axis="y", color="#DADADA", linewidth=0.55)
    ax.grid(axis="x", visible=False)
    for spine in ("left", "right", "bottom"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(FACET_BORDER_COLOR)
        ax.spines[spine].set_linewidth(FACET_BORDER_LW)
    ax.spines["top"].set_visible(False)


def _draw_oracle_boxplot(
    plot_df: pd.DataFrame,
    ax: plt.Axes,
    *,
    palette: list[str],
    show_points: bool,
    show_fliers: bool,
    show_mean: bool,
    title: str | None = None,
    ylabel: str = "Overall",
    show_ylabel: bool = True,
    y_top: float | None = None,
    facet_axis: bool = False,
) -> None:
    """AxBench-style boxplot: Tukey whiskers (whis=1.5), fliers hidden unless requested."""
    sns.boxplot(
        data=plot_df,
        x="policy",
        y="overall",
        hue="policy",
        order=POLICY_ORDER,
        hue_order=POLICY_ORDER,
        palette=dict(zip(POLICY_ORDER, palette)),
        dodge=False,
        legend=False,
        ax=ax,
        width=0.52,
        linewidth=0.85,
        fliersize=4 if show_fliers else 0,
        showfliers=show_fliers,
        whis=1.5,
        boxprops={"edgecolor": "#222222", "linewidth": 0.85},
        medianprops={"color": "#111111", "linewidth": 1.1},
        whiskerprops={"color": "#222222", "linewidth": 0.75},
        capprops={"color": "#222222", "linewidth": 0.75},
    )
    if show_points and not show_fliers:
        sns.stripplot(
            data=plot_df,
            x="policy",
            y="overall",
            order=POLICY_ORDER,
            ax=ax,
            color="#333333",
            alpha=0.38,
            size=3.0,
            jitter=0.14,
            zorder=0,
        )
    if show_mean:
        for i, policy in enumerate(POLICY_ORDER):
            vals = plot_df.loc[plot_df["policy"] == policy, "overall"]
            whisker = _upper_whisker(vals)
            y = whisker + 0.05
            ax.text(
                i,
                y,
                rf"$\mu={vals.mean():.2f}$",
                ha="center",
                va="bottom",
                fontsize=8,
                color="black",
                clip_on=False,
            )
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=9, pad=2)
    if title:
        ax.set_title(title, fontsize=10, pad=8, color="#333333")
    if facet_axis and y_top is not None:
        _style_lift_axis(ax, ylabel=ylabel, y_top=y_top, show_ylabel=show_ylabel)
    else:
        ax.set_ylabel(ylabel if show_ylabel else "")
        if not show_ylabel:
            ax.set_yticklabels([])
        style_score_axis(ax, ylabel=ylabel if show_ylabel else "")
        ax.grid(axis="x", visible=False)
    ax.set_facecolor("white")


FACET_HEADER_COLOR = "#E8E8E8"
FACET_BORDER_COLOR = "#000000"
FACET_BORDER_LW = 0.9
FACET_HEADER_FONT = {"family": "serif", "size": 11, "color": "black"}


def _style_axbench_facet_panel(ax_header: plt.Axes, ax_plot: plt.Axes, title: str) -> None:
    """AxBench-style facet: grey title strip + white plot in one black-bordered panel."""
    ax_header.set_facecolor(FACET_HEADER_COLOR)
    ax_plot.set_facecolor("white")
    ax_header.set_xticks([])
    ax_header.set_yticks([])
    ax_header.set_xlim(0, 1)
    ax_header.set_ylim(0, 1)

    for spine in ax_header.spines.values():
        spine.set_visible(True)
        spine.set_color(FACET_BORDER_COLOR)
        spine.set_linewidth(FACET_BORDER_LW)

    ax_plot.spines["top"].set_visible(False)

    ax_header.text(
        0.5,
        0.50,
        title,
        ha="center",
        va="center",
        transform=ax_header.transAxes,
        clip_on=False,
        **FACET_HEADER_FONT,
    )


def _add_panel_header(
    fig: plt.Figure,
    ax_plot: plt.Axes,
    title: str,
    *,
    style: str,
    accent: str | None = None,
) -> None:
    """Panel title above the plot area."""
    pos = ax_plot.get_position()
    header_h = 0.045
    gap = 0.006
    y_top = pos.y1 + gap
    y_bot = y_top + header_h
    x0, x1 = pos.x0, pos.x1

    if style == "grey_band":
        band = fig.add_axes([x0, y_bot, x1 - x0, header_h], zorder=10)
        band.set_facecolor("#ECECEC")
        band.set_xticks([])
        band.set_yticks([])
        for spine in band.spines.values():
            spine.set_visible(False)
        band.text(0.5, 0.5, title, ha="center", va="center", fontsize=10, color="#333333", transform=band.transAxes)
    elif style == "accent_stripe":
        band = fig.add_axes([x0, y_bot, x1 - x0, header_h], zorder=10)
        band.set_facecolor("#EFEFEF")
        band.set_xticks([])
        band.set_yticks([])
        for spine in band.spines.values():
            spine.set_visible(False)
        if accent:
            stripe = fig.add_axes([x0, y_bot, 0.008, header_h], zorder=11)
            stripe.set_facecolor(accent)
            stripe.set_xticks([])
            stripe.set_yticks([])
            for spine in stripe.spines.values():
                spine.set_visible(False)
        band.text(0.5, 0.5, title, ha="center", va="center", fontsize=10, fontweight="medium", color="#222222", transform=band.transAxes)
    elif style == "fig_text":
        fig.text(
            (x0 + x1) / 2,
            y_bot + header_h * 0.55,
            title,
            ha="center",
            va="center",
            fontsize=10,
            color="#333333",
            bbox=dict(boxstyle="square,pad=0.35", facecolor="#ECECEC", edgecolor="#D0D0D0", linewidth=0.6),
        )
    else:
        ax_plot.set_title(title, fontsize=10, pad=14, color="#333333")


def _save_triptych(
    dfs: dict[str, pd.DataFrame],
    out: Path,
    *,
    show_points: bool,
    show_fliers: bool,
    show_mean: bool,
    header_style: str | None,
) -> None:
    apply_axbench_style()
    use_facet = header_style in ("axbench_facet", "axbench_facet_wide", "axbench_facet_v2")
    y_top = _lift_y_top(dfs) if use_facet else None
    fig_h = 3.25 if header_style in ("axbench_facet_wide", "axbench_facet_v2") else (3.55 if use_facet else (3.75 if header_style else 3.5))
    fig = plt.figure(figsize=(9.6, fig_h))

    if use_facet:
        hr = 0.18 if header_style in ("axbench_facet_wide", "axbench_facet_v2") else 0.12
        gs = GridSpec(2, 3, figure=fig, height_ratios=[hr, 1.0], hspace=0.0, wspace=0.16)
        fig.subplots_adjust(left=0.10, right=0.99, top=0.98, bottom=0.17)
        plot_axes: list[plt.Axes] = []
        for col, (key, meta) in enumerate(REPULSION_ARMS.items()):
            ax_h = fig.add_subplot(gs[0, col])
            ax_p = fig.add_subplot(gs[1, col])
            _style_axbench_facet_panel(ax_h, ax_p, meta["label"])
            plot_axes.append(ax_p)
        axes = plot_axes
    elif header_style:
        gs = GridSpec(2, 3, figure=fig, height_ratios=[0.11, 1.0], hspace=0.06, wspace=0.28)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.14)
        plot_axes: list[plt.Axes] = []
        for col, (key, meta) in enumerate(REPULSION_ARMS.items()):
            ax_h = fig.add_subplot(gs[0, col])
            ax_p = fig.add_subplot(gs[1, col])
            ax_h.set_facecolor("#ECECEC" if header_style == "grey_band" else "#EFEFEF")
            ax_h.set_xticks([])
            ax_h.set_yticks([])
            for spine in ax_h.spines.values():
                spine.set_visible(False)
            if header_style == "accent_stripe":
                accent = ARM_BOX_COLORS[key]["best"]
                ax_h.axvline(0.02, color=accent, linewidth=3, ymin=0.15, ymax=0.85, clip_on=False, zorder=5)
            ax_h.text(
                0.5,
                0.5,
                meta["label"],
                ha="center",
                va="center",
                fontsize=10,
                fontweight="medium" if header_style == "accent_stripe" else "normal",
                color="#222222" if header_style == "accent_stripe" else "#333333",
                transform=ax_h.transAxes,
            )
            plot_axes.append(ax_p)
        axes = plot_axes
    else:
        axes = fig.subplots(1, 3)
        fig.subplots_adjust(wspace=0.28, left=0.08, right=0.98, top=0.88, bottom=0.14)

    for ax, (key, meta) in zip(axes, REPULSION_ARMS.items()):
        _draw_oracle_boxplot(
            dfs[key],
            ax,
            palette=_palette_for_arm(key),
            title=None if header_style else meta["label"],
            show_ylabel=(key == "latent_corr"),
            show_points=show_points,
            show_fliers=show_fliers,
            show_mean=show_mean,
            y_top=y_top,
            facet_axis=use_facet,
        )
        if header_style == "fig_text":
            _add_panel_header(fig, ax, meta["label"], style="fig_text")

    save_pub_fig(fig, out, pad_inches=0.08 if use_facet else 0.04)


def run_selection_lift_figures(base_root: Path, out: Path) -> Path:
    """Write the three-panel mean-of-K vs best-of-K selection lift figure."""
    out.mkdir(parents=True, exist_ok=True)
    dfs: dict[str, pd.DataFrame] = {}
    for key, meta in REPULSION_ARMS.items():
        root = base_root / meta["root_suffix"]
        df = load_policy_df(root, particles=meta["particles"])
        dfs[key] = df
        df.to_csv(out / f"selection_lift_{key}.csv", index=False)

    triptych = out / "selection_lift_triptych.png"
    _save_triptych(
        dfs,
        triptych,
        show_points=False,
        show_fliers=False,
        show_mean=True,
        header_style="axbench_facet_v2",
    )
    return triptych


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-root", type=Path, default=repo / "data/particlesteering")
    p.add_argument(
        "--out",
        type=Path,
        default=repo / "paper_outputs/figures/selection_lift",
    )
    args = p.parse_args()
    path = run_selection_lift_figures(args.base_root, args.out)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

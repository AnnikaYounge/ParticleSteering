# Adapted from AxBench — https://github.com/stanfordnlp/axbench (Apache-2.0).
"""AxBench paper styling (arXiv:2505.20809) — Fig. 3 / Fig. 4 aesthetic."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

# Gemma-2 layer palette from AxBench Fig. 3 (purple → blue → teal → green)
AXBENCH_LAYER_COLORS = ["#3B2864", "#5B6BAF", "#4A8FA8", "#6FB98F", "#A8D5BA", "#C8E6C9"]

METRIC_ORDER_AXBENCH = [
    ("concept", "Concept Score"),
    ("fluency", "Fluency Score"),
    ("instruct", "Instruction Score"),
    ("overall", "Overall Score"),
]

SCORE_YLIM = (0.0, 2.05)
LIFT_MEAN = "#7A7A7A"
LIFT_BEST = "#5B6BAF"
LIFT_GAIN = "#4A8FA8"


def axbench_particle_palette(k: int) -> list[str]:
    """Layer-colored palette used in main-paper judge-score figures."""
    n = max(int(k), 1)
    return [AXBENCH_LAYER_COLORS[i % len(AXBENCH_LAYER_COLORS)] for i in range(n)]


def apply_axbench_style() -> None:
    """theme_bw-like: white panel, horizontal grid, tight typography."""
    sns.set_theme(style="white", context="paper", font_scale=1.0)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "axes.titleweight": "normal",
            "axes.linewidth": 0.75,
            "axes.edgecolor": "#222222",
            "axes.facecolor": "white",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#E0E0E0",
            "grid.linewidth": 0.6,
            "grid.alpha": 1.0,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def save_pub_fig(fig: plt.Figure, path: Path, *, pad_inches: float = 0.04) -> None:
    """Save publication figure at 300 dpi and close."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=pad_inches)
    plt.close(fig)


def style_score_axis(ax: plt.Axes, *, ylabel: str = "Score") -> None:
    ax.set_ylim(*SCORE_YLIM)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.grid(axis="y", which="major")
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax, left=False, bottom=False)


def label_panel_right(ax: plt.Axes, text: str) -> None:
    ax.text(
        1.01,
        0.5,
        text,
        transform=ax.transAxes,
        va="center",
        ha="left",
        fontsize=9,
        rotation=-90,
        color="#333333",
    )


def add_axbench_boxplot(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    order: list,
    palette: list[str],
    show_points: bool = False,
    width: float = 0.62,
) -> None:
    pal_map = dict(zip(order, palette[: len(order)]))
    sns.boxplot(
        data=df,
        x=x,
        y=y,
        hue=x,
        order=order,
        hue_order=order,
        palette=pal_map,
        dodge=False,
        legend=False,
        ax=ax,
        width=width,
        linewidth=0.85,
        fliersize=0,
        showfliers=False,
        whis=1.5,
        boxprops={"edgecolor": "#222222", "linewidth": 0.85},
        medianprops={"color": "#111111", "linewidth": 1.1},
        whiskerprops={"color": "#222222", "linewidth": 0.75},
        capprops={"color": "#222222", "linewidth": 0.75},
    )
    if show_points:
        sns.stripplot(
            data=df,
            x=x,
            y=y,
            order=order,
            ax=ax,
            color="#333333",
            alpha=0.35,
            size=2.8,
            jitter=0.15,
            zorder=0,
        )


def style_heatmap_ax(ax: plt.Axes, *, cbar_label: str = "") -> None:
    ax.tick_params(length=0)
    if ax.collections:
        cbar = ax.collections[0].colorbar
        if cbar is not None:
            cbar.ax.tick_params(labelsize=8)
            if cbar_label:
                cbar.set_label(cbar_label, fontsize=9)

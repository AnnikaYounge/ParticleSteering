# Adapted from AxBench — https://github.com/stanfordnlp/axbench (Apache-2.0).
"""Matplotlib styling aligned with AxBench / analyse.ipynb (theme_bw, clean sans)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Seaborn deep palette — ggplot-adjacent, publication-friendly
PARTICLE_PALETTE_NAME = "deep"
LIFT_COLOR_MEAN = "#7f7f7f"
LIFT_COLOR_BEST = "#4c72b0"
LIFT_COLOR_GAIN = "#55a868"

METRIC_TITLES = {
    "concept": "Concept Score",
    "instruct": "Instruction Score",
    "fluency": "Fluency Score",
    "overall": "Overall Score",
}


def apply_paper_style() -> None:
    """ggplot-like matplotlib defaults for geometry and exploratory figures."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05, palette=PARTICLE_PALETTE_NAME)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "normal",
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#333333",
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def save_fig(fig: plt.Figure, path: Path) -> None:
    """Save figure at 300 dpi and close."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def particle_palette(k: int) -> list:
    """Seaborn deep palette with ``k`` colors."""
    return sns.color_palette(PARTICLE_PALETTE_NAME, max(k, 3))[:k]


def particle_palette_muted(k: int) -> list:
    return sns.color_palette("muted", max(k, 3))[:k]


def particle_palette_alt(k: int) -> list:
    """Secondary palette (muted) for paired metric panels."""
    return particle_palette_muted(k)


def add_ggplot_boxplot(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    k: int,
    palette: list | None = None,
    show_points: bool = True,
    point_alpha: float = 0.45,
) -> None:
    """Boxplot matching analyse.ipynb: Tukey whiskers, outliers hidden (outlier_shape='').

    Quartiles: box = [Q1, Q3], line = median, whiskers = min/max within Q1−1.5·IQR, Q3+1.5·IQR.
    """
    order = sorted(df[x].unique())
    pal = palette or particle_palette(len(order))
    sns.boxplot(
        data=df,
        x=x,
        y=y,
        hue=x,
        order=order,
        hue_order=order,
        palette=dict(zip(order, pal)),
        dodge=False,
        legend=False,
        ax=ax,
        width=0.55,
        linewidth=0.9,
        fliersize=0,
        showfliers=False,
        whis=1.5,
    )
    if show_points:
        sns.stripplot(
            data=df,
            x=x,
            y=y,
            order=order,
            ax=ax,
            color="#333333",
            alpha=point_alpha,
            size=3.2,
            jitter=0.18,
            zorder=0,
        )

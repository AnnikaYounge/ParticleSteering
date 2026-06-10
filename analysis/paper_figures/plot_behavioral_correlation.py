#!/usr/bin/env python3
# Adapted from AxBench — https://github.com/stanfordnlp/axbench (Apache-2.0).
"""Pairwise particle behavioral-correlation heatmaps across repulsion arms."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import squareform

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from loaders import load_single_arm_per_row, mean_over_prompts, per_particle_fstar_single
from paper_style_axbench import apply_axbench_style, save_pub_fig
CMAP_SEQ = sns.color_palette("light:#5B6BAF", as_cmap=True)

# Per-repulsion sequential palettes (distinct, publication-friendly)
REPULSION_ARMS: dict[str, dict] = {
    "latent_corr": {
        "label": "Latent-corr",
        "cmap": sns.color_palette("light:#4A6FA5", as_cmap=True),
        "root_suffix": "repulsion_latent_corr/form_additive/w0.5",
        "particles": None,
    },
    "rbf": {
        "label": "RBF",
        "cmap": sns.color_palette("light:#2E7D6F", as_cmap=True),
        "root_suffix": "repulsion_rbf/form_additive/w0.5",
        "particles": None,
    },
    "cosine": {
        "label": "Cosine",
        "cmap": sns.color_palette("light:#7D4E6F", as_cmap=True),
        "root_suffix": "repulsion_cosine/form_additive/w0.5",
        "particles": [0, 1, 2, 3, 4],  # K=10 trained; show first five for K=5 comparison
    },
}


def load_corr_from_root(root: Path, *, particles: list[int] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pearson correlation matrix of per-particle overall scores and the f* table used."""
    per_row, _ = load_single_arm_per_row(root)
    if per_row.empty:
        raise FileNotFoundError(f"No eval data under {root / 'evaluate'}")
    if particles is not None:
        per_row = per_row[per_row["particle_id"].isin(particles)]
    fstar = per_particle_fstar_single(mean_over_prompts(per_row))
    return _behavioral_corr(fstar), fstar


def _behavioral_corr(fstar: pd.DataFrame) -> pd.DataFrame:
    """Correlate particle overall-score vectors across concepts."""
    mat = fstar.pivot(index="concept_id", columns="particle_id", values="overall")
    mat = mat.sort_index(axis=0).sort_index(axis=1)
    return mat.corr()


def _particle_labels(ids) -> list[str]:
    return [f"$p_{{{int(i)}}}$" for i in ids]


def _mask_upper(corr: pd.DataFrame) -> np.ndarray:
    n = corr.shape[0]
    return np.triu(np.ones((n, n), dtype=bool), k=1)


def _off_diag_values(corr: pd.DataFrame) -> np.ndarray:
    arr = corr.values.astype(float)
    return arr[~np.eye(arr.shape[0], dtype=bool)]


def _color_limits(corr: pd.DataFrame) -> tuple[float, float]:
    off = _off_diag_values(corr)
    vmin = max(0.0, float(np.floor(off.min() * 20) / 20) - 0.05)
    return vmin, 1.0


def _annot_text_color(val: float, cmap, norm) -> str:
    r, g, b, _ = cmap(norm(val))
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if lum < 0.52 else "#1a1a1a"


def _style_corr_ax(ax: plt.Axes, *, xlabel: str = "Particle", ylabel: str = "Particle") -> None:
    ax.set_xlabel(xlabel, labelpad=6)
    ax.set_ylabel(ylabel, labelpad=6)
    ax.tick_params(length=0, pad=2)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color("#CCCCCC")


def _add_cbar(fig, ax, mappable, label: str, *, ticks: list[float] | None = None) -> None:
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04, shrink=0.92)
    cbar.ax.tick_params(labelsize=8, length=2, width=0.6)
    cbar.set_label(label, fontsize=9, labelpad=6)
    if ticks is not None:
        cbar.set_ticks(ticks)


def _cluster_linkage(corr: pd.DataFrame):
    dist = 1.0 - corr.values.astype(float)
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    return linkage(condensed, method="average")


def _reorder_corr(corr: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    link = _cluster_linkage(corr)
    order_idx = leaves_list(link)
    order = [corr.index[i] for i in order_idx]
    return corr.loc[order, order], order


def _style_heatmap_annotations(hm, corr: pd.DataFrame, cmap, norm, *, mask: np.ndarray | None = None) -> None:
    vals = corr.values if mask is None else corr.values[~mask]
    for text, val in zip(hm.texts, vals.ravel() if mask is None else vals):
        if np.isfinite(val):
            text.set_color(_annot_text_color(float(val), cmap, norm))


def _draw_clustered_heatmap(
    corr: pd.DataFrame,
    out: Path | None,
    *,
    annot: bool,
    dendro: str,
    figsize: tuple[float, float],
    cmap=None,
    title: str | None = None,
    ax: plt.Axes | None = None,
    show_ylabel: bool = True,
) -> tuple[plt.Figure | None, plt.Axes, float]:
    """Cluster-ordered square heatmap. Returns (fig, ax, vmin) for compositing."""
    apply_axbench_style()
    cmap = cmap or CMAP_SEQ
    corr_ord, order = _reorder_corr(corr)
    labels = _particle_labels(order)
    vmin, vmax = _color_limits(corr)
    norm = Normalize(vmin=vmin, vmax=vmax)
    link = _cluster_linkage(corr)
    owns_fig = ax is None

    if owns_fig:
        if dendro == "top":
            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(2, 1, height_ratios=[0.18, 1.0], hspace=0.04)
            ax_top = fig.add_subplot(gs[0])
            ax = fig.add_subplot(gs[1])
            dendrogram(link, ax=ax_top, color_threshold=0, above_threshold_color="#555555", no_labels=True)
            ax_top.set_xticks([])
            ax_top.set_yticks([])
            for spine in ax_top.spines.values():
                spine.set_visible(False)
        elif dendro == "both":
            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(
                2, 2, width_ratios=[0.28, 1.0], height_ratios=[0.18, 1.0], wspace=0.06, hspace=0.04,
            )
            ax_top = fig.add_subplot(gs[0, 1])
            ax_left = fig.add_subplot(gs[1, 0])
            ax = fig.add_subplot(gs[1, 1])
            dendrogram(link, ax=ax_top, color_threshold=0, above_threshold_color="#555555", no_labels=True)
            dendrogram(
                link, ax=ax_left, orientation="left", color_threshold=0, above_threshold_color="#555555", no_labels=True,
            )
            for dax in (ax_top, ax_left):
                dax.set_xticks([])
                dax.set_yticks([])
                for spine in dax.spines.values():
                    spine.set_visible(False)
        else:
            fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    hm = sns.heatmap(
        corr_ord,
        cmap=cmap,
        norm=norm,
        square=True,
        ax=ax,
        annot=annot,
        fmt=".2f",
        annot_kws={"size": 9 if annot else 8},
        cbar=False,
        linewidths=0.8,
        linecolor="white",
        xticklabels=labels,
        yticklabels=labels if show_ylabel else False,
    )
    if annot:
        _style_heatmap_annotations(hm, corr_ord, cmap, norm)
    _style_corr_ax(ax, ylabel="Particle" if show_ylabel else "")
    if not show_ylabel:
        ax.set_ylabel("")
    if title:
        ax.set_title(title, fontsize=10, pad=8, color="#333333")
    if owns_fig and out is not None:
        _add_cbar(fig, ax, hm.collections[0], "Pearson $r$", ticks=[round(vmin, 2), 0.6, 0.8, 1.0])
        if dendro == "top":
            fig.subplots_adjust(left=0.14, right=0.88, top=0.98, bottom=0.16)
        elif dendro == "both":
            fig.subplots_adjust(left=0.16, right=0.88, top=0.96, bottom=0.16)
        else:
            fig.tight_layout()
        save_pub_fig(fig, out)
    return fig, ax, vmin


def run_repulsion_triptych(base_root: Path, out: Path) -> Path:
    """Three-panel behavioral correlation (latent-corr, RBF, cosine)."""
    apply_axbench_style()
    out.mkdir(parents=True, exist_ok=True)
    arms = list(REPULSION_ARMS.items())
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.35))
    fig.subplots_adjust(wspace=0.38, left=0.06, right=0.92, top=0.88, bottom=0.18)

    summary_rows: list[dict] = []
    mappables = []
    for ax, (key, meta) in zip(axes, arms):
        root = base_root / meta["root_suffix"]
        corr, fstar = load_corr_from_root(root, particles=meta["particles"])
        corr.to_csv(out / f"behavioral_correlation_{key}.csv")
        off = _off_diag_values(corr)
        summary_rows.append(
            {
                "repulsion": key,
                "mean_r": float(off.mean()),
                "min_r": float(off.min()),
                "max_r": float(off.max()),
                "n_concepts": int(fstar["concept_id"].nunique()),
            }
        )
        _, _, vmin = _draw_clustered_heatmap(
            corr,
            None,
            annot=True,
            dendro="none",
            figsize=(3.2, 3.0),
            cmap=meta["cmap"],
            title=meta["label"],
            ax=ax,
            show_ylabel=(key == "latent_corr"),
        )
        mappable = ax.collections[0]
        mappables.append((ax, mappable, vmin))

    for ax, mappable, vmin in mappables:
        cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04, shrink=0.95)
        cbar.ax.tick_params(labelsize=7, length=2, width=0.5)
        cbar.set_label("Pearson $r$", fontsize=8, labelpad=4)
        cbar.set_ticks([round(vmin, 2), 0.6, 0.8, 1.0])

    triptych = out / "behavioral_correlation_triptych.png"
    save_pub_fig(fig, triptych)
    pd.DataFrame(summary_rows).to_csv(out / "behavioral_correlation_summary.csv", index=False)
    return triptych


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base-root",
        type=Path,
        default=repo / "data/particlesteering",
        help="Parent of repulsion_* dirs",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=repo / "paper_outputs/figures/behavioral_correlation",
    )
    args = p.parse_args()
    print(f"Writing triptych to {args.out}")
    path = run_repulsion_triptych(args.base_root, args.out)
    print(f"  {path.name}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Adapted from AxBench — https://github.com/stanfordnlp/axbench (Apache-2.0).
"""Per-concept steering heterogeneity figures (latent-corr, K=5, 20 concepts)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from loaders import load_single_arm_per_row, mean_over_prompts, multiplicity_summary_single, per_particle_fstar_single
from paper_style_axbench import (
    LIFT_BEST,
    LIFT_GAIN,
    LIFT_MEAN,
    apply_axbench_style,
    axbench_particle_palette,
    style_score_axis,
)

PARTICLE_PAL = axbench_particle_palette(5)
EXEMPLAR_EDGE = "#C44E52"

# Main-text exemplars (fixed roles for cross-figure consistency).
EXEMPLARS: dict[int, str] = {
    9: "volatile",   # large oracle gap / particle disagreement
    2: "hard",       # low ceiling, concept-limited
    5: "flat",       # particles agree; little selection value
    16: "easy",      # high ceiling, responsive steering
}

# Main-text 2×2 exemplar panel (concept id, role label).
PAPER_EXEMPLAR_QUAD: list[tuple[int, str]] = [
    (16, "Highest ceiling"),
    (15, "High ceiling"),
    (2, "Lowest ceiling"),
    (10, "Low mean, moderate best"),
]

SHORT_LABELS: dict[int, str] = {
    0: "Rental services",
    1: "Science research",
    2: "C/C++ syntax",
    3: "Academic papers",
    4: "UI layout",
    5: "Math roots",
    6: "Saying / expressing",
    7: "Entity statements",
    8: "Biography",
    9: "Fantasy worlds",
    10: "Chemistry vocab.",
    11: "Debugging",
    12: "Qualifiers",
    13: "Uncertainty sci.",
    14: "Math/code syntax",
    15: "Possession / time",
    16: "Ease verbs",
    17: "Code comments",
    18: "Code snippets",
    19: "XML/HTML attrs.",
}

METRIC_ORDER = ("concept", "instruct", "fluency")
METRIC_LABELS = {"concept": "Concept", "instruct": "Instruction", "fluency": "Fluency"}


def _save(fig: plt.Figure, stem: Path) -> None:
    """Write PNG and PDF for ``stem`` and close the figure."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.04)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white", pad_inches=0.04)
    plt.close(fig)


def _load_metadata(meta_path: Path) -> pd.DataFrame:
    """Concept id, label, and genre from train metadata.jsonl."""
    rows: list[dict[str, Any]] = []
    with meta_path.open() as f:
        for line in f:
            m = json.loads(line)
            genre = list(m["concept_genres_map"].values())[0][0]
            rows.append(
                {
                    "concept_id": int(m["concept_id"]),
                    "concept_full": m["concept"],
                    "genre": genre,
                    "short_label": SHORT_LABELS.get(int(m["concept_id"]), m["concept"][:28]),
                }
            )
    return pd.DataFrame(rows)


def _policy_metrics_from_fstar(fstar: pd.DataFrame) -> pd.DataFrame:
    """Mean-of-K and best-of-K subscores per concept from per-particle f* rows."""
    rows: list[dict[str, Any]] = []
    for cid, g in fstar.groupby("concept_id"):
        mean_row = g.mean(numeric_only=True)
        best_row = g.loc[g["overall"].idxmax()]
        for policy, src in (("Mean-of-K", mean_row), ("Best-of-K", best_row)):
            for metric in METRIC_ORDER:
                rows.append(
                    {
                        "concept_id": int(cid),
                        "policy": policy,
                        "metric_key": metric,
                        "score": float(src[metric]),
                    }
                )
    return pd.DataFrame(rows)


def _build_table(mult: pd.DataFrame, meta: pd.DataFrame, box: pd.DataFrame) -> pd.DataFrame:
    """Join multiplicity, metadata, and per-policy subscores into one summary table."""
    best_sub = (
        box[box["policy"] == "Best-of-K"]
        .pivot(index="concept_id", columns="metric_key", values="score")
        .reset_index()
    )
    tab = mult.merge(meta, on="concept_id", how="left")
    tab = tab.merge(
        best_sub.rename(
            columns={
                "concept": "best_concept",
                "instruct": "best_instruct",
                "fluency": "best_fluency",
                "overall": "best_overall_chk",
            }
        ),
        on="concept_id",
        how="left",
    )
    tab["pattern"] = tab.apply(_classify_pattern, axis=1)
    tab = tab.sort_values("oracle_gap", ascending=False).reset_index(drop=True)
    return tab


def _classify_pattern(row: pd.Series) -> str:
    """Heuristic heterogeneity bucket from oracle gap and spread."""
    gap = float(row["oracle_gap"])
    best = float(row["best_of_k_overall"])
    spread = float(row["spread_overall"])
    bc = float(row.get("best_concept", np.nan))
    bi = float(row.get("best_instruct", np.nan))
    bf = float(row.get("best_fluency", np.nan))

    if best < 0.55:
        base = "low-ceiling"
    elif gap >= 0.33 or spread >= 0.7:
        base = "high-variance"
    elif gap <= 0.10:
        base = "low-variance"
    else:
        base = "moderate"

    if bc < 0.70 and bi >= 1.25 and bf >= 1.0:
        return f"{base}; concept-limited"
    if best >= 1.15 and bc >= 1.0:
        return f"{base}; responsive"
    return base


def _plot_lift_scatter(tab: pd.DataFrame, out: Path) -> None:
    """Mean-of-K vs best-of-K scatter for all concepts."""
    apply_axbench_style()
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    colors = []
    for cid in tab["concept_id"]:
        colors.append("#C44E52" if cid in EXEMPLARS else LIFT_BEST)

    ax.scatter(
        tab["mean_of_k_overall"],
        tab["best_of_k_overall"],
        c=colors,
        s=52,
        alpha=0.9,
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
    )
    lim = max(tab["best_of_k_overall"].max(), tab["mean_of_k_overall"].max()) * 1.08
    ax.plot([0, lim], [0, lim], color=LIFT_MEAN, ls="--", lw=1.0, zorder=1)
    ax.set_xlim(-0.02, lim)
    ax.set_ylim(-0.02, lim)
    ax.set_aspect("equal", adjustable="box")
    style_score_axis(ax, ylabel="Best-of-$K$ overall")
    ax.set_xlabel("Mean-of-$K$ overall")
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))

    annotate = {
        9: "volatile",
        2: "hard",
        5: "flat",
        16: "easy",
    }
    for cid, tag in annotate.items():
        r = tab.loc[tab["concept_id"] == cid].iloc[0]
        ax.annotate(
            f"{SHORT_LABELS[cid]} ({tag})",
            (r["mean_of_k_overall"], r["best_of_k_overall"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7.5,
            color="#333333",
        )

    ax.text(
        0.03,
        0.97,
        "Above diagonal: oracle selection helps",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="#555555",
    )
    _save(fig, out / "fig_hetero_lift_scatter")


def _plot_lift_dumbbell(tab: pd.DataFrame, out: Path) -> None:
    """Per-concept oracle lift dumbbells sorted by gap."""
    apply_axbench_style()
    plot = tab.sort_values("oracle_gap", ascending=True).reset_index(drop=True)
    n = len(plot)
    fig_h = max(4.5, 0.22 * n + 1.2)
    fig, ax = plt.subplots(figsize=(5.2, fig_h))
    y = np.arange(n)
    highlight = set(EXEMPLARS)

    for i, row in plot.iterrows():
        cid = int(row["concept_id"])
        color = "#C44E52" if cid in highlight else LIFT_GAIN
        lw = 2.4 if cid in highlight else 1.8
        ax.plot(
            [row["mean_of_k_overall"], row["best_of_k_overall"]],
            [i, i],
            color=color,
            lw=lw,
            zorder=1,
        )

    ax.scatter(plot["mean_of_k_overall"], y, color=LIFT_MEAN, s=30, zorder=2, label="Mean-of-$K$")
    ax.scatter(plot["best_of_k_overall"], y, color=LIFT_BEST, s=30, zorder=3, label="Best-of-$K$")
    ax.set_yticks(y)
    ax.set_yticklabels(plot["short_label"], fontsize=7.5)
    ax.set_xlabel("Overall score at per-particle $f^*$")
    style_score_axis(ax, ylabel="")
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    ax.grid(axis="x")
    fig.subplots_adjust(left=0.30)
    _save(fig, out / "fig_hetero_lift_dumbbell")


def _plot_exemplar_subscores(box: pd.DataFrame, out: Path) -> None:
    """Concept / instruction / fluency breakdown for fixed exemplar concepts."""
    apply_axbench_style()
    cids = list(EXEMPLARS.keys())
    roles = [EXEMPLARS[c] for c in cids]
    titles = [f"{SHORT_LABELS[c]} ({roles[i]})" for i, c in enumerate(cids)]

    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.2), sharey=True)
    order = ["Mean-of-K", "Best-of-K"]
    colors = [LIFT_MEAN, LIFT_BEST]

    for ax, cid, title in zip(axes.flat, cids, titles):
        sub = box[(box["concept_id"] == cid) & (box["metric_key"].isin(METRIC_ORDER))].copy()
        sub["metric_key"] = pd.Categorical(sub["metric_key"], categories=list(METRIC_ORDER), ordered=True)
        x = np.arange(len(METRIC_ORDER))
        width = 0.36
        for j, policy in enumerate(order):
            vals = [
                float(sub[(sub["policy"] == policy) & (sub["metric_key"] == m)]["score"].iloc[0])
                for m in METRIC_ORDER
            ]
            ax.bar(x + (j - 0.5) * width, vals, width=width, color=colors[j], label=policy if ax is axes[0, 0] else None)
        ax.set_xticks(x)
        ax.set_xticklabels([METRIC_LABELS[m] for m in METRIC_ORDER], fontsize=8)
        ax.set_title(title, fontsize=9)
        style_score_axis(ax)
        ax.set_ylabel("")

    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.supylabel("LM judge score", fontsize=10)
    fig.tight_layout()
    _save(fig, out / "fig_hetero_exemplars_subscore")


def _draw_particle_spread_panel(
    ax: plt.Axes,
    g: pd.DataFrame,
    *,
    title: str,
    highlight: bool = False,
    show_legend: bool = False,
    marker_size: float = 55,
) -> tuple[float, float]:
    """One concept: particle scores at f* with mean-of-K and best-of-K reference lines."""
    g = g.sort_values("particle_id")
    x = g["particle_id"].astype(int).tolist()
    y = g["overall"].tolist()
    ax.scatter(
        x,
        y,
        c=[PARTICLE_PAL[i % len(PARTICLE_PAL)] for i in x],
        s=marker_size,
        edgecolors="white",
        linewidths=0.4,
        zorder=2,
    )
    ax.plot(x, y, color="#CCCCCC", lw=0.9, zorder=1)
    mean_y = float(g["overall"].mean())
    best_y = float(g["overall"].max())
    ax.axhline(mean_y, color=LIFT_MEAN, ls=":", lw=1.0)
    ax.axhline(best_y, color=LIFT_BEST, ls="--", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x], fontsize=7)
    ax.set_title(title, fontsize=8 if marker_size < 50 else 9)
    style_score_axis(ax)
    if highlight:
        for spine in ax.spines.values():
            spine.set_edgecolor(EXEMPLAR_EDGE)
            spine.set_linewidth(1.6)
    if show_legend:
        ax.legend(
            handles=[
                plt.Line2D([0], [0], color=LIFT_MEAN, ls=":", lw=1.2, label="Mean-of-$K$"),
                plt.Line2D([0], [0], color=LIFT_BEST, ls="--", lw=1.2, label="Best-of-$K$"),
            ],
            fontsize=7,
            loc="lower right",
            frameon=False,
        )
    return mean_y, best_y


def _concept_header(cid: int, label: str, *, style: str) -> str:
    """Format concept id + short label for panel titles."""
    if style == "concept_id":
        return f"Concept {cid:02d} · {label}"
    return f"c{cid:02d} · {label}"


def _set_panel_header(
    ax: plt.Axes,
    header: str,
    stats_line: str | None,
    *,
    bold_header: bool,
) -> None:
    """Place title and optional stats line above a panel (bold-header layout)."""
    ax.set_title("")
    header_fs = 9.5 if bold_header else 8
    # Anchor title above stats: header hangs down from y_head, stats hangs down from y_stats.
    y_head = 1.40 if stats_line else 1.10
    y_stats = 1.26
    ax.text(
        0.5,
        y_head,
        header,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=header_fs,
        fontweight="bold" if bold_header else "normal",
        color="#111111",
        clip_on=False,
    )
    if stats_line:
        ax.text(
            0.5,
            y_stats,
            stats_line,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7.5,
            color="#555555",
            clip_on=False,
        )


def _plot_particle_spread_grid(
    fstar: pd.DataFrame,
    tab: pd.DataFrame,
    out: Path,
    *,
    stem: str,
    concept_order: list[int],
    ncols: int,
    figsize_per_panel: tuple[float, float],
    suptitle: str,
    suptitle_sub: str | None = None,
    subtitle_fn: Any | None = None,
    highlight: set[int] | None = None,
    header_style: str = "c_id",
    bold_header: bool = False,
) -> None:
    """Multi-panel grid of per-concept particle spread plots."""
    apply_axbench_style()
    n = len(concept_order)
    nrows = int(np.ceil(n / ncols))
    fig_w = figsize_per_panel[0] * ncols
    top_margin = 0.85 if bold_header else 0.35
    fig_h = figsize_per_panel[1] * nrows + top_margin
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharey=True, squeeze=False)
    highlight = highlight or set()

    stats: list[dict[str, Any]] = []
    for idx, cid in enumerate(concept_order):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        g = fstar[fstar["concept_id"] == cid]
        row = tab.loc[tab["concept_id"] == cid].iloc[0]
        label = SHORT_LABELS.get(cid, str(cid))
        header = _concept_header(cid, label, style=header_style)
        stats_line = subtitle_fn(row) if subtitle_fn is not None else None

        mean_y, best_y = _draw_particle_spread_panel(
            ax,
            g,
            title="",
            highlight=cid in highlight,
            marker_size=42 if n > 8 else 55,
        )
        if bold_header:
            _set_panel_header(ax, header, stats_line, bold_header=True)
        else:
            title = f"{header}\n{stats_line}" if stats_line else header
            ax.set_title(title, fontsize=8 if n > 8 else 9)

        stats.append({"concept_id": cid, "mean": mean_y, "best": best_y})
        if idx % ncols == 0:
            ax.set_ylabel("Overall at $f^*$", fontsize=9)
        if r == nrows - 1:
            ax.set_xlabel("Particle", fontsize=8)
        else:
            ax.set_xlabel("")

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")

    handles = [
        plt.Line2D([0], [0], color=LIFT_MEAN, ls=":", lw=1.4, label="Mean-of-$K$"),
        plt.Line2D([0], [0], color=LIFT_BEST, ls="--", lw=1.4, label="Best-of-$K$"),
    ]
    legend_y = 1.02 if suptitle_sub else 1.0
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, legend_y), frameon=False)
    if suptitle_sub:
        fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=1.06)
        fig.text(0.5, 1.025, suptitle_sub, ha="center", va="top", fontsize=9.5, color="#444444")
    else:
        fig.suptitle(suptitle, fontsize=11, y=1.03)
    if bold_header:
        fig.subplots_adjust(top=0.84, hspace=0.82, wspace=0.22)
    else:
        fig.tight_layout()
    _save(fig, out / stem)
    pd.DataFrame(stats).to_csv(out / f"{stem}.csv", index=False)


def _style_all_y_axes(axes: np.ndarray, *, ylabel: str = "Overall at $f^*$") -> None:
    """Same y-limits and tick labels on every panel (fixes hidden right-column ticks under sharey)."""
    import matplotlib.ticker as mticker

    nrows, ncols = axes.shape
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]
            ax.set_ylim(0.0, 2.05)
            ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
            ax.set_ylabel(ylabel if c == 0 else "")
            ax.tick_params(axis="y", which="both", labelleft=True, labelright=False, left=True)
            ax.grid(axis="y", which="major")
            ax.grid(axis="x", visible=False)


def _exemplar_stats_line(row: pd.Series, role: str) -> str:
    """One-line mean / best / gap summary for an exemplar panel."""
    return (
        f"{role} · mean {row.mean_of_k_overall:.2f} · "
        f"best {row.best_of_k_overall:.2f} · gap {row.oracle_gap:.2f}"
    )


def _plot_four_exemplar_panel(
    fstar: pd.DataFrame,
    tab: pd.DataFrame,
    out: Path,
    *,
    concepts: list[tuple[int, str]],
    stem: str = "particle_spread_exemplars",
) -> None:
    """2×2 panel of exemplar concepts."""
    apply_axbench_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.4), sharey=False)

    stats: list[dict[str, Any]] = []
    for idx, (cid, role) in enumerate(concepts):
        r, c = divmod(idx, 2)
        ax = axes[r, c]
        g = fstar[fstar["concept_id"] == cid]
        row = tab.loc[tab["concept_id"] == cid].iloc[0]
        header = _concept_header(cid, SHORT_LABELS[cid], style="concept_id")
        mean_y, best_y = _draw_particle_spread_panel(ax, g, title="", highlight=False, marker_size=58)
        _set_panel_header(ax, header, _exemplar_stats_line(row, role), bold_header=True)
        if r == axes.shape[0] - 1:
            ax.set_xlabel("Particle", fontsize=9)
        else:
            ax.set_xlabel("")
        stats.append({"concept_id": cid, "role": role, "mean": mean_y, "best": best_y})

    _style_all_y_axes(axes)

    handles = [
        plt.Line2D([0], [0], color=LIFT_MEAN, ls=":", lw=1.4, label="Mean-of-$K$"),
        plt.Line2D([0], [0], color=LIFT_BEST, ls="--", lw=1.4, label="Best-of-$K$"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.suptitle(
        "Steering heterogeneity across four exemplar concepts",
        fontsize=12,
        fontweight="bold",
        y=1.06,
    )
    fig.text(
        0.5,
        1.025,
        "Overall LM-judge score at each particle's $f^*$ (latent-corr repulsion, $K{=}5$)",
        ha="center",
        va="top",
        fontsize=9.5,
        color="#444444",
    )
    fig.subplots_adjust(top=0.80, hspace=0.78, wspace=0.30, bottom=0.10)
    _save(fig, out / stem)
    pd.DataFrame(stats).to_csv(out / f"{stem}.csv", index=False)


def _plot_paper_particle_spread(
    fstar: pd.DataFrame,
    tab: pd.DataFrame,
    out: Path,
) -> None:
    """Main-paper particle spread figures: all-20 grid + exemplar quad."""
    by_spread = tab.sort_values("spread_overall", ascending=False)["concept_id"].astype(int).tolist()

    def gap_sub(row: pd.Series) -> str:
        return f"mean {row.mean_of_k_overall:.2f} · best {row.best_of_k_overall:.2f} · gap {row.oracle_gap:.2f}"

    _plot_particle_spread_grid(
        fstar,
        tab,
        out,
        stem="particle_spread_all20_by_spread",
        concept_order=by_spread,
        ncols=5,
        figsize_per_panel=(2.75, 2.15),
        suptitle="Particle disagreement across 20 SAE steering targets",
        suptitle_sub=(
            "Overall LM-judge score at each particle's $f^*$, sorted by spread "
            "(concepts where particles disagree most appear first)"
        ),
        subtitle_fn=gap_sub,
        highlight=set(),
        header_style="concept_id",
        bold_header=True,
    )
    _plot_four_exemplar_panel(fstar, tab, out, concepts=PAPER_EXEMPLAR_QUAD)


def _plot_appendix_particle_spread(
    fstar: pd.DataFrame,
    tab: pd.DataFrame,
    out: Path,
) -> None:
    """Appendix-only heterogeneity figures (lift scatter, dumbbell, tables)."""
    by_gap = tab.sort_values("oracle_gap", ascending=False)["concept_id"].astype(int).tolist()
    policy = _policy_metrics_from_fstar(fstar)
    _plot_lift_scatter(tab, out)
    _plot_lift_dumbbell(tab, out)
    _plot_exemplar_subscores(policy, out)
    _plot_particle_spread_overview(fstar, tab, out, order=by_gap)
    _write_tables(tab, out)


def _plot_particle_spread_overview(fstar: pd.DataFrame, tab: pd.DataFrame, out: Path, *, order: list[int]) -> None:
    """Single-axis overview: concepts on y, score on x, one dot per particle."""
    apply_axbench_style()
    n = len(order)
    ypos = {cid: i for i, cid in enumerate(order)}
    fig, ax = plt.subplots(figsize=(6.5, max(5.5, 0.28 * n + 1.5)))

    for cid in order:
        g = fstar[fstar["concept_id"] == cid].sort_values("particle_id")
        y = ypos[cid]
        xs = g["overall"].tolist()
        ys = [y] * len(xs)
        pids = g["particle_id"].astype(int).tolist()
        ax.scatter(xs, ys, c=[PARTICLE_PAL[p % len(PARTICLE_PAL)] for p in pids], s=36, zorder=3, edgecolors="white", linewidths=0.3)
        mean_x = g["overall"].mean()
        best_x = g["overall"].max()
        ax.plot([mean_x, best_x], [y, y], color=LIFT_GAIN, lw=2.2, zorder=2, solid_capstyle="round")
        ax.scatter([mean_x], [y], color=LIFT_MEAN, s=22, zorder=4, marker="|")
        ax.scatter([best_x], [y], color=LIFT_BEST, s=40, zorder=4, marker="D", edgecolors="white", linewidths=0.3)

    labels = [f"c{cid:02d} {SHORT_LABELS[cid]}" for cid in order]
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Overall at $f^*$")
    ax.set_xlim(-0.02, 2.05)
    ax.set_ylabel("")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=PARTICLE_PAL[0], markersize=6, label="Particle"),
        plt.Line2D([0], [0], color=LIFT_GAIN, lw=2.2, label="Mean $\\rightarrow$ best"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=LIFT_BEST, markersize=6, label="Best-of-$K$"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=False)
    ax.set_title("Particle spread overview (sorted by oracle gap)", fontsize=10)
    fig.tight_layout()
    _save(fig, out / "fig_particle_spread_overview_by_gap")


def _tex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def _write_tables(tab: pd.DataFrame, out: Path) -> None:
    """LaTeX tables for full concept list and exemplar subset."""
    ex = tab[tab["concept_id"].isin(EXEMPLARS.keys())].copy()
    ex = ex.set_index("concept_id").loc[list(EXEMPLARS.keys())].reset_index()

    lines_full = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Per-concept steering heterogeneity (latent-corr repulsion, $K{=}5$, $n{=}20$ SAE features). "
        "Scores are LM-judge ratings at each particle's $f^*$; Mean/Best aggregate over five particles. "
        "Gap $=$ Best $-$ Mean. Spread is max$-$min overall across particles.}",
        "\\label{tab:heterogeneity-full}",
        "\\small",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}rlcccccc@{}}",
        "\\toprule",
        "ID & Short label & Genre & Mean & Best & Gap & Spread & Pattern \\\\",
        "\\midrule",
    ]
    for _, r in tab.iterrows():
        lines_full.append(
            f"{int(r.concept_id)} & {_tex_escape(r.short_label)} & {r.genre} & "
            f"{r.mean_of_k_overall:.2f} & {r.best_of_k_overall:.2f} & {r.oracle_gap:.2f} & "
            f"{r.spread_overall:.2f} & {_tex_escape(str(r.pattern))} \\\\"
        )
    lines_full += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]

    lines_ex = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Exemplar concepts illustrating three heterogeneity axes: "
        "steering ceiling (Best), selection value (Gap), and subscore profile (Concept vs.\\ Instruction/Fluency at Best-of-$K$).}",
        "\\label{tab:heterogeneity-exemplars}",
        "\\begin{tabular}{@{}l l c c c c c c@{}}",
        "\\toprule",
        "Role & Label & Mean & Best & Gap & $C$ & $I$ & $F$ \\\\",
        "\\midrule",
    ]
    role_names = {"volatile": "High-variance", "hard": "Low-ceiling", "flat": "Low-variance", "easy": "Responsive"}
    for _, r in ex.iterrows():
        role = role_names[EXEMPLARS[int(r.concept_id)]]
        lines_ex.append(
            f"{role} & {_tex_escape(r.short_label)} & "
            f"{r.mean_of_k_overall:.2f} & {r.best_of_k_overall:.2f} & {r.oracle_gap:.2f} & "
            f"{r.best_concept:.2f} & {r.best_instruct:.2f} & {r.best_fluency:.2f} \\\\"
        )
    lines_ex += [
        "\\bottomrule",
        "\\multicolumn{8}{@{}p{0.95\\linewidth}@{}}{\\footnotesize "
        "$C/I/F$: concept, instruction, and fluency subscores for the best particle at $f^*$. "
        "Low-ceiling concepts can remain fluent and on-instruction while failing to express the target feature (e.g., C/C++ syntax).}",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]

    (out / "table_heterogeneity_full.tex").write_text("\n".join(lines_full))
    (out / "table_heterogeneity_exemplars.tex").write_text("\n".join(lines_ex))


def _load_fstar_from_dump(root: Path) -> pd.DataFrame:
    dump = root / "repulsion_latent_corr" / "form_additive" / "w0.5"
    per_row, _ = load_single_arm_per_row(dump)
    return per_particle_fstar_single(mean_over_prompts(per_row))


def run(data_root: Path, out: Path) -> None:
    """Section 4.5 heterogeneity figures from latent-corr eval parquets."""
    root = data_root / "particlesteering"
    meta_path = root / "repulsion_latent_corr" / "form_additive" / "w0.5" / "train" / "rank_0_metadata.jsonl"

    fstar = _load_fstar_from_dump(root)
    mult = multiplicity_summary_single(fstar)
    meta = _load_metadata(meta_path)
    policy = _policy_metrics_from_fstar(fstar)

    out.mkdir(parents=True, exist_ok=True)
    tab = _build_table(mult, meta, policy)
    tab.to_csv(out / "heterogeneity_summary.csv", index=False)
    _plot_paper_particle_spread(fstar, tab, out)

    print(f"Wrote heterogeneity assets to {out}")
    for p in sorted(out.iterdir()):
        print(f"  {p.name}")


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=repo / "data")
    parser.add_argument(
        "--out",
        type=Path,
        default=repo / "paper_outputs" / "figures" / "heterogeneity",
    )
    args = parser.parse_args()
    run(args.data_root, args.out)


if __name__ == "__main__":
    main()

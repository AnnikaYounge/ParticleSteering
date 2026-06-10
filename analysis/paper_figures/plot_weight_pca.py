#!/usr/bin/env python3
# Adapted from AxBench — https://github.com/stanfordnlp/axbench (Apache-2.0).
"""Figure 1: weight-space PCA strip for selected concepts (RBF repulsion, K=5)."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from loaders import find_weight_path
from paper_style import apply_paper_style, particle_palette, save_fig

PAPER_CONCEPT_ORDER = [0, 1, 2, 3]
K_DEFAULT = 5


def discover_repulsion_dump(root: Path, rep: str) -> Path | None:
    rep_dir = root / f"repulsion_{rep}"
    for dump in sorted(rep_dir.rglob("form_additive/w0.5")):
        if find_weight_path(dump) is not None:
            return dump
    return None


def load_concept_labels(dump: Path) -> dict[int, str]:
    labels: dict[int, str] = {}
    for name in ("metadata.jsonl", "rank_0_metadata.jsonl"):
        path = dump / "train" / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            cid = int(rec.get("concept_id", rec.get("id", -1)))
            if cid >= 0:
                labels[cid] = str(rec.get("concept", rec.get("input_concept", f"c{cid}")))
        if labels:
            return labels
    return labels


def load_weight_matrix(path: Path) -> np.ndarray:
    try:
        import torch

        return torch.load(path, map_location="cpu", weights_only=True).float().numpy()
    except Exception:
        with zipfile.ZipFile(path) as zf:
            raw_key = next(k for k in zf.namelist() if k.endswith("/data/0"))
            raw = zf.read(raw_key)
        arr = np.frombuffer(raw, dtype=np.float32)
        n_rows = arr.size // 1152
        return arr.reshape(n_rows, -1)


def weights_by_concept(path: Path, k: int, concept_ids: list[int]) -> dict[int, np.ndarray]:
    w = load_weight_matrix(path)
    return {cid: w[cid * k : (cid + 1) * k].astype(np.float64) for cid in concept_ids}


def unit_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


def pca_coords(x: np.ndarray, n_components: int = 2) -> np.ndarray:
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:n_components].T


def concept_short(label: str, max_len: int = 44) -> str:
    s = label.replace("\n", " ").strip()
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def _fill_hull_2d(ax: plt.Axes, xy: np.ndarray) -> None:
    if xy.shape[0] < 3:
        return
    try:
        from scipy.spatial import ConvexHull

        hull_pts = xy[ConvexHull(xy).vertices]
        hull_pts = np.vstack([hull_pts, hull_pts[0]])
        ax.fill(hull_pts[:, 0], hull_pts[:, 1], color="#cccccc", alpha=0.25, zorder=1)
        ax.plot(hull_pts[:, 0], hull_pts[:, 1], color="#888888", lw=0.8, zorder=2)
    except Exception:
        pass


def _scatter_particles(ax: plt.Axes, xy: np.ndarray, *, k: int) -> None:
    colors = particle_palette(k)
    cx, cy = xy.mean(axis=0)
    _fill_hull_2d(ax, xy)
    for i, (x, y) in enumerate(xy):
        ax.annotate(
            "",
            xy=(x, y),
            xytext=(cx, cy),
            arrowprops=dict(arrowstyle="-|>", color=colors[i % len(colors)], lw=1.4, alpha=0.85),
            zorder=3,
        )
    ax.scatter(xy[:, 0], xy[:, 1], c=colors[: xy.shape[0]], s=70, edgecolors="white", linewidths=0.9, zorder=4)
    for i, (x, y) in enumerate(xy):
        ax.text(x, y, f" p{i}", fontsize=8, va="center", zorder=5)
    ax.scatter([cx], [cy], marker="x", c="black", s=40, zorder=6)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axhline(0, color="#bbbbbb", lw=0.6, zorder=0)
    ax.axvline(0, color="#bbbbbb", lw=0.6, zorder=0)


def plot_concept_strip(
    weights: dict[int, np.ndarray],
    labels: dict[int, str],
    out_path: Path,
    *,
    concept_ids: list[int],
) -> None:
    ids = [c for c in concept_ids if c in weights]
    if not ids:
        return
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    n = len(ids)
    ncols = 2 if n == 4 else min(n, 2)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.9 * nrows))
    for ax, cid, pidx in zip(np.atleast_1d(axes).ravel(), ids, range(n)):
        xy = pca_coords(unit_rows(weights[cid]), 2)
        _scatter_particles(ax, xy, k=weights[cid].shape[0])
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        short = concept_short(labels.get(cid, ""))
        ax.text(0.5, 1.10, f"({letters[pidx]}) Concept {cid}", transform=ax.transAxes, ha="center", fontsize=10, fontweight="bold")
        ax.text(0.5, 1.02, short, transform=ax.transAxes, ha="center", fontsize=9, color="#333333")
    for ax in np.atleast_1d(axes).ravel()[n:]:
        ax.axis("off")
    fig.subplots_adjust(top=0.88, hspace=0.52, wspace=0.28)
    save_fig(fig, out_path)


def run(root: Path, out: Path, *, rep: str = "rbf", k: int = K_DEFAULT, concept_ids: list[int] | None = None) -> Path:
    apply_paper_style()
    dump = discover_repulsion_dump(root, rep)
    if dump is None:
        raise FileNotFoundError(f"No {rep} dump with weights under {root}")
    wp = find_weight_path(dump)
    assert wp is not None
    ids = concept_ids or PAPER_CONCEPT_ORDER
    weights = weights_by_concept(wp, k, ids)
    labels = load_concept_labels(dump)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"weight_pca_concept_strip_{rep}.png"
    plot_concept_strip(weights, labels, path, concept_ids=ids)
    return path


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=repo / "data/particlesteering")
    p.add_argument("--out", type=Path, default=repo / "paper_outputs/figures/geometry")
    p.add_argument("--rep", type=str, default="rbf")
    p.add_argument("--k", type=int, default=K_DEFAULT)
    p.add_argument("--concept-order", type=str, default=",".join(map(str, PAPER_CONCEPT_ORDER)))
    args = p.parse_args()
    ids = [int(x.strip()) for x in args.concept_order.split(",") if x.strip()]
    path = run(args.root.resolve(), args.out.resolve(), rep=args.rep, k=args.k, concept_ids=ids)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

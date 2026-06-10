#!/usr/bin/env python3
# Adapted from AxBench — https://github.com/stanfordnlp/axbench (Apache-2.0).
"""Build main-paper tables (CSV + LaTeX) from eval export parquets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_PKG = Path(__file__).resolve().parent / "paper_figures"
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from loaders import (
    bootstrap_mean_ci,
    judge_cols_for_method,
    mean_over_prompts,
    per_particle_fstar_single,
)

COSINE_SUBSET = [4, 5, 7, 8, 9]

L20_PS_ARMS: list[dict[str, Any]] = [
    {
        "id": "rbf",
        "label": "RBF",
        "table_label": "RBF",
        "path": "particlesteering/repulsion_rbf/form_additive/w0.5",
        "k": 5,
        "particles": None,
        "group": "ours",
        "bold": True,
    },
    {
        "id": "latent_corr",
        "label": "Latent-corr",
        "table_label": "Latent-corr",
        "path": "particlesteering/repulsion_latent_corr/form_additive/w0.5",
        "k": 5,
        "particles": None,
        "group": "ours",
        "bold": True,
    },
    {
        "id": "cosine",
        "label": "Cosine",
        "table_label": "Cosine",
        "path": "particlesteering/repulsion_cosine/form_additive/w0.5",
        "k": 5,
        "particles": COSINE_SUBSET,
        "group": "ours",
        "bold": True,
        "note": "particles {4,5,7,8,9} of K=10",
    },
]

L20_BASELINES = [
    ("baselines/lsreft", "LsReFT", "ReFT-r1", False),
    ("baselines/no_grad", "GemmaScopeSAE", "SAE", False),
    ("baselines/no_grad", "DiffMean", "Diff-in-means", False),
    ("baselines/no_grad", "LAT", "LAT", False),
    ("baselines/no_grad", "PCA", "PCA", False),
    ("baselines/no_grad", "PromptSteering", "Prompting", True),
]


def _load_ps_fstar(root: Path, arm: dict[str, Any]) -> pd.DataFrame:
    """Per-particle f* rows for one Particle-Steering repulsion arm."""
    pq = root / arm["path"] / "evaluate" / "steering_data.parquet"
    df = pd.read_parquet(pq)
    jc = judge_cols_for_method("ParticleSteering")
    per = df[
        ["concept_id", "input_id", "factor", "particle_id"] + list(jc.values())
    ].rename(
        columns={
            jc["overall"]: "overall",
            jc["concept"]: "concept",
            jc["instruct"]: "instruct",
            jc["fluency"]: "fluency",
        }
    )
    particles = arm.get("particles")
    if particles is not None:
        per = per[per["particle_id"].isin(particles)]
    return per_particle_fstar_single(mean_over_prompts(per))


def _load_baseline_fstar(root: Path, dump: str, method_raw: str) -> pd.DataFrame:
    """Per-concept f* row for a single-vector baseline method."""
    pq = root / dump / "evaluate" / "steering_data.parquet"
    df = pd.read_parquet(pq)
    jc = judge_cols_for_method(method_raw)
    per = df[["concept_id", "input_id", "factor"] + list(jc.values())].rename(
        columns={
            jc["overall"]: "overall",
            jc["concept"]: "concept",
            jc["instruct"]: "instruct",
            jc["fluency"]: "fluency",
        }
    )
    per["particle_id"] = 0
    agg = mean_over_prompts(per)
    rows = []
    for _, g in agg.groupby("concept_id"):
        rows.append(g.loc[g["overall"].idxmax()])
    return pd.DataFrame(rows)


def _macro_mean_per_concept(fstar: pd.DataFrame) -> pd.Series:
    """Mean overall score across particles at f*, one value per concept."""
    return fstar.groupby("concept_id")["overall"].mean()


def _mci(mean: float, lo: float, hi: float) -> str:
    """LaTeX macro for mean with bootstrap CI brackets."""
    return f"\\mci{{{mean:.2f}}}{{{lo:.2f}}}{{{hi:.2f}}}"


def _quality_metrics(fstar: pd.DataFrame) -> dict[str, Any]:
    """Macro mean per metric plus worst/best particle range for multi-particle arms."""
    out: dict[str, Any] = {}
    for metric in ("overall", "concept", "instruct", "fluency"):
        per_p = fstar.groupby("particle_id")[metric].mean()
        out[metric] = float(per_p.mean())
        if len(per_p) > 1:
            out[f"{metric}_lo"] = float(per_p.min())
            out[f"{metric}_hi"] = float(per_p.max())
    return out


def build_l20_tables(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """Layer-20 overall and per-metric tables with LaTeX fragments."""
    overall_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []

    for arm in L20_PS_ARMS:
        fstar = _load_ps_fstar(root, arm)
        macro = _macro_mean_per_concept(fstar)
        m, lo, hi = bootstrap_mean_ci(macro.to_numpy())
        q = _quality_metrics(fstar)
        overall_rows.append(
            {
                "method": arm["table_label"],
                "k": arm["k"],
                "mean": m,
                "ci_lo": lo,
                "ci_hi": hi,
                "group": arm["group"],
                "bold": arm.get("bold", False),
                "note": arm.get("note", ""),
            }
        )
        quality_rows.append({"method": arm["table_label"], "k": arm["k"], "group": "ours", **q})

    for dump, raw, label, is_prompt in L20_BASELINES:
        fstar = _load_baseline_fstar(root, dump, raw)
        m, lo, hi = bootstrap_mean_ci(fstar["overall"].to_numpy())
        overall_rows.append(
            {
                "method": label,
                "k": None if is_prompt else 1,
                "mean": m,
                "ci_lo": lo,
                "ci_hi": hi,
                "group": "prompt" if is_prompt else "baseline",
                "bold": False,
                "note": "",
            }
        )
        quality_rows.append(
            {
                "method": label,
                "k": None if is_prompt else 1,
                "group": "prompt" if is_prompt else "baseline",
                "overall": float(fstar["overall"].mean()),
                "concept": float(fstar["concept"].mean()),
                "instruct": float(fstar["instruct"].mean()),
                "fluency": float(fstar["fluency"].mean()),
            }
        )

    overall_df = pd.DataFrame(overall_rows).sort_values("mean", ascending=False)
    quality_df = pd.DataFrame(quality_rows)
    overall_pub = overall_df[~overall_df["method"].isin(["No repulsion"])].copy()

    # tab:main-overall (exclude K=1 no-repulsion; it lives in tab:main-quality only)
    tex_overall = [
        "\\begin{table}[H]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Mean overall steering quality on AxBench (Gemma-2-2B, layer~20, 20 concepts). "
        "Overall is the harmonic mean of concept, instruction-following, and fluency ($0$--$2$). "
        "Rows sorted by mean overall; brackets are 95\\% bootstrap CIs over concepts. "
        "$f^*$ chosen per particle (per concept for baselines) on the eval factor grid. "
        "Cosine: five-particle subset $\\{4,5,7,8,9\\}$ of $K{=}10$.}",
        "\\label{tab:main-overall}",
        "\\begin{tabular}{@{}l c r@{}}",
        "\\toprule",
        "Method & $K$ & Overall \\\\",
        "\\midrule",
    ]
    for _, r in overall_pub.iterrows():
        k = "---" if r["k"] is None or pd.isna(r["k"]) else str(int(r["k"]))
        name = f"\\textbf{{{r['method']}}}" if r["bold"] else r["method"]
        tex_overall.append(f"{name} & {k} & {_mci(r['mean'], r['ci_lo'], r['ci_hi'])} \\\\")
    tex_overall += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]

    tex_quality = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Per-particle steering quality on AxBench (Gemma-2-2B, layer~20, 20 concepts). "
        "Overall is the harmonic mean of concept, instruction-following, and fluency (each $0$--$2$), "
        "with the steering factor selected per concept. For \\textsc{Particle-Steering} ($K{>}1$), "
        "Overall is the mean over particles with the [worst, best] particle in brackets; "
        "concept, instruction, and fluency are means over particles. Baselines are single vectors. "
        "Cosine: particles $\\{4,5,7,8,9\\}$ of $K{=}10$.}",
        "\\label{tab:main-quality}",
        "\\begin{tabular}{@{}l c c ccc@{}}",
        "\\toprule",
        "Method & $K$ & Overall & Concept & Instr. & Flu. \\\\",
        "\\midrule",
        "\\multicolumn{6}{@{}l}{\\textit{\\textsc{Particle-Steering} (ours)}}\\\\",
    ]
    for _, r in quality_df[quality_df["group"] == "ours"].iterrows():
        overall = f"\\mci{{{r['overall']:.2f}}}{{{r.get('overall_lo', r['overall']):.2f}}}{{{r.get('overall_hi', r['overall']):.2f}}}" if pd.notna(r.get("overall_lo")) else f"{r['overall']:.2f}"
        tex_quality.append(
            f"\\quad {r['method']} & {int(r['k'])} & {overall} & {r['concept']:.2f} & {r['instruct']:.2f} & {r['fluency']:.2f} \\\\"
        )
    tex_quality += ["\\midrule", "\\multicolumn{6}{@{}l}{\\textit{Single-vector baselines}}\\\\"]
    for _, r in quality_df[quality_df["group"] == "baseline"].iterrows():
        tex_quality.append(
            f"\\quad {r['method']} & 1 & {r['overall']:.2f} & {r['concept']:.2f} & {r['instruct']:.2f} & {r['fluency']:.2f} \\\\"
        )
    tex_quality += ["\\midrule", "\\multicolumn{6}{@{}l}{\\textit{Non-steering reference}}\\\\"]
    for _, r in quality_df[quality_df["group"] == "prompt"].iterrows():
        tex_quality.append(
            f"\\quad {r['method']} & --- & {r['overall']:.2f} & {r['concept']:.2f} & {r['instruct']:.2f} & {r['fluency']:.2f} \\\\"
        )
    tex_quality += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]

    return overall_df, quality_df, "\n".join(tex_overall), "\n".join(tex_quality)


def build_prose_snippets(overall_df: pd.DataFrame, quality_df: pd.DataFrame) -> str:
    """Short LaTeX paragraph summarizing Table 1 headline numbers."""
    rbf = overall_df.loc[overall_df["method"] == "RBF"].iloc[0]
    latent = overall_df.loc[overall_df["method"] == "Latent-corr"].iloc[0]
    reft = overall_df.loc[overall_df["method"] == "ReFT-r1"].iloc[0]
    cosine = overall_df.loc[overall_df["method"] == "Cosine"].iloc[0]

    return f"""Training under repulsion does not cost steering quality (\\cref{{tab:main-overall}}).
Under latent-score repulsion the mean particle reaches an overall score of ${latent['mean']:.2f}$, matching ReFT-r1 (${reft['mean']:.2f}$), with cosine repulsion close behind (${cosine['mean']:.2f}$) and RBF competitive (${rbf['mean']:.2f}$).
Every repulsion arm clears the localized baselines by a wide margin (SAE ${overall_df.loc[overall_df['method']=='SAE','mean'].iloc[0]:.2f}$, difference-in-means ${overall_df.loc[overall_df['method']=='Diff-in-means','mean'].iloc[0]:.2f}$).
Prompting scores higher than any steering method (${overall_df.loc[overall_df['method']=='Prompting','mean'].iloc[0]:.2f}$); we report it as a non-steering reference.
"""


def run(root: Path, out: Path) -> None:
    """Write all table CSVs, TeX, and validity prose snippet to ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    overall_df, quality_df, tex_overall, tex_quality = build_l20_tables(root)
    prose = build_prose_snippets(overall_df, quality_df)

    overall_df.to_csv(out / "table_main_overall.csv", index=False)
    quality_df.to_csv(out / "table_main_quality.csv", index=False)
    (out / "table_main_overall.tex").write_text(tex_overall)
    (out / "table_main_quality.tex").write_text(tex_quality)
    (out / "section_validity.tex").write_text(prose)
    print(f"Wrote {out}")
    print(overall_df.to_string(index=False))


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=repo / "data")
    p.add_argument("--out-dir", type=Path, default=repo / "paper_outputs" / "tables")
    args = p.parse_args()
    run(args.data_root.resolve(), args.out_dir.resolve())


if __name__ == "__main__":
    main()

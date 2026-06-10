# Adapted from AxBench — https://github.com/stanfordnlp/axbench (Apache-2.0).
"""Shared loaders and aggregation policies for paper figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

METHOD_MAP = {
    "LsReFT": "ReFT-r1",
    "SteeringVector": "SSV",
    "SparseLinearProbe": "Probe-SL",
    "DiffMean": "DiffMean",
    "PCA": "PCA",
    "LAT": "LAT",
    "GemmaScopeSAE": "SAE",
    "IntegratedGradients": "IG",
    "InputXGradients": "IxG",
    "LinearProbe": "Probe",
    "PromptSteering": "Prompt",
    "PromptDetection": "Prompt",
    "LoReFT": "LoReFT",
    "LoRA": "LoRA",
    "SFT": "SFT",
    "GemmaScopeSAEMaxAUC": "SAE-A",
    "BoW": "BoW",
    "ParticleSteering": "ParticleSteering",
    "PreferenceVector": "PreferenceVector",
}

# Older eval exports may still use this method prefix (pre-rename checkpoints).
LEGACY_STEERING_METHODS = ("PreferenceSet",)

MODEL_MAP = {"2b": "Gemma-2-2B", "9b": "Gemma-2-9B"}
LAYER_MAP = {"l10": "L10", "l20": "L20", "l31": "L31"}

METRIC_RENAME = {
    "max_lm_judge_rating": "Overall Score",
    "max_fluency_rating": "Fluency Score",
    "max_relevance_concept_rating": "Concept Score",
    "max_relevance_instruction_rating": "Instruct Score",
    "max_factor": "Steering Factor",
    "roc_auc": "AUROC",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
}

STEERING_METRICS = [
    "Overall Score",
    "Concept Score",
    "Instruct Score",
    "Fluency Score",
]

NO_FACTOR = {"Prompt", "LoReFT", "LoRA", "SFT"}
BASELINE_IGNORE = {"SAE-c", "Prompt"}
BASELINE_ORDER = ["DiffMean", "PCA", "LAT", "SAE", "Probe", "ReFT-r1"]

# Baseline steering f*: argmax mean concept score over factors per (concept, method).
BASELINE_FSTAR_METRIC = "concept"

REPULSION_LABELS = {
    "none": "none",
    "cosine": "cosine",
    "rbf": "rbf",
    "latent_corr": "latent_corr",
}
REPULSION_COLORS = {
    "none": "#d62728",
    "cosine": "#1f77b4",
    "rbf": "#2ca02c",
    "latent_corr": "#9467bd",
}

RUN_METHODS: list[tuple[str, list[str]]] = [
    ("no_grad", ["DiffMean", "PCA", "LAT", "GemmaScopeSAE", "PromptSteering"]),
    ("probe", ["LinearProbe"]),
    ("lsreft", ["LsReFT"]),
]

STEERING_METHOD_CANDIDATES = (
    "ParticleSteering",
    *LEGACY_STEERING_METHODS,
    "PreferenceVector",
)


def judge_cols_for_method(method: str) -> dict[str, str]:
    """LM-judge parquet column names for a steering method prefix."""
    return {
        "overall": f"{method}_LMJudgeEvaluator",
        "concept": f"{method}_LMJudgeEvaluator_relevance_concept_ratings",
        "instruct": f"{method}_LMJudgeEvaluator_relevance_instruction_ratings",
        "fluency": f"{method}_LMJudgeEvaluator_fluency_ratings",
    }


def discover_steering_method(root: Path) -> str | None:
    """Detect LM-judge method prefix from evaluate parquet or steering.jsonl."""
    pq = root / "evaluate" / "steering_data.parquet"
    if pq.exists():
        cols = set(pd.read_parquet(pq).columns)
        for method in STEERING_METHOD_CANDIDATES:
            if judge_cols_for_method(method)["overall"] in cols:
                return method
    for rec in load_jsonl(root / "evaluate" / "steering.jsonl"):
        judge = rec.get("results", {}).get("LMJudgeEvaluator", {})
        for method in STEERING_METHOD_CANDIDATES:
            if method in judge:
                return method
    return None


def find_weight_path(dump: Path) -> Path | None:
    train = dump / "train"
    if not train.is_dir():
        return None
    for name in ("ParticleSteering_weight.pt",):
        p = train / name
        if p.exists():
            return p
    legacy = train / "PreferenceSet_weight.pt"
    if legacy.exists():
        return legacy
    matches = sorted(train.glob("*_weight.pt"))
    return matches[0] if matches else None


def load_single_arm_curves(root: Path) -> tuple[pd.DataFrame, str | None]:
    """Factor curves from steering.jsonl (mean over prompts in jsonl aggregation)."""
    method = discover_steering_method(root)
    if method is None:
        return pd.DataFrame(), None
    rows: list[dict[str, Any]] = []
    for rec in load_jsonl(root / "evaluate" / "steering.jsonl"):
        ps = rec["results"]["LMJudgeEvaluator"][method]
        cid = int(rec.get("concept_id", 0))
        for f, concept, instruct, fluency, agg in zip(
            ps["factor"],
            ps["relevance_concept_ratings"],
            ps["relevance_instruction_ratings"],
            ps["fluency_ratings"],
            ps["lm_judge_rating"],
        ):
            rows.append(
                {
                    "concept_id": cid,
                    "factor": float(f),
                    "concept": float(concept),
                    "instruct": float(instruct),
                    "fluency": float(fluency),
                    "overall": float(agg),
                }
            )
    return pd.DataFrame(rows), method


def load_single_arm_per_row(root: Path) -> tuple[pd.DataFrame, str | None]:
    path = root / "evaluate" / "steering_data.parquet"
    if not path.exists():
        return pd.DataFrame(), None
    method = discover_steering_method(root)
    if method is None:
        return pd.DataFrame(), None
    jc = judge_cols_for_method(method)
    if jc["overall"] not in pd.read_parquet(path).columns:
        return pd.DataFrame(), None
    df = pd.read_parquet(path)
    cols = ["concept_id", "factor", "particle_id", jc["overall"], jc["concept"], jc["instruct"], jc["fluency"]]
    if "input_id" in df.columns:
        cols.insert(3, "input_id")
    out = df[cols].copy().rename(
        columns={
            jc["overall"]: "overall",
            jc["concept"]: "concept",
            jc["instruct"]: "instruct",
            jc["fluency"]: "fluency",
        }
    )
    out["method"] = method
    return out, method


def mean_over_prompts(per_row: pd.DataFrame) -> pd.DataFrame:
    """Aggregate eval rows: mean judge scores per (concept, particle, factor)."""
    if per_row.empty:
        return per_row
    keys = ["concept_id", "particle_id", "factor"]
    return (
        per_row.groupby(keys, as_index=False)[["overall", "concept", "instruct", "fluency"]]
        .mean()
        .sort_values(keys)
    )


def per_particle_fstar_single(agg: pd.DataFrame) -> pd.DataFrame:
    """One row per (concept_id, particle_id) with f* and scores at f*."""
    if agg.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (cid, pid), g in agg.groupby(["concept_id", "particle_id"]):
        idx = g["overall"].idxmax()
        best = g.loc[idx]
        rows.append(
            {
                "concept_id": int(cid),
                "particle_id": int(pid),
                "f_star": float(best["factor"]),
                "overall": float(best["overall"]),
                "concept": float(best["concept"]),
                "instruct": float(best["instruct"]),
                "fluency": float(best["fluency"]),
            }
        )
    return pd.DataFrame(rows)


def concept_fstar_axbench(curves_jsonl: pd.DataFrame) -> pd.DataFrame:
    """AxBench/analyse.ipynb f*: argmax overall on steering.jsonl curve per concept.

    Matches ``max_lm_judge_rating_idx = argmax(lm_judge_rating)`` in analyse.ipynb.
    """
    if curves_jsonl.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for cid, g in curves_jsonl.groupby("concept_id"):
        idx = g["overall"].idxmax()
        best = g.loc[idx]
        rows.append(
            {
                "concept_id": int(cid),
                "f_star_concept": float(best["factor"]),
                "overall": float(best["overall"]),
                "concept": float(best["concept"]),
                "instruct": float(best["instruct"]),
                "fluency": float(best["fluency"]),
            }
        )
    return pd.DataFrame(rows)


def scores_at_concept_fstar(agg: pd.DataFrame, concept_fstar: pd.DataFrame) -> pd.DataFrame:
    """Particle scores at the shared per-concept f* (AxBench boxplot policy)."""
    if agg.empty or concept_fstar.empty:
        return pd.DataFrame()
    merged = agg.merge(concept_fstar[["concept_id", "f_star_concept"]], on="concept_id", how="inner")
    return merged[merged["factor"] == merged["f_star_concept"]].drop(columns=["f_star_concept"])


def multiplicity_summary_single(fstar: pd.DataFrame) -> pd.DataFrame:
    """Per-concept best/mean/spread/oracle-gap from per-particle f* scores."""
    rows: list[dict[str, Any]] = []
    for cid, g in fstar.groupby("concept_id"):
        rows.append(
            {
                "concept_id": int(cid),
                "best_of_k_overall": float(g["overall"].max()),
                "mean_of_k_overall": float(g["overall"].mean()),
                "best_of_k_concept": float(g["concept"].max()),
                "spread_overall": float(g["overall"].max() - g["overall"].min()),
                "oracle_gap": float(g["overall"].max() - g["overall"].mean()),
                "n_particles": len(g),
            }
        )
    return pd.DataFrame(rows)


def aggregate_factor_curves_single(agg: pd.DataFrame) -> pd.DataFrame:
    """Mean judge scores at each steering factor (macro over concepts and particles)."""
    if agg.empty:
        return pd.DataFrame()
    return (
        agg.groupby("factor", as_index=False)[["concept", "instruct", "fluency", "overall"]]
        .mean()
        .sort_values("factor")
    )


def particle_stats_by_factor_single(agg: pd.DataFrame) -> pd.DataFrame:
    """Concept-score mean/min/max/std per factor."""
    if agg.empty:
        return pd.DataFrame()
    return (
        agg.groupby("factor")["concept"]
        .agg(mean="mean", min="min", max="max", std="std")
        .reset_index()
    )


def load_latent_concept_summary(root: Path, method: str | None = None) -> pd.DataFrame:
    """Mean peak activation per concept from inference latent parquet."""
    path = root / "inference" / "latent_data.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "concept_id" not in df.columns:
        return pd.DataFrame()
    method = method or discover_steering_method(root)
    act_col = None
    tok_col = None
    if method:
        act_col = f"{method}_max_act"
        tok_col = f"{method}_max_token"
    if act_col and act_col not in df.columns:
        act_col = next((c for c in df.columns if c.endswith("_max_act")), None)
    if not act_col or act_col not in df.columns:
        return pd.DataFrame()
    cols = ["concept_id", act_col]
    if tok_col and tok_col in df.columns:
        cols.append(tok_col)
    return df[cols].groupby("concept_id", as_index=False).mean(numeric_only=True)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file as a list of dicts (empty if missing)."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def best_idx(ratings: list[float] | np.ndarray) -> int:
    """Index of maximum rating."""
    return int(np.argmax(ratings))


def baseline_fstar_idx(ps: dict[str, Any]) -> int:
    """Per (concept, method): argmax factor by mean concept score on the eval curve."""
    if BASELINE_FSTAR_METRIC == "overall":
        return best_idx(ps["lm_judge_rating"])
    return best_idx(ps["relevance_concept_ratings"])


def resolve_baselines_root(root: Path) -> Path:
    """Return first root with evaluate steering.jsonl under no_grad, probe, or lsreft."""
    candidates = [
        root,
        Path("results/old_contrastive/n20_baselines/l20_n20"),
    ]
    for cand in candidates:
        cand = cand.resolve()
        for run_dir, _ in RUN_METHODS:
            if (cand / run_dir / "evaluate" / "steering.jsonl").exists():
                return cand
    return root.resolve()


def baseline_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean / median / std per method × metric at concept-f*."""
    rows: list[dict[str, Any]] = []
    sub = df[~df["Method"].isin(BASELINE_IGNORE)]
    for method, g in sub.groupby("Method"):
        for metric in STEERING_METRICS:
            if metric not in g.columns:
                continue
            vals = g[metric].astype(float)
            rows.append(
                {
                    "Method": method,
                    "metric": metric,
                    "mean": float(vals.mean()),
                    "median": float(vals.median()),
                    "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                    "n_concepts": len(g),
                }
            )
    return pd.DataFrame(rows)


def discover_repulsion_eval_dumps(root: Path) -> list[tuple[str, Path]]:
    """One eval dump per repulsion_* arm (first form_additive/w0.5 with steering.jsonl)."""
    dumps: list[tuple[str, Path]] = []
    for rep_dir in sorted(root.glob("repulsion_*")):
        if not rep_dir.is_dir():
            continue
        rep = rep_dir.name.replace("repulsion_", "")
        for dump in sorted(rep_dir.rglob("form_additive/w0.5")):
            if (dump / "evaluate" / "steering.jsonl").exists():
                dumps.append((rep, dump))
                break
    return dumps


def load_arm_curves_jsonl(dump: Path) -> pd.DataFrame:
    """Factor curves from steering.jsonl for ParticleSteering."""
    rows: list[dict[str, Any]] = []
    for rec in load_jsonl(dump / "evaluate" / "steering.jsonl"):
        ps = rec["results"]["LMJudgeEvaluator"]["ParticleSteering"]
        cid = int(rec.get("concept_id", 0))
        for f, concept, instruct, fluency, agg in zip(
            ps["factor"],
            ps["relevance_concept_ratings"],
            ps["relevance_instruction_ratings"],
            ps["fluency_ratings"],
            ps["lm_judge_rating"],
        ):
            rows.append(
                {
                    "concept_id": cid,
                    "factor": float(f),
                    "concept": float(concept),
                    "instruct": float(instruct),
                    "fluency": float(fluency),
                    "overall": float(agg),
                }
            )
    return pd.DataFrame(rows)


def load_arm_per_row(dump: Path, repulsion: str) -> pd.DataFrame:
    """Per-row eval scores from steering_data.parquet for one repulsion arm."""
    per_row, _ = load_single_arm_per_row(dump)
    if per_row.empty:
        return per_row
    out = per_row.copy()
    out["repulsion"] = repulsion
    return out


def aggregate_factor_curves(per_row: pd.DataFrame) -> pd.DataFrame:
    """Mean over prompts and particles per (repulsion, factor)."""
    if per_row.empty:
        return pd.DataFrame()
    gcols = ["repulsion", "factor"]
    return (
        per_row.groupby(gcols, as_index=False)[["concept", "instruct", "fluency", "overall"]]
        .mean()
        .sort_values(gcols)
    )


def particle_stats_by_factor(per_row: pd.DataFrame) -> pd.DataFrame:
    """Concept-score stats per (repulsion, factor)."""
    if per_row.empty:
        return pd.DataFrame()
    return (
        per_row.groupby(["repulsion", "factor"])["concept"]
        .agg(mean="mean", min="min", max="max", std="std")
        .reset_index()
    )


def per_particle_fstar_table(per_row: pd.DataFrame) -> pd.DataFrame:
    """One row per (repulsion, concept_id, particle_id) with f* and scores at f*."""
    if per_row.empty:
        return pd.DataFrame()
    prompt_cols = ["input_id"] if "input_id" in per_row.columns else []
    agg = (
        per_row.groupby(["repulsion", "concept_id", "particle_id", "factor"], as_index=False)[
            ["overall", "concept", "instruct", "fluency"]
        ]
        .mean()
    )
    rows: list[dict[str, Any]] = []
    for keys, g in agg.groupby(["repulsion", "concept_id", "particle_id"]):
        rep, cid, pid = keys
        idx = g["overall"].idxmax()
        best = g.loc[idx]
        rows.append(
            {
                "repulsion": rep,
                "concept_id": int(cid),
                "particle_id": int(pid),
                "f_star": float(best["factor"]),
                "overall": float(best["overall"]),
                "concept": float(best["concept"]),
                "instruct": float(best["instruct"]),
                "fluency": float(best["fluency"]),
            }
        )
    return pd.DataFrame(rows)


def multiplicity_summary(fstar: pd.DataFrame) -> pd.DataFrame:
    """Per repulsion × concept: best-of-K, mean-of-K, oracle, spread."""
    rows: list[dict[str, Any]] = []
    for (rep, cid), g in fstar.groupby(["repulsion", "concept_id"]):
        rows.append(
            {
                "repulsion": rep,
                "concept_id": int(cid),
                "best_of_k_overall": float(g["overall"].max()),
                "mean_of_k_overall": float(g["overall"].mean()),
                "best_of_k_concept": float(g["concept"].max()),
                "spread_overall": float(g["overall"].max() - g["overall"].min()),
                "oracle_gap": float(g["overall"].max() - g["overall"].mean()),
                "n_particles": len(g),
            }
        )
    return pd.DataFrame(rows)


def load_k1_curves(k1_root: Path) -> pd.DataFrame:
    """Factor curves for a K=1 (no repulsion) run."""
    path = k1_root / "evaluate" / "steering.jsonl"
    if not path.exists():
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for rec in load_jsonl(path):
        judge = rec["results"]["LMJudgeEvaluator"]
        method = next(iter(judge))
        ps = judge[method]
        cid = int(rec.get("concept_id", 0))
        for f, concept, instruct, fluency, agg in zip(
            ps["factor"],
            ps["relevance_concept_ratings"],
            ps["relevance_instruction_ratings"],
            ps["fluency_ratings"],
            ps["lm_judge_rating"],
        ):
            rows.append(
                {
                    "concept_id": cid,
                    "factor": float(f),
                    "concept": float(concept),
                    "instruct": float(instruct),
                    "fluency": float(fluency),
                    "overall": float(agg),
                }
            )
    return pd.DataFrame(rows)


def k1_score_at_fstar(curves: pd.DataFrame) -> dict[str, float]:
    """Scores at argmax-overall factor for a single-concept K=1 curve."""
    if curves.empty:
        return {}
    idx = curves["overall"].idxmax()
    row = curves.loc[idx]
    return {
        "f_star": float(row["factor"]),
        "overall": float(row["overall"]),
        "concept": float(row["concept"]),
        "fluency": float(row["fluency"]),
        "instruct": float(row["instruct"]),
    }


def weight_geometry(dump: Path) -> dict[str, float] | None:
    """Pairwise cosine spread summary for particle weight rows."""
    if torch is None:
        return None
    wp = find_weight_path(dump)
    if wp is None:
        return None
    w = torch.load(wp, map_location="cpu", weights_only=True).float()
    wn = w / w.norm(dim=1, keepdim=True).clamp(min=1e-8)
    cos = wn @ wn.T
    k = int(cos.shape[0])
    off = cos[~torch.eye(k, dtype=bool)]
    return {
        "n_particles": k,
        "pairwise_cos_mean": float(off.mean()),
        "pairwise_cos_max": float(off.max()),
        "pairwise_cos_min": float(off.min()),
    }


def weight_cos_matrix(dump: Path) -> np.ndarray | None:
    """Unit-normalized particle weight cosine similarity matrix."""
    if torch is None:
        return None
    wp = find_weight_path(dump)
    if wp is None:
        return None
    w = torch.load(wp, map_location="cpu", weights_only=True).float()
    wn = w / w.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return (wn @ wn.T).cpu().numpy()


def load_latent_peak_summary(dump: Path) -> pd.DataFrame:
    """Peak activation per (concept, particle) from latent inference."""
    path = dump / "inference" / "latent_data.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "particle_id" not in df.columns:
        return pd.DataFrame()
    method = discover_steering_method(dump)
    act = f"{method}_max_act" if method else None
    tok = f"{method}_max_token" if method else None
    if act and act not in df.columns:
        act = next((c for c in df.columns if c.endswith("_max_act")), None)
    if not act or act not in df.columns:
        return pd.DataFrame()
    cols = ["concept_id", "particle_id", act]
    if tok and tok in df.columns:
        cols.append(tok)
    num = df[cols].groupby(["concept_id", "particle_id"], as_index=False).mean(numeric_only=True)
    if tok and tok in df.columns:
        tok_df = df.groupby(["concept_id", "particle_id"], as_index=False)[tok].first()
        num = num.merge(tok_df, on=["concept_id", "particle_id"], how="left")
    return num


def build_baseline_steering_df(root: Path, split: str = "n20") -> pd.DataFrame:
    """Baseline steering scores at per-concept f* in analyse.ipynb format."""
    rows: list[dict[str, Any]] = []
    for run_dir, methods in RUN_METHODS:
        dump = root / run_dir
        for rec in load_jsonl(dump / "evaluate" / "steering.jsonl"):
            cid = int(rec["concept_id"])
            judge = rec.get("results", {}).get("LMJudgeEvaluator", {})
            for method in methods:
                if method not in judge:
                    continue
                ps = judge[method]
                factors = list(ps["factor"])
                ratings = list(ps["lm_judge_rating"])
                concept = list(ps["relevance_concept_ratings"])
                instruct = list(ps["relevance_instruction_ratings"])
                fluency = list(ps["fluency_ratings"])
                idx = baseline_fstar_idx(ps)
                rows.append(
                    {
                        "concept_id": cid,
                        "method": METHOD_MAP.get(method, method),
                        "model": MODEL_MAP["2b"],
                        "layer": LAYER_MAP["l20"],
                        "split": split,
                        "factor": factors,
                        "lm_judge_rating": ratings,
                        "relevance_concept_ratings": concept,
                        "relevance_instruction_ratings": instruct,
                        "fluency_ratings": fluency,
                        "max_lm_judge_rating": float(ratings[idx]),
                        "max_relevance_concept_rating": float(concept[idx]),
                        "max_relevance_instruction_rating": float(instruct[idx]),
                        "max_fluency_rating": float(fluency[idx]),
                        "max_factor": float(factors[idx]),
                    }
                )
    df = pd.DataFrame(rows)
    return df.rename(columns=METRIC_RENAME)


def expand_factor_rows(df: pd.DataFrame) -> pd.DataFrame:
    """One row per factor from nested list columns in baseline steering df."""
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        factors = row["factor"]
        if not isinstance(factors, list):
            factors = [factors]
        method = row.get("Method", row.get("method"))
        for i, fac in enumerate(factors):
            rows.append(
                {
                    "concept_id": row["concept_id"],
                    "Method": method,
                    "model": row["model"],
                    "layer": row["layer"],
                    "Steering Factor": float(fac),
                    "Overall Score": float(row["lm_judge_rating"][i]),
                    "Concept Score": float(row["relevance_concept_ratings"][i]),
                    "Instruct Score": float(row["relevance_instruction_ratings"][i]),
                    "Fluency Score": float(row["fluency_ratings"][i]),
                }
            )
    return pd.DataFrame(rows)


def load_latent_rows(root: Path, methods_by_run: list[tuple[str, list[str]]] | None = None) -> pd.DataFrame:
    """Concept-detection metrics from baseline latent.jsonl exports."""
    rows: list[dict[str, Any]] = []
    for run_dir, methods in methods_by_run or RUN_METHODS:
        for rec in load_jsonl(root / run_dir / "evaluate" / "latent.jsonl"):
            cid = int(rec["concept_id"])
            results = rec.get("results", {})
            for method in methods:
                if method not in results:
                    continue
                for evaluator, metrics in results[method].items():
                    if not isinstance(metrics, dict):
                        continue
                    row: dict[str, Any] = {
                        "concept_id": cid,
                        "method": METHOD_MAP.get(method, method),
                        "evaluator": evaluator,
                    }
                    for k, v in metrics.items():
                        if k != "roc_curve":
                            row[k] = v
                    rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).rename(columns=METRIC_RENAME)


def count_distinct_factors(root: Path) -> dict[str, int]:
    """Number of steering factors per baseline method in eval jsonl."""
    out: dict[str, int] = {}
    for run_dir, methods in RUN_METHODS:
        recs = load_jsonl(root / run_dir / "evaluate" / "steering.jsonl")
        if not recs:
            continue
        judge = recs[0]["results"]["LMJudgeEvaluator"]
        for method in methods:
            if method in judge:
                out[run_dir + "/" + method] = len(judge[method]["factor"])
    return out


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95% percentile CI from bootstrap resampling over ``values``."""
    rng = np.random.default_rng(seed)
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(x) == 1:
        return float(x[0]), float(x[0]), float(x[0])
    means = [float(rng.choice(x, size=len(x), replace=True).mean()) for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


# Backward-compatible aliases (older scripts used "smoke" naming).
discover_smoke_dumps = discover_repulsion_eval_dumps
load_smoke_curves_jsonl = load_arm_curves_jsonl
load_smoke_per_row = load_arm_per_row

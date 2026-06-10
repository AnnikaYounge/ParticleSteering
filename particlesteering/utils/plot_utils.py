# Adapted from AxBench — https://github.com/stanfordnlp/axbench (Apache-2.0).
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotnine import *

logger = logging.getLogger(__name__)


def plot_metrics(jsonl_data, configs, write_to_path=None, report_to=None, wandb_name=None, mode=None):
    data = []
    for config in configs:
        evaluator_name = config["evaluator_name"]
        metric_name = config["metric_name"]
        y_label = config["y_label"]
        use_log_scale = config["use_log_scale"]

        for entry in jsonl_data:
            results = entry.get("results", {}).get(evaluator_name, {})
            for method, res in results.items():
                factors = res.get("factor", [])
                metrics = res.get(metric_name, [])
                if not isinstance(factors, list):
                    factors = [factors]
                if not isinstance(metrics, list):
                    metrics = [metrics]
                for f, m in zip(factors, metrics):
                    data.append({
                        "Factor": f,
                        "Value": m,
                        "Method": method,
                        "Metric": y_label,
                        "UseLogScale": use_log_scale,
                    })

    df = pd.DataFrame(data)
    df = df.groupby(["Method", "Factor", "Metric", "UseLogScale"], as_index=False).mean()
    df["TransformedValue"] = df.apply(
        lambda row: np.log10(row["Value"]) if row["UseLogScale"] else row["Value"],
        axis=1,
    )

    p = (
        ggplot(df, aes(x="Factor", y="TransformedValue", color="Method", group="Method"))
        + geom_line()
        + geom_point()
        + theme_bw()
        + labs(x="Factor", y="Value")
        + facet_wrap("~ Metric", scales="free_y", nrow=1)
        + theme(
            subplots_adjust={"wspace": 0.1},
            figure_size=(1.5 * len(configs), 3),
            legend_position="right",
            legend_title=element_text(size=4),
            legend_text=element_text(size=6),
            axis_title=element_text(size=6),
            axis_text=element_text(size=6),
            axis_text_x=element_text(rotation=90, hjust=1),
            strip_text=element_text(size=6),
        )
    )

    if write_to_path:
        p.save(filename=str(write_to_path / f"{mode}_plot.png"), dpi=300, bbox_inches="tight")
    elif report_to is not None and "wandb" in report_to:
        import wandb

        line_series_plots = {}
        for metric in df["Metric"].unique():
            metric_data = df[df["Metric"] == metric]
            xs = metric_data["Factor"].unique().tolist()
            ys = [
                metric_data[metric_data["Method"] == method]["TransformedValue"].tolist()
                for method in metric_data["Method"].unique()
            ]
            keys = [f"{method}" for method in metric_data["Method"].unique()]
            line_series_plots[f"{mode}/{metric}"] = wandb.plot.line_series(
                xs=xs, ys=ys, keys=keys, title=f"{metric}", xname="Factor"
            )
        wandb.log(line_series_plots)


def plot_metrics_multiple_datasets(
    data_path, write_to_path=None, report_to=None, wandb_name=None, mode=None, rule=False
):
    df = pd.read_parquet(data_path)
    suffix = "_LMJudgeEvaluator_relevance_concept_ratings"
    method_names = [a[: len(a) - len(suffix)] for a in df.columns if suffix in a]

    for method in method_names:
        col = method + suffix
        norm_col = method + "_normalized_LMJudgeEvaluator_relevance_concept_ratings"
        df[norm_col] = 0.0
        for dataset in df["dataset_name"].unique():
            mask_dataset = df["dataset_name"] == dataset
            for concept_id in df.loc[mask_dataset, "concept_id"].unique():
                mask_concept = mask_dataset & (df["concept_id"] == concept_id)
                for input_id in df.loc[mask_concept, "input_id"].unique():
                    mask_input = mask_concept & (df["input_id"] == input_id)
                    mask_minus2 = mask_input & (df["factor"] == -2)
                    if mask_minus2.any():
                        base_val = 2 - df.loc[mask_minus2, col].values[0]
                        val = (2 - df.loc[mask_input, col]) - base_val
                        df.loc[mask_input, norm_col] = val.clip(lower=0)

    for method in method_names:
        norm_concept_col = method + "_normalized_LMJudgeEvaluator_relevance_concept_ratings"
        instr_col = method + "_LMJudgeEvaluator_relevance_instruction_ratings"
        fluency_col = method + "_LMJudgeEvaluator_fluency_ratings"
        new_col = method + "_normalized_LMJudgeEvaluator"

        def safe_hmean(row):
            vals = [row.get(norm_concept_col, 0), row.get(instr_col, 0), row.get(fluency_col, 0)]
            if 0 in vals:
                return 0
            return 3 / sum(1 / v for v in vals)

        df[new_col] = df.apply(safe_hmean, axis=1)

    metrics = [
        "_normalized_LMJudgeEvaluator",
        "_normalized_LMJudgeEvaluator_relevance_concept_ratings",
        "_LMJudgeEvaluator_relevance_instruction_ratings",
        "_LMJudgeEvaluator_fluency_ratings",
    ]
    metrics_names = [
        "Normalized Overall",
        "Normalized Relevance Concept",
        "Relevance Instruction",
        "Fluency",
    ]

    plot_data = []
    for dataset in df["dataset_name"].unique():
        dataset_data = df[df["dataset_name"] == dataset]
        for method in method_names:
            for factor in dataset_data["factor"].unique():
                factor_data = dataset_data[dataset_data["factor"] == factor]
                for idx, metric in enumerate(metrics):
                    col = method + metric
                    if col in factor_data.columns:
                        plot_data.append({
                            "Dataset": dataset,
                            "Method": method,
                            "Factor": factor,
                            "Metric": metrics_names[idx],
                            "Value": factor_data[col].mean(),
                        })

    plot_df = pd.DataFrame(plot_data)
    if plot_df.empty:
        logger.warning("plot_metrics_multiple_datasets: no plottable metrics in %s", data_path)
        return None

    plot_df["Metric"] = pd.Categorical(plot_df["Metric"], categories=metrics_names, ordered=True)
    unique_combinations = plot_df[["Dataset", "Metric"]].drop_duplicates()
    ncols = 4
    nrows = int(np.ceil(len(unique_combinations) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 4 * nrows))
    axes = axes.flatten()
    unique_methods = plot_df["Method"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_methods)))
    method_colors = dict(zip(unique_methods, colors))

    for idx, (dataset, metric) in enumerate(unique_combinations.values):
        ax = axes[idx]
        subset = plot_df[(plot_df["Dataset"] == dataset) & (plot_df["Metric"] == metric)]
        for method in unique_methods:
            method_data = subset[subset["Method"] == method]
            if not method_data.empty:
                ax.plot(
                    method_data["Factor"],
                    method_data["Value"],
                    color=method_colors[method],
                    label=method,
                    marker="o",
                )
        ax.set_title(metric)
        ax.set_xlabel("Factor")
        ax.set_ylabel("Score")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()

    if write_to_path:
        plt.savefig(str(write_to_path / f"{mode}_combined_plot.png"), dpi=300, bbox_inches="tight")
    plt.close()
    return fig

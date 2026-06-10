# Adapted from AxBench — https://github.com/stanfordnlp/axbench (Apache-2.0).
import asyncio
import json
import logging
import multiprocessing
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import httpx
import pandas as pd
from openai import AsyncOpenAI

import particlesteering
from particlesteering.models.language_models import LanguageModel
from particlesteering.scripts.args.eval_args import EvalArgs
from particlesteering.scripts.inference import STEERING_EXCLUDE_MODELS
from particlesteering.utils.plot_utils import plot_metrics, plot_metrics_multiple_datasets

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.WARN,
)
logger = logging.getLogger(__name__)

STATE_FILE = "evaluate_state.pkl"


def data_generator(data_dir, mode):
    df = pd.read_parquet(os.path.join(data_dir, "steering_data.parquet"))
    for concept_id, group in df.groupby("concept_id"):
        yield concept_id, group.copy()


def save_results(dump_dir, state, concept_id, partition, eval_results, eval_df=None):
    dump_dir.mkdir(parents=True, exist_ok=True)
    state_path = os.path.join(dump_dir, f"{partition}_{STATE_FILE}")
    with open(state_path, "wb") as f:
        pickle.dump(state, f)

    result_path = Path(dump_dir) / f"{partition}.jsonl"
    with open(result_path, "a") as f:
        f.write(json.dumps({"concept_id": int(concept_id), "results": eval_results}) + "\n")

    if not eval_df:
        return

    evaluator_name = next(iter(eval_df))
    model_name = next(iter(eval_df[evaluator_name]))
    current_df = eval_df[evaluator_name][model_name]

    df_path = os.path.join(dump_dir, f"{partition}_data.parquet")
    if os.path.exists(df_path):
        combined_df = pd.concat([pd.read_parquet(df_path), current_df], ignore_index=True)
    else:
        combined_df = current_df
    combined_df.to_parquet(df_path, index=False)


def load_state(dump_dir, mode):
    state_path = os.path.join(dump_dir, f"{mode}_{STATE_FILE}")
    if os.path.exists(state_path):
        with open(state_path, "rb") as f:
            return pickle.load(f)
    return None


def process_jsonl_file(jsonl_lines):
    for data in jsonl_lines:
        data["results"]["LMJudgeEvaluator"] = data["results"]["LMJudgeEvaluator"]
    return jsonl_lines


def plot_steering(aggregated_results, dump_dir, report_to=None, wandb_name=None, mode=None):
    configs = [
        {"evaluator_name": "LMJudgeEvaluator", "metric_name": "relevance_concept_ratings", "y_label": "Concept", "use_log_scale": False},
        {"evaluator_name": "LMJudgeEvaluator", "metric_name": "relevance_instruction_ratings", "y_label": "Instruct", "use_log_scale": False},
        {"evaluator_name": "LMJudgeEvaluator", "metric_name": "fluency_ratings", "y_label": "Fluency", "use_log_scale": False},
        {"evaluator_name": "LMJudgeEvaluator", "metric_name": "lm_judge_rating", "y_label": "Aggregated", "use_log_scale": False},
    ]
    try:
        plot_metrics(
            jsonl_data=aggregated_results,
            configs=configs,
            write_to_path=dump_dir,
            report_to=report_to,
            wandb_name=wandb_name,
            mode=mode,
        )
    except Exception as e:
        logger.warning("Failed to plot: %s", e)


def _steering_eval_task(concept_id, current_df, evaluator_name, model_name, args):
    return (
        concept_id,
        current_df,
        evaluator_name,
        model_name,
        args.dump_dir,
        args.lm_model,
        args.steer_data_type,
        float(getattr(args, "openai_timeout", 120.0) or 120.0),
        int(getattr(args, "judge_batch_size", 8) or 8),
        args.master_data_dir or "particlesteering/data",
    )


def eval_steering_single_task(args_tuple):
    (
        concept_id,
        current_df,
        evaluator_name,
        model_name,
        dump_dir,
        lm_model_name,
        steer_dataset_type,
        openai_timeout,
        judge_batch_size,
        master_data_dir,
    ) = args_tuple

    client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=openai_timeout,
        http_client=httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=100, max_connections=1000),
            headers={"Connection": "close"},
        ),
        max_retries=3,
    )
    lm_model = LanguageModel(
        lm_model_name,
        client,
        dump_dir=dump_dir,
        use_cache=True,
        cache_level="prompt",
        cache_tag="evaluate",
        master_data_dir=master_data_dir,
        temperature=0.0,
    )

    try:
        evaluator_class = getattr(particlesteering, evaluator_name)
        evaluator = evaluator_class(
            model_name,
            dump_dir=dump_dir,
            concept_id=concept_id,
            lm_model=lm_model,
            steer_dataset_type=steer_dataset_type,
            judge_batch_size=judge_batch_size,
        )
        eval_result = evaluator.compute_metrics(current_df)
        return (
            concept_id,
            evaluator.__str__(),
            str(model_name),
            eval_result,
            lm_model.stats.get_report(),
            lm_model.cache_in_mem,
            current_df,
        )
    finally:
        asyncio.run(client.close())


def eval_steering(args):
    state = load_state(args.dump_dir, mode=args.mode)
    start_concept_id = state.get("concept_id", 0) if state else 0
    logger.warning("Starting concept_id: %s", start_concept_id)

    all_tasks = [
        _steering_eval_task(concept_id, current_df, evaluator_name, model_name, args)
        for concept_id, current_df in data_generator(args.data_dir, args.mode)
        if concept_id >= start_concept_id
        for evaluator_name in args.steering_evaluators
        for model_name in args.models
        if model_name not in STEERING_EXCLUDE_MODELS
    ]

    if not hasattr(args, "num_of_workers") or args.num_of_workers is None:
        args.num_of_workers = max(1, multiprocessing.cpu_count() - 1)

    all_results = {}
    eval_dfs = {}
    lm_reports = []

    with ProcessPoolExecutor(max_workers=args.num_of_workers) as executor:
        for concept_id, evaluator_str, model_str, result, lm_report, _, current_df in executor.map(
            eval_steering_single_task, all_tasks
        ):
            all_results.setdefault(concept_id, {}).setdefault(evaluator_str, {})[model_str] = result
            if evaluator_str == "LMJudgeEvaluator":
                df = current_df.copy()
                df[f"{model_str}_{evaluator_str}"] = result["raw_aggregated_ratings"]
                df[f"{model_str}_{evaluator_str}_relevance_concept_ratings"] = result["raw_relevance_concept_ratings"]
                df[f"{model_str}_{evaluator_str}_relevance_instruction_ratings"] = result["raw_relevance_instruction_ratings"]
                df[f"{model_str}_{evaluator_str}_fluency_ratings"] = result["raw_fluency_ratings"]
                df[f"{model_str}_{evaluator_str}_relevance_concept_completions"] = result["relevance_concept_completions"]
                df[f"{model_str}_{evaluator_str}_relevance_instruction_completions"] = result["relevance_instruction_completions"]
                df[f"{model_str}_{evaluator_str}_fluency_completions"] = result["fluency_completions"]
                eval_dfs.setdefault(concept_id, {}).setdefault(evaluator_str, {})[model_str] = df
            lm_reports.append(lm_report)

    for concept_id, eval_results in sorted(all_results.items()):
        save_results(
            args.dump_dir,
            {"concept_id": concept_id + 1},
            concept_id,
            args.mode,
            eval_results,
            eval_dfs.get(concept_id),
        )

    if args.mode == "train_data":
        return

    aggregated_results = process_jsonl_file(load_jsonl(args.dump_dir / f"{args.mode}.jsonl"))
    aggregated_lm_report = {
        "total_calls": sum(r["total_calls"] for r in lm_reports),
        "total_cache_hits": sum(r["total_cache_hits"] for r in lm_reports),
        "total_price": sum(r["total_price"] for r in lm_reports),
    }
    logger.warning(
        "Total calls: %s, cache hits: %s, price: $%s",
        aggregated_lm_report["total_calls"],
        aggregated_lm_report["total_cache_hits"],
        aggregated_lm_report["total_price"],
    )

    if os.environ.get("AXBENCH_SKIP_EVAL_PLOTS", "").strip().lower() not in {"1", "true", "yes"}:
        plot_steering(aggregated_results, args.dump_dir, args.report_to, args.wandb_name, args.mode)
        steering_parquet = Path(getattr(args, "experiment_root", args.dump_dir.parent)) / "inference" / "steering_data.parquet"
        plot_metrics_multiple_datasets(
            str(steering_parquet), args.dump_dir, args.report_to, args.wandb_name, args.mode
        )


def load_jsonl(jsonl_path):
    with open(jsonl_path, "r") as f:
        return [json.loads(line) for line in f]


def main():
    args = EvalArgs(section="evaluate", ignore_unknown=True)
    if args.mode == "train_data":
        args.data_dir = (
            f"{args.dump_dir}/generate"
            if args.overwrite_inference_dump_dir is None
            else Path(args.overwrite_inference_dump_dir)
        )
    else:
        args.data_dir = (
            f"{args.dump_dir}/inference"
            if args.overwrite_inference_dump_dir is None
            else Path(args.overwrite_inference_dump_dir)
        )

    experiment_root = Path(args.dump_dir).resolve()
    dump_dir = Path(args.dump_dir) / "evaluate" if args.overwrite_evaluate_dump_dir is None else Path(args.overwrite_evaluate_dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    args.experiment_root = experiment_root
    args.dump_dir = dump_dir

    if "steering" in args.mode or args.mode == "train_data":
        eval_steering(args)
    else:
        raise ValueError(f"Unsupported evaluate mode: {args.mode}")


if __name__ == "__main__":
    main()

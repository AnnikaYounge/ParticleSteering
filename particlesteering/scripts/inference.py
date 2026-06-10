# Adapted from AxBench — https://github.com/stanfordnlp/axbench (Apache-2.0).
# inference.py: Inference with existing subspaces.
#
# example launch command:
#     torchrun --nproc_per_node=NUM_GPUS particlesteering/scripts/inference.py --config configs/inference.yaml --mode latent
import os, argparse, yaml, json, glob, pickle, time, itertools, datetime
import shutil
import pandas as pd
from tqdm.auto import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import atexit

from particlesteering.utils.dataset import (
    DatasetFactory,
    SteeringDatasetFactory
)
from particlesteering.utils.constants import * 
from particlesteering.utils.model_utils import get_prefix_length, get_suffix_length
from particlesteering.models.particle_utils import expand_steering_dataframe_for_particles
from particlesteering.scripts.args.dataset_args import DatasetArgs
from particlesteering.scripts.args.training_args import TrainingArgs
from transformers import set_seed

# all supported methods
import particlesteering
from openai import AsyncOpenAI
import httpx, asyncio

import logging
import torch.distributed as dist
import sys

# Initialize the logger
logger = logging.getLogger(__name__)

MAX_RETRIES = 5
RETRY_DELAY = 1  # in seconds
STATE_FILE = "inference_state.pkl"
CONFIG_FILE = "config.json"
METADATA_FILE = "metadata.jsonl"
STEERING_WITH_SHARED_MODELS: set[str] = set()
STEERING_EXCLUDE_MODELS = {"IntegratedGradients", "InputXGradients", "PromptDetection", "BoW"}
LATENT_EXCLUDE_MODELS = {"PromptSteering", "PromptBaseline", "DiReFT", "LoReFT", "LoRA", "SFT"}
PARTICLE_STEER_MODELS = {"ParticleSteering"}
LATENT_PROMPT_PREFIX = "Generate a random sentence."

def load_config(config_path):
    """
    Load metadata from a JSON lines file.
    """
    if not os.path.exists(Path(config_path) / CONFIG_FILE):
        return None
    with open(Path(config_path) / CONFIG_FILE) as f:
        d = json.load(f)
    return d


def load_state(dump_dir, mode, rank, subfolder="inference"):
    """
    Load the state from a file if it exists.
    """
    state_path = os.path.join(f"{dump_dir}/{subfolder}", f"{mode}_{STATE_FILE}_rank_{rank}")
    if os.path.exists(state_path):
        with open(state_path, "rb") as f:
            return pickle.load(f)
    return None


def save_state(dump_dir, state, partition, rank):
    if not isinstance(dump_dir, Path):
        dump_dir = Path(dump_dir)
        
    dump_dir.mkdir(parents=True, exist_ok=True)
    # Save state
    state_path = os.path.join(dump_dir, f"{partition}_{STATE_FILE}_rank_{rank}")
    with open(state_path, "wb") as f:
        pickle.dump(state, f)


def load_metadata_flatten(metadata_path):
    """
    Load flatten metadata from a JSON lines file.
    """
    metadata = []
    with open(Path(metadata_path) / METADATA_FILE, 'r') as f:
        for line in f:
            data = json.loads(line)
            concept, ref = data["concept"], data["ref"]
            concept_genres_map = data["concept_genres_map"][concept]
            ref = data["ref"]
            flatten_data = {
                "concept": concept,
                "ref": ref,
                "concept_genres_map": {concept: concept_genres_map},
                "concept_id": data["concept_id"]
            }
            metadata += [flatten_data]  # Return the metadata as is
    return metadata


def save(
    dump_dir, partition,
    current_df, rank):
    # This function saves DataFrames per rank per partition (latent or steering)
    dump_dir = Path(dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    # Save DataFrame
    df_path = os.path.join(dump_dir, f"rank_{rank}_{partition}_data.parquet")
    
    if "defense" in current_df.columns:
        current_df["defense"] = current_df["defense"].apply(lambda x: "["+','.join(x)+"]" if isinstance(x, list) else str(x))
        
    if os.path.exists(df_path):
        existing_df = pd.read_parquet(df_path)
        
        if "defense" in existing_df.columns:
            # Convert defense column to string format if it exists
            existing_df["defense"] = existing_df["defense"].apply(lambda x: "["+','.join(x)+"]" if isinstance(x, list) else str(x))
        combined_df = pd.concat([existing_df, current_df], ignore_index=True)
    else:
        combined_df = current_df

    combined_df.to_parquet(df_path, engine='pyarrow')


def partition_concept_ids(concept_ids, world_size):
    concept_ids_per_rank = []
    n = len(concept_ids)
    chunk_size = n // world_size
    remainder = n % world_size
    start = 0
    for i in range(world_size):
        end = start + chunk_size + (1 if i < remainder else 0)
        concept_ids_per_rank.append(concept_ids[start:end])
        start = end
    return concept_ids_per_rank


def _resolve_max_concepts(inference_args, training_args):
    """max_concepts may be set on the inference or train YAML section / CLI."""
    for source in (inference_args, training_args):
        n = getattr(source, "max_concepts", None)
        if n is not None:
            return int(n)
    return None


def _apply_max_concepts(concept_ids, inference_args, training_args, logger):
    n = _resolve_max_concepts(inference_args, training_args)
    if n is None:
        return concept_ids
    limited = concept_ids[:n]
    logger.warning(
        "Limiting inference to %s concepts (max_concepts=%s; metadata has %s)",
        len(limited),
        n,
        len(concept_ids),
    )
    return limited


def create_data_latent(dataset_factory, metadata, concept_id, num_of_examples, args):
    # prepare concept related data.
    concept = metadata[concept_id]["concept"]
    sae_link = metadata[concept_id]["ref"]
    sae_id = int(sae_link.split("/")[-1]) 
    concept_genres_map = metadata[concept_id]["concept_genres_map"]
    _, eval_contrast_concepts_map = \
        dataset_factory.prepare_concepts(
            [concept], 
            concept_genres_map=concept_genres_map,
            contrast_concepts_map={}, api_tag="inference")
    current_df = dataset_factory.create_eval_df(
        [concept], num_of_examples, concept_genres_map, {},
        eval_contrast_concepts_map, input_length=args.input_length, 
        output_length=args.output_length, concept_id=concept_id
    )
    current_df["concept_id"] = concept_id
    current_df["sae_link"] = sae_link
    current_df["sae_id"] = sae_id
    return current_df


def create_data_steering(
    dataset_factory, metadata, concept_id, num_of_examples, 
    n_steering_factors, steering_datasets, args, generate_args):

    # prepare concept related data.
    concept = metadata[concept_id]["concept"]
    sae_link = metadata[concept_id]["ref"]
    try:
        sae_id = int(sae_link.split("/")[-1]) 
    except:
        sae_id = 0

    current_df = dataset_factory.create_eval_df(
        [concept], num_of_examples, n_steering_factors, steering_datasets, concept_id=concept_id,
        steering_model_name=args.steering_model_name, steer_data_type=generate_args.steer_data_type,
        n_shots=args.n_shot, defense=args.defense, dump_dir=args.dump_dir, multishot_factors_parquet=args.multishot_factors_parquet,
        suppress_eval_dir=args.suppress_eval_dir
    )
    current_df["concept_id"] = concept_id
    current_df["sae_link"] = sae_link
    current_df["sae_id"] = sae_id

    return current_df, (concept_id, sae_link, sae_id)


def _steering_results_sort_columns(df):
    cols = ["concept_id", "input_id", "factor"]
    if "particle_id" in df.columns:
        cols.append("particle_id")
    return cols


def _benchmark_num_particles(benchmark_model):
    """Particle count for steering merge."""
    if hasattr(benchmark_model, "particle_head"):
        return int(benchmark_model.particle_head.num_particles)
    return int(getattr(benchmark_model, "num_particles", 1))


def _attach_steer_results_to_df(current_df, model_name, results, benchmark_model):
    """Merge predict_steer outputs; expand rows when all K particles were steered."""
    results = dict(results)
    steer_all = results.pop("_steer_all_particles", False)
    num_particles = _benchmark_num_particles(benchmark_model)

    if steer_all:
        current_df = expand_steering_dataframe_for_particles(current_df, num_particles)
    particle_ids = results.pop("particle_id", None)
    if particle_ids is not None:
        if len(particle_ids) != len(current_df):
            raise ValueError(
                f"{model_name}: got {len(particle_ids)} particle outputs for "
                f"{len(current_df)} dataframe rows"
            )
        current_df = current_df.copy()
        current_df["particle_id"] = particle_ids

    for key, val in results.items():
        if key.startswith("_"):
            continue
        current_df[f"{model_name}_{key}"] = val
    return current_df


def _particle_steer_kwargs(model_name, training_args):
    if model_name not in PARTICLE_STEER_MODELS:
        return {}
    mp = training_args.models[model_name]
    return {
        "steer_all_particles": getattr(mp, "steer_all_particles", True),
        "steering_particle_policy": getattr(mp, "steering_particle_policy", "all"),
        "steering_particle_index": getattr(mp, "steering_particle_index", None),
    }


def _particle_model_load_kwargs(model_name, training_args):
    if model_name not in PARTICLE_STEER_MODELS:
        return {}
    return {
        "num_particles": int(training_args.models[model_name].num_particles),
    }


def _particle_bm_init_kwargs(model_name, training_args, device):
    kw = {"device": device}
    if model_name in PARTICLE_STEER_MODELS:
        kw["training_args"] = training_args.models[model_name]
    return kw


def _apply_latent_results_to_df(current_df, model_name, results):
    for k, v in results.items():
        if k in ("tokens", "particle_id"):
            if k == "tokens":
                if "tokens" not in current_df:
                    current_df["tokens"] = v
            continue
        current_df[f"{model_name}_{k}"] = v
    return current_df


def _predict_latent_into_df(benchmark_model, model_name, current_df, **predict_kw):
    """Run predict_latent; expand rows per particle when steer-all K vectors are used."""
    n_particles = _benchmark_num_particles(benchmark_model)
    if model_name not in PARTICLE_STEER_MODELS or n_particles <= 1:
        results = benchmark_model.predict_latent(current_df, **predict_kw)
        return _apply_latent_results_to_df(current_df, model_name, results)

    chunks = []
    for particle_id in range(n_particles):
        results = benchmark_model.predict_latent(
            current_df, particle_id=particle_id, **predict_kw
        )
        chunk = current_df.copy()
        chunk["particle_id"] = particle_id
        _apply_latent_results_to_df(chunk, model_name, results)
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


def prepare_df(current_df, tokenizer, is_chat_model, model_name):
    suffix_length, _ = get_suffix_length(tokenizer)
    if is_chat_model:
        if model_name == "meta-llama/Llama-3.1-8B-Instruct":
            def apply_chat_template(row):
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."}, 
                    {"role": "user", "content": row["input"]},
                    {"role": "assistant", "content": row["output"]}
                ]
                tokens = tokenizer.apply_chat_template(messages, tokenize=True)[1:-suffix_length]
                return tokenizer.decode(tokens)
            current_df['input'] = current_df.apply(apply_chat_template, axis=1)
        else:
            def apply_chat_template(row):
                messages = [
                    {"role": "user", "content": row["input"]},
                    {"role": "assistant", "content": row["output"]}
                ]
                tokens = tokenizer.apply_chat_template(messages, tokenize=True)[1:-suffix_length]
                return tokenizer.decode(tokens)
            current_df['input'] = current_df.apply(apply_chat_template, axis=1)
    return current_df


def infer_steering(args, rank, world_size, device, logger, training_args, generate_args, suppress_eval_dir=None):
    data_dir = args.data_dir
    train_dir = args.train_dir
    dump_dir = args.dump_dir
    overwrite_inference_dump_dir = Path(args.overwrite_inference_dump_dir) if args.overwrite_inference_dump_dir is not None else Path(dump_dir) / "inference"
    num_of_examples = args.steering_num_of_examples
    config = load_config(train_dir)
    metadata = load_metadata_flatten(data_dir)
    layer = int(args.steering_layer) if args.steering_layer is not None else config["layer"] if config else 0  # default layer for prompt baselines
    steering_layers = args.steering_layers if args.steering_layers is not None else [layer]
    steering_factors = args.steering_factors
    steering_datasets = args.steering_datasets

    state = load_state(args.dump_dir, "steering", rank)
    last_concept_id_processed = state.get("last_concept_id", None) if state else None
    logger.warning(f"Rank {rank} last concept_id processed: {last_concept_id_processed}")

    # Get list of all concept_ids
    concept_ids = [metadata[i]["concept_id"] for i in range(len(metadata))]
    concept_ids = _apply_max_concepts(concept_ids, args, training_args, logger)

    # Partition concept_ids among ranks sequentially
    concept_ids_per_rank = partition_concept_ids(concept_ids, world_size)
    my_concept_ids = concept_ids_per_rank[rank]

    if last_concept_id_processed is not None:
        if last_concept_id_processed in my_concept_ids:
            idx = my_concept_ids.index(last_concept_id_processed)
            my_concept_ids = my_concept_ids[idx+1:]
        else:
            # If last_concept_id_processed is not in my_concept_ids, process all
            pass

    if len(my_concept_ids) == 0:

        # Synchronize all processes
        dist.barrier()

        # Rank 0 merges results
        if rank == 0:
            logger.warning("Rank 0 is merging results.")
            # Merge per-rank results
            all_parquet_files = list((overwrite_inference_dump_dir).glob("rank_*_steering_data.parquet"))
            # Parse filenames to extract rank
            import re
            pattern = re.compile(r'rank_(\d+)_steering_data\.parquet')

            file_info_list = []
            for parquet_file in all_parquet_files:
                match = pattern.match(parquet_file.name)
                if match:
                    rank_str = match.group(1)
                    rank_int = int(rank_str)
                    file_info_list.append({
                        'rank': rank_int,
                        'file': parquet_file
                    })
                else:
                    logger.warning(f"Filename {parquet_file.name} does not match the expected pattern.")

            # Sort the file_info_list by rank
            file_info_list.sort(key=lambda x: x['rank'])

            # Read and concatenate dataframes
            dfs = []
            for info in file_info_list:
                df = pd.read_parquet(info['file'])
                dfs.append(df)
            if len(dfs) > 0:
                combined_df = pd.concat(dfs, ignore_index=True)
                # Optionally sort combined_df by 'concept_id' if needed
                combined_df = combined_df.sort_values(
                    by=_steering_results_sort_columns(combined_df)
                ).reset_index(drop=True)
                combined_df.to_parquet(overwrite_inference_dump_dir / "steering_data.parquet", engine='pyarrow')
                logger.warning(f"Saved combined steering inference results to {overwrite_inference_dump_dir / 'steering_data.parquet'}")
            else:
                logger.warning("No results to merge.")

            # Optionally, delete per-rank files
            for info in file_info_list:
                os.remove(info['file'])
                logger.warning(f"Deleted {info['file']}")

        logger.warning(f"Rank {rank} has no concepts to process. Exiting.")
        return

    # Create a new OpenAI client.
    lm_client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=60.0,
        http_client=httpx.AsyncClient(
            limits=httpx.Limits(
                max_keepalive_connections=100,
                max_connections=1000
            ),
            headers={"Connection": "close"},
        ),
        max_retries=3,
    )

    # Initialize the dataset factory with the tokenizer.
    if "google/gemma-3" in args.steering_model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.steering_model_name, use_fast=False, model_max_length=128000)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.steering_model_name, use_fast=False, model_max_length=1024)
    tokenizer.padding_side = "right"
    has_prompt_steering = "PromptSteering" in args.models
    dataset_factory = SteeringDatasetFactory(
        tokenizer, dump_dir,
        master_data_dir=args.master_data_dir, lm_client=lm_client,
        lm_model=args.lm_model,
        has_prompt_steering=has_prompt_steering
    )
    is_chat_model = True if args.model_name in CHAT_MODELS else False
    prefix_length = 1 # prefix is default to 1 for all models due to the BOS token.
    if is_chat_model:
        prefix_length = get_prefix_length(tokenizer)
        logger.warning(f"Chat model prefix length: {prefix_length}")
        
    # Load model instance onto device
    if args.use_bf16:
        logger.warning(f"Using bfloat16 for model {args.model_name}")
    if "gemma-3" in args.model_name:
        from transformers import Gemma3ForCausalLM
        model_instance = Gemma3ForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16 if args.use_bf16 else None, device_map=device)
    else:
        model_instance = AutoModelForCausalLM.from_pretrained(
            args.steering_model_name if args.steering_model_name else args.model_name, 
            torch_dtype=torch.bfloat16 if args.use_bf16 else None, device_map=device
        )
    model_instance = model_instance.eval()

    if tokenizer.unk_token == None and tokenizer.pad_token == None:
        # raw llama3
        print("adding a special padding token...")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        need_resize = True
    else:
        need_resize = False
    if need_resize:
        model_instance.resize_token_embeddings(len(tokenizer))

    # Prepare data per concept
    data_per_concept = {}
    for concept_id in my_concept_ids:
        
        current_df, (_, sae_link, sae_id) = create_data_steering(
            dataset_factory, metadata, concept_id, num_of_examples,
            steering_factors, steering_datasets, args, generate_args
        )
        data_per_concept[concept_id] = (current_df, sae_link, sae_id)
    
    # Preload models that are shared across concepts (none in this repo).
    preloaded_models = dict()
    for model_name in training_args.models.keys():
        if model_name in STEERING_EXCLUDE_MODELS:
            continue
        if model_name in STEERING_WITH_SHARED_MODELS:
            model_class = getattr(particlesteering, model_name)
            logger.warning(f"Loading {model_class} on {device}.")
            
            benchmark_model = model_class(
                model_instance, tokenizer, layer=layer,
                low_rank_dimension=len(metadata),
                device=device,
                training_args=training_args.models[model_name],
                lm_model_name=training_args.model_name,
            )
            benchmark_model.load(
                dump_dir=train_dir, low_rank_dimension=1, mode="steering", 
                hypernet_initialize_from_pretrained=training_args.models[model_name].hypernet_initialize_from_pretrained,
                hypernet_name_or_path=training_args.models[model_name].hypernet_name_or_path,
                num_hidden_layers=training_args.models[model_name].num_hidden_layers,
            )
            preloaded_models[model_name] = benchmark_model

    # Now loop over concept_ids and use preloaded models
    for concept_id in my_concept_ids:
        current_df, sae_link, sae_id = data_per_concept[concept_id]
        for model_name in args.models:
            if model_name in STEERING_EXCLUDE_MODELS:
                continue
            
            if model_name not in STEERING_WITH_SHARED_MODELS:
                model_class = getattr(particlesteering, model_name)
                logger.warning(f"Loading {model_class} on {device}.")

                benchmark_model = model_class(
                    model_instance, tokenizer, layer=layer,
                    training_args=training_args.models[model_name] if model_name not in {"PromptSteering", "GemmaScopeSAE"} else None, # we init with training args as well
                    low_rank_dimension=len(metadata),
                    device=device, steering_layers=steering_layers,
                )
                if model_name in {"PromptSteering", "GemmaScopeSAE"}:
                    lr = 1
                else:
                    lr = training_args.models[model_name].low_rank_dimension if training_args.models[model_name].low_rank_dimension else 1
                load_kw = dict(
                    dump_dir=train_dir,
                    sae_path=metadata[0]["ref"],
                    mode="steering",
                    priority_mode="compute_priority",
                    intervention_type=args.steering_intervention_type,
                    concept_id=concept_id,
                    low_rank_dimension=lr,
                )
                if model_name in PARTICLE_STEER_MODELS:
                    load_kw.update(_particle_model_load_kwargs(model_name, training_args))
                benchmark_model.load(**load_kw)
                benchmark_model.to(device)
                if hasattr(benchmark_model, 'ax') and args.use_bf16:
                    if model_name not in {"PreferenceLoReFT", "ConceptLoReFT",}:
                        if isinstance(benchmark_model.ax, list):
                            for ax in benchmark_model.ax:
                                ax.eval()
                            ax.to(torch.bfloat16)
                        else:
                            benchmark_model.ax.eval()
                            benchmark_model.ax.to(torch.bfloat16)
            else:
                benchmark_model = preloaded_models[model_name]
                
                benchmark_model.to(device)
                if hasattr(benchmark_model, 'ax') and args.use_bf16:
                    benchmark_model.ax.eval()
                    benchmark_model.ax.to(torch.bfloat16)
                
            # Pre-compute mean activations once
            if model_name not in {"LoReFT", "BoW"} and model_name not in LATENT_EXCLUDE_MODELS:
                benchmark_model.pre_compute_mean_activations(
                    os.path.join(dump_dir, "inference"), 
                    master_data_dir=args.master_data_dir,
                    disable_neuronpedia_max_act=args.disable_neuronpedia_max_act,
                    metadata=metadata,
                )
            unique_concept_ids = list(set(current_df["concept_id"].tolist()))
            logger.warning(f"Inference steering with {model_name} on {device} for concept {concept_id}.")
            # Run prediction
            steer_kwargs = dict(
                concept_id=unique_concept_ids[0] if len(unique_concept_ids) == 1 else unique_concept_ids,
                sae_link=None,
                sae_id=None,
                batch_size=int(args.steering_batch_size),
                eval_output_length=int(args.steering_output_length),
                temperature=float(args.temperature),
                prefix_length=prefix_length,
                positions=training_args.models[model_name].intervention_positions
                if model_name not in {"PromptSteering", "GemmaScopeSAE"}
                else None,
                use_synergy=False,
                disable_neuronpedia_max_act=args.disable_neuronpedia_max_act,
                intervene_on_prompt=args.intervene_on_prompt if args.intervene_on_prompt is not None else True,
                return_vector=False,
            )
            steer_kwargs.update(_particle_steer_kwargs(model_name, training_args))
            results = benchmark_model.predict_steer(current_df, **steer_kwargs)
            if model_name in PARTICLE_STEER_MODELS:
                current_df = _attach_steer_results_to_df(current_df, model_name, results, benchmark_model)
            else:
                for k, v in results.items():
                    current_df[f"{model_name}_{k}"] = v
                
            if model_name not in STEERING_WITH_SHARED_MODELS:
                del benchmark_model
            else:
                benchmark_model = benchmark_model.to("cpu") # move shared model to cpu to save memory
                
            torch.cuda.empty_cache()
        save(overwrite_inference_dump_dir, 'steering', current_df, rank)
        logger.warning(f"Saved inference results for concept {concept_id} to rank_{rank}_steering_data.parquet")
        # After processing, save state
        current_state = {'last_concept_id': concept_id}
        save_state(overwrite_inference_dump_dir, current_state, 'steering', rank)

    # Synchronize all processes
    dist.barrier()

    # Rank 0 merges results
    if rank == 0:
        logger.warning("Rank 0 is merging results.")
        # Merge per-rank results
        all_parquet_files = list(overwrite_inference_dump_dir.glob("rank_*_steering_data.parquet"))
        # Parse filenames to extract rank
        import re
        pattern = re.compile(r'rank_(\d+)_steering_data\.parquet')

        file_info_list = []
        for parquet_file in all_parquet_files:
            match = pattern.match(parquet_file.name)
            if match:
                rank_str = match.group(1)
                rank_int = int(rank_str)
                file_info_list.append({
                    'rank': rank_int,
                    'file': parquet_file
                })
            else:
                logger.warning(f"Filename {parquet_file.name} does not match the expected pattern.")

        # Sort the file_info_list by rank
        file_info_list.sort(key=lambda x: x['rank'])

        # Read and concatenate dataframes
        dfs = []
        for info in file_info_list:
            df = pd.read_parquet(info['file'])
            dfs.append(df)
        if len(dfs) > 0:
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df = combined_df.sort_values(
                by=_steering_results_sort_columns(combined_df)
            ).reset_index(drop=True)
            merged_path = overwrite_inference_dump_dir / "steering_data.parquet"
            combined_df.to_parquet(merged_path, engine='pyarrow')
            logger.warning(f"Saved combined steering inference results to {merged_path}")
        else:
            logger.warning("No results to merge.")

        # Optionally, delete per-rank files
        for info in file_info_list:
            os.remove(info['file'])
            logger.warning(f"Deleted {info['file']}")


def infer_latent(args, rank, world_size, device, logger, training_args, generate_args):
    data_dir = args.data_dir
    train_dir = args.train_dir
    dump_dir = Path(args.dump_dir) / "inference"
    num_of_examples = args.latent_num_of_examples
    config = load_config(train_dir)
    metadata = load_metadata_flatten(data_dir)
    layer = config["layer"] if config else 0  # default layer for prompt baselines

    state = load_state(args.dump_dir, "latent", rank)
    last_concept_id_processed = state.get("last_concept_id", None) if state else None
    logger.warning(f"Rank {rank} last concept_id processed: {last_concept_id_processed}")

    # Get list of all concept_ids
    concept_ids = [metadata[i]["concept_id"] for i in range(len(metadata))]
    concept_ids = _apply_max_concepts(concept_ids, args, training_args, logger)

    # Partition concept_ids among ranks sequentially
    concept_ids_per_rank = partition_concept_ids(concept_ids, world_size)
    my_concept_ids = concept_ids_per_rank[rank]

    if last_concept_id_processed is not None:
        if last_concept_id_processed in my_concept_ids:
            idx = my_concept_ids.index(last_concept_id_processed)
            my_concept_ids = my_concept_ids[idx+1:]
        else:
            # If last_concept_id_processed is not in my_concept_ids, process all
            pass

    if len(my_concept_ids) == 0:
        logger.warning(f"Rank {rank} has no concepts to process. Exiting.")
        return

    # Create a new OpenAI client.
    client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=60.0,
        http_client=httpx.AsyncClient(
            limits=httpx.Limits(
                max_keepalive_connections=100,
                max_connections=1000
            ),
            headers={"Connection": "close"},
        ),
        max_retries=3,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, model_max_length=1024)
    tokenizer.padding_side = "right"

    # Load model instance onto device
    if args.use_bf16:
        logger.warning(f"Using bfloat16 for model {args.model_name}")
    model_instance = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        torch_dtype=torch.bfloat16 if args.use_bf16 else None, 
        device_map=device
    )
    is_chat_model = True if args.model_name in CHAT_MODELS else False
    model_instance = model_instance.eval()

    if tokenizer.unk_token == None and tokenizer.pad_token == None:
        # raw llama3
        print("adding a special padding token...")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        need_resize = True
    else:
        need_resize = False
    if need_resize:
        model_instance.resize_token_embeddings(len(tokenizer))

    prefix_length = 1 # prefix is default to 1 for all models due to the BOS token.
    if is_chat_model:
        prefix_length = get_prefix_length(tokenizer)
        logger.warning(f"Chat model prefix length: {prefix_length}")

    # Load dataset factory for evals.
    dataset_factory = DatasetFactory(
        None, client, tokenizer, generate_args.dataset_category, None, None, args.dump_dir,
        use_cache=False, master_data_dir=args.master_data_dir,
        lm_model=args.lm_model, logger=logger, is_inference=True,
        overwrite_inference_data_dir=training_args.overwrite_inference_data_dir
    )
    atexit.register(dataset_factory.save_cache)
    atexit.register(dataset_factory.reset_stats)

    has_latent_model = False
    for model_name in args.models:
        # load model on the fly to save memory
        if model_name not in LATENT_EXCLUDE_MODELS:
            has_latent_model = True
            break

    if not has_latent_model:
        logger.warning("No latent model to infer. Exiting.")
        return

    # Now loop over concept_ids and use preloaded models
    cache_df = {}
    for concept_id in my_concept_ids:
        for model_name in args.models:
            # load model on the fly to save memory
            if model_name in LATENT_EXCLUDE_MODELS:
                continue
            model_class = getattr(particlesteering, model_name)
            logger.warning(f"Loading {model_class} on {device}.")
            bm_kw = dict(low_rank_dimension=len(metadata), device=device)
            bm_kw.update(_particle_bm_init_kwargs(model_name, training_args, device))
            benchmark_model = model_class(
                model_instance, tokenizer, layer=layer,
                **bm_kw,
            )
            benchmark_model.load(
                dump_dir=train_dir, sae_path=metadata[0]["ref"], mode="latent",
                concept_id=concept_id
            )
            benchmark_model.to(device)
            if hasattr(benchmark_model, 'ax') and args.use_bf16:
                benchmark_model.ax.eval()
                benchmark_model.ax.to(torch.bfloat16)

            dataset_category = generate_args.dataset_category
            if (concept_id, dataset_category) not in cache_df:
                current_df = create_data_latent(
                    dataset_factory, metadata, concept_id, num_of_examples, args)
                logger.warning(f"Inference latent with {model_name} on {device} for concept {concept_id}.")
                current_df = prepare_df(current_df, tokenizer, is_chat_model, args.model_name)
                cache_df[(concept_id, dataset_category)] = current_df
            else:
                current_df = cache_df[(concept_id, dataset_category)]

            current_df = _predict_latent_into_df(
                benchmark_model,
                model_name,
                current_df,
                batch_size=args.latent_batch_size,
                prefix_length=prefix_length,
            )
            del benchmark_model
            torch.cuda.empty_cache()
        save(dump_dir, 'latent', current_df, rank)
        logger.warning(f"Saved inference results for concept {concept_id} to rank_{rank}_latent_data.parquet")
        # After processing, save state
        current_state = {'last_concept_id': concept_id}
        save_state(args.dump_dir, current_state, 'latent', rank)

    # Synchronize all processes
    dist.barrier()

    # Rank 0 merges results
    if rank == 0:
        logger.warning("Rank 0 is merging results.")
        # Merge per-rank results
        all_parquet_files = list(dump_dir.glob("rank_*_latent_data.parquet"))
        # Parse filenames to extract rank
        import re
        pattern = re.compile(r'rank_(\d+)_latent_data\.parquet')

        file_info_list = []
        for parquet_file in all_parquet_files:
            match = pattern.match(parquet_file.name)
            if match:
                rank_str = match.group(1)
                rank_int = int(rank_str)
                file_info_list.append({
                    'rank': rank_int,
                    'file': parquet_file
                })
            else:
                logger.warning(f"Filename {parquet_file.name} does not match the expected pattern.")

        # Sort the file_info_list by rank
        file_info_list.sort(key=lambda x: x['rank'])

        # Read and concatenate dataframes
        dfs = []
        for info in file_info_list:
            df = pd.read_parquet(info['file'])
            dfs.append(df)
        if len(dfs) > 0:
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df.to_parquet(dump_dir / "latent_data.parquet", engine='pyarrow')
            logger.warning(f"Saved combined latent inference results to {dump_dir / 'latent_data.parquet'}")
        else:
            logger.warning("No results to merge.")

        # Optionally, delete per-rank files
        for info in file_info_list:
            os.remove(info['file'])
            logger.warning(f"Deleted {info['file']}")


def main():
    custom_args = [
        {
            'args': ['--mode'],
            'kwargs': {
                'type': str,
                'default': "all",
                'help': 'The inference mode.'
            }
        }
    ]
    training_args = TrainingArgs(custom_args=custom_args, section="train", ignore_unknown=True)
    generate_args = DatasetArgs(custom_args=custom_args, section="generate", ignore_unknown=True)
    inference_args = DatasetArgs(custom_args=custom_args, section="inference", ignore_unknown=True)

    if training_args.overwrite_metadata_dir is not None and os.path.exists(training_args.overwrite_metadata_dir):
        inference_args.data_dir = training_args.overwrite_metadata_dir # since we only load metadata from this dir
    else:
        inference_args.data_dir = f"{inference_args.dump_dir}/generate"
    inference_args.train_dir = f"{inference_args.dump_dir}/train"
    logger.warning("Inferencing with following configuration:")
    logger.warning(inference_args)
    set_seed(inference_args.seed)

    # Initialize the process group
    dist.init_process_group(backend='nccl', init_method='env://', 
                          timeout=datetime.timedelta(seconds=60000))


    # Get the rank and world_size from environment variables
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    # Set the device for this process
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    # Configure the logger per rank
    logger.setLevel(logging.WARNING)  # Set the logging level as desired

    # Create a logging formatter that includes the rank
    formatter = logging.Formatter(
        fmt=f'%(asctime)s,%(msecs)03d %(levelname)-8s [Rank {rank}] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S'
    )

    # Create a console handler and set its formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    if not logger.handlers:
        logger.addHandler(console_handler)

    if inference_args.mode == "latent":
        infer_latent(inference_args, rank, world_size, device, logger, training_args, generate_args)
    elif inference_args.mode == "steering":
        infer_steering(inference_args, rank, world_size, device, logger, training_args, generate_args)
    elif inference_args.mode == "all":
        infer_latent(inference_args, rank, world_size, device, logger, training_args, generate_args)
        infer_steering(inference_args, rank, world_size, device, logger, training_args, generate_args)

    # Finalize the process group
    dist.destroy_process_group()

    # Remove handlers to prevent duplication if the script is run multiple times
    logger.removeHandler(console_handler)
    # If file_handler is used, remove it as well
    # logger.removeHandler(file_handler)


if __name__ == "__main__":
    main()


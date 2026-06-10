# Adapted from AxBench — https://github.com/stanfordnlp/axbench (Apache-2.0).
from .model import Model
from pyvene import (
    IntervenableConfig,
    IntervenableModel
)

import os, requests, torch
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
from .interventions import (
    JumpReLUSAECollectIntervention, 
    AdditionIntervention,
    SubspaceIntervention,
    DictionaryAdditionIntervention,
    DictionaryMinClampingIntervention,
)
from huggingface_hub import hf_hub_download

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)


def load_metadata_flatten(metadata_path):
    """
    Load flatten metadata from a JSON lines file.
    """
    metadata = []
    concept_id = 0
    with open(metadata_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            concept, ref = data["concept"], data["ref"]
            concept_genres_map = data["concept_genres_map"][concept]
            ref = data["ref"]
            flatten_data = {
                "concept": concept,
                "ref": ref,
                "concept_genres_map": {concept: concept_genres_map},
                "concept_id": concept_id
            }
            metadata += [flatten_data]  # Return the metadata as is
            concept_id += 1
    return metadata


def save_pruned_sae(
    metadata_path, dump_dir, modified_refs=None, savefile="GemmaScopeSAE.pt", sae_params=None
):
    # Save SAE weights and biases for inference
    logger.warning("Saving SAE weights and biases for inference")
    flatten_metadata = load_metadata_flatten(metadata_path)
    # logger.warning(flatten_metadata)

    # mess with SAE refs if needed (for DBM stuff)
    if modified_refs is not None:
        for i, ref in enumerate(modified_refs):
            flatten_metadata[i]["ref"] = "/".join(flatten_metadata[i]["ref"].split("/")[:-1]) + "/" + str(ref)

    # Save pruned SAE weights and biases
    # https://www.neuronpedia.org/llama3.1-8b/20-llamascope-res-131k/65591
    # https://www.neuronpedia.org/api/feature/llama3.1-8b/20-llamascope-res-131k/65591
    sae_path = flatten_metadata[0]["ref"].split("https://www.neuronpedia.org/")[-1]
    sae_url = f"https://www.neuronpedia.org/api/feature/{sae_path}"
    headers = {"X-Api-Key": os.environ.get("NP_API_KEY")}
    response = requests.get(sae_url, headers=headers).json()
    hf_repo = response["source"]["hfRepoId"]
    hf_folder = response["source"]["hfFolderId"]
    hf_filename = f"{hf_folder}/params.npz"
    if hf_repo is not None and "gemma" in hf_repo:
        if sae_params is None:
            logger.warning(f"Loading GemmaScope SAE params from {hf_repo}/{hf_filename}")
            path_to_params = hf_hub_download(
                repo_id=hf_repo,
                filename=hf_filename,
                force_download=False,
            )
            sae_params = np.load(path_to_params)
            logger.warning(f"Loaded SAE params from {path_to_params}")
            logger.warning(f"SAE params: {list(sae_params.keys())}")
    else:
        return None
    sae_pt_params = {k: torch.from_numpy(v) for k, v in sae_params.items()}
    pruned_sae_pt_params = {
        "b_dec": sae_pt_params["b_dec"],
        "W_dec": [],
        "W_enc": [],
        "b_enc": [],
        "threshold": []
    }
    for concept_id, m in enumerate(flatten_metadata):
        sae_id = int(m["ref"].split("/")[-1])
        pruned_sae_pt_params["W_dec"].append(sae_pt_params["W_dec"][[sae_id], :])
        pruned_sae_pt_params["W_enc"].append(sae_pt_params["W_enc"][:, [sae_id]])
        pruned_sae_pt_params["b_enc"].append(sae_pt_params["b_enc"][[sae_id]])
        pruned_sae_pt_params["threshold"].append(sae_pt_params["threshold"][[sae_id]])
    for k, v in pruned_sae_pt_params.items():
        if k == "b_dec":
            continue
        if k == "W_enc":
            pruned_sae_pt_params[k] = torch.cat(v, dim=1)
        else:
            pruned_sae_pt_params[k] = torch.cat(v, dim=0)
    torch.save(pruned_sae_pt_params, dump_dir / savefile) # sae only has one file
    return sae_params


class GemmaScopeSAE(Model):
    def __str__(self):
        return 'GemmaScopeSAE'
    
    def make_model(self, **kwargs):
        mode = kwargs.get("mode", "latent")
        intervention_type = kwargs.get("intervention_type", "addition")
        if mode == "steering":
            if intervention_type == "addition":
                ax = AdditionIntervention(
                    embed_dim=self.model.config.hidden_size, 
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),
                )
            elif intervention_type == "clamping":
                ax = DictionaryAdditionIntervention(
                    embed_dim=self.model.config.hidden_size, 
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),
                )
            elif intervention_type == "min_clamping":
                ax = DictionaryMinClampingIntervention(
                    embed_dim=self.model.config.hidden_size, 
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),
                )
            else:
                raise ValueError(f"Invalid intervention type for steering: {intervention_type}")
        else:
            ax = JumpReLUSAECollectIntervention(
                embed_dim=self.model.config.hidden_size, 
                low_rank_dimension=kwargs.get("low_rank_dimension", 1),
            )
        ax = ax.train()
        ax_config = IntervenableConfig(representations=[{
            "layer": l,
            "component": f"model.layers[{l}].output",
            "low_rank_dimension": kwargs.get("low_rank_dimension", 1),
            "intervention": ax} for l in [self.layer]])
        ax_model = IntervenableModel(ax_config, self.model)
        ax_model.set_device(self.device)
        self.ax = ax
        self.ax_model = ax_model

    def load(self, dump_dir=None, **kwargs):
        model_name = kwargs.get("model_name", self.__str__())
        logger.warning(f"Loading SAE from {dump_dir}/{model_name}.pt")
        params = torch.load(
            f"{dump_dir}/{model_name}.pt"
        )
        pt_params = {k: v.to(self.device) for k, v in params.items()}
        self.make_model(low_rank_dimension=params['W_enc'].shape[1], **kwargs)
        if isinstance(self.ax, SubspaceIntervention) or isinstance(self.ax, AdditionIntervention):
            self.ax.proj.weight.data = pt_params['W_dec']
        else:
            try:
                self.ax.load_state_dict(pt_params, strict=True)
            except Exception as e:
                logger.warning(f"Error loading state dict: {e}")
                self.ax.load_state_dict(pt_params, strict=False)

    def pre_compute_mean_activations(self, dump_dir, **kwargs):
        if kwargs.get("disable_neuronpedia_max_act", False):
            logger.warning(f"Using AxBench dataset max activations instead of Neuronpedia.")
            # we use max/mean activation from AxBench dataset.
            return super().pre_compute_mean_activations(dump_dir, **kwargs)

        # Loop over all praqut files in dump_dir.
        sae_links = []
        for file in os.listdir(dump_dir):
            if file.endswith(".parquet") and file.startswith("latent_data"):
                df = pd.read_parquet(os.path.join(dump_dir, file))
                # sort by concept_id from small to large and enumerate through all concept_ids.
                for sae_link in sorted(df["sae_link"].unique()):
                    sae_links += [sae_link]

        model_name, sae_name = sae_links[0].split("/")[-3], sae_links[0].split("/")[-2]
        max_activations = {} # sae_id to max_activation

        # Load existing max activations file and skip if exists.
        max_activations_file = os.path.join(
            kwargs.get("master_data_dir", "particlesteering/data"), 
            f"{model_name}_{sae_name}_max_activations.json")
        if os.path.exists(max_activations_file):
            with open(max_activations_file, "r") as f:
                max_activations = json.load(f)
            max_activations = {int(k): v for k, v in max_activations.items()}
        
        has_new = False
        for sae_link in tqdm(sae_links):
            sae_path = sae_link.split("https://www.neuronpedia.org/")[-1]
            sae_id = int(sae_link.split("/")[-1])
            if sae_id in max_activations:
                continue
            url = f"https://www.neuronpedia.org/api/feature/{sae_path}"
            headers = {"X-Api-Key": os.environ["NP_API_KEY"]}
            response = requests.get(url, headers=headers)
            max_activation = response.json()["activations"][0]["maxValue"]
            max_activations[sae_id] = max_activation if max_activation > 0 else 50
            has_new = True

        if has_new:
            with open(max_activations_file, "w") as f:
                json.dump(max_activations, f)

        self.max_activations = max_activations
        return max_activations

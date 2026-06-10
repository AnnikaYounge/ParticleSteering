# Derived from AxBench (preference_set.py) — https://github.com/stanfordnlp/axbench (Apache-2.0).
"""K-particle contrastive steering with optional repulsion (see particle_objectives.py)."""

from __future__ import annotations

import logging
import os
import random
from typing import List, Optional, Tuple

import pandas as pd
import torch
from pyvene import IntervenableConfig, IntervenableModel
from tqdm.auto import tqdm
from transformers import get_scheduler, set_seed

from .particle_utils import (
    _read_particle_init_config,
    _select_particle_indices,
    _steering_row_sampling_seed,
    _use_all_particles,
    init_particle_proj_weights,
    log_particle_weight_geometry,
)
from .particle_objectives import (
    aggregate_validity_loss,
    compute_repulsion_similarity,
    decompose_particle_objective,
    format_particle_loss_breakdown,
    normalize_particle_training_config,
    particle_config_to_dict,
    resolve_particle_weights_slice,
)
from .preference_model import (
    PreferenceModel,
    _get_batch_logps,
    preference_loss,
)
from .interventions import (
    AdditionIntervention,
    AdditionSuppressionIntervention,
    PreferenceVectorIntervention,
)
from ..utils.model_utils import (
    gather_residual_activations,
    get_lr,
    remove_gradient_parallel_to_decoder_directions,
    set_decoder_norm_to_unit_norm,
)

logger = logging.getLogger(__name__)


def _use_max_act_steering_calibration(training_args) -> bool:
    """SDL/LsReFT path: scale infer adds by calibrated max ReLU(h·w). Default False (RePS)."""
    if training_args is None:
        return False
    return bool(getattr(training_args, "use_max_act_steering_calibration", False))


def _use_unit_norm_steering_vectors(training_args) -> bool:
    """Keep ||w||=1 during train (LsReFT-style); infer magnitude via max_act * factor when calibrated."""
    if training_args is None:
        return False
    return bool(getattr(training_args, "use_unit_norm_steering_vectors", False))


def _use_sdl_training_intervention(training_args) -> bool:
    """Train with AdditionIntervention (mag * max_act * w), matching SDL infer geometry."""
    if training_args is None:
        return False
    return bool(getattr(training_args, "use_sdl_training_intervention", False))


def _resolve_steering_shared_rng_per_row(model: "ParticleSteering", kwargs: dict) -> bool:
    if "steering_shared_rng_per_row" in kwargs:
        return bool(kwargs["steering_shared_rng_per_row"])
    ta = model.training_args
    if ta is not None:
        return bool(getattr(ta, "steering_shared_rng_per_row", False))
    return False


def _intervention_location_mask(
    intervention_locations: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """True at steered token indices (first intervention group), excluding padding."""
    bsz, seq_len = attention_mask.shape
    device = attention_mask.device
    mask = torch.zeros(bsz, seq_len, dtype=torch.bool, device=device)
    locs = intervention_locations[:, 0, :].long()
    valid = (locs >= 0) & (locs < seq_len)
    batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand_as(locs)
    mask[batch_idx[valid], locs[valid]] = True
    return mask & attention_mask.bool()


def _sample_train_steering_factor(
    train_factors: List[float],
    *,
    randomize: bool,
) -> float:
    # Match PreferenceModel.train: random.choice(steering_factors) per example when
    # randomize is set or the list has more than one value (RePS overwrite_append grid).
    if randomize or len(train_factors) > 1:
        return float(random.choice(train_factors))
    return float(train_factors[0])


def _use_flat_weight_merge(n_weights: int, num_particles: int) -> bool:
    """True when proj holds merged multi-concept rows (infer compute_priority path)."""
    return n_weights > num_particles


def _steer_row_index(
    concept_id: int,
    particle_id: int,
    *,
    num_particles: int,
    particle_rank: int,
    use_flat_merge: bool,
) -> int:
    """Map (concept, particle) -> proj.weight row. Train uses local rows particle_id * R."""
    r = particle_rank
    if use_flat_merge:
        return int(concept_id) * num_particles * r + int(particle_id) * r
    return int(particle_id) * r


def _max_act_lookup(max_activations: dict, concept_id: int, particle_id: Optional[int] = None) -> float:
    cid = int(concept_id)
    if particle_id is not None:
        key = (cid, int(particle_id))
        if key in max_activations:
            return float(max_activations[key])
        str_key = f"{cid}:{int(particle_id)}"
        if str_key in max_activations:
            return float(max_activations[str_key])
    if cid in max_activations:
        return float(max_activations[cid])
    return 1.0


def _load_max_activations_from_latent_parquet(dump_dir: str, model_name: str) -> dict:
    """Per-(concept, particle) peak detector scores for SDL-style steering calibration."""
    max_activations: dict = {}
    col = f"{model_name}_max_act"
    if not os.path.isdir(dump_dir):
        return max_activations
    for file in os.listdir(dump_dir):
        if not (file.startswith("latent_") and file.endswith(".parquet")):
            continue
        latent_path = os.path.join(dump_dir, file)
        latent = pd.read_parquet(latent_path)
        if col not in latent.columns:
            continue
        if "particle_id" in latent.columns:
            for (concept_id, particle_id), grp in latent.groupby(
                ["concept_id", "particle_id"], dropna=False
            ):
                max_act = float(grp[col].max())
                if max_act <= 0:
                    max_act = 50.0
                cid = int(concept_id)
                pid = int(particle_id)
                max_activations[(cid, pid)] = max_act
                max_activations[f"{cid}:{pid}"] = max_act
        for concept_id in sorted(latent["concept_id"].unique()):
            concept_latent = latent[latent["concept_id"] == concept_id]
            max_act = float(concept_latent[col].max())
            if max_act <= 0:
                max_act = 50.0
            max_activations[int(concept_id)] = max_act
    return max_activations


class ParticleSteering(PreferenceModel):
    """K-vector preference steering with particle repulsion (contrastive validity)."""

    preference_pairs = ["orig_add"]

    def __str__(self):
        return "ParticleSteering"

    @property
    def num_particles(self) -> int:
        ta = self.training_args
        v = getattr(ta, "num_particles", None)
        if v is not None:
            return int(v)
        return 1

    @property
    def particle_rank(self) -> int:
        ta = self.training_args
        return int(getattr(ta, "low_rank_dimension", None) or 1)

    def _proj_row_count(self, n: int, r: int, low_rank_dimension: Optional[int]) -> int:
        if low_rank_dimension is not None and low_rank_dimension > n * r:
            return int(low_rank_dimension)
        return n * r

    def make_model(self, **kwargs):
        mode = kwargs.get("mode", "latent")
        overwrite_component = kwargs.get("overwrite_component", None)
        n = self.num_particles
        r = self.particle_rank
        if r != 1:
            raise ValueError(
                "ParticleSteering currently requires yaml low_rank_dimension=1 (rank R per particle)"
            )
        # Match LsReFTParticles: yaml R=1, proj / IntervenableConfig use N*R storage rows.
        proj_rows = self._proj_row_count(n, r, kwargs.get("low_rank_dimension"))
        kwargs = dict(kwargs, low_rank_dimension=proj_rows)

        if mode == "steering":
            intervention_type = kwargs.get("intervention_type", "addition")
            if intervention_type == "addition":
                ax = AdditionIntervention(
                    embed_dim=kwargs.get("embed_dim", self.model.config.hidden_size),
                    low_rank_dimension=proj_rows,
                )
            elif intervention_type == "addition_suppression":
                ax = AdditionSuppressionIntervention(
                    embed_dim=kwargs.get("embed_dim", self.model.config.hidden_size),
                    low_rank_dimension=proj_rows,
                )
            else:
                raise ValueError(f"Intervention type {intervention_type} not supported")
        elif _use_sdl_training_intervention(self.training_args):
            ax = AdditionIntervention(
                embed_dim=kwargs.get("embed_dim", self.model.config.hidden_size),
                low_rank_dimension=proj_rows,
            )
        else:
            ax = PreferenceVectorIntervention(
                embed_dim=kwargs.get("embed_dim", self.model.config.hidden_size),
                low_rank_dimension=proj_rows,
                dropout=kwargs.get("dropout", 0.0),
                intervention_positions_dropout=kwargs.get(
                    "intervention_positions_dropout", 0.0
                ),
            )

        self.intervention_type = kwargs.get("intervention_type", "addition")
        self._sdl_training_intervention = _use_sdl_training_intervention(self.training_args)
        layers = self.steering_layers if self.steering_layers else [self.layer]
        self.ax = ax.to(self.device)
        self.ax.train()
        ax_config = IntervenableConfig(
            representations=[
                {
                    "layer": l,
                    "component": (
                        f"model.layers[{l}].output"
                        if overwrite_component is None
                        else overwrite_component
                    ),
                    "low_rank_dimension": proj_rows,
                    "intervention": self.ax,
                }
                for l in layers
            ]
        )
        ax_model = IntervenableModel(ax_config, self.model)
        ax_model.set_device(self.device)
        self.ax_model = ax_model
        self.preference_pairs = kwargs.get("preference_pairs", ["orig_add"])

        if n > 1 and mode != "steering":
            ta = self.training_args
            init_mode, scale_min, scale_max = _read_particle_init_config(ta)
            cid = int(kwargs.get("concept_id", 0) or 0)
            base_seed = int(getattr(ta, "seed", self.seed) or 0)
            seed = base_seed if init_mode == "orthogonal" else base_seed + cid
            gen = torch.Generator(device=self.device)
            gen.manual_seed(seed)
            init_particle_proj_weights(
                self.ax.proj,
                n,
                r,
                mode=init_mode,
                scale_min=scale_min,
                scale_max=scale_max,
                generator=gen,
            )
            set_decoder_norm_to_unit_norm(self.ax)
            # RePS default: no per-step renorm (||w|| learnable). SDL: use_unit_norm_steering_vectors
            # reprojects each optimizer step (see train loop).

    def _stack_preference_minibatch(
        self,
        winning_inputs: dict,
        losing_inputs: dict,
        start_idx: int,
        end_idx: int,
        steering_factors: torch.Tensor,
    ) -> dict:
        stacked = {
            k: torch.stack(
                winning_inputs[k][start_idx:end_idx] + losing_inputs[k][start_idx:end_idx],
                dim=0,
            ).to(self.device)
            for k in winning_inputs
            if k != "steering_factors"
        }
        stacked["steering_factors"] = steering_factors.to(self.device)
        return stacked

    @torch.no_grad()
    def _compute_prompt_max_act_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        row_idx: int,
    ) -> torch.Tensor:
        """Peak ReLU(h·w) on prompt tokens (label == -100), aligned with latent max_act calibration."""
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        h = gather_residual_activations(self.model, self.layer, inputs)
        w = self.ax.proj.weight[row_idx].to(device=h.device, dtype=h.dtype)
        if _use_unit_norm_steering_vectors(self.training_args):
            w = w / w.norm().clamp(min=1e-8)
        scores = torch.relu(torch.matmul(h, w))
        prompt_mask = labels.eq(-100) & attention_mask.bool()
        scores = scores.masked_fill(~prompt_mask, 0.0)
        return scores.max(dim=-1).values.clamp(min=1.0).float()

    def _dpo_subspaces_for_minibatch(
        self, minibatch_inputs: dict, row_idx: int
    ) -> List[dict]:
        """Build pyvene subspaces for one DPO minibatch (legacy PV vs SDL Addition)."""
        bsz = minibatch_inputs["input_ids"].shape[0]
        if not _use_sdl_training_intervention(self.training_args):
            return [
                {
                    "idx": row_idx,
                    "steering_factor": minibatch_inputs["steering_factors"],
                }
            ] * self.num_of_layers

        half = bsz // 2
        peak_w = self._compute_prompt_max_act_batch(
            minibatch_inputs["input_ids"][:half],
            minibatch_inputs["attention_mask"][:half],
            minibatch_inputs["labels"][:half],
            row_idx,
        )
        max_acts = torch.cat([peak_w, peak_w], dim=0).to(self.device)
        idx_t = torch.full((bsz,), row_idx, dtype=torch.long, device=self.device)
        mag = minibatch_inputs["steering_factors"].float().to(self.device)
        return [
            {"idx": idx_t, "mag": mag, "max_act": max_acts}
        ] * self.num_of_layers

    def _dpo_loss_for_minibatch(
        self,
        minibatch_inputs: dict,
        unit_locations: dict,
        row_idx: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """DPO loss for one particle row on win+lose stacked minibatch."""
        bsz = minibatch_inputs["input_ids"].shape[0]
        subspaces = self._dpo_subspaces_for_minibatch(minibatch_inputs, row_idx)

        ref_outputs, policy_outputs = self.ax_model(
            base={
                "input_ids": minibatch_inputs["input_ids"],
                "attention_mask": minibatch_inputs["attention_mask"],
            },
            unit_locations=unit_locations,
            output_original_output=True,
            subspaces=subspaces,
            use_cache=False,
        )

        policy_logps = _get_batch_logps(
            policy_outputs.logits, minibatch_inputs["labels"], average_log_prob=False
        )
        ref_logps = _get_batch_logps(
            ref_outputs.logits, minibatch_inputs["labels"], average_log_prob=False
        )

        half = bsz // 2
        pos_loss_kwargs = {
            "beta": self.training_args.beta,
            "gemma": self.training_args.gemma,
            "simpo_scaler": self.training_args.simpo_scaler,
            "reference_free": self.training_args.reference_free,
            "label_smoothing": self.training_args.label_smoothing,
            "loss_type": self.training_args.loss_type,
            "winning_lens": minibatch_inputs["attention_mask"][:half].sum(dim=-1),
            "losing_lens": minibatch_inputs["attention_mask"][half:].sum(dim=-1),
        }
        steer_losses, _, _ = preference_loss(
            policy_logps[:half],
            policy_logps[half:],
            ref_logps[:half],
            ref_logps[half:],
            **pos_loss_kwargs,
        )
        loss = steer_losses.mean()

        latent_tensor = None
        win_mask = None
        if (
            self.ax_model.full_intervention_outputs
            and len(self.ax_model.full_intervention_outputs) > 0
        ):
            latent_pack = self.ax_model.full_intervention_outputs[0].latent
            if latent_pack is not None:
                lat = latent_pack[0] if isinstance(latent_pack, (tuple, list)) else latent_pack
                latent_tensor = lat[:half]
                win_mask = _intervention_location_mask(
                    minibatch_inputs["intervention_locations"][:half],
                    minibatch_inputs["attention_mask"][:half],
                )

        return loss, latent_tensor, win_mask

    def train(self, examples, **kwargs):
        if not kwargs.get("use_dpo_loss", False):
            raise ValueError(
                "ParticleSteering requires contrastive (DPO) training data. "
                "Run train.py with --use_dpo_loss (and dpo_train_data.parquet)."
            )
        if self.use_wandb:
            import wandb

            logging_metadata = kwargs["logging_metadata"]
            run_name = (
                f"{logging_metadata['model_name']}_{logging_metadata['layer']}_"
                f"{logging_metadata['concept_id']}"
            )
            wandb_proj = kwargs.get("wandb_project", None)
            wandb_name = kwargs.get("wandb_name", None)
            run = wandb.init(
                project=f"{wandb_proj}",
                entity=wandb_name,
                name=run_name,
                dir="wandb",
            )

        train_dataloader = self.make_preference_dataloader(examples, **kwargs)
        torch.cuda.empty_cache()

        optimizer = torch.optim.AdamW(
            self.ax_model.parameters(),
            lr=self.training_args.lr,
            weight_decay=self.training_args.weight_decay,
        )
        num_training_steps = self.training_args.n_epochs * (
            len(train_dataloader) // self.training_args.gradient_accumulation_steps
        )
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

        ta = self.training_args
        obj_cfg = normalize_particle_training_config(ta)
        N = self.num_particles
        R = self.particle_rank
        train_factors = list(getattr(ta, "steering_factors", None) or [1.0])
        if not train_factors:
            train_factors = [1.0]
        randomize_train_factors = bool(
            getattr(ta, "randomize_train_steering_factors", False)
        )
        use_max_act_cal = _use_max_act_steering_calibration(ta)
        use_unit_norm = _use_unit_norm_steering_vectors(ta)
        use_sdl_train = _use_sdl_training_intervention(ta)
        rank = torch.distributed.get_rank()

        if rank == 0:
            init_mode, scale_min, scale_max = _read_particle_init_config(ta)
            print("[ParticleSteering] objective:", particle_config_to_dict(obj_cfg), flush=True)
            print(
                "[ParticleSteering] weight_init:",
                {
                    "particle_weight_init": init_mode,
                    "particle_init_scale_min": scale_min,
                    "particle_init_scale_max": scale_max,
                    "train_steering_factors": train_factors,
                    "randomize_train_steering_factors": randomize_train_factors,
                    "use_max_act_steering_calibration": use_max_act_cal,
                    "use_unit_norm_steering_vectors": use_unit_norm,
                    "use_sdl_training_intervention": use_sdl_train,
                    "train_intervention": (
                        "AdditionIntervention (mag*max_act*w)"
                        if use_sdl_train
                        else "PreferenceVectorIntervention ((alpha+bias)*v)"
                    ),
                },
                flush=True,
            )
            if use_sdl_train and not use_unit_norm:
                print(
                    "[ParticleSteering] WARNING: use_sdl_training_intervention without "
                    "use_unit_norm_steering_vectors — ||w|| may drift from infer unit-norm policy.",
                    flush=True,
                )
            if use_sdl_train and obj_cfg.repulsion_type == "latent_corr":
                print(
                    "[ParticleSteering] latent_corr: functional repulsion on ReLU(h·w) "
                    "from AdditionIntervention (SDL train path).",
                    flush=True,
                )
            if use_unit_norm and not use_max_act_cal:
                print(
                    "[ParticleSteering] WARNING: use_unit_norm_steering_vectors without "
                    "use_max_act_steering_calibration — infer will use ||w||≈1 (RePS factor*w only).",
                    flush=True,
                )
            if N > 1:
                log_particle_weight_geometry(
                    self.ax.proj.weight,
                    N,
                    R,
                    rbf_bandwidth=obj_cfg.repulsion_bandwidth,
                    label="pre-train",
                    init_mode=init_mode,
                )
            if N > 1 and not obj_cfg.active_repulsion():
                print(
                    "[ParticleSteering] WARNING: repulsion inactive "
                    "(repulsion_type=none or strength=0).",
                    flush=True,
                )

        progress_bar = tqdm(range(num_training_steps), position=rank, leave=True)
        curr_step = 0
        collect_latents = obj_cfg.needs_latent_tensors_for_repulsion() and N > 1

        for epoch in range(self.training_args.n_epochs):
            for step, batch in enumerate(train_dataloader):
                expanded_batch_size = self.training_args.batch_size * len(
                    self.preference_pairs
                )
                minibatch_size = self.training_args.batch_size
                num_minibatches = (
                    expanded_batch_size + minibatch_size - 1
                ) // minibatch_size

                winning_inputs = {
                    "input_ids": [],
                    "attention_mask": [],
                    "labels": [],
                    "intervention_locations": [],
                    "steering_factors": [],
                }
                losing_inputs = {
                    "input_ids": [],
                    "attention_mask": [],
                    "labels": [],
                    "intervention_locations": [],
                    "steering_factors": [],
                }

                for i in range(self.training_args.batch_size):
                    for pair in self.preference_pairs:
                        winning_inputs["input_ids"].append(
                            batch[f"{pair}_winning_input_ids"][i]
                        )
                        winning_inputs["attention_mask"].append(
                            batch[f"{pair}_winning_attention_mask"][i]
                        )
                        winning_inputs["labels"].append(
                            batch[f"{pair}_winning_labels"][i]
                        )
                        winning_inputs["intervention_locations"].append(
                            batch[f"{pair}_winning_intervention_locations"][i]
                        )
                        winning_inputs["steering_factors"].append(
                            torch.tensor(
                                _sample_train_steering_factor(
                                    train_factors,
                                    randomize=randomize_train_factors,
                                )
                            )
                        )
                        losing_inputs["input_ids"].append(
                            batch[f"{pair}_losing_input_ids"][i]
                        )
                        losing_inputs["attention_mask"].append(
                            batch[f"{pair}_losing_attention_mask"][i]
                        )
                        losing_inputs["labels"].append(
                            batch[f"{pair}_losing_labels"][i]
                        )
                        losing_inputs["intervention_locations"].append(
                            batch[f"{pair}_losing_intervention_locations"][i]
                        )
                        losing_inputs["steering_factors"].append(
                            torch.tensor(
                                _sample_train_steering_factor(
                                    train_factors,
                                    randomize=randomize_train_factors,
                                )
                            )
                        )

                per_particle_losses: List[torch.Tensor] = []
                latent_per_particle: List[torch.Tensor] = []
                latent_masks: List[torch.Tensor] = []

                for particle_id in range(N):
                    row_idx = particle_id * R
                    particle_mb_losses: List[torch.Tensor] = []
                    particle_latents: List[torch.Tensor] = []
                    particle_masks: List[torch.Tensor] = []

                    for mb in range(num_minibatches):
                        start_idx = mb * minibatch_size
                        end_idx = min((mb + 1) * minibatch_size, expanded_batch_size)
                        if start_idx >= expanded_batch_size:
                            break

                        mb_win_factors = [
                            float(x.item()) if torch.is_tensor(x) else float(x)
                            for x in winning_inputs["steering_factors"][start_idx:end_idx]
                        ]
                        mb_lose_factors = [
                            float(x.item()) if torch.is_tensor(x) else float(x)
                            for x in losing_inputs["steering_factors"][start_idx:end_idx]
                        ]
                        mb_factors = torch.tensor(
                            mb_win_factors + mb_lose_factors,
                            dtype=torch.float32,
                            device=self.device,
                        )
                        minibatch_inputs = self._stack_preference_minibatch(
                            winning_inputs,
                            losing_inputs,
                            start_idx,
                            end_idx,
                            mb_factors,
                        )
                        unit_locations = {
                            "sources->base": (
                                None,
                                minibatch_inputs["intervention_locations"]
                                .permute(1, 0, 2)
                                .tolist(),
                            )
                        }

                        loss_mb, lat_mb, lat_mask = self._dpo_loss_for_minibatch(
                            minibatch_inputs, unit_locations, row_idx
                        )
                        particle_mb_losses.append(loss_mb)
                        if collect_latents and lat_mb is not None:
                            particle_latents.append(lat_mb)
                            if lat_mask is not None:
                                particle_masks.append(lat_mask)

                    if not particle_mb_losses:
                        continue
                    per_particle_losses.append(torch.stack(particle_mb_losses).mean())
                    if collect_latents and particle_latents:
                        latent_per_particle.append(torch.cat(particle_latents, dim=0))
                        if particle_masks:
                            latent_masks.append(torch.cat(particle_masks, dim=0))

                if not per_particle_losses:
                    continue

                l_valid = aggregate_validity_loss(
                    torch.stack(per_particle_losses), obj_cfg
                )
                r_sim = l_valid.new_zeros(())
                if N > 1:
                    W = resolve_particle_weights_slice(self.ax.proj.weight, N, R)
                    if use_unit_norm:
                        W = torch.nn.functional.normalize(W.float(), dim=1)
                    # Winning-side attention mask is shared across particles (same prompts).
                    intervention_mask = latent_masks[0] if latent_masks else None
                    if collect_latents and latent_per_particle:
                        r_sim = compute_repulsion_similarity(
                            obj_cfg,
                            weights=W,
                            latent_per_particle=latent_per_particle,
                            intervention_mask=intervention_mask,
                        )
                    else:
                        r_sim = compute_repulsion_similarity(
                            obj_cfg,
                            weights=W,
                        )

                loss_bd = decompose_particle_objective(l_valid, r_sim, obj_cfg, curr_step)
                loss_accum = loss_bd.total / self.training_args.gradient_accumulation_steps
                loss_accum.backward()

                if (step + 1) % self.training_args.gradient_accumulation_steps == 0 or (
                    step + 1
                ) == len(train_dataloader):
                    torch.nn.utils.clip_grad_norm_(self.ax_model.parameters(), 1.0)
                    if use_unit_norm:
                        set_decoder_norm_to_unit_norm(self.ax)
                        if self.ax.proj.weight.grad is not None:
                            remove_gradient_parallel_to_decoder_directions(self.ax)
                    curr_step += 1
                    curr_lr = get_lr(optimizer)
                    optimizer.step()
                    if use_unit_norm:
                        set_decoder_norm_to_unit_norm(self.ax)
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    progress_bar.set_description(
                        "lr %.4f | %s | %s/%s"
                        % (
                            curr_lr,
                            format_particle_loss_breakdown(loss_bd),
                            obj_cfg.repulsion_type,
                            obj_cfg.repulsion_formulation,
                        )
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "loss/train": float(loss_bd.total.detach()),
                                "loss/validity": float(loss_bd.l_validity.detach()),
                                "loss/repulsion": float(loss_bd.repulsion_term.detach()),
                            },
                            step=curr_step,
                        )

        progress_bar.close()
        if self.use_wandb:
            run.finish()

    @torch.no_grad()
    def predict_latent(self, examples, **kwargs):
        self.ax.eval()
        batch_size = kwargs.get("batch_size", 32)
        return_max_act_only = kwargs.get("return_max_act_only", False)
        is_chat_model = kwargs.get("is_chat_model", False)
        eager_prepare_df = kwargs.get("eager_prepare_df", False)
        overwrite_concept_id = kwargs.get("overwrite_concept_id", None)
        particle_id = int(kwargs.get("particle_id", 0))
        R = self.particle_rank

        from ..scripts.inference import prepare_df

        all_acts = []
        all_max_act = []
        all_max_act_idx = []
        all_max_token = []
        all_tokens = []
        n_weights = self.ax.proj.weight.shape[0]

        progress_bar = tqdm(range(0, len(examples), batch_size), desc="Processing batches")
        for i in progress_bar:
            batch = examples.iloc[i : i + batch_size]
            if eager_prepare_df:
                batch = prepare_df(batch, self.tokenizer, is_chat_model)

            inputs = self.tokenizer(
                batch["input"].tolist(),
                return_tensors="pt",
                add_special_tokens=True,
                padding=True,
                truncation=True,
            ).to(self.device)

            gather_acts = gather_residual_activations(self.model, self.layer, inputs)
            use_flat_merge = _use_flat_weight_merge(n_weights, self.num_particles)
            row_idx = [
                _steer_row_index(
                    int(overwrite_concept_id) if overwrite_concept_id is not None else int(c),
                    particle_id,
                    num_particles=self.num_particles,
                    particle_rank=R,
                    use_flat_merge=use_flat_merge,
                )
                for c in batch["concept_id"].tolist()
            ]

            row_tensor = torch.tensor(row_idx, dtype=torch.long, device=self.device)
            tail = gather_acts[:, kwargs["prefix_length"] :]
            if getattr(self, "_sdl_training_intervention", False):
                # SDL train/infer alignment path: AdditionIntervention has no latent detector.
                # Compute detector scores directly: s_t = ReLU(h_t · w_row) (no bias).
                w = self.ax.proj.weight[row_tensor].to(device=tail.device, dtype=tail.dtype)
                if _use_unit_norm_steering_vectors(self.training_args):
                    w = w / w.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                ax_acts = torch.relu(torch.einsum("bsh,bh->bs", tail, w)).detach().cpu().float()
            else:
                outputs = self.ax(
                    tail,
                    subspaces={"subspaces": row_tensor},
                )
                ax_acts = outputs.latent[0].float().detach().cpu()

            seq_lens = inputs["attention_mask"].sum(dim=1) - kwargs["prefix_length"]
            for seq_idx, ax_seq in enumerate(ax_acts):
                acts = ax_seq[: seq_lens[seq_idx]].flatten().data.numpy().tolist()
                acts = [round(x, 3) for x in acts]
                max_act = max(acts) if acts else 0.0
                all_max_act.append(max_act)
                if not return_max_act_only:
                    max_act_indices = [j for j, x in enumerate(acts) if x == max_act]
                    max_act_idx = max_act_indices[0] if max_act_indices else 0
                    tokens = self.tokenizer.tokenize(batch.iloc[seq_idx]["input"])[
                        kwargs["prefix_length"] - 1 :
                    ]
                    max_token = tokens[max_act_idx] if max_act_idx < len(tokens) else ""
                    all_acts.append(acts)
                    all_max_act_idx.append(max_act_idx)
                    all_max_token.append(max_token)
                    all_tokens.append(tokens)

            del ax_acts
            del gather_acts
            torch.cuda.empty_cache()

        if return_max_act_only:
            return {"max_act": all_max_act, "particle_id": particle_id}
        return {
            "acts": all_acts,
            "max_act": all_max_act,
            "max_act_idx": all_max_act_idx,
            "max_token": all_max_token,
            "tokens": all_tokens,
            "particle_id": particle_id,
        }

    @torch.no_grad()
    def predict_steer(self, examples, **kwargs):
        self.ax.eval()
        self.tokenizer.padding_side = "left"
        concept_id_col = (
            "sae_id"
            if "sae" in self.__str__().lower()
            and not kwargs.get("disable_neuronpedia_max_act", False)
            else "concept_id"
        )
        use_synergy = kwargs.get("use_synergy", False)

        batch_size = kwargs.get("batch_size", 64)
        eval_output_length = kwargs.get("eval_output_length", 128)
        temperature = kwargs.get("temperature", 1.0)
        prefix_length = kwargs["prefix_length"]

        policy = kwargs.get("steering_particle_policy", "all")
        particle_index = kwargs.get("steering_particle_index", None)
        steer_all_particles = kwargs.get("steer_all_particles", True)
        steering_shared_rng_per_row = _resolve_steering_shared_rng_per_row(self, kwargs)
        steering_seed = int(
            kwargs.get("steering_seed", getattr(self.training_args, "seed", 42))
        )
        N = self.num_particles
        R = self.particle_rank
        use_all_k = _use_all_particles(policy, particle_index, steer_all_particles) and N > 1
        _shared_rng_warned = False
        n_weights = self.ax.proj.weight.shape[0]
        use_flat_merge = _use_flat_weight_merge(n_weights, N)

        rank = torch.distributed.get_rank()
        progress_bar = tqdm(range(0, len(examples), batch_size), position=rank, leave=True)

        all_generations = []
        all_strengths = []
        all_particle_ids = []

        for i in range(0, len(examples), batch_size):
            batch_examples = examples.iloc[i : i + batch_size]
            if use_synergy:
                input_strings = batch_examples["steered_input"].tolist()
            else:
                input_strings = batch_examples["input"].tolist()
            mag = torch.tensor(batch_examples["factor"].tolist()).to(self.device)
            concept_ids = batch_examples[concept_id_col].tolist()

            inputs = self.tokenizer(
                input_strings, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            bsz = len(concept_ids)

            def run_generate(idxs, particle_ids_for_batch):
                if _use_max_act_steering_calibration(self.training_args):
                    max_acts = torch.tensor(
                        [
                            _max_act_lookup(
                                self.max_activations, concept_ids[bi], particle_ids_for_batch[bi]
                            )
                            for bi in range(bsz)
                        ],
                        device=self.device,
                    )
                else:
                    # RePS: AdditionIntervention uses factor * w only (max_act fixed at 1).
                    max_acts = torch.ones(bsz, device=self.device)
                idx_t = torch.tensor(idxs, dtype=torch.long, device=self.device)
                _, generations = self.ax_model.generate(
                    inputs,
                    unit_locations=None,
                    intervene_on_prompt=True,
                    subspaces=[
                        {
                            "idx": idx_t,
                            "mag": mag,
                            "max_act": max_acts,
                            "prefix_length": prefix_length,
                        }
                    ]
                    * self.num_of_layers,
                    max_new_tokens=eval_output_length,
                    do_sample=True,
                    temperature=temperature,
                )
                input_lengths = [len(ids) for ids in inputs.input_ids]
                return [
                    self.tokenizer.decode(gen[il:], skip_special_tokens=True)
                    for gen, il in zip(generations, input_lengths)
                ]

            if use_all_k:
                per_particle_gens = []
                if steering_shared_rng_per_row and bsz > 1 and not _shared_rng_warned:
                    logger.warning(
                        "steering_shared_rng_per_row requires steering_batch_size=1; "
                        "skipping per-row RNG reset (bsz=%s).",
                        bsz,
                    )
                    _shared_rng_warned = True
                for j in range(N):
                    if steering_shared_rng_per_row and bsz == 1:
                        row = batch_examples.iloc[0]
                        set_seed(
                            _steering_row_sampling_seed(
                                steering_seed,
                                int(row["concept_id"]),
                                int(row["input_id"]),
                                float(row["factor"]),
                            )
                        )
                    steer_idxs = [
                        _steer_row_index(
                            int(c),
                            j,
                            num_particles=N,
                            particle_rank=R,
                            use_flat_merge=use_flat_merge,
                        )
                        for c in concept_ids
                    ]
                    pids = [j] * bsz
                    per_particle_gens.append(run_generate(steer_idxs, pids))
                for bi in range(bsz):
                    for j in range(N):
                        mag_val = float(mag[bi].item())
                        if _use_max_act_steering_calibration(self.training_args):
                            max_a = _max_act_lookup(self.max_activations, concept_ids[bi], j)
                            strength = mag_val * max_a
                        else:
                            strength = mag_val
                        all_generations.append(per_particle_gens[j][bi])
                        all_strengths.append(strength)
                        all_particle_ids.append(j)
            else:
                pidx = _select_particle_indices(bsz, N, policy, particle_index, self.device)
                steer_idxs = [
                    _steer_row_index(
                        int(c),
                        int(pidx[bi].item()),
                        num_particles=N,
                        particle_rank=R,
                        use_flat_merge=use_flat_merge,
                    )
                    for bi, c in enumerate(concept_ids)
                ]
                pids = [int(pidx[bi].item()) for bi in range(bsz)]
                gens = run_generate(steer_idxs, pids)
                all_generations.extend(gens)
                for bi in range(bsz):
                    pid = int(pidx[bi].item())
                    mag_val = float(mag[bi].item())
                    if _use_max_act_steering_calibration(self.training_args):
                        strength = mag_val * _max_act_lookup(
                            self.max_activations, concept_ids[bi], pid
                        )
                    else:
                        strength = mag_val
                    all_strengths.append(strength)
                all_particle_ids.extend(pids)

            progress_bar.update(1)

        progress_bar.close()

        return {
            "steered_generation": all_generations,
            "strength": all_strengths,
            "particle_id": all_particle_ids,
            "_steer_all_particles": use_all_k,
        }

    def pre_compute_mean_activations(self, dump_dir, **kwargs):
        """Load latent peak scores only when use_max_act_steering_calibration is set."""
        if not _use_max_act_steering_calibration(self.training_args):
            # RePS default: same as PreferenceModel — infer uses factor * w, not max_act.
            self.max_activations = {}
            return self.max_activations
        self.max_activations = _load_max_activations_from_latent_parquet(
            dump_dir, self.__str__()
        )
        return self.max_activations

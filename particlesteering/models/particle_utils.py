# Adapted from AxBench — https://github.com/stanfordnlp/axbench (Apache-2.0).
"""Shared helpers for K-particle init, geometry logging, and steering inference."""

from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd
import torch

from .particle_objectives import (
    ParticleTrainingConfig,
    compute_repulsion_similarity,
    resolve_particle_weights_slice,
)

PARTICLE_WEIGHT_INIT_MODES = ("legacy", "random", "orthogonal")


def _read_particle_init_config(ta: object) -> Tuple[str, float, float]:
    mode = getattr(ta, "particle_weight_init", None) or "orthogonal"
    mode = str(mode).strip().lower()
    if mode not in PARTICLE_WEIGHT_INIT_MODES:
        raise ValueError(
            f"particle_weight_init must be one of {PARTICLE_WEIGHT_INIT_MODES}, got {mode!r}"
        )
    scale_min = float(getattr(ta, "particle_init_scale_min", None) or 0.01)
    scale_max = float(getattr(ta, "particle_init_scale_max", None) or 0.03)
    if scale_min <= 0.0 or scale_max <= 0.0 or scale_min > scale_max:
        raise ValueError(
            f"particle_init_scale_min/max must be positive with min <= max, got {scale_min}, {scale_max}"
        )
    return mode, scale_min, scale_max


@torch.no_grad()
def init_particle_proj_weights(
    proj: torch.nn.Linear,
    num_particles: int,
    particle_rank: int,
    *,
    mode: str = "random",
    scale_min: float = 0.01,
    scale_max: float = 0.03,
    generator: Optional[torch.Generator] = None,
) -> None:
    """(Re)initialize the first K steering rows in proj.weight."""
    n = num_particles
    r = particle_rank
    n_rows = n * r
    if n_rows <= 0:
        return
    mode = mode.lower()
    if mode not in PARTICLE_WEIGHT_INIT_MODES:
        raise ValueError(f"Unknown particle_weight_init mode: {mode}")

    block = proj.weight[:n_rows]
    if mode == "legacy":
        block.fill_(scale_min)
        if proj.bias is not None:
            proj.bias.zero_()
        return

    if mode == "orthogonal":
        torch.nn.init.orthogonal_(block, generator=generator)
    elif mode == "random":
        block.normal_(0.0, 1.0, generator=generator)
        block.div_(block.norm(dim=1, keepdim=True).clamp(min=1e-8))

    if proj.bias is not None:
        proj.bias.zero_()


@torch.no_grad()
def log_particle_weight_geometry(
    proj_weight: torch.Tensor,
    num_particles: int,
    particle_rank: int,
    *,
    rbf_bandwidth: float = 1.0,
    label: str = "init",
    init_mode: Optional[str] = None,
) -> None:
    """Print mean RBF similarity R(W) over K particle rows (lower = more spread)."""
    w = resolve_particle_weights_slice(proj_weight, num_particles, particle_rank)
    if w.shape[0] < 2:
        return
    r_cfg = ParticleTrainingConfig(
        repulsion_type="rbf", repulsion_bandwidth=rbf_bandwidth
    )
    r = float(compute_repulsion_similarity(r_cfg, weights=w).item())
    wn = torch.nn.functional.normalize(w.float(), dim=1)
    cos = wn @ wn.T
    k = cos.shape[0]
    mask = ~torch.eye(k, dtype=torch.bool, device=cos.device)
    off = cos[mask]
    mean_abs_cos = float(off.abs().mean().item())
    max_abs_cos = float(off.abs().max().item())
    mode_tag = f" init={init_mode}" if init_mode else ""
    print(
        f"[ParticleSteering] {label} geometry{mode_tag}: "
        f"R_rbf={r:.4f} mean_|cos|={mean_abs_cos:.4f} max_|cos|={max_abs_cos:.4f}",
        flush=True,
    )
    if init_mode == "orthogonal" and max_abs_cos > 1e-3:
        print(
            "[ParticleSteering] WARNING: particle_weight_init=orthogonal but "
            f"max_|cos|={max_abs_cos:.4f} (expected ~0). Check init code / GPU sync.",
            flush=True,
        )
    elif init_mode == "random" and max_abs_cos < 1e-5:
        print(
            "[ParticleSteering] NOTE: particle_weight_init=random but max_|cos|≈0 "
            "(looks like orthogonal init was applied).",
            flush=True,
        )


def _steering_row_sampling_seed(
    base_seed: int, concept_id: int, input_id: int, factor: float
) -> int:
    """Deterministic seed for one (concept, prompt, factor); excludes particle_id."""
    return (
        int(base_seed)
        + int(concept_id) * 100_000
        + int(input_id) * 1_000
        + int(round(float(factor) * 100))
    )


def _use_all_particles(
    policy: str,
    particle_index: Optional[int],
    steer_all_particles: Optional[bool],
) -> bool:
    if particle_index is not None:
        return False
    if steer_all_particles is False:
        return False
    if steer_all_particles is True:
        return True
    return policy in ("all", "every")


def _select_particle_indices(
    batch_size: int,
    num_particles: int,
    policy: str,
    index: Optional[int],
    device: torch.device,
) -> torch.Tensor:
    if policy == "first" or index is not None:
        j = 0 if index is None else int(index)
        return torch.full((batch_size,), j, device=device, dtype=torch.long)
    if policy == "random":
        return torch.randint(0, num_particles, (batch_size,), device=device)
    raise ValueError(
        "steering_particle_policy must be 'all'/'every' (via steer_all_particles), "
        "'first', or 'random'"
    )


def expand_steering_dataframe_for_particles(df: pd.DataFrame, num_particles: int) -> pd.DataFrame:
    """
    Expand each eval row into K rows with particle_id 0..K-1 (stable head index).

    Order: (row0, p0), (row0, p1), ..., (row1, p0), ... — matches predict_steer output order.
    """
    if num_particles <= 1:
        out = df.copy()
        if "particle_id" not in out.columns:
            out["particle_id"] = 0
        return out.reset_index(drop=True)
    rows = []
    for _, row in df.iterrows():
        for j in range(num_particles):
            r = row.copy()
            r["particle_id"] = int(j)
            rows.append(r)
    return pd.DataFrame(rows).reset_index(drop=True)

# Adapted from AxBench — https://github.com/stanfordnlp/axbench (Apache-2.0).
"""Validity aggregation and repulsion penalties (rbf, cosine, latent_corr)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, NamedTuple, Optional, Sequence

import torch
import torch.nn.functional as F

# L_validity: Particle-ReFT-Mean / Mean+Std / soft worst-particle (repulsion_plan SS4)
VALIDITY_MODES = ("mean", "mean_plus_std", "soft_max")

# R(W): geometric (rbf, cosine) vs detector-level (latent_corr); see notes/diversity_types.tex
REPULSION_TYPES = ("none", "rbf", "cosine", "latent_corr")

# How L_validity is combined with similarity R(W)
REPULSION_FORMULATIONS = ("additive", "scaled_penalty", "barrier")


@dataclass
class ParticleTrainingConfig:
    """Resolved L_validity + R settings for one training run (saved in train/config.json)."""

    validity_mode: str = "mean"
    validity_std_weight: float = 0.0  # eta; only for mean_plus_std
    validity_softmax_temp: float = 0.1  # tau; only for soft_max
    repulsion_type: str = "none"
    repulsion_formulation: str = "additive"  # additive | scaled_penalty | barrier
    repulsion_weight: float = 0.0  # gamma (additive) or fallback mu
    repulsion_lambda: Optional[float] = None  # mu for scaled_penalty / barrier
    repulsion_bandwidth: float = 1.0  # sigma; only for rbf
    repulsion_reference_r: float = 1.0  # r_0 for scaling / barrier denom
    repulsion_threshold_tau: float = 0.95  # tau: penalize R > tau (barrier; audit-aligned)
    repulsion_validity_scale: float = 0.0  # ell_0; 0 => use L_valid.detach() per step
    repulsion_warmup_steps: int = 0

    def repulsion_strength(self) -> float:
        """mu or gamma depending on formulation."""
        if self.repulsion_formulation == "additive":
            return self.repulsion_weight
        if self.repulsion_lambda is not None and self.repulsion_lambda > 0.0:
            return self.repulsion_lambda
        return self.repulsion_weight

    def active_repulsion(self) -> bool:
        return self.repulsion_type != "none" and self.repulsion_strength() > 0.0

    def needs_latent_tensors_for_repulsion(self) -> bool:
        # Post-ReLU detector scores s_k = ReLU(H w_k); functional not geometric (repulsion_plan)
        return self.repulsion_type == "latent_corr"


def _get_float(ta: Any, *names: str, default: Optional[float] = None) -> Optional[float]:
    for name in names:
        v = getattr(ta, name, None)
        if v is not None:
            return float(v)
    return default


def _get_str(ta: Any, *names: str, default: Optional[str] = None) -> Optional[str]:
    for name in names:
        v = getattr(ta, name, None)
        if v is not None and str(v).strip():
            return str(v).strip().lower()
    return default


def normalize_particle_training_config(ta: Any) -> ParticleTrainingConfig:
    """Parse ModelParams; validate mode-specific fields; apply legacy aliases."""
    validity_mode = _get_str(ta, "particle_validity_mode", default=None)
    if validity_mode is None:
        lm_agg = _get_str(ta, "lm_loss_aggregation", default="mean")
        validity_mode = lm_agg if lm_agg in VALIDITY_MODES else "mean"
    validity_mode = validity_mode.lower()
    if validity_mode not in VALIDITY_MODES:
        raise ValueError(
            f"particle_validity_mode must be one of {VALIDITY_MODES}, got {validity_mode!r}"
        )

    repulsion_type = _get_str(ta, "repulsion_type", default=None)
    legacy_lam = _get_float(ta, "repulsion_weight", "particle_diversity_weight", default=0.0) or 0.0
    if repulsion_type is None:
        repulsion_type = "rbf" if legacy_lam > 0.0 else "none"
    repulsion_type = repulsion_type.lower()
    if repulsion_type not in REPULSION_TYPES:
        raise ValueError(
            f"repulsion_type must be one of {REPULSION_TYPES}, got {repulsion_type!r}"
        )

    repulsion_weight = _get_float(ta, "repulsion_weight", "particle_diversity_weight", default=0.0) or 0.0
    if repulsion_type == "none":
        repulsion_weight = 0.0

    bandwidth = _get_float(ta, "repulsion_bandwidth", "particle_rbf_bandwidth", default=1.0) or 1.0
    warmup = int(getattr(ta, "repulsion_warmup_steps", 0) or 0)

    std_weight = _get_float(ta, "particle_validity_std_weight", default=0.0) or 0.0
    softmax_temp = _get_float(ta, "particle_validity_softmax_temp", default=0.1) or 0.1

    if validity_mode == "mean_plus_std":
        if std_weight <= 0.0:
            raise ValueError(
                "particle_validity_mode='mean_plus_std' requires particle_validity_std_weight > 0"
            )
    else:
        std_weight = 0.0

    if validity_mode == "soft_max":
        if softmax_temp <= 0.0:
            raise ValueError(
                "particle_validity_mode='soft_max' requires particle_validity_softmax_temp > 0"
            )
    else:
        softmax_temp = 0.1

    strength_probe = repulsion_weight
    lam_probe = _get_float(ta, "repulsion_lambda", default=None)
    if lam_probe is not None and lam_probe > 0.0:
        strength_probe = lam_probe
    if repulsion_type == "rbf" and strength_probe > 0.0 and bandwidth <= 0.0:
        raise ValueError("repulsion_type='rbf' requires repulsion_bandwidth > 0")

    formulation = _get_str(ta, "repulsion_formulation", default="additive") or "additive"
    formulation = formulation.lower()
    if formulation not in REPULSION_FORMULATIONS:
        raise ValueError(
            f"repulsion_formulation must be one of {REPULSION_FORMULATIONS}, got {formulation!r}"
        )

    repulsion_lambda = _get_float(ta, "repulsion_lambda", default=None)
    r_ref = _get_float(ta, "repulsion_reference_r", default=1.0) or 1.0
    tau = _get_float(ta, "repulsion_threshold_tau", default=None)
    if tau is None:
        # RBF on unit vecs: collapsed ~1, orthogonal pair ~0.37; cosine: 1 vs 0
        tau = 0.95 if repulsion_type == "rbf" else 0.90
    ell_0 = _get_float(ta, "repulsion_validity_scale", default=0.0) or 0.0

    strength = repulsion_lambda if (repulsion_lambda is not None and repulsion_lambda > 0) else repulsion_weight
    if formulation in ("scaled_penalty", "barrier") and strength > 0.0:
        if r_ref <= 0.0:
            raise ValueError("repulsion_reference_r must be > 0")
        if formulation == "barrier" and tau >= r_ref:
            raise ValueError(
                f"repulsion_threshold_tau ({tau}) must be < repulsion_reference_r ({r_ref})"
            )

    return ParticleTrainingConfig(
        validity_mode=validity_mode,
        validity_std_weight=std_weight,
        validity_softmax_temp=softmax_temp,
        repulsion_type=repulsion_type,
        repulsion_formulation=formulation,
        repulsion_weight=repulsion_weight,
        repulsion_lambda=repulsion_lambda,
        repulsion_bandwidth=bandwidth,
        repulsion_reference_r=r_ref,
        repulsion_threshold_tau=tau,
        repulsion_validity_scale=ell_0,
        repulsion_warmup_steps=max(0, warmup),
    )


def particle_config_to_dict(cfg: ParticleTrainingConfig) -> Dict[str, Any]:
    return asdict(cfg)


def particle_config_from_training_args(ta: Any) -> Dict[str, Any]:
    return particle_config_to_dict(normalize_particle_training_config(ta))


def repulsion_scale(cfg: ParticleTrainingConfig, step: int) -> float:
    """Ramp mu/gamma to full strength after repulsion_warmup_steps optimizer steps."""
    if not cfg.active_repulsion():
        return 0.0
    target = cfg.repulsion_strength()
    if cfg.repulsion_warmup_steps <= 0:
        return target
    return target * min(1.0, float(step) / float(cfg.repulsion_warmup_steps))


def aggregate_validity_loss(
    per_particle_losses: torch.Tensor,
    cfg: ParticleTrainingConfig,
) -> torch.Tensor:
    """
    L_validity from per-particle ReFT-r1 losses L_k (shape K).

    mean          (1/K) sum_k L_k  — Particle-ReFT-Mean; default Method A baseline.
    mean_plus_std mean_k L_k + eta * std_k L_k  — penalize one bad particle hiding in the mean.
    soft_max      tau log((1/K) sum_k exp(L_k/tau))  — smooth emphasis on worst particle.
    """
    if per_particle_losses.numel() == 0:
        raise ValueError("per_particle_losses is empty")
    k = per_particle_losses.numel()
    if cfg.validity_mode == "mean":
        return per_particle_losses.mean()
    if cfg.validity_mode == "mean_plus_std":
        mean = per_particle_losses.mean()
        if k < 2:
            return mean
        std = per_particle_losses.std(unbiased=False)
        return mean + cfg.validity_std_weight * std
    if cfg.validity_mode == "soft_max":
        tau = cfg.validity_softmax_temp
        return tau * torch.logsumexp(per_particle_losses / tau, dim=0) - tau * math.log(k)
    raise ValueError(f"Unknown validity_mode: {cfg.validity_mode}")


def _pairwise_upper_mean(matrix: torch.Tensor) -> torch.Tensor:
    k = matrix.shape[0]
    if k < 2:
        return matrix.new_zeros(())
    idx = torch.triu_indices(k, k, offset=1, device=matrix.device)
    return matrix[idx[0], idx[1]].mean()


def _pairwise_rbf_mean(weights: torch.Tensor, sigma: float) -> torch.Tensor:
    # R_RBF: mean_{i<j} exp(-||w_i - w_j||^2 / 2 sigma^2). Low-level geometric diversity.
    k = weights.shape[0]
    if k < 2:
        return weights.new_zeros(())
    w = weights.unsqueeze(0)
    diff = w.unsqueeze(2) - w.unsqueeze(1)
    dist2 = (diff * diff).sum(-1)
    kmat = torch.exp(-dist2 / (2.0 * sigma * sigma))
    eye = torch.eye(k, device=weights.device, dtype=weights.dtype)
    mask = 1.0 - eye
    return (kmat * mask.unsqueeze(0)).sum() / mask.sum().clamp(min=1.0)


def _pairwise_cosine_sq_mean(weights: torch.Tensor) -> torch.Tensor:
    # R_cos: mean_{i<j} cos(w_i, w_j)^2 on unit directions. Cleaner than RBF when ||w_k|| ~ 1.
    k = weights.shape[0]
    if k < 2:
        return weights.new_zeros(())
    w = F.normalize(weights, dim=1)
    cos = w @ w.T
    return _pairwise_upper_mean(cos * cos)


def _masked_flatten(latent: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if latent.dim() == 3:
        latent = latent.squeeze(-1)
    if mask is None:
        return latent.reshape(-1)
    m = mask.to(device=latent.device, dtype=torch.bool)
    if m.shape != latent.shape:
        return latent.reshape(-1)
    return latent[m]


def _pairwise_corr_sq_mean(vectors: Sequence[torch.Tensor]) -> torch.Tensor:
    # R_latent: mean_{i<j} corr(s_i, s_j)^2 over masked batch tokens (detector diversity).
    k = len(vectors)
    if k < 2:
        return vectors[0].new_zeros(()) if k == 1 else torch.tensor(0.0)
    mats = []
    for v in vectors:
        v = v.float().flatten()
        if v.numel() < 2:
            return v.new_zeros(())
        v = v - v.mean()
        std = v.std(unbiased=False).clamp(min=1e-8)
        mats.append(v / std)
    x = torch.stack(mats, dim=0)
    c = (x @ x.T) / x.shape[1]
    return _pairwise_upper_mean(c * c)


def compute_repulsion_similarity(
    cfg: ParticleTrainingConfig,
    *,
    weights: torch.Tensor,
    latent_per_particle: Optional[List[torch.Tensor]] = None,
    intervention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Raw similarity R(W) in [~0, 1] for common settings. Minimize to spread particles.

    none        R = 0 (rep0 collapse baseline).
    rbf         weight-space Euclidean similarity (Particle-ReFT-RBF).
    cosine      directional similarity; audit via pairwise_abs_cosine.
    latent_corr post-ReLU score correlation — functional; low cosine but high
                latent_corr still means geometric_only in eval (diversity_types.tex).
    """
    k = weights.shape[0]
    if cfg.repulsion_type == "none" or k < 2:
        return weights.new_zeros(())

    if cfg.repulsion_type == "rbf":
        return _pairwise_rbf_mean(weights.float(), cfg.repulsion_bandwidth)

    if cfg.repulsion_type == "cosine":
        return _pairwise_cosine_sq_mean(weights.float())

    if cfg.repulsion_type == "latent_corr":
        if not latent_per_particle or len(latent_per_particle) < 2:
            raise ValueError(
                "repulsion_type='latent_corr' requires per-particle latents from the train loop"
            )
        flat = [
            _masked_flatten(lat.float(), intervention_mask) for lat in latent_per_particle
        ]
        return _pairwise_corr_sq_mean(flat)

    raise ValueError(f"Unknown repulsion_type: {cfg.repulsion_type}")


def compute_repulsion_loss(
    cfg: ParticleTrainingConfig,
    *,
    weights: torch.Tensor,
    latent_per_particle: Optional[List[torch.Tensor]] = None,
    intervention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Alias for compute_repulsion_similarity (backward compatible)."""
    return compute_repulsion_similarity(
        cfg,
        weights=weights,
        latent_per_particle=latent_per_particle,
        intervention_mask=intervention_mask,
    )


class ParticleLossBreakdown(NamedTuple):
    """Scalars for logging (same graph as training when tensors require grad)."""

    l_validity: torch.Tensor  # mean particle ReFT loss (LM + ramped L1)
    r_similarity: torch.Tensor  # raw R(W); ~1 = collapsed for RBF/cosine
    repulsion_term: torch.Tensor  # contribution added to l_validity
    total: torch.Tensor  # l_validity + repulsion_term


def decompose_particle_objective(
    l_validity: torch.Tensor,
    r_similarity: torch.Tensor,
    cfg: ParticleTrainingConfig,
    step: int,
) -> ParticleLossBreakdown:
    """
    Split the training objective for interpretable logs.

    repulsion_term is what combine_particle_objective adds on top of l_validity.
    """
    mu = repulsion_scale(cfg, step)
    zero = l_validity.new_zeros(())
    if mu <= 0.0 or cfg.repulsion_type == "none":
        return ParticleLossBreakdown(l_validity, r_similarity.detach(), zero, l_validity)

    if cfg.repulsion_formulation == "additive":
        rep = mu * r_similarity
        return ParticleLossBreakdown(l_validity, r_similarity, rep, l_validity + rep)

    if cfg.repulsion_validity_scale > 0.0:
        ell = r_similarity.new_tensor(cfg.repulsion_validity_scale)
    else:
        ell = l_validity.detach()

    r0 = max(float(cfg.repulsion_reference_r), 1e-8)

    if cfg.repulsion_formulation == "scaled_penalty":
        rep = mu * (r_similarity / r0) * ell
        return ParticleLossBreakdown(l_validity, r_similarity, rep, l_validity + rep)

    if cfg.repulsion_formulation == "barrier":
        tau = float(cfg.repulsion_threshold_tau)
        denom = max(r0 - tau, 1e-8)
        excess = torch.relu(r_similarity - tau) / denom
        rep = mu * (excess * excess) * ell
        return ParticleLossBreakdown(l_validity, r_similarity, rep, l_validity + rep)

    raise ValueError(f"Unknown repulsion_formulation: {cfg.repulsion_formulation}")


def format_particle_loss_breakdown(bd: ParticleLossBreakdown) -> str:
    """Compact tqdm description fragment."""
    lv = float(bd.l_validity.detach())
    r = float(bd.r_similarity.detach())
    rep = float(bd.repulsion_term.detach())
    tot = float(bd.total.detach())
    return f"lv={lv:.3f} R={r:.3f} rep={rep:.4f} tot={tot:.3f}"


def combine_particle_objective(
    l_validity: torch.Tensor,
    r_similarity: torch.Tensor,
    cfg: ParticleTrainingConfig,
    step: int,
) -> torch.Tensor:
    """Total training loss = l_validity + repulsion_term."""
    return decompose_particle_objective(l_validity, r_similarity, cfg, step).total


def resolve_particle_weights_slice(
    proj_weight: torch.Tensor,
    num_particles: int,
    particle_rank: int,
) -> torch.Tensor:
    """Current concept's particle rows W in proj.weight, shape (K, d)."""
    r = particle_rank
    n = num_particles
    return proj_weight[0::r][:n].float()

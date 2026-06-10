# ParticleSteering

Course project code for **Discovering Distinct Steering Interventions** — learning multiple distinct steering vectors per concept on Gemma-2-2B (20 AxBench concepts, layer 20).

## Repo map


| Path                                             | What it is                                     |
| ------------------------------------------------ | ---------------------------------------------- |
| `particlesteering/`                              | Train / infer / eval package                   |
| `particlesteering/models/particle_steering.py`   | K-particle contrastive training                |
| `particlesteering/models/particle_objectives.py` | Repulsion penalties (RBF, cosine, latent-corr) |
| `configs/particlesteering_l20.yaml`              | Main experiment config                         |
| `experiments/run_particlesteering_l20.sh`        | End-to-end pipeline (train → infer → eval)     |
| `analysis/`                                      | Paper tables and figures                       |
| `data/particlesteering/`, `data/baselines/`      | Bundled eval exports for tables/figures        |


Generated outputs: `paper_outputs/` (tables and figures).

## Setup

```bash
pip install -e .
export OPENAI_API_KEY=...

python particlesteering/data/download-seed-sentences.py
cd particlesteering/data && bash download-2b.sh && bash download-alpaca.sh
```

You also need [Gemma-2-2B-it](https://huggingface.co/google/gemma-2-2b-it) and the concept JSON referenced in `configs/particlesteering_l20.yaml`.

## Run experiments

```bash
# ParticleSteering: K=5, repulsion arms rbf / cosine / latent_corr
bash experiments/run_particlesteering_l20.sh
```

The script runs train → latent inference → steering inference → LM-judge evaluate. New runs write under `data/runs/` by default.

Baseline eval exports for paper tables are bundled under `data/baselines/` (see `data/README.md`).

## Figures and tables

```bash
bash analysis/run_paper_figures.sh
```

Reads bundled results from `data/particlesteering/` and `data/baselines/`. Outputs: `paper_outputs/tables/`, `paper_outputs/figures/`.

## Acknowledgments

Most of `particlesteering/` is adapted from [AxBench](https://github.com/stanfordnlp/axbench) (Apache-2.0). New method code: `particle_steering.py`, `particle_objectives.py`, `particle_utils.py`.

```bibtex
@inproceedings{wu2025axbench,
  title={{AxBench}: Steering {LLM}s? Even Simple Baselines Outperform Sparse Autoencoders},
  author={Wu, Zhengxuan and others},
  booktitle={ICML}, year={2025}
}
```

---

## Walkthrough

This section explains the method end to end: how we train steering vectors, why we learn **K particles per concept**, and how **repulsion** keeps them distinct.

**Setup.** We build on the [AxBench](https://github.com/stanfordnlp/axbench) benchmark (Gemma-2-2B, 20 GemmaScope SAE concepts, layer 20). Each concept has preference pairs: a base prompt `x`, a positive completion `y⁺` that expresses the concept, and a negative completion `y⁻` that does not.

**Core idea.** Every particle is trained with the same contrastive objective (RePS / DPO on steered log-probabilities). Without an extra penalty, K independent runs usually **collapse** to nearly the same direction. We add a repulsion term so the particles stay spread out in weight space, direction space, or activation space.

### 1. Activation steering (single vector)

At layer ℓ, token `i` has hidden state `hᵢ ∈ ℝᵈ`. A steering vector `wᶜ` adds a shift to the residual stream.

Following SDL / LsReFT-style **max-activation calibration**, the vector also scores how much the concept is present at each position:

$$
\alpha_i = \mathrm{ReLU}(h_i^\top w_c), \qquad m_c = \max_i \alpha_i
$$

The intervention scales the unit direction `ŵᶜ` by the max activation and a strength knob ρ (the **steering factor**, swept at eval time):

$$
h_i \leftarrow h_i + \rho  m_c  \hat{w}_c
$$

### 2. Contrastive training (one vector)

Let `pθ` be the steered model and `p₀` the frozen base model. The **DPO reward** on completion `y` is the log-ratio of steered vs. base token probabilities:

$$
r_\theta(x, y) = \beta \sum_t \log \frac{p_\theta(y_t \mid x, y_{\lt t})}{p_0(y_t \mid x, y_{\lt t})}
$$

Training prefers `y⁺` over `y⁻` via the Bradley–Terry / DPO loss:

$$
\mathcal{L}(w_c) = -\log \sigma\big(r_\theta(x, y^{+}) - r_\theta(x, y^{-})\big)
$$

Implemented as `preference_loss` in `preference_model.py` (`loss_type: dpo` in the config). The loss only contrasts tokens where `y⁺` and `y⁻` differ, so it targets the concept rather than shared fluent text.

### 3. K particles per concept

Instead of one vector `wᶜ`, ParticleSteering learns **K rank-1 directions** per concept: `Wᶜ = {wᶜ,₁, …, wᶜ,ₖ}`. Each particle `k` has its own DPO loss `ℒₖ`.

**Validity** (how well steering works) is aggregated over particles — by default, the mean:

$$
\mathcal{L}*{\mathrm{steer}} = \frac{1}{K}\sum*{k=1}^K \mathcal{L}_k
$$

**Repulsion** discourages similar particles. Let `R(Wᶜ)` be a pairwise similarity (higher = more alike; we **minimize** it). The full objective is:

$$
\mathcal{L}*{\mathrm{total}} = \mathcal{L}*{\mathrm{steer}} + \gamma  R(W_c)
$$

where γ is `repulsion_weight` in the config.

### 4. Repulsion penalties

All three variants are implemented in `particle_objectives.py`. Each returns a scalar similarity averaged over unordered particle pairs. **Lower R ⇒ more diverse particles.**


| Config value  | What it penalizes                      | Intuition                                      |
| ------------- | -------------------------------------- | ---------------------------------------------- |
| `rbf`         | Euclidean closeness in weight space    | Spread particles apart geometrically           |
| `cosine`      | Alignment of unit directions           | Spread particles apart angularly               |
| `latent_corr` | Correlation of detector score profiles | Spread particles apart functionally on prompts |


**RBF** (paper eq. 9):

$$
R_{\mathrm{RBF}}(W_c) = \frac{2}{K(K-1)} \sum_{k \lt l} \exp\left(-\frac{w_{c,k} - w_{c,l}^2}{2\tau^2}\right)
$$

Bandwidth τ = `repulsion_bandwidth`.

**Cosine** (paper eq. 10):

$$
R_{\mathrm{cosine}}(W_c) = \frac{2}{K(K-1)} \sum_{k \lt l} (\hat{w}*{c,k} \cdot \hat{w}*{c,l})^2
$$

**Latent correlation** (paper eq. 11). Each particle induces scores `αₖ,ᵢ = ReLU(hᵢᵀ wᶜ,ₖ)` over prompt tokens; let `sₖ` be the resulting profile:

$$
R_{\mathrm{latent}}(W_c) = \frac{2}{K(K-1)} \sum_{k \lt l} \mathrm{corr}(s_k, s_l)^2
$$

Particles can be far apart in weight space but still fire on the same tokens (or the reverse). Latent-corr repulsion targets functional diversity.

At training time, call `compute_repulsion_similarity` with a `ParticleTrainingConfig` (`repulsion_type`, `repulsion_bandwidth`, etc.).

### 5. Combined loss

Default (**additive**): `ℒ_total = ℒ_steer + γ R(W)`.

Alternatives in config: `scaled_penalty`, `barrier` (soft cap on R). See `combine_particle_objective` in `particle_objectives.py`.

### 6. Training loop

`ParticleSteering.train` in `particle_steering.py`:

1. Initialize K rows of `proj.weight` (orthogonal init by default).
2. For each batch of preference pairs:
  - Sample a steering factor ρ from a grid.
  - Run each particle through the model with its intervention.
  - Compute per-particle DPO loss `ℒₖ`; aggregate to `ℒ_steer`.
  - Compute repulsion `R(W)` (using latents when `repulsion_type: latent_corr`).
  - Backprop `ℒ_total`. Optionally renormalize rows each step (`use_unit_norm_steering_vectors`).
3. Save `ParticleSteering_weight.pt` per concept.

Paper settings (L20, K=5) in `configs/particlesteering_l20.yaml`, with overrides in `experiments/run_particlesteering_l20.sh`:

```yaml
ParticleSteering:
  num_particles: 5
  repulsion_type: rbf       # rbf | cosine | latent_corr
  repulsion_weight: 0.5
  repulsion_bandwidth: 1.0
  use_unit_norm_steering_vectors: true
  use_max_act_steering_calibration: true
  loss_type: dpo
```

### 7. Inference and evaluation

**Per particle:** sweep ρ on held-out prompts; pick `f`* that maximizes mean LM-judge overall score (AxBench protocol).

**Across K particles:**

- **Mean-of-K** — average scores at each concept's `f`*.
- **Best-of-K (oracle)** — take the max; measures how much a good particle beats the average.

Eval uses an LM judge (concept relevance, instruction following, fluency → overall) on 20 GemmaScope concepts. Paper repulsion arms: `rbf`, `cosine`, `latent_corr`.

**Takeaway.** Repulsion-trained particles match strong single-vector baselines (ReFT-r1) on average while maintaining diversity on a defined repulsion axis.

### 8. Code map


| Component                        | File                                             |
| -------------------------------- | ------------------------------------------------ |
| `ParticleSteering` model         | `particlesteering/models/particle_steering.py`   |
| Repulsion + loss aggregation     | `particlesteering/models/particle_objectives.py` |
| DPO / preference loss            | `particlesteering/models/preference_model.py`    |
| Particle init + steering helpers | `particlesteering/models/particle_utils.py`      |
| Experiment config                | `configs/particlesteering_l20.yaml`              |
| Bundled baseline eval exports    | `data/baselines/`                                |


**Related work:** [AxBench](https://proceedings.mlr.press/v267/wu25a.html) (Wu et al., ICML 2025); [RePS preference steering](https://arxiv.org/pdf/2505.20809) (Wu et al., 2025).
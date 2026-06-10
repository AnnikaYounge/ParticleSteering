#!/usr/bin/env bash
# L20 ParticleSteering: N=20 concepts, K=5, repulsion rbf/cosine/latent_corr.
# Override: DATA, PT_ROOT, K_PARTICLES, REP_WEIGHT, NPROC.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export DATA="${DATA:-data/concept500/prod_2b_l20_v1}"
export INFER_DATA="${INFER_DATA:-$DATA/inference}"
export CFG="${CFG:-configs/particlesteering_l20.yaml}"
export MODEL=ParticleSteering
export N_CONCEPTS="${N_CONCEPTS:-20}"
export K_PARTICLES="${K_PARTICLES:-5}"
export NPROC="${NPROC:-1}"
export PT_ROOT="${PT_ROOT:-data/runs/particlesteering_n20_k5}"
export REP_WEIGHT="${REP_WEIGHT:-0.5}"
export OPENAI_TIMEOUT="${OPENAI_TIMEOUT:-120}"
export JUDGE_BATCH_SIZE="${JUDGE_BATCH_SIZE:-8}"

train_arm() {
  local REP="$1"
  local DUMP="${PT_ROOT}/repulsion_${REP}/form_additive/w${REP_WEIGHT}"
  mkdir -p "$DUMP"/{train,inference,evaluate}

  echo "=== train rep=${REP} -> ${DUMP} ==="
  torchrun --nproc_per_node="$NPROC" particlesteering/scripts/train.py \
    --config "$CFG" --dump_dir "$DUMP" \
    --overwrite_data_dir "$DATA/generate" \
    --overwrite_metadata_dir "$DATA/generate" \
    --max_concepts "$N_CONCEPTS" \
    --model_param "${MODEL}.num_particles=${K_PARTICLES}" \
    --model_param "${MODEL}.repulsion_type=${REP}" \
    --model_param "${MODEL}.repulsion_formulation=additive" \
    --model_param "${MODEL}.repulsion_weight=${REP_WEIGHT}" \
    --model_param "${MODEL}.particle_weight_init=orthogonal" \
    --model_param "${MODEL}.use_unit_norm_steering_vectors=true" \
    --model_param "${MODEL}.use_max_act_steering_calibration=true" \
    --model_param "${MODEL}.use_sdl_training_intervention=true"

  echo "=== latent ==="
  torchrun --nproc_per_node="$NPROC" particlesteering/scripts/inference.py \
    --config "$CFG" --dump_dir "$DUMP" --mode latent \
    --max_concepts "$N_CONCEPTS" \
    --overwrite_data_dir "$DATA/generate" \
    --overwrite_inference_data_dir "$INFER_DATA"

  echo "=== steering ==="
  torchrun --nproc_per_node="$NPROC" particlesteering/scripts/inference.py \
    --config "$CFG" --dump_dir "$DUMP" --mode steering \
    --max_concepts "$N_CONCEPTS" --steering_batch_size 1 \
    --overwrite_inference_data_dir "$INFER_DATA"

  echo "=== evaluate ==="
  python particlesteering/scripts/evaluate.py \
    --config "$CFG" --dump_dir "$DUMP" --mode steering \
    --openai_timeout "$OPENAI_TIMEOUT" --judge_batch_size "$JUDGE_BATCH_SIZE"
}

for REP in rbf cosine latent_corr; do
  train_arm "$REP"
done

echo "Done. Eval parquets under ${PT_ROOT}/repulsion_*/form_additive/w${REP_WEIGHT}/evaluate/"

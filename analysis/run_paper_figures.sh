#!/usr/bin/env bash
# Tables + figures → paper_outputs/. See analysis/README.md.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/.mplconfig}"
export MPLBACKEND=Agg
export PYTHONPATH="$ROOT/analysis/paper_figures${PYTHONPATH:+:$PYTHONPATH}"

DATA="${DATA:-data}"
OUT="$ROOT/paper_outputs"
mkdir -p "$OUT/tables" "$OUT/figures"

python analysis/generate_paper_tables.py \
  --data-root "$DATA" \
  --out-dir "$OUT/tables"

python analysis/paper_figures/plot_behavioral_correlation.py \
  --base-root "$DATA/particlesteering" \
  --out "$OUT/figures/behavioral_correlation"

python analysis/paper_figures/plot_selection_lift.py \
  --base-root "$DATA/particlesteering" \
  --out "$OUT/figures/selection_lift"

python analysis/paper_figures/plot_concept_heterogeneity.py \
  --data-root "$DATA" \
  --out "$OUT/figures/heterogeneity"

python analysis/paper_figures/plot_weight_pca.py \
  --root "$DATA/particlesteering" \
  --out "$OUT/figures/geometry" \
  --rep rbf --k 5

echo "Wrote tables to $OUT/tables and figures to $OUT/figures"

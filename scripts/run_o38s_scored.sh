#!/usr/bin/env bash
# One o38s run plus the scoring chain, Windows/Git-Bash twin of run_e2_scored.sh (same four scripts, same order).
# Usage: bash scripts/run_o38s_scored.sh SEED T_S TAG LAYOUT
# Env: O38S_PY (venv python), JAXFNE_SITE (verified 351f6048 export), JAXFNE_EDGE_DELIVERY=event.
# -B: never write bytecode (JAXFNE_SITE is a read-only export).
cd "$(dirname "$0")/.." || exit 1
seed=$1 T=$2 tag=$3 lay=$4
py=${O38S_PY:?set O38S_PY}
[[ -f results/laminar_e2/${tag}_verdict.json ]] && { echo "${tag} done"; exit 0; }
mkdir -p results/laminar_e2
"$py" -B -u scripts/run_e2_stack.py "$seed" "$T" "$tag" "$lay" > "results/laminar_e2/${tag}.log" 2>&1 \
  && "$py" -B scripts/e2_motif_figure.py "$tag" 2 >> "results/laminar_e2/${tag}.log" 2>&1 \
  && "$py" -B scripts/e2_motif_similarity.py "$tag" >> "results/laminar_e2/${tag}.log" 2>&1 \
  && "$py" -B scripts/e2_scorer.py "$tag" >> "results/laminar_e2/${tag}.log" 2>&1
echo "${tag} rc=$?"

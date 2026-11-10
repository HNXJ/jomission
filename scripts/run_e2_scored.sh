#!/bin/zsh
# One E2 run plus the W2 scoring chain (as run_t1_axis.sh run_one), for exploratory layouts.
# Usage: zsh scripts/run_e2_scored.sh SEED T_S TAG LAYOUT     (env JAXFNE_SITE / JAXFNE_EDGE_DELIVERY pass through)
cd "${0:A:h}/.." || exit 1
seed=$1 T=$2 tag=$3 lay=$4
[[ -f results/laminar_e2/${tag}_verdict.json ]] && { echo "${tag} done"; exit 0; }
/usr/local/bin/python3 -u scripts/run_e2_stack.py "$seed" "$T" "$tag" "$lay" > "results/laminar_e2/${tag}.log" 2>&1 \
  && /usr/local/bin/python3 scripts/e2_motif_figure.py "$tag" 2 >> "results/laminar_e2/${tag}.log" 2>&1 \
  && /usr/local/bin/python3 scripts/e2_motif_similarity.py "$tag" >> "results/laminar_e2/${tag}.log" 2>&1 \
  && /usr/local/bin/python3 scripts/e2_scorer.py "$tag" >> "results/laminar_e2/${tag}.log" 2>&1
echo "${tag} rc=$?"

#!/usr/bin/env bash
#
# Batch backfill + resolve ownership for the remaining real, unowned models
# in the registry, to org 1 (Default Organization).
#
# This is NOT a dynamic scan -- the model list below is an explicit, reviewed
# set, confirmed against the registry on 2026-09-03 by listing every
# tasks/*/models/* directory and checking each one's ownership.json:
#
#   - 46x crispr_guide_efficiency/guide_lstm_pytest_*  -- test fixtures, excluded
#   - amr_gene_classification/amr_cnn_real_v1          -- excluded (0 versions)
#   - regression/missing                               -- excluded (0 versions)
#   - regression/phase516-26c49cd3e8                   -- already owned by org
#                                                          472 (not ours) -- excluded
#   - protein_stability_ddg/ddg_mlp                     -- already resolved to
#                                                          org 1 this session
#   - crispr_guide_efficiency/guide_lstm_smoketest      -- already resolved to
#                                                          org 1 in a prior session
#
# That leaves exactly the 3 models below as real + currently unowned.
#
# Each model goes through the same two already-reviewed, already-verified
# steps used for ddg_mlp:
#   1. scripts/backfill_ownership_single_model.py --yes
#      (creates ownership.json with status=legacy_unowned; refuses if one
#      already exists -- write-once, so this script cannot clobber
#      phase516-26c49cd3e8 or anything else already owned)
#   2. omr resolve-ownership --org-id 1
#
# Usage:
#   ./scripts/batch_resolve_remaining_ownership.sh            # dry run (default)
#   ./scripts/batch_resolve_remaining_ownership.sh --execute  # actually writes
#
# Fails fast: set -e means the whole batch stops at the first error instead
# of silently continuing past a broken model.

set -euo pipefail

CONTAINER=omnibioai-studio-model-registry-1
ORG_ID=1
ACTOR="manish@omnibioai.org"
SCRIPT_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/backfill_ownership_single_model.py"
SCRIPT_DST="/app/scripts/backfill_ownership_single_model.py"

MODELS=(
  "crispr_guide_efficiency guide_lstm_smoketest_epoch0"
  "citeseq_celltype citeseq_mlp"
  "spatial_domain spatial_gat"
)

EXECUTE=0
if [[ "${1:-}" == "--execute" ]]; then
  EXECUTE=1
fi

echo "Refreshing backfill script inside container (from $SCRIPT_SRC)..."
docker exec "$CONTAINER" mkdir -p /app/scripts
docker cp "$SCRIPT_SRC" "$CONTAINER:$SCRIPT_DST"
echo

for entry in "${MODELS[@]}"; do
  read -r task model <<< "$entry"
  echo "=== $task/$model ==="

  echo "-- backfill preview (dry run, always) --"
  docker exec -e PYTHONPATH=/app "$CONTAINER" \
    python "$SCRIPT_DST" --task "$task" --model "$model" --json

  if [[ "$EXECUTE" -eq 0 ]]; then
    echo "(dry run only -- pass --execute to actually write ownership + resolve)"
    echo
    continue
  fi

  echo "-- backfill (writing ownership.json, status=legacy_unowned) --"
  docker exec -e PYTHONPATH=/app "$CONTAINER" \
    python "$SCRIPT_DST" --task "$task" --model "$model" --yes --json

  echo "-- resolve-ownership (org $ORG_ID) --"
  docker exec -e PYTHONPATH=/app "$CONTAINER" \
    omr resolve-ownership --task "$task" --model "$model" --org-id "$ORG_ID" --actor "$ACTOR" --json

  echo
done

if [[ "$EXECUTE" -eq 0 ]]; then
  echo "Dry run complete for all ${#MODELS[@]} models. Re-run with --execute to apply."
else
  echo "Batch complete for all ${#MODELS[@]} models."
  echo "Next: verify via GET /v1/models and the ModelHub UI (webstudio.omnibioai.org)."
fi

#!/usr/bin/bash
set -euo pipefail

GATE_COUNTS_CSV="../data/gate_counts.csv"
PREFIX="../data/chosen_circuits"
MAX_GATE_COUNT=51219
LOG_FILE="$(date '+%Y-%m-%d-%H%M%S').log"

tail -n +2 "${GATE_COUNTS_CSV}" | \
while IFS= read -r line; do
  CIRCUIT=$(echo "$line" | cut -f 1 -d, | sed 's/\.aig$//')
  GATE_COUNT=$(echo "$line" | cut -f 2 -d,)
  MAX_DEPTH=$(echo "$line" | cut -f 7 -d,)

  if [[ $((MAX_DEPTH)) -gt 1 ]] && [[ $((GATE_COUNT)) -lt $((MAX_GATE_COUNT)) ]]; then
    ./create_embedding_db.mk CIRCUIT="${PREFIX}/${CIRCUIT}.aig" MAX_DEPTH="${MAX_DEPTH}" PREFIX="${PREFIX}/${CIRCUIT}" && \
    echo "Circuit ${CIRCUIT} completed to its max depth ${MAX_DEPTH}" >> "${LOG_FILE}"
  fi
done

#!/bin/bash
# Casanovo v5.1.2 with the v5.0.0 checkpoint on the same nine-species samples, plain sequence mode (its evaluate mode
# cannot parse the benchmark's label notation), 5 beams as in the fork's baseline README, GPU. Outputs: <out>/<species>.mztab
# and <species>.log; chain.log as for the fork.
#   bash run_casanovo.sh <samples dir> <out dir> <checkpoint> <config yaml>
set -uo pipefail
SAMPLES=$1; OUT=$2; CKPT=$3; CFG=$4
CASA=/home/sabrina/Documents/experiments/Arc/envs/casanovo/bin/casanovo
mkdir -p "$OUT"
for mgf in "$SAMPLES"/*.mgf; do
  sp=$(basename "$mgf" .mgf)
  [ -s "$OUT/$sp.mztab" ] && { echo "$(date '+%F %T') $sp exists, skipped" | tee -a "$OUT/chain.log"; continue; }
  t0=$(date +%s)
  echo "$(date '+%F %T') $sp start" | tee -a "$OUT/chain.log"
  "$CASA" sequence -m "$CKPT" -c "$CFG" -d "$OUT" -o "$sp" -f "$mgf" > "$OUT/$sp.run.log" 2>&1
  echo "$(date '+%F %T') $sp exit $? after $(( $(date +%s) - t0 )) s" | tee -a "$OUT/chain.log"
done
echo "$(date '+%F %T') all done" | tee -a "$OUT/chain.log"

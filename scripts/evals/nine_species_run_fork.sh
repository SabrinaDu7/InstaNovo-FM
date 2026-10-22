#!/bin/bash
# The fork's fine-tuned sequencer (instanovo-fm-denovo-v0.1.0) on the nine-species samples, one species after another,
# greedy decoding then beam search with 5 beams (the setting the fork's baseline README uses for Casanovo). Evaluation
# mode with the benchmark's modification notation added to the residue remapping (configs/inference/denovo_ninespecies.yaml):
# without it the fork drops every labelled-with-modification spectrum from its own metrics. Outputs: <out>/<mode>/<species>.csv
# plus <species>.log; chain.log records start, exit code and wall time per run.
#   bash run_fork.sh <samples dir> <out dir> <checkpoint> <config dir> [greedy|beam5|both]
set -uo pipefail
SAMPLES=$1; OUT=$2; CKPT=$3; CFG=$4; MODES=${5:-both}
FM=/home/sabrina/Documents/experiments/Arc/InstaNovo-FM
export INSTANOVO_FM_DATA_DIR=$FM/data
run() {  # mode, beams
  local mode=$1 beams=$2
  mkdir -p "$OUT/$mode"
  for mgf in "$SAMPLES"/*.mgf; do
    sp=$(basename "$mgf" .mgf)
    [ "$(wc -l < "$OUT/$mode/$sp.csv" 2>/dev/null || echo 0)" -gt 1 ] && { echo "$(date '+%F %T') $mode $sp exists, skipped" | tee -a "$OUT/chain.log"; continue; }  # header-only means a failed run
    t0=$(date +%s)
    echo "$(date '+%F %T') $mode $sp start" | tee -a "$OUT/chain.log"
    "$FM/.venv/bin/instanovo-fm" denovo predict --data-path "$mgf" --output-path "$OUT/$mode/$sp.csv" --denovo-model "$CKPT" --evaluation \
      --config-path "$CFG" --config-name denovo_ninespecies num_beams=$beams > "$OUT/$mode/$sp.log" 2>&1
    echo "$(date '+%F %T') $mode $sp exit $? after $(( $(date +%s) - t0 )) s" | tee -a "$OUT/chain.log"
  done
}
case "$MODES" in greedy) run greedy 1 ;; beam5) run beam5 5 ;; both) run greedy 1; run beam5 5 ;; *) echo "mode?"; exit 2 ;; esac
echo "$(date '+%F %T') all done" | tee -a "$OUT/chain.log"

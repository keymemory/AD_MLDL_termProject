#!/bin/bash
# Phase3 실험A(taskaware) 런처: results_phase3_merge 격리, N-GPU 병렬.
set -u
TP=/home/jhlee/CLUST_KETI/AD_MLDL_termProject/term_project
GPUS="${1:-1 2}"
RESREL=exp_runner/results_phase3_merge.tsv
LOCKREL=exp_runner/locks_p3
JOBS=exp_runner/exp_jobs_phase3.tsv

[ -f "$TP/$RESREL" ] || : > "$TP/$RESREL"
echo -e "ID\tBENCH\tM2\tCLUST\tM1\tMETHOD\tR\tMETRIC\tVALUE\tGEN\tSELMETHOD\tAVG_M1\tAVG_R\tFLOOR\tCAP" > "$TP/exp_runner/results_phase3_header.tsv"
rm -rf "$TP/$LOCKREL"; mkdir -p "$TP/$LOCKREL" "$TP/exp_runner/logs"

for G in $GPUS; do
  RES="$RESREL" LOCK="$LOCKREL" bash "$TP/exp_runner/worker.sh" "$G" "$JOBS" &
done
wait
echo "PHASE3_DONE -> $TP/$RESREL"

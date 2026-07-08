#!/bin/bash
# Phase2 묶음B 런처: results 격리(append, resume), locks 리셋, N-GPU 병렬.
set -u
TP=/home/jhlee/CLUST_KETI/AD_MLDL_termProject/term_project
GPUS="${1:-1 2}"
RESREL=exp_runner/results_phase2B.tsv
LOCKREL=exp_runner/locks_p2B
JOBS=exp_runner/exp_jobs_phase2_B.tsv

[ -f "$TP/$RESREL" ] || : > "$TP/$RESREL"
echo -e "ID\tBENCH\tM2\tCLUST\tM1\tMETHOD\tR\tMETRIC\tVALUE\tGEN\tSELMETHOD\tAVG_M1\tAVG_R\tFLOOR\tCAP" > "$TP/exp_runner/results_phase2B_header.tsv"
rm -rf "$TP/$LOCKREL"; mkdir -p "$TP/$LOCKREL" "$TP/exp_runner/logs"

for G in $GPUS; do
  RES="$RESREL" LOCK="$LOCKREL" bash "$TP/exp_runner/worker.sh" "$G" "$JOBS" &
done
wait
echo "PHASE2B_DONE -> $TP/$RESREL"

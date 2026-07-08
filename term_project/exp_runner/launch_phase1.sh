#!/bin/bash
# Phase1 E2 전용 런처: 기존 results.tsv 보호(RES/LOCK 격리), N-GPU 병렬.
# 사용: bash launch_phase1.sh "2 1 0"   (쓸 GPU 목록; 기본 "1 0")
set -u
TP=/home/jhlee/CLUST_KETI/AD_MLDL_termProject/term_project
GPUS="${1:-1 0}"
RESREL=exp_runner/results_phase1.tsv
LOCKREL=exp_runner/locks_p1
JOBS=exp_runner/exp_jobs_phase1.tsv

: > "$TP/$RESREL"
echo -e "ID\tBENCH\tM2\tCLUST\tM1\tMETHOD\tR\tMETRIC\tVALUE\tGEN\tSELMETHOD\tAVG_M1\tAVG_R\tFLOOR\tCAP" > "$TP/exp_runner/results_phase1_header.tsv"
rm -rf "$TP/$LOCKREL"; mkdir -p "$TP/$LOCKREL" "$TP/exp_runner/logs"

for G in $GPUS; do
  RES="$RESREL" LOCK="$LOCKREL" bash "$TP/exp_runner/worker.sh" "$G" "$JOBS" &
done
wait
echo "PHASE1_E2_DONE -> $TP/$RESREL"

#!/bin/bash
# Phase2 묶음A 런처: results 격리, locks 리셋(resume), N-GPU 병렬.
# results는 비우지 않음(append) → 재시작 시 완료 job은 answers resume + 재채점. 중복은 분석에서 dedup.
set -u
TP=/home/jhlee/CLUST_KETI/AD_MLDL_termProject/term_project
GPUS="${1:-1 0}"
RESREL=exp_runner/results_phase2A.tsv
LOCKREL=exp_runner/locks_p2A
JOBS=exp_runner/exp_jobs_phase2_A.tsv

[ -f "$TP/$RESREL" ] || : > "$TP/$RESREL"
echo -e "ID\tBENCH\tM2\tCLUST\tM1\tMETHOD\tR\tMETRIC\tVALUE\tGEN\tSELMETHOD\tAVG_M1\tAVG_R\tFLOOR\tCAP" > "$TP/exp_runner/results_phase2A_header.tsv"
rm -rf "$TP/$LOCKREL"; mkdir -p "$TP/$LOCKREL" "$TP/exp_runner/logs"

for G in $GPUS; do
  RES="$RESREL" LOCK="$LOCKREL" bash "$TP/exp_runner/worker.sh" "$G" "$JOBS" &
done
wait
echo "PHASE2A_DONE -> $TP/$RESREL"

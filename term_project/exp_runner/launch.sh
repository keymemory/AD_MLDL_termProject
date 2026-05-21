#!/bin/bash
# 3-GPU 병렬 워커 런치 (락 기반 job 분배). 모든 워커 종료까지 대기.
TP=/home/jhlee/CLUST_KETI/SKKU_Works/Y1_S1/Advanced_ML_DL/experiments/term_project
cd "$TP/exp_runner"
: > results.tsv
echo -e "ID\tBENCH\tM2\tCLUST\tM1\tMETHOD\tR\tMETRIC\tVALUE\tGEN" > results_header.tsv
rm -rf locks; mkdir -p locks logs
bash worker.sh 2 &
bash worker.sh 1 &
bash worker.sh 0 &
wait
echo "ALL_WORKERS_DONE"

#!/bin/bash
# Experiment W-C: TextVQA weighted_avg M1 Scaling — 2-GPU nohup 런처
# 사용: cd term_project && bash exp_runner/launch_wc.sh

set -e
cd /home/dxlab/jupyter/heejung/AD_MLDL_termProject/term_project

RES=exp_runner/results_scaling_wc.tsv
if [ ! -f "$RES" ]; then
  echo -e "ID\tBENCH\tM2\tCLUST\tM1\tMETHOD\tR\tMETRIC\tVALUE\tGEN" > "$RES"
fi

echo "[$(date)] Experiment W-C (TextVQA weighted_avg) 시작"
echo "총 job 수: $(grep -c '^[^#]' exp_runner/exp_jobs_scaling_wc.tsv)"

nohup bash exp_runner/worker_wc.sh 0 > /tmp/nohup_wc_g0.log 2>&1 &
PID0=$!
nohup bash exp_runner/worker_wc.sh 1 > /tmp/nohup_wc_g1.log 2>&1 &
PID1=$!

echo "PID0=$PID0  PID1=$PID1"
echo "로그: tail -f /tmp/worker_wc_g0.log  또는  tail -f /tmp/worker_wc_g1.log"
echo "결과: tail -f $RES"

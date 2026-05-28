#!/bin/bash
# Experiment W-D: ScienceQA weighted_avg M1 Scaling — 2-GPU nohup 런처
# 사용: cd term_project && bash exp_runner/launch_wd.sh

set -e
cd /home/dxlab/jupyter/heejung/AD_MLDL_termProject/term_project

SQA_BASE=/data1/heejung/datasets/scienceqa
if [ ! -f "$SQA_BASE/problems.json" ] || [ ! -f "$SQA_BASE/pid_splits.json" ]; then
  echo "[ERROR] ScienceQA 미준비. setup_scienceqa.py를 먼저 실행하세요."
  exit 1
fi

RES=exp_runner/results_scaling_wd.tsv
if [ ! -f "$RES" ]; then
  echo -e "ID\tBENCH\tM2\tCLUST\tM1\tMETHOD\tR\tMETRIC\tVALUE\tGEN\tIMG_ACC" > "$RES"
fi

echo "[$(date)] Experiment W-D (ScienceQA weighted_avg) 시작"
echo "총 job 수: $(grep -c '^[^#]' exp_runner/exp_jobs_scaling_wd.tsv)"

nohup bash exp_runner/worker_wd.sh 0 > /tmp/nohup_wd_g0.log 2>&1 &
PID0=$!
nohup bash exp_runner/worker_wd.sh 1 > /tmp/nohup_wd_g1.log 2>&1 &
PID1=$!

echo "PID0=$PID0  PID1=$PID1"
echo "로그: tail -f /tmp/worker_wd_g0.log  또는  tail -f /tmp/worker_wd_g1.log"
echo "결과: tail -f $RES"

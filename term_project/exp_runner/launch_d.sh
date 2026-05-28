#!/bin/bash
# Experiment D: ScienceQA M1 Scaling — 2-GPU nohup 런처
# 사용: cd term_project && bash exp_runner/launch_d.sh

set -e
cd /home/dxlab/jupyter/heejung/AD_MLDL_termProject/term_project

# 셋업 완료 확인
SQA_BASE=/data1/heejung/datasets/scienceqa
if [ ! -f "$SQA_BASE/problems.json" ] || [ ! -f "$SQA_BASE/pid_splits.json" ]; then
  echo "[ERROR] ScienceQA 미준비. setup_scienceqa.py를 먼저 실행하세요."
  exit 1
fi
echo "[OK] ScienceQA base: $(ls $SQA_BASE)"
echo "[OK] Image folders: $(ls $SQA_BASE/images/test | wc -l)"

# results 헤더 (없을 때만)
RES=exp_runner/results_scaling_d.tsv
if [ ! -f "$RES" ]; then
  echo -e "ID\tBENCH\tM2\tCLUST\tM1\tMETHOD\tR\tMETRIC\tVALUE\tGEN\tIMG_ACC" > "$RES"
fi

echo "[$(date)] Experiment D ScienceQA 시작"
nohup bash exp_runner/worker_d.sh 0 > /tmp/nohup_d_g0.log 2>&1 &
PID0=$!
nohup bash exp_runner/worker_d.sh 1 > /tmp/nohup_d_g1.log 2>&1 &
PID1=$!
echo "PID0=$PID0  PID1=$PID1"
echo "로그: tail -f /tmp/worker_d_g0.log"
echo "결과: tail -f $RES"

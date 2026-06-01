#!/bin/bash
# Experiment B: M1 Scaling Law 런처 (2-GPU, nohup)
# 사용: bash launch_b.sh
#   → nohup으로 GPU 0, 1 각 1개 워커를 백그라운드에서 실행
#   → 로그: /tmp/worker_b_g0.log, /tmp/worker_b_g1.log
#   → 결과: exp_runner/results/results_scaling_b.tsv

TP=/home/dxlab/jupyter/heejung/AD_MLDL_termProject/term_project
cd "$TP"

# 결과 파일 헤더 초기화 (없으면 생성, 있으면 유지 — append 모드)
RES=exp_runner/results/results_scaling_b.tsv
if [ ! -f "$RES" ]; then
  echo -e "ID\tBENCH\tM2\tCLUST\tM1\tMETHOD\tR\tMETRIC\tVALUE\tGEN" > "$RES"
fi

# 락/로그 디렉토리 초기화
rm -rf exp_runner/scaling_b_locks
mkdir -p exp_runner/scaling_b_locks exp_runner/scaling_b_logs

echo "=== Experiment B Launch: $(date) ==="
echo "Jobs: exp_runner/jobs/exp_jobs_scaling_b.tsv"
echo "GPU 0 로그: /tmp/worker_b_g0.log"
echo "GPU 1 로그: /tmp/worker_b_g1.log"
echo "결과: $RES"
echo ""

nohup bash exp_runner/workers/worker_b.sh 0 > /tmp/nohup_b_g0.log 2>&1 &
PID0=$!
echo "GPU 0 worker PID: $PID0"

nohup bash exp_runner/workers/worker_b.sh 1 > /tmp/nohup_b_g1.log 2>&1 &
PID1=$!
echo "GPU 1 worker PID: $PID1"

echo ""
echo "백그라운드 실행 중. 진행상황 확인:"
echo "  tail -f /tmp/worker_b_g0.log"
echo "  tail -f /tmp/worker_b_g1.log"
echo "결과 확인:"
echo "  cat $RES"

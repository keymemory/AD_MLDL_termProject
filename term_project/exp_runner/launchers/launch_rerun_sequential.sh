#!/bin/bash
# W-B(미완료) → W-C → W-D 순차 실행 (GPU 경합 방지)
# 각 단계가 완전히 끝난 후 다음 단계 시작
# 사용: nohup bash exp_runner/launchers/launch_rerun_sequential.sh > /tmp/rerun_all.log 2>&1 &

set -u
cd /home/dxlab/jupyter/heejung/AD_MLDL_termProject/term_project

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a /tmp/rerun_all.log; }

# ── Stage 1: W-B 미완료 jobs 재실행 ────────────────────────────────────────────
log "=== Stage 1: W-B (POPE+GQA weighted_avg) 재실행 시작 ==="
log "  미완료 jobs: WM-128_gqa, WM-192/256/384/576 pope+gqa, WM32-48_gqa (10개)"

bash exp_runner/workers/worker_wb.sh 0 >> /tmp/rerun_wb_g0.log 2>&1 &
PID_WB0=$!
bash exp_runner/workers/worker_wb.sh 1 >> /tmp/rerun_wb_g1.log 2>&1 &
PID_WB1=$!

log "  W-B workers 시작: PID0=$PID_WB0, PID1=$PID_WB1"
wait $PID_WB0 $PID_WB1
log "=== Stage 1: W-B 완료 ==="

# ── Stage 2: W-C 전체 실행 ────────────────────────────────────────────────────
log "=== Stage 2: W-C (TextVQA weighted_avg) 시작 ==="

bash exp_runner/workers/worker_wc.sh 0 >> /tmp/rerun_wc_g0.log 2>&1 &
PID_WC0=$!
bash exp_runner/workers/worker_wc.sh 1 >> /tmp/rerun_wc_g1.log 2>&1 &
PID_WC1=$!

log "  W-C workers 시작: PID0=$PID_WC0, PID1=$PID_WC1"
wait $PID_WC0 $PID_WC1
log "=== Stage 2: W-C 완료 ==="

# ── Stage 3: W-D 전체 실행 ────────────────────────────────────────────────────
log "=== Stage 3: W-D (ScienceQA weighted_avg) 시작 ==="

bash exp_runner/workers/worker_wd.sh 0 >> /tmp/rerun_wd_g0.log 2>&1 &
PID_WD0=$!
bash exp_runner/workers/worker_wd.sh 1 >> /tmp/rerun_wd_g1.log 2>&1 &
PID_WD1=$!

log "  W-D workers 시작: PID0=$PID_WD0, PID1=$PID_WD1"
wait $PID_WD0 $PID_WD1
log "=== Stage 3: W-D 완료 ==="

log "=== 전체 재실행 완료 ==="
log "결과 확인:"
log "  W-B: wc -l exp_runner/results/results_scaling_wb.tsv"
log "  W-C: wc -l exp_runner/results/results_scaling_wc.tsv"
log "  W-D: wc -l exp_runner/results/results_scaling_wd.tsv"

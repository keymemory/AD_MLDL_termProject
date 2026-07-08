#!/bin/bash
# Phase2 묶음A GPU2 공존 worker. sihwang님 작업(26GB)을 건드리지 않고 여유분에 worker 1개만.
# 메모리 OOM 방지를 위해 GPU2는 worker 1개만(14.6GB). 같은 lock/results 공유 → job 분배.
[ -f /tmp/phase2A_done.marker ] && exit 0
exec >> /tmp/phase2A_gpu2.log 2>&1
echo "=== GPU2_START $(date) ==="
cd /home/jhlee/CLUST_KETI/AD_MLDL_termProject/term_project || exit 1
source /home/jhlee/miniconda3/etc/profile.d/conda.sh
conda activate vispruner
RES=exp_runner/results_phase2A.tsv LOCK=exp_runner/locks_p2A bash exp_runner/worker.sh 2 exp_runner/exp_jobs_phase2_A.tsv &
RES=exp_runner/results_phase2A.tsv LOCK=exp_runner/locks_p2A bash exp_runner/worker.sh 2 exp_runner/exp_jobs_phase2_A.tsv &
wait
echo "=== GPU2_DONE $(date) ==="

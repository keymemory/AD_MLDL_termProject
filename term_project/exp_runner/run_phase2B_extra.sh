#!/bin/bash
# Phase2 묶음B 추가 worker (GPU 1,2 각 2번째). 같은 lock/results 공유 → job 분배, GPU 활용도↑.
[ -f /tmp/phase2B_done.marker ] && exit 0
exec >> /tmp/phase2B_extra.log 2>&1
echo "=== EXTRA_START $(date) ==="
cd /home/jhlee/CLUST_KETI/AD_MLDL_termProject/term_project || exit 1
source /home/jhlee/miniconda3/etc/profile.d/conda.sh
conda activate vispruner
RES=exp_runner/results_phase2B.tsv LOCK=exp_runner/locks_p2B bash exp_runner/worker.sh 1 exp_runner/exp_jobs_phase2_B.tsv &
RES=exp_runner/results_phase2B.tsv LOCK=exp_runner/locks_p2B bash exp_runner/worker.sh 2 exp_runner/exp_jobs_phase2_B.tsv &
wait
echo "=== EXTRA_DONE $(date) ==="

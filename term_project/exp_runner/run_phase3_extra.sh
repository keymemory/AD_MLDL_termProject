#!/bin/bash
# Phase3 추가 worker (GPU 1,2 각 2번째). 묶음B 완료 후에만.
[ -f /tmp/phase3_done.marker ] && exit 0
[ ! -f /tmp/phase2B_done.marker ] && exit 0
exec >> /tmp/phase3_extra.log 2>&1
echo "=== EXTRA_START $(date) ==="
cd /home/jhlee/CLUST_KETI/AD_MLDL_termProject/term_project || exit 1
source /home/jhlee/miniconda3/etc/profile.d/conda.sh
conda activate vispruner
RES=exp_runner/results_phase3_merge.tsv LOCK=exp_runner/locks_p3 bash exp_runner/worker.sh 1 exp_runner/exp_jobs_phase3.tsv &
RES=exp_runner/results_phase3_merge.tsv LOCK=exp_runner/locks_p3 bash exp_runner/worker.sh 2 exp_runner/exp_jobs_phase3.tsv &
wait
echo "=== EXTRA_DONE $(date) ==="

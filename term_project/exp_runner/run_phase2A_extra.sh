#!/bin/bash
# Phase2 묶음A 추가 worker (GPU당 2번째). 기존 cron chain과 같은 lock/results 공유 → job 분배.
# GPU 0/1에 worker 1개씩 더 붙여 GPU 활용도↑. flock으로 중복 방지, 완료 marker면 종료.
[ -f /tmp/phase2A_done.marker ] && exit 0
exec >> /tmp/phase2A_extra.log 2>&1
echo "=== EXTRA_START $(date) ==="
cd /home/jhlee/CLUST_KETI/AD_MLDL_termProject/term_project || exit 1
source /home/jhlee/miniconda3/etc/profile.d/conda.sh
conda activate vispruner
RES=exp_runner/results_phase2A.tsv LOCK=exp_runner/locks_p2A bash exp_runner/worker.sh 1 exp_runner/exp_jobs_phase2_A.tsv &
wait
echo "=== EXTRA_DONE $(date) ==="

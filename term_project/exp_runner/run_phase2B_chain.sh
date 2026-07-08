#!/bin/bash
# Phase2 묶음B 체인 (cron+flock detach): launch(58 job) → 완료 marker.
[ -f /tmp/phase2B_done.marker ] && exit 0
exec >> /tmp/phase2B_chain.log 2>&1
echo "=== PHASE2B_START $(date) ==="
cd /home/jhlee/CLUST_KETI/AD_MLDL_termProject/term_project || exit 1
source /home/jhlee/miniconda3/etc/profile.d/conda.sh
conda activate vispruner
bash exp_runner/launch_phase2B.sh "1 2" > /tmp/launch_p2B.log 2>&1
touch /tmp/phase2B_done.marker
echo "=== PHASE2B_COMPLETE $(date) ==="

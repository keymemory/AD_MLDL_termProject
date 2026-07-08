#!/bin/bash
# Phase3 실험A 체인 (cron+flock). 묶음B 완료(marker) 후에만 시작 → GPU 1,2 자동 사용.
[ -f /tmp/phase3_done.marker ] && exit 0
[ ! -f /tmp/phase2B_done.marker ] && exit 0   # 묶음B 완료 전엔 대기
exec >> /tmp/phase3_chain.log 2>&1
echo "=== PHASE3_START $(date) ==="
cd /home/jhlee/CLUST_KETI/AD_MLDL_termProject/term_project || exit 1
source /home/jhlee/miniconda3/etc/profile.d/conda.sh
conda activate vispruner
bash exp_runner/launch_phase3.sh "1 2" > /tmp/launch_p3.log 2>&1
python exp_runner/generate_phase3_report.py > /home/jhlee/CLUST_KETI/AD_MLDL_termProject/develop_md/phase3_taskaware_merge_result.md 2>/tmp/gen_p3.log
touch /tmp/phase3_done.marker
echo "=== PHASE3_COMPLETE $(date) ==="

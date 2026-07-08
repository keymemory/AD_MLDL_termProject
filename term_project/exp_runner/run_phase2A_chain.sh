#!/bin/bash
# Phase2 묶음A 실행 체인 (cron+flock detach): launch(210 job, 며칠) → 완료 후 결과 md 자동 생성.
# 중복 방지: 완료 마커 있으면 즉시 종료. cron이 매분 flock으로 호출, 1회만 실제 실행.
[ -f /tmp/phase2A_done.marker ] && exit 0
exec >> /tmp/phase2A_chain.log 2>&1
echo "=== PHASE2A_START $(date) ==="
cd /home/jhlee/CLUST_KETI/AD_MLDL_termProject/term_project || exit 1
source /home/jhlee/miniconda3/etc/profile.d/conda.sh
conda activate vispruner
bash exp_runner/launch_phase2A.sh "1" > /tmp/launch_p2A.log 2>&1
python exp_runner/generate_phase2A_report.py > /home/jhlee/CLUST_KETI/AD_MLDL_termProject/develop_md/phase2_part2_A_result.md 2>/tmp/gen_p2A.log
touch /tmp/phase2A_done.marker
echo "=== PHASE2A_COMPLETE $(date) ==="

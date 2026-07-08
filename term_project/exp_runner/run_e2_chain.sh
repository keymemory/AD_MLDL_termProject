#!/bin/bash
# Phase1 E2 실행 체인: launch(24 job) → 완료 후 결과 md 자동 생성.
# Claude/VSCode와 무관하게 cron(flock) 또는 nohup으로 단독 실행되도록 자족적으로 작성.
# 중복 방지: 완료 마커가 있으면 즉시 종료 (cron 매분 재호출 대비).
[ -f /tmp/e2_done.marker ] && exit 0
exec >> /tmp/e2_chain.log 2>&1
echo "=== E2_CHAIN_START $(date) ==="
cd /home/jhlee/CLUST_KETI/AD_MLDL_termProject/term_project || exit 1
source /home/jhlee/miniconda3/etc/profile.d/conda.sh
conda activate vispruner
bash exp_runner/launch_phase1.sh "1 0" > /tmp/launch_p1.log 2>&1
python exp_runner/generate_e2_report.py > develop_md/phase1_e2_result.md 2>/tmp/gen_report.log
touch /tmp/e2_done.marker
echo "=== E2_CHAIN_COMPLETE $(date) ==="

#!/bin/bash
# Experiment C: TextVQA M1 Scaling — 2-GPU nohup 런처
# 실행 전 setup_textvqa.py가 완료되어야 함
# 사용: cd term_project && bash exp_runner/launchers/launch_c.sh

set -e
cd /home/dxlab/jupyter/heejung/AD_MLDL_termProject/term_project

# 이미지 추출 완료 확인
IMG_DIR=/data1/heejung/datasets/textvqa_val_images
ANN=/home/dxlab/jupyter/heejung/AD_MLDL_termProject/term_project/exp_runner/textvqa_val_annotations.json
if [ ! -f "$ANN" ] || [ $(ls "$IMG_DIR" 2>/dev/null | wc -l) -lt 100 ]; then
  echo "[ERROR] TextVQA 이미지/annotation 미준비. setup_textvqa.py를 먼저 실행하세요."
  exit 1
fi
echo "[OK] TextVQA images: $(ls $IMG_DIR | wc -l)  annotation: $ANN"

# results 헤더 (없을 때만)
RES=exp_runner/results/results_scaling_c.tsv
if [ ! -f "$RES" ]; then
  echo -e "ID\tBENCH\tM2\tCLUST\tM1\tMETHOD\tR\tMETRIC\tVALUE\tGEN" > "$RES"
fi

echo "[$(date)] Experiment C TextVQA 시작"
nohup bash exp_runner/workers/worker_c.sh 0 > /tmp/nohup_c_g0.log 2>&1 &
PID0=$!
nohup bash exp_runner/workers/worker_c.sh 1 > /tmp/nohup_c_g1.log 2>&1 &
PID1=$!
echo "PID0=$PID0  PID1=$PID1"
echo "로그: tail -f /tmp/nohup_c_g0.log  /tmp/worker_c_g0.log"
echo "결과: tail -f $RES"

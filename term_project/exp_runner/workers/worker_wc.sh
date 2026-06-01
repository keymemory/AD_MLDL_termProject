#!/bin/bash
# Experiment W-C: M1 Scaling Law — TextVQA, weighted_avg 전용 워커
# worker_c.sh와 동일 구조, 출력 경로만 wc로 분리
# 사용: bash exp_runner/workers/worker_wc.sh <GPU_INDEX> [jobs_file]
set -u

GPU="${1:?GPU index required}"
JOBS="${2:-exp_runner/jobs/exp_jobs_scaling_wc.tsv}"

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
PYTHON=/home/dxlab/anaconda3/envs/vispruner/bin/python
TP=/home/dxlab/jupyter/heejung/AD_MLDL_termProject/term_project
MODEL=/data1/heejung/hf/llava-v1.5-7b

TEXTVQA_QF=/home/dxlab/jupyter/heejung/VisPruner/playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl
TEXTVQA_IMG=/data1/heejung/datasets/textvqa_val_images
TEXTVQA_ANN=/home/dxlab/jupyter/heejung/AD_MLDL_termProject/term_project/exp_runner/textvqa_val_annotations.json

# 출력 경로 (wc = weighted C)
ANSBASE=$TP/exp_runner/scaling_wc_answers
LOGD=$TP/exp_runner/scaling_wc_logs
LOCK=$TP/exp_runner/scaling_wc_locks
RES=$TP/exp_runner/results/results_scaling_wc.tsv

mkdir -p "$ANSBASE/textvqa" "$LOGD" "$LOCK"

# ── 환경 변수 ──────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=$GPU
export CUDA_LAUNCH_BLOCKING=1
export PYTHONPATH="$TP:${PYTHONPATH:-}"

W=/tmp/worker_wc_g${GPU}.log; : > "$W"
echo "[g$GPU] worker_wc.sh 시작 (jobs=$JOBS)" | tee -a "$W"

TEXTVQA_TOT=$(wc -l < "$TEXTVQA_QF")

while IFS=$'\t' read -r ID BENCH M2 CLUST M1 METHOD R; do
  [[ "$ID" =~ ^#.*$ || -z "${ID:-}" ]] && continue
  [ "$BENCH" != "textvqa" ] && continue

  KEY="${ID}_${BENCH}"
  mkdir -p "$LOCK"
  mkdir "$LOCK/$KEY" 2>/dev/null || continue
  echo "[g$GPU] START $KEY  M2=$M2 CLUST=$CLUST M1=$M1 METHOD=$METHOD r=$R" | tee -a "$W"

  TOT=$TEXTVQA_TOT
  ANS=$ANSBASE/textvqa/${ID}.jsonl
  mkdir -p "$(dirname "$ANS")"

  CL_ARGS=""
  if [ "$CLUST" = "1" ]; then
    CL_ARGS="--enable_clustering --stage1_tokens $M1 --merge_method $METHOD"
  fi

  for attempt in $(seq 1 25); do
    C=0; [ -f "$ANS" ] && C=$(wc -l < "$ANS")
    [ "$C" -ge "$TOT" ] && break
    echo "[g$GPU] $KEY 추론 시도 #$attempt ($C/$TOT)" >> "$W"
    cd "$TP"
    $PYTHON -m llava.eval.model_vqa_loader \
      --model-path "$MODEL" \
      --question-file "$TEXTVQA_QF" \
      --image-folder "$TEXTVQA_IMG" \
      --answers-file "$ANS" \
      --visual_token_num "$M2" \
      --important_ratio "$R" \
      $CL_ARGS \
      --temperature 0 \
      --conv-mode vicuna_v1 >> "$W" 2>&1 \
    || echo "[g$GPU] $KEY 추론 실패 #$attempt" >> "$W"
  done
  GEN=$(wc -l < "$ANS" 2>/dev/null || echo 0)

  EV=$LOGD/${KEY}_textvqa_eval.txt
  mkdir -p "$LOGD"
  VAL="-"; METRIC="Acc"

  cd "$TP"
  $PYTHON llava/eval/eval_textvqa.py \
    --annotation-file "$TEXTVQA_ANN" \
    --result-file "$ANS" > "$EV" 2>&1
  VAL=$(grep -iE "^Accuracy:" "$EV" | tail -1 | awk '{print $2}' | tr -d '%')

  (
    flock 9
    echo -e "${ID}\t${BENCH}\t${M2}\t${CLUST}\t${M1}\t${METHOD}\t${R}\t${METRIC}\t${VAL}\t${GEN}/${TOT}" >> "$RES"
  ) 9>>"$RES.lock"

  echo "[g$GPU] DONE $KEY → $METRIC=$VAL ($GEN/$TOT)" | tee -a "$W"
done < "$JOBS"

echo "[g$GPU] WORKER_WC DONE" | tee -a "$W"

#!/bin/bash
# Experiment C: M1 Scaling Law — TextVQA benchmark 전용 워커
# 사용: bash exp_runner/workers/worker_c.sh <GPU_INDEX> [jobs_file]
# worker_b.sh 구조 동일, textvqa 경로·채점만 변경
set -u

GPU="${1:?GPU index required}"
JOBS="${2:-exp_runner/jobs/exp_jobs_scaling_c.tsv}"

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
PYTHON=/home/dxlab/anaconda3/envs/vispruner/bin/python
TP=/home/dxlab/jupyter/heejung/AD_MLDL_termProject/term_project
MODEL=/data1/heejung/hf/llava-v1.5-7b

# TextVQA 경로 (HF Arrow에서 추출한 이미지 + 생성된 annotation)
TEXTVQA_QF=/home/dxlab/jupyter/heejung/VisPruner/playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl
TEXTVQA_IMG=/data1/heejung/datasets/textvqa_val_images
TEXTVQA_ANN=/home/dxlab/jupyter/heejung/AD_MLDL_termProject/term_project/exp_runner/textvqa_val_annotations.json

# 출력 경로
ANSBASE=$TP/exp_runner/scaling_c_answers
LOGD=$TP/exp_runner/scaling_c_logs
LOCK=$TP/exp_runner/scaling_c_locks
RES=$TP/exp_runner/results/results_scaling_c.tsv

mkdir -p "$ANSBASE/textvqa" "$LOGD" "$LOCK"

# ── 환경 변수 ──────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=$GPU
export CUDA_LAUNCH_BLOCKING=1
export PYTHONPATH="$TP:${PYTHONPATH:-}"

W=/tmp/worker_c_g${GPU}.log; : > "$W"
echo "[g$GPU] worker_c.sh 시작 (jobs=$JOBS)" | tee -a "$W"

# TextVQA 질문 수
TEXTVQA_TOT=$(wc -l < "$TEXTVQA_QF")

while IFS=$'\t' read -r ID BENCH M2 CLUST M1 METHOD R; do
  # 주석·빈 줄 skip
  [[ "$ID" =~ ^#.*$ || -z "${ID:-}" ]] && continue
  # BENCH 확인 (이 워커는 textvqa 전용)
  [ "$BENCH" != "textvqa" ] && continue

  KEY="${ID}_${BENCH}"
  mkdir -p "$LOCK"
  mkdir "$LOCK/$KEY" 2>/dev/null || continue
  echo "[g$GPU] START $KEY  M2=$M2 CLUST=$CLUST M1=$M1 METHOD=$METHOD r=$R" | tee -a "$W"

  TOT=$TEXTVQA_TOT
  ANS=$ANSBASE/textvqa/${ID}.jsonl
  mkdir -p "$(dirname "$ANS")"

  # ── Clustering 인자 구성 ────────────────────────────────────────────────────
  CL_ARGS=""
  if [ "$CLUST" = "1" ]; then
    CL_ARGS="--enable_clustering --stage1_tokens $M1 --merge_method $METHOD"
  fi

  # ── 추론 (resume 지원) ────────────────────────────────────────────────────
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

  # ── 채점 (eval_textvqa.py) ───────────────────────────────────────────────
  EV=$LOGD/${KEY}_textvqa_eval.txt
  mkdir -p "$LOGD"
  VAL="-"; METRIC="Acc"

  cd "$TP"
  $PYTHON llava/eval/eval_textvqa.py \
    --annotation-file "$TEXTVQA_ANN" \
    --result-file "$ANS" > "$EV" 2>&1
  # eval_textvqa.py 출력: "Accuracy: XX.XX%"
  VAL=$(grep -iE "^Accuracy:" "$EV" | tail -1 | awk '{print $2}' | tr -d '%')

  # ── 결과 기록 ────────────────────────────────────────────────────────────
  (
    flock 9
    echo -e "${ID}\t${BENCH}\t${M2}\t${CLUST}\t${M1}\t${METHOD}\t${R}\t${METRIC}\t${VAL}\t${GEN}/${TOT}" >> "$RES"
  ) 9>>"$RES.lock"

  echo "[g$GPU] DONE $KEY → $METRIC=$VAL ($GEN/$TOT)" | tee -a "$W"
done < "$JOBS"

echo "[g$GPU] WORKER_C DONE" | tee -a "$W"

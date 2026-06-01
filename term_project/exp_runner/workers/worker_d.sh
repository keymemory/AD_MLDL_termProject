#!/bin/bash
# Experiment D: M1 Scaling Law — ScienceQA benchmark 전용 워커
# 사용: bash exp_runner/workers/worker_d.sh <GPU_INDEX> [jobs_file]
set -u

GPU="${1:?GPU index required}"
JOBS="${2:-exp_runner/jobs/exp_jobs_scaling_d.tsv}"

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
PYTHON=/home/dxlab/anaconda3/envs/vispruner/bin/python
TP=/home/dxlab/jupyter/heejung/AD_MLDL_termProject/term_project
MODEL=/data1/heejung/hf/llava-v1.5-7b

# ScienceQA 경로
SQA_QF=$TP/llava/eval/../../../VisPruner/playground/data/eval/scienceqa/llava_test_CQM-A.json
# 위 경로 단순화
SQA_QF=/home/dxlab/jupyter/heejung/VisPruner/playground/data/eval/scienceqa/llava_test_CQM-A.json
SQA_IMG=/data1/heejung/datasets/scienceqa/images/test   # {id}/image.png
SQA_BASE=/data1/heejung/datasets/scienceqa              # problems.json, pid_splits.json

SQA_TOT=4241

# 출력 경로
ANSBASE=$TP/exp_runner/scaling_d_answers
LOGD=$TP/exp_runner/scaling_d_logs
LOCK=$TP/exp_runner/scaling_d_locks
RES=$TP/exp_runner/results/results_scaling_d.tsv

mkdir -p "$ANSBASE/sqa" "$LOGD" "$LOCK"

# ── 환경 변수 ──────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=$GPU
export CUDA_LAUNCH_BLOCKING=1
export PYTHONPATH="$TP:${PYTHONPATH:-}"

W=/tmp/worker_d_g${GPU}.log; : > "$W"
echo "[g$GPU] worker_d.sh 시작 (jobs=$JOBS)" | tee -a "$W"

while IFS=$'\t' read -r ID BENCH M2 CLUST M1 METHOD R; do
  [[ "$ID" =~ ^#.*$ || -z "${ID:-}" ]] && continue
  [ "$BENCH" != "sqa" ] && continue

  KEY="${ID}_${BENCH}"
  mkdir -p "$LOCK"
  mkdir "$LOCK/$KEY" 2>/dev/null || continue
  echo "[g$GPU] START $KEY  M2=$M2 CLUST=$CLUST M1=$M1" | tee -a "$W"

  ANS=$ANSBASE/sqa/${ID}.jsonl
  mkdir -p "$(dirname "$ANS")"

  CL_ARGS=""
  if [ "$CLUST" = "1" ]; then
    CL_ARGS="--enable_clustering --stage1_tokens $M1 --merge_method $METHOD"
  fi

  # ── 추론 (resume 지원) ────────────────────────────────────────────────────
  for attempt in $(seq 1 25); do
    C=0; [ -f "$ANS" ] && C=$(wc -l < "$ANS")
    [ "$C" -ge "$SQA_TOT" ] && break
    echo "[g$GPU] $KEY 추론 시도 #$attempt ($C/$SQA_TOT)" >> "$W"
    cd "$TP"
    $PYTHON -m llava.eval.model_vqa_science \
      --model-path "$MODEL" \
      --question-file "$SQA_QF" \
      --image-folder "$SQA_IMG" \
      --answers-file "$ANS" \
      --visual_token_num "$M2" \
      --important_ratio "$R" \
      $CL_ARGS \
      --single-pred-prompt \
      --temperature 0 \
      --conv-mode vicuna_v1 >> "$W" 2>&1 \
    || echo "[g$GPU] $KEY 추론 실패 #$attempt" >> "$W"
  done
  GEN=$(wc -l < "$ANS" 2>/dev/null || echo 0)

  # ── 채점 (eval_science_qa.py) ───────────────────────────────────────────────
  EV=$LOGD/${KEY}_sqa_eval.txt
  OUT_DETAIL=$LOGD/${KEY}_sqa_output.jsonl
  OUT_RESULT=$LOGD/${KEY}_sqa_result.json
  mkdir -p "$LOGD"
  VAL="-"; METRIC="Acc"

  cd "$TP"
  $PYTHON llava/eval/eval_science_qa.py \
    --base-dir "$SQA_BASE" \
    --result-file "$ANS" \
    --output-file "$OUT_DETAIL" \
    --output-result "$OUT_RESULT" > "$EV" 2>&1
  # 출력 형식: "Total: 4241, Correct: XXXX, Accuracy: XX.XX%, IMG-Accuracy: XX.XX%"
  VAL=$(grep -iE "^Total:" "$EV" | tail -1 | grep -oP "Accuracy: \K[0-9.]+")
  IMG_ACC=$(grep -iE "^Total:" "$EV" | tail -1 | grep -oP "IMG-Accuracy: \K[0-9.]+")

  # ── 결과 기록 ──────────────────────────────────────────────────────────────
  (
    flock 9
    echo -e "${ID}\t${BENCH}\t${M2}\t${CLUST}\t${M1}\t${METHOD}\t${R}\t${METRIC}\t${VAL}\t${GEN}/${SQA_TOT}\t${IMG_ACC}" >> "$RES"
  ) 9>>"$RES.lock"

  echo "[g$GPU] DONE $KEY → Acc=$VAL  IMG-Acc=$IMG_ACC  ($GEN/$SQA_TOT)" | tee -a "$W"
done < "$JOBS"

echo "[g$GPU] WORKER_D DONE" | tee -a "$W"

#!/bin/bash
# Experiment B: M1 Scaling Law 전용 워커
# 사용: bash worker_b.sh <GPU_INDEX> [jobs_file]
# 원본 worker.sh에서 dxlab 환경(경로·conda·데이터)에 맞게 수정
set -u

GPU="${1:?GPU index required}"
JOBS="${2:-exp_runner/jobs/exp_jobs_scaling_b.tsv}"

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
PYTHON=/home/dxlab/anaconda3/envs/vispruner/bin/python
TP=/home/dxlab/jupyter/heejung/AD_MLDL_termProject/term_project
MODEL=/data1/heejung/hf/llava-v1.5-7b

# 데이터 경로 (broken symlink 우회 — 직접 경로 지정)
POPE_QF=/home/dxlab/jupyter/heejung/VisPruner/playground/data/eval/pope/llava_pope_test.jsonl
POPE_IMG=/data1/heejung/datasets/pope/val2014
POPE_ANN=/data1/heejung/datasets/pope/coco

GQA_QF=/home/dxlab/jupyter/heejung/VisPruner/playground/data/eval/gqa/llava_gqa_testdev_balanced.jsonl
GQA_IMG=/data1/heejung/datasets/gqa/data/images
GQA_EVAL_DIR=/home/dxlab/jupyter/heejung/VisPruner/playground/data/eval/gqa/data
GQA_Q_PATH=/data1/heejung/datasets/gqa/data/questions

# 출력 경로
ANSBASE=$TP/exp_runner/scaling_b_answers
LOGD=$TP/exp_runner/scaling_b_logs
LOCK=$TP/exp_runner/scaling_b_locks
RES=$TP/exp_runner/results/results_scaling_b.tsv

mkdir -p "$ANSBASE/pope" "$ANSBASE/gqa" "$LOGD" "$LOCK"

# ── 환경 변수 ──────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=$GPU
export CUDA_LAUNCH_BLOCKING=1
# term_project를 Python 경로에 추가 (pip install -e . 대신)
export PYTHONPATH="$TP:${PYTHONPATH:-}"

W=/tmp/worker_b_g${GPU}.log; : > "$W"
echo "[g$GPU] worker_b.sh 시작 (jobs=$JOBS)" | tee -a "$W"

total_q() { case "$1" in pope) echo 8910;; gqa) echo 12578;; esac; }

while IFS=$'\t' read -r ID BENCH M2 CLUST M1 METHOD R; do
  # 주석·빈 줄 skip
  [[ "$ID" =~ ^#.*$ || -z "${ID:-}" ]] && continue

  KEY="${ID}_${BENCH}"
  # [FIX] 락 부모 디렉토리가 삭제됐을 경우 재생성 후 점유 시도
  mkdir -p "$LOCK"
  mkdir "$LOCK/$KEY" 2>/dev/null || continue
  echo "[g$GPU] START $KEY  M2=$M2 CLUST=$CLUST M1=$M1 METHOD=$METHOD r=$R" | tee -a "$W"

  TOT=$(total_q "$BENCH")
  ANS=$ANSBASE/$BENCH/${ID}.jsonl
  mkdir -p "$(dirname "$ANS")"

  # ── Clustering 인자 구성 ────────────────────────────────────────────────────
  CL_ARGS=""
  if [ "$CLUST" = "1" ]; then
    CL_ARGS="--enable_clustering --stage1_tokens $M1 --merge_method $METHOD"
  fi

  # ── 추론 (resume 지원: 기존 답변 이어쓰기) ──────────────────────────────────
  for attempt in $(seq 1 25); do
    C=0; [ -f "$ANS" ] && C=$(wc -l < "$ANS")
    [ "$C" -ge "$TOT" ] && break
    echo "[g$GPU] $KEY 추론 시도 #$attempt ($C/$TOT)" >> "$W"
    cd "$TP"
    $PYTHON -m llava.eval.model_vqa_loader \
      --model-path "$MODEL" \
      --question-file "$( [ "$BENCH" = pope ] && echo "$POPE_QF" || echo "$GQA_QF" )" \
      --image-folder "$( [ "$BENCH" = pope ] && echo "$POPE_IMG" || echo "$GQA_IMG" )" \
      --answers-file "$ANS" \
      --visual_token_num "$M2" \
      --important_ratio "$R" \
      $CL_ARGS \
      --temperature 0 \
      --conv-mode vicuna_v1 >> "$W" 2>&1 \
    || echo "[g$GPU] $KEY 추론 실패 #$attempt" >> "$W"
  done
  GEN=$(wc -l < "$ANS" 2>/dev/null || echo 0)

  # ── 채점 ────────────────────────────────────────────────────────────────────
  METRIC="-"; VAL="-"

  if [ "$BENCH" = "pope" ]; then
    EV=$LOGD/${KEY}_pope_eval.txt
    mkdir -p "$LOGD"   # [FIX] log 디렉토리 삭제 대비 재생성
    cd "$TP"
    $PYTHON llava/eval/eval_pope.py \
      --annotation-dir "$POPE_ANN" \
      --question-file "$POPE_QF" \
      --result-file "$ANS" > "$EV" 2>&1
    VAL=$(grep "Average F1 score:" "$EV" | tail -1 | awk '{print $NF}')
    METRIC="AvgF1"

  elif [ "$BENCH" = "gqa" ]; then
    EV=$LOGD/${KEY}_gqa_eval.txt
    PRED=$ANSBASE/gqa/${ID}_pred.json
    mkdir -p "$LOGD" "$ANSBASE/gqa"  # [FIX] log/answer 디렉토리 삭제 대비 재생성
    cd "$TP"
    $PYTHON scripts/convert_gqa_for_eval.py --src "$ANS" --dst "$PRED" >> "$W" 2>&1
    (
      cd "$GQA_EVAL_DIR"
      $PYTHON eval/eval.py \
        --path "$GQA_Q_PATH" \
        --tier testdev_balanced \
        --predictions "$PRED"
    ) > "$EV" 2>&1
    # eval.py 출력: "Accuracy: XX.XX%" 또는 "accuracy: XX.XX"
    VAL=$(grep -iE "^accuracy:" "$EV" | tail -1 | awk '{print $2}' | tr -d '%')
    METRIC="Acc"
  fi

  # ── 결과 기록 (flock으로 동시쓰기 보호) ─────────────────────────────────────
  (
    flock 9
    echo -e "${ID}\t${BENCH}\t${M2}\t${CLUST}\t${M1}\t${METHOD}\t${R}\t${METRIC}\t${VAL}\t${GEN}/${TOT}" >> "$RES"
  ) 9>>"$RES.lock"

  echo "[g$GPU] DONE $KEY → $METRIC=$VAL ($GEN/$TOT)" | tee -a "$W"
done < "$JOBS"

echo "[g$GPU] WORKER_B DONE" | tee -a "$W"

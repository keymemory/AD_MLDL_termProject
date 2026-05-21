#!/bin/bash
# 락 기반 실험 워커: exp_jobs.tsv의 미점유 job을 잡아 추론(resume/retry)+로컬채점.
# 사용: bash worker.sh <GPU>
set -u
GPU="${1:?GPU index}"
TP=/home/jhlee/CLUST_KETI/SKKU_Works/Y1_S1/Advanced_ML_DL/experiments/term_project
cd "$TP"
source /home/jhlee/miniconda3/etc/profile.d/conda.sh
conda activate vispruner
export CUDA_VISIBLE_DEVICES=$GPU CUDA_LAUNCH_BLOCKING=1
E=playground/data/eval
JOBS="${2:-exp_runner/exp_jobs.tsv}"
LOCK=exp_runner/locks
RES=exp_runner/results.tsv
LOGD=exp_runner/logs
mkdir -p "$LOCK" "$LOGD"
W=/tmp/exp_worker_g${GPU}.log; : > "$W"

total_q(){ case "$1" in pope) echo 8910;; gqa) echo 12578;; textvqa) echo 5000;; vqav2) echo 6000;; esac; }

while read -r ID BENCH M2 CLUST M1 METHOD R; do
  [ -z "${ID:-}" ] && continue
  KEY="${ID}_${BENCH}"
  mkdir "$LOCK/$KEY" 2>/dev/null || continue   # 다른 워커가 점유 → skip
  echo "[g$GPU] START $KEY (M2=$M2 clust=$CLUST M1=$M1 $METHOD r=$R)" | tee -a "$W"
  TOT=$(total_q "$BENCH")
  CL_ARGS=""
  [ "$CLUST" = "1" ] && CL_ARGS="--enable_clustering --stage1_tokens $M1 --merge_method $METHOD"

  case "$BENCH" in
    pope)     QF=$E/pope/llava_pope_test.jsonl;            IMG=$E/pope/val2014 ;;
    gqa)      QF=$E/gqa/llava_gqa_testdev_balanced.jsonl;  IMG=$E/gqa/data/images ;;
    textvqa)  QF=$E/textvqa/llava_textvqa_val_v051_ocr.jsonl; IMG=$E/textvqa/train_images ;;
    vqav2)    QF=$E/vqav2/llava_vqav2_val_subset.jsonl;      IMG=$E/vqav2/val2014 ;;
  esac
  ANS=$E/$BENCH/answers/EXP/$ID/r_${R}.jsonl
  mkdir -p "$(dirname "$ANS")"

  for i in $(seq 1 25); do
    C=0; [ -f "$ANS" ] && C=$(wc -l < "$ANS")
    [ "$C" -ge "$TOT" ] && break
    python -m llava.eval.model_vqa_loader --model-path models/llava-v1.5-7b \
      --question-file "$QF" --image-folder "$IMG" --answers-file "$ANS" \
      --visual_token_num "$M2" --important_ratio "$R" $CL_ARGS \
      --temperature 0 --conv-mode vicuna_v1 >> "$W" 2>&1 || echo "[g$GPU] $KEY 재시도 #$i" >> "$W"
  done
  GEN=$(wc -l < "$ANS" 2>/dev/null || echo 0)

  # ---- 로컬 채점 ----
  METRIC="-"; VAL="-"
  if [ "$BENCH" = "pope" ]; then
    EV=$LOGD/${ID}_pope_eval.txt
    python llava/eval/eval_pope.py --annotation-dir $E/pope/coco \
      --question-file "$QF" --result-file "$ANS" > "$EV" 2>&1
    VAL=$(grep "Average F1 score:" "$EV" | tail -1 | awk '{print $NF}')
    METRIC="AvgF1"
  elif [ "$BENCH" = "gqa" ]; then
    PRED=$E/gqa/data/pred_${ID}.json
    python scripts/convert_gqa_for_eval.py --src "$ANS" --dst "$PRED" >> "$W" 2>&1
    EV=$LOGD/${ID}_gqa_eval.txt
    ( cd $E/gqa/data && python eval/eval.py --tier testdev_balanced \
        --questions questions/testdev_balanced_questions.json \
        --predictions "pred_${ID}.json" ) > "$EV" 2>&1
    VAL=$(grep -i "^Accuracy:" "$EV" | tail -1 | awk '{print $2}' | tr -d '%')
    METRIC="Acc"
  elif [ "$BENCH" = "textvqa" ]; then
    EV=$LOGD/${ID}_textvqa_eval.txt
    python -m llava.eval.eval_textvqa --annotation-file $E/textvqa/TextVQA_0.5.1_val.json \
      --result-file "$ANS" > "$EV" 2>&1
    VAL=$(grep -i "Accuracy:" "$EV" | tail -1 | awk '{print $NF}' | tr -d '%')
    METRIC="Acc"
  elif [ "$BENCH" = "vqav2" ]; then
    EV=$LOGD/${ID}_vqav2_eval.txt
    python exp_runner/vqa_eval.py "$ANS" $E/vqav2/val_subset_gt.json by_type > "$EV" 2>&1
    VAL=$(grep -i "Overall Acc" "$EV" | tail -1 | awk '{print $NF}')
    METRIC="Acc"
  fi
  ( flock 9
    echo -e "${ID}\t${BENCH}\t${M2}\t${CLUST}\t${M1}\t${METHOD}\t${R}\t${METRIC}\t${VAL}\t${GEN}/${TOT}" >> "$RES"
  ) 9>>"$RES.lock"
  echo "[g$GPU] DONE $KEY -> $METRIC=$VAL ($GEN/$TOT)" | tee -a "$W"
done < "$JOBS"
echo "[g$GPU] WORKER_DONE" | tee -a "$W"

#!/bin/bash
# Exp2 worker: automatic Stage-1 token selection experiments.
# 사용: bash exp_runner/workers/worker_exp2.sh <GPU_INDEX> <jobs_file>
set -u

GPU="${1:?GPU index required}"
JOBS="${2:-exp_runner/jobs/exp2_attngain_greedygain_pope_gqa_jobs.tsv}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TP="${TP:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
REPO_ROOT="$(cd "$TP/.." && pwd)"

PYTHON="${PYTHON:-$(command -v python)}"
MODEL="${MODEL:-/data1/heejung/hf/llava-v1.5-7b}"
EXP2="${EXP2:-$REPO_ROOT/vispruner_md/exp2}"

POPE_QF="${POPE_QF:-/home/dxlab/jupyter/heejung/VisPruner/playground/data/eval/pope/llava_pope_test.jsonl}"
POPE_IMG="${POPE_IMG:-/data1/heejung/datasets/pope/val2014}"
POPE_ANN="${POPE_ANN:-/data1/heejung/datasets/pope/coco}"

GQA_QF="${GQA_QF:-/home/dxlab/jupyter/heejung/VisPruner/playground/data/eval/gqa/llava_gqa_testdev_balanced.jsonl}"
GQA_IMG="${GQA_IMG:-/data1/heejung/datasets/gqa/data/images}"
GQA_EVAL_DIR="${GQA_EVAL_DIR:-/home/dxlab/jupyter/heejung/VisPruner/playground/data/eval/gqa/data}"
GQA_Q_PATH="${GQA_Q_PATH:-/data1/heejung/datasets/gqa/data/questions}"

TEXTVQA_QF="${TEXTVQA_QF:-/home/dxlab/jupyter/heejung/VisPruner/playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl}"
TEXTVQA_IMG="${TEXTVQA_IMG:-/data1/heejung/datasets/textvqa_val_images}"
TEXTVQA_ANN="${TEXTVQA_ANN:-$TP/exp_runner/textvqa_val_annotations.json}"

SQA_QF="${SQA_QF:-/home/dxlab/jupyter/heejung/VisPruner/playground/data/eval/scienceqa/llava_test_CQM-A.json}"
SQA_IMG="${SQA_IMG:-/data1/heejung/datasets/scienceqa/images/test}"
SQA_BASE="${SQA_BASE:-/data1/heejung/datasets/scienceqa}"

ANSBASE=$TP/exp_runner/exp2_answers
LOGD=$TP/exp_runner/exp2_logs
LOCK=$TP/exp_runner/exp2_locks
RES=$EXP2/exp2_results.tsv
STATD=$EXP2/attn_stats

die() {
  echo "[worker_exp2] ERROR: $*" >&2
  exit 2
}

require_file() {
  [ -f "$1" ] || die "required file not found: $1"
}

require_dir() {
  [ -d "$1" ] || die "required directory not found: $1"
}

validate_benchmark() {
  case "$1" in
    pope)
      require_file "$POPE_QF"
      require_dir "$POPE_IMG"
      require_dir "$POPE_ANN"
      ;;
    gqa)
      require_file "$GQA_QF"
      require_dir "$GQA_IMG"
      require_file "$GQA_EVAL_DIR/eval/eval.py"
      require_dir "$GQA_Q_PATH"
      ;;
    textvqa)
      require_file "$TEXTVQA_QF"
      require_dir "$TEXTVQA_IMG"
      require_file "$TEXTVQA_ANN"
      ;;
    sqa)
      require_file "$SQA_QF"
      require_dir "$SQA_IMG"
      require_file "$SQA_BASE/problems.json"
      require_file "$SQA_BASE/pid_splits.json"
      ;;
    *) die "unsupported benchmark: $1";;
  esac
}

require_file "$MODEL/config.json"
require_file "$JOBS"

mkdir -p "$ANSBASE" "$LOGD" "$LOCK" "$STATD" "$EXP2"
[ -f "$RES" ] || echo -e "ID\tPHASE\tBENCH\tM2\tMETHOD\tSELECT\tDIVERSE\tSTAGE1\tR\tTAU\tMETRIC\tVALUE\tMETRIC2\tVALUE2\tGEN\tM1_MEAN\tM1_STD\tM1_MIN\tM1_MAX\tINFER_SEC\tLATENCY_SEC_PER_Q" > "$RES"

export CUDA_VISIBLE_DEVICES=$GPU
export CUDA_LAUNCH_BLOCKING=1
export PYTHONPATH="$TP:${PYTHONPATH:-}"

W=/tmp/worker_exp2_g${GPU}.log; : > "$W"
echo "[g$GPU] worker_exp2 시작 jobs=$JOBS" | tee -a "$W"

total_q() {
  case "$1" in
    pope) echo 8910;;
    gqa) echo 12578;;
    textvqa) wc -l < "$TEXTVQA_QF";;
    sqa) echo 4241;;
    *) echo 0;;
  esac
}

run_infer() {
  local bench="$1" ans="$2" m2="$3" method="$4" select="$5" diverse="$6" stage1="$7"
  local qf img module extra
  module="llava.eval.model_vqa_loader"
  extra=""
  case "$bench" in
    pope) qf="$POPE_QF"; img="$POPE_IMG";;
    gqa) qf="$GQA_QF"; img="$GQA_IMG";;
    textvqa) qf="$TEXTVQA_QF"; img="$TEXTVQA_IMG";;
    sqa) qf="$SQA_QF"; img="$SQA_IMG"; module="llava.eval.model_vqa_science"; extra="--single-pred-prompt";;
    *) return 2;;
  esac
  $PYTHON -m "$module" \
    --model-path "$MODEL" \
    --question-file "$qf" \
    --image-folder "$img" \
    --answers-file "$ans" \
    --visual_token_num "$m2" \
    --important_ratio 0.5 \
    --enable_clustering \
    --stage1_tokens "$stage1" \
    --merge_method "$method" \
    --select_mode "$select" \
    --diverse_mode "$diverse" \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    $extra
}

eval_one() {
  local bench="$1" id="$2" ans="$3" ev="$4"
  case "$bench" in
    pope)
      $PYTHON "$TP/llava/eval/eval_pope.py" --annotation-dir "$POPE_ANN" --question-file "$POPE_QF" --result-file "$ans" > "$ev" 2>&1
      local avg_f1 avg_acc
      avg_f1=$(grep "Average F1 score:" "$ev" | tail -1 | awk '{print $NF}')
      avg_acc=$(awk '/^Accuracy:/ {sum += $2; n += 1} END {if (n) printf "%.10f", sum/n}' "$ev")
      echo -e "AvgF1\t${avg_f1}\tAvgAcc\t${avg_acc}"
      ;;
    gqa)
      local pred="$ANSBASE/gqa/${id}_pred.json"
      $PYTHON "$TP/scripts/convert_gqa_for_eval.py" --src "$ans" --dst "$pred" >> "$W" 2>&1
      (cd "$GQA_EVAL_DIR" && $PYTHON eval/eval.py --path "$GQA_Q_PATH" --tier testdev_balanced --predictions "$pred") > "$ev" 2>&1
      grep -iE "^accuracy:" "$ev" | tail -1 | awk '{gsub("%","",$2); print "Acc\t"$2"\t-\t-"}'
      ;;
    textvqa)
      $PYTHON "$TP/llava/eval/eval_textvqa.py" --annotation-file "$TEXTVQA_ANN" --result-file "$ans" > "$ev" 2>&1
      grep -iE "^Accuracy:" "$ev" | tail -1 | awk '{gsub("%","",$2); print "Acc\t"$2"\t-\t-"}'
      ;;
    sqa)
      local out_detail="$LOGD/${id}_sqa_output.jsonl"
      local out_result="$LOGD/${id}_sqa_result.json"
      $PYTHON "$TP/llava/eval/eval_science_qa.py" --base-dir "$SQA_BASE" --result-file "$ans" --output-file "$out_detail" --output-result "$out_result" > "$ev" 2>&1
      grep -iE "^Total:" "$ev" | tail -1 | grep -oP "(?<!IMG-)Accuracy: \\K[0-9.]+" | awk '{print "Acc\t"$1"\t-\t-"}'
      ;;
  esac
}

stats_one() {
  local sf="$1"
  $PYTHON - "$sf" <<'PY'
import json, math, sys
vals = []
for line in open(sys.argv[1]):
    try:
        vals.append(float(json.loads(line)["m1"]))
    except Exception:
        pass
if not vals:
    print("-\t-\t-\t-")
else:
    mean = sum(vals) / len(vals)
    var = sum((x - mean) ** 2 for x in vals) / len(vals)
    print(f"{mean:.2f}\t{math.sqrt(var):.2f}\t{min(vals):.0f}\t{max(vals):.0f}")
PY
}

while IFS=$'\t' read -r ID PHASE BENCH M2 METHOD SELECT DIVERSE STAGE1; do
  [[ "$ID" =~ ^#.*$ || -z "${ID:-}" ]] && continue
  validate_benchmark "$BENCH"
  KEY="${ID}_${BENCH}"
  mkdir -p "$LOCK"
  mkdir "$LOCK/$KEY" 2>/dev/null || continue

  TOT=$(total_q "$BENCH")
  ANS=$ANSBASE/$BENCH/${ID}.jsonl
  STAT=$STATD/${KEY}.jsonl
  EV=$LOGD/${KEY}_eval.txt
  mkdir -p "$(dirname "$ANS")" "$LOGD"
  touch "$STAT"

  echo "[g$GPU] START $KEY M2=$M2 method=$METHOD select=$SELECT diverse=$DIVERSE stage1=$STAGE1" | tee -a "$W"
  INITIAL_GEN=0; [ -f "$ANS" ] && INITIAL_GEN=$(wc -l < "$ANS")
  INFER_START_NS=$(date +%s%N)
  for attempt in $(seq 1 25); do
    C=0; [ -f "$ANS" ] && C=$(wc -l < "$ANS")
    [ "$C" -ge "$TOT" ] && break
    echo "[g$GPU] $KEY inference attempt #$attempt ($C/$TOT)" >> "$W"
    cd "$TP"
    EXP2_SELECTION_LOG="$STAT" run_infer "$BENCH" "$ANS" "$M2" "$METHOD" "$SELECT" "$DIVERSE" "$STAGE1" >> "$W" 2>&1 \
      || echo "[g$GPU] $KEY inference failed #$attempt" >> "$W"
  done
  INFER_END_NS=$(date +%s%N)

  GEN=$(wc -l < "$ANS" 2>/dev/null || echo 0)
  INFER_SEC=$(awk -v start="$INFER_START_NS" -v end="$INFER_END_NS" 'BEGIN {printf "%.6f", (end-start)/1000000000}')
  if [ "$INITIAL_GEN" -eq 0 ] && [ "$GEN" -gt 0 ]; then
    LATENCY_SEC_PER_Q=$(awk -v sec="$INFER_SEC" -v n="$GEN" 'BEGIN {printf "%.6f", sec/n}')
  else
    LATENCY_SEC_PER_Q="RESUMED"
  fi
  METVAL=$(eval_one "$BENCH" "$ID" "$ANS" "$EV")
  METRIC=$(echo "$METVAL" | cut -f1)
  VALUE=$(echo "$METVAL" | cut -f2)
  METRIC2=$(echo "$METVAL" | cut -f3)
  VALUE2=$(echo "$METVAL" | cut -f4)
  STATS=$(stats_one "$STAT")

  (
    flock 9
    echo -e "${ID}\t${PHASE}\t${BENCH}\t${M2}\t${METHOD}\t${SELECT}\t${DIVERSE}\t${STAGE1}\t-\t-\t${METRIC}\t${VALUE}\t${METRIC2}\t${VALUE2}\t${GEN}/${TOT}\t${STATS}\t${INFER_SEC}\t${LATENCY_SEC_PER_Q}" >> "$RES"
  ) 9>>"$RES.lock"
  echo "[g$GPU] DONE $KEY $METRIC=$VALUE ($GEN/$TOT) infer=${INFER_SEC}s latency=${LATENCY_SEC_PER_Q}s/q" | tee -a "$W"
done < "$JOBS"

echo "[g$GPU] worker_exp2 DONE" | tee -a "$W"

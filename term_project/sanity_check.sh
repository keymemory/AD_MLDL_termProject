#!/bin/bash
# Two-Stage Framework sanity check 3종 (POPE 300 subset, 최종 64토큰, r=0.5)
set -u
cd /home/jhlee/CLUST_KETI/SKKU_Works/Y1_S1/Advanced_ML_DL/experiments/term_project
source /home/jhlee/miniconda3/etc/profile.d/conda.sh
conda activate vispruner
export CUDA_VISIBLE_DEVICES=2 CUDA_LAUNCH_BLOCKING=1
P=playground/data/eval/pope
Q=$P/llava_pope_sanity.jsonl
IMG=$P/val2014
ANNO=$P/coco_sanity
LOG=/tmp/sanity_twostage.log; : > $LOG

score(){ python llava/eval/eval_pope.py --annotation-dir $ANNO --question-file $Q --result-file "$1" 2>&1 | grep -E "Average F1"; }

echo "===== (1) VisPruner only: 64토큰, r=0.5 (clustering off) =====" | tee -a $LOG
A1=$P/answers/sanity/visp_only_n64.jsonl
python -m llava.eval.model_vqa_loader --model-path models/llava-v1.5-7b \
  --question-file $Q --image-folder $IMG --answers-file $A1 \
  --visual_token_num 64 --important_ratio 0.5 \
  --temperature 0 --conv-mode vicuna_v1 >> $LOG 2>&1
echo "[1] 생성 $(wc -l < $A1)/300" | tee -a $LOG
echo -n "[1] VisPruner-only F1 = " | tee -a $LOG; score $A1 | tee -a $LOG

echo "===== (2) Two-stage simple_avg: M1=128 -> M2=64 =====" | tee -a $LOG
A2=$P/answers/sanity/two_simple_n64.jsonl
python -m llava.eval.model_vqa_loader --model-path models/llava-v1.5-7b \
  --question-file $Q --image-folder $IMG --answers-file $A2 \
  --enable_clustering --stage1_tokens 128 --visual_token_num 64 \
  --merge_method simple_avg --important_ratio 0.5 \
  --temperature 0 --conv-mode vicuna_v1 >> $LOG 2>&1
echo "[2] 생성 $(wc -l < $A2)/300" | tee -a $LOG
echo -n "[2] two-stage simple_avg F1 = " | tee -a $LOG; score $A2 | tee -a $LOG

echo "===== (3) Two-stage weighted_avg: M1=128 -> M2=64 =====" | tee -a $LOG
A3=$P/answers/sanity/two_weighted_n64.jsonl
python -m llava.eval.model_vqa_loader --model-path models/llava-v1.5-7b \
  --question-file $Q --image-folder $IMG --answers-file $A3 \
  --enable_clustering --stage1_tokens 128 --visual_token_num 64 \
  --merge_method weighted_avg --important_ratio 0.5 \
  --temperature 0 --conv-mode vicuna_v1 >> $LOG 2>&1
echo "[3] 생성 $(wc -l < $A3)/300" | tee -a $LOG
echo -n "[3] two-stage weighted_avg F1 = " | tee -a $LOG; score $A3 | tee -a $LOG
echo "SANITY_DONE" | tee -a $LOG

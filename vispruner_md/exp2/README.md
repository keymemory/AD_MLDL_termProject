# AGG Reproduction Guide

이 문서는 `agg-token-selection` 브랜치의 AGG(Attention Gain + Greedy diversity Gain) 실험을 다른 환경에서 처음부터 재현하기 위한 안내서다. 모델과 데이터는 용량 때문에 GitHub에 포함하지 않는다.

## 1. 재현 범위

AGG는 LLaVA-1.5-7B의 576개 image patch token 중 Stage-1 token 수 `M1`을 자동으로 결정하고, spherical k-means로 최종 `M2`개 token을 만든다.

- Important: CLS-to-patch attention을 내림차순 정렬하고, 인접 attention 차분 곡선의 elbow까지 선택
- Diverse: 선택 집합과의 최대 cosine similarity가 가장 낮은 token을 greedy하게 추가하고, marginal diversity gain 곡선의 elbow에서 정지
- `M1 = n_important + n_diverse`
- 사용자가 지정하는 token 관련 값은 최종 token 수 `M2`뿐이다.
- 평가 데이터셋: POPE, GQA, TextVQA, ScienceQA-IMG
- 설정: `M2={32,64,128}` x `{simple_avg,weighted_avg}`

방법과 결과 해석은 [exp2_plan.md](exp2_plan.md), 최종 결론은 [result.md](result.md), 공개된 행 단위 결과는 [exp2_results.tsv](exp2_results.tsv)를 참고한다. POPE weighted latency 전용 계획과 결과는 [pope_agg_weighted_latency_plan.md](pope_agg_weighted_latency_plan.md), [pope_agg_weighted_latency_results.tsv](pope_agg_weighted_latency_results.tsv)에 분리해 둔다.

## 2. 공개 파일 구조

```text
AD_MLDL_termProject/
├── README.md
├── dataset/README.md
├── term_project/
│   ├── pyproject.toml
│   ├── llava/
│   │   ├── eval/
│   │   │   ├── model_vqa_loader.py       # POPE/GQA/TextVQA inference CLI
│   │   │   └── model_vqa_science.py      # ScienceQA inference CLI
│   │   └── model/
│   │       ├── llava_arch.py              # AGG Stage-1 선택 및 Stage-2 호출
│   │       ├── spherical_kmeans.py        # Stage-2 cosine k-means/merge
│   │       └── language_model/
│   │           └── llava_llama.py         # AGG 옵션 저장/getter
│   ├── scripts/
│   │   └── convert_gqa_for_eval.py        # LLaVA answer를 GQA 평가 형식으로 변환
│   └── exp_runner/
│       ├── exp2_selection_smoke.py        # AGG 핵심 함수 단위 smoke test
│       ├── textvqa_val_annotations.json   # TextVQA local evaluation annotation
│       ├── workers/
│       │   └── worker_exp2.sh             # inference, evaluation, 통계 기록
│       └── jobs/
│           ├── exp2_attngain_greedygain_pope_gqa_jobs.tsv
│           ├── exp2_attngain_greedygain_textvqa_sqa_jobs.tsv
│           └── exp2_agg_weighted_pope_latency_jobs.tsv
└── vispruner_md/exp2/
    ├── README.md                           # 현재 문서
    ├── exp2_plan.md                        # 방법, 설정, 전체 요약
    ├── exp2_results.tsv                    # AGG + fixed 50:50 공개 결과
    ├── pope_agg_weighted_latency_plan.md   # POPE latency 측정 조건과 해석
    ├── pope_agg_weighted_latency_results.tsv # POPE latency 결과 TSV
    └── result.md                           # 최종 결론
```

실행 후 아래 생성물은 `.gitignore` 대상이다.

```text
term_project/exp_runner/exp2_answers/       # 모델 answer JSONL
term_project/exp_runner/exp2_logs/          # benchmark evaluator 출력
term_project/exp_runner/exp2_locks/         # 중복 실행 방지 lock 디렉터리
vispruner_md/exp2/attn_stats/                # 샘플별 M1/important/diverse 통계
reproduced_results/                          # 권장 재현 결과 저장 위치
```

## 3. 요구 환경

원 실험 환경은 다음과 같다.

| 항목 | 버전/설정 |
|---|---|
| OS | Linux |
| Python | 3.10 |
| CUDA | 12.1 계열 |
| GPU | NVIDIA RTX A6000 48GB |
| PyTorch | 2.1.2 |
| torchvision | 0.16.2 |
| transformers | 4.37.2 |
| tokenizers | 0.15.1 |
| inference | fp16, temperature 0 |

LLaVA-1.5-7B inference는 약 15GB의 GPU memory를 사용했다. CUDA와 GPU driver 조합이 다르면 동일한 PyTorch wheel을 설치할 수 있는지 먼저 확인한다.

## 4. 저장소 및 환경 설치

```bash
git clone https://github.com/keymemory/AD_MLDL_termProject.git
cd AD_MLDL_termProject
git switch agg-token-selection

conda create -n vispruner python=3.10 -y
conda activate vispruner

cd term_project
python -m pip install --upgrade pip
python -m pip install -e .
cd ..
```

설치를 확인한다.

```bash
conda run -n vispruner python -c \
  "import torch, transformers; print(torch.__version__, transformers.__version__)"
```

예상 주요 버전은 `2.1.2`와 `4.37.2`다.

## 5. 모델 준비

Hugging Face의 `liuhaotian/llava-v1.5-7b` checkpoint를 사용한다. 아래는 저장소 바깥의 `/path/to/models`에 받는 예시다.

```bash
conda activate vispruner
python -m pip install "huggingface_hub[cli]"

huggingface-cli download liuhaotian/llava-v1.5-7b \
  --local-dir /path/to/models/llava-v1.5-7b
```

checkpoint의 `config.json`에서 `mm_vision_tower`를 확인한다.

```bash
python - <<'PY'
import json
p = "/path/to/models/llava-v1.5-7b/config.json"
print(json.load(open(p))["mm_vision_tower"])
PY
```

값이 `openai/clip-vit-large-patch14-336`이면 최초 실행 시 Hugging Face에서 자동으로 받는다. 완전한 offline 실행이 필요하면 CLIP도 미리 받고 `config.json`의 값을 로컬 절대경로로 변경한다.

```bash
huggingface-cli download openai/clip-vit-large-patch14-336 \
  --local-dir /path/to/models/clip-vit-large-patch14-336
```

## 6. 데이터 준비

worker는 데이터가 저장소 내부에 있을 것을 강제하지 않는다. 아래 환경변수로 임의의 위치를 지정할 수 있다. 공개 결과와 같은 평가를 위해 질문 파일의 문항 수와 split을 반드시 맞춘다.

### 6.1 POPE

필요 항목:

```text
/path/to/data/pope/
├── llava_pope_test.jsonl            # 8,910 lines
├── val2014/                          # COCO val2014 images
│   └── COCO_val2014_*.jpg
└── coco/                             # POPE category annotations
    ├── coco_adversarial.json
    ├── coco_popular.json
    └── coco_random.json
```

환경변수:

```bash
export POPE_QF=/path/to/data/pope/llava_pope_test.jsonl
export POPE_IMG=/path/to/data/pope/val2014
export POPE_ANN=/path/to/data/pope/coco
```

### 6.2 GQA

필요 항목:

```text
/path/to/data/gqa/
├── llava_gqa_testdev_balanced.jsonl  # 12,578 lines
├── data/
│   ├── images/
│   └── questions/                    # official GQA question files
└── eval_root/
    └── eval/eval.py                  # official GQA evaluator
```

환경변수:

```bash
export GQA_QF=/path/to/data/gqa/llava_gqa_testdev_balanced.jsonl
export GQA_IMG=/path/to/data/gqa/data/images
export GQA_Q_PATH=/path/to/data/gqa/data/questions
export GQA_EVAL_DIR=/path/to/data/gqa/eval_root
```

`GQA_EVAL_DIR`에서 아래 명령 형태가 동작해야 한다.

```bash
cd "$GQA_EVAL_DIR"
python eval/eval.py --help
```

### 6.3 TextVQA

필요 항목:

```text
/path/to/data/textvqa/
├── llava_textvqa_val_v051_ocr.jsonl  # 5,000 lines
└── train_images/                     # validation에 사용되는 image files
```

평가 annotation은 저장소의 `term_project/exp_runner/textvqa_val_annotations.json`을 기본 사용한다.

```bash
export TEXTVQA_QF=/path/to/data/textvqa/llava_textvqa_val_v051_ocr.jsonl
export TEXTVQA_IMG=/path/to/data/textvqa/train_images
# 다른 annotation을 쓸 때만 지정
# export TEXTVQA_ANN=/path/to/TextVQA_0.5.1_val.json
```

### 6.4 ScienceQA-IMG

필요 항목:

```text
/path/to/data/scienceqa/
├── llava_test_CQM-A.json             # image subset, 4,241 questions
├── problems.json
├── pid_splits.json
└── images/test/<question_id>/image.png
```

환경변수:

```bash
export SQA_QF=/path/to/data/scienceqa/llava_test_CQM-A.json
export SQA_IMG=/path/to/data/scienceqa/images/test
export SQA_BASE=/path/to/data/scienceqa
```

데이터 출처와 추가 준비 기록은 [dataset/README.md](../../dataset/README.md), [05_additional_benchmarks.md](../05_additional_benchmarks.md), LLaVA 형식 설명은 [term_project/EVAL.md](../../term_project/EVAL.md)를 함께 참고한다.

## 7. 실행 전 검증

저장소 루트에서 다음 변수들을 설정한다. `/path/to/...`는 실제 경로로 바꾼다.

```bash
export REPO_ROOT="$PWD"
export MODEL=/path/to/models/llava-v1.5-7b
export EXP2="$REPO_ROOT/reproduced_results/agg"

# 위 데이터 준비 절에서 필요한 환경변수도 export한다.
```

파일 존재 여부와 문항 수를 확인한다.

```bash
test -f "$MODEL/config.json"
wc -l "$POPE_QF"       # 8910
wc -l "$GQA_QF"        # 12578
wc -l "$TEXTVQA_QF"    # 5000
test -f "$SQA_QF"
test -f "$SQA_BASE/problems.json"
nvidia-smi
```

AGG 핵심 함수 smoke test를 실행한다.

```bash
cd "$REPO_ROOT/term_project"
PYTHONPATH="$PWD" conda run -n vispruner \
  python exp_runner/exp2_selection_smoke.py
```

성공 시 다음 문구가 출력된다.

```text
AGG selection smoke: ok
```

## 8. 전체 실험 실행

모든 공식 재현 실행은 `nohup`으로 시작한다. worker의 첫 번째 인자는 물리 GPU index, 두 번째 인자는 job TSV다.

### 8.1 순차 실행: GPU 1개

```bash
cd "$REPO_ROOT/term_project"
mkdir -p "$REPO_ROOT/reproduced_results/agg"

nohup env \
  MODEL="$MODEL" EXP2="$EXP2" \
  POPE_QF="$POPE_QF" POPE_IMG="$POPE_IMG" POPE_ANN="$POPE_ANN" \
  GQA_QF="$GQA_QF" GQA_IMG="$GQA_IMG" \
  GQA_EVAL_DIR="$GQA_EVAL_DIR" GQA_Q_PATH="$GQA_Q_PATH" \
  conda run -n vispruner --no-capture-output \
  bash exp_runner/workers/worker_exp2.sh 1 \
  exp_runner/jobs/exp2_attngain_greedygain_pope_gqa_jobs.tsv \
  > "$EXP2/pope_gqa.nohup.log" 2>&1 &

echo $! > "$EXP2/pope_gqa.pid"
```

POPE/GQA가 끝난 뒤 TextVQA/SQA를 실행한다.

```bash
nohup env \
  MODEL="$MODEL" EXP2="$EXP2" \
  TEXTVQA_QF="$TEXTVQA_QF" TEXTVQA_IMG="$TEXTVQA_IMG" \
  SQA_QF="$SQA_QF" SQA_IMG="$SQA_IMG" SQA_BASE="$SQA_BASE" \
  conda run -n vispruner --no-capture-output \
  bash exp_runner/workers/worker_exp2.sh 1 \
  exp_runner/jobs/exp2_attngain_greedygain_textvqa_sqa_jobs.tsv \
  > "$EXP2/textvqa_sqa.nohup.log" 2>&1 &

echo $! > "$EXP2/textvqa_sqa.pid"
```

### 8.2 병렬 실행: GPU 2개

두 job 파일은 서로 다른 benchmark를 사용하므로 GPU가 2개라면 동시에 실행할 수 있다. 첫 worker에는 GPU 0, 둘째에는 GPU 1을 지정한다. 두 worker가 같은 `exp2_results.tsv`에 쓸 때 `flock`으로 행 추가를 보호한다.

### 8.3 POPE weighted latency 재현

논문/보고서의 latency 비교용으로는 POPE, weighted merge, `M2={128,64,32}`만 순차 실행한다. 이 job은 F1과 Accuracy를 동시에 기록하고, 조건별 end-to-end inference 시간을 문항 수로 나눈 `LATENCY_SEC_PER_Q`를 남긴다.

측정 조건:

- Model: LLaVA-1.5-7B + AGG
- Dataset: POPE full test, 8,910 questions
- Merge: `weighted_avg`
- `M2`: 128 -> 64 -> 32 순차 실행
- GPU: worker 첫 번째 인자로 지정한 단일 GPU
- Inference: batch size 1, fp16, temperature 0
- Latency 포함 범위: model/vision tower loading, image I/O, preprocessing, AGG selection, spherical k-means, weighted merge, generation, answer write
- Latency 제외 범위: POPE evaluator 실행 시간, job 사이 shell 대기 시간

실행 예시는 다음과 같다.

```bash
cd "$REPO_ROOT"
export EXP2="$REPO_ROOT/reproduced_results/pope_agg_weighted_latency"
mkdir -p "$EXP2"

cd "$REPO_ROOT/term_project"
nohup env \
  MODEL="$MODEL" EXP2="$EXP2" \
  POPE_QF="$POPE_QF" POPE_IMG="$POPE_IMG" POPE_ANN="$POPE_ANN" \
  conda run -n vispruner --no-capture-output \
  bash exp_runner/workers/worker_exp2.sh 1 \
  exp_runner/jobs/exp2_agg_weighted_pope_latency_jobs.tsv \
  > "$EXP2/nohup.log" 2>&1 &

echo $! > "$EXP2/pope_latency.pid"
```

완료 확인:

```bash
tail -f "$EXP2/nohup.log"
column -ts $'\t' "$EXP2/exp2_results.tsv" | less -S
```

공개 재현 결과:

| M2 | AvgF1 | AvgAcc | M1 mean | INFER_SEC | Latency sec/q |
|---:|---:|---:|---:|---:|---:|
| 32 | 0.8099 | 0.8288 | 105.17 | 2726.45 | 0.3060 |
| 64 | 0.8470 | 0.8572 | 105.20 | 2711.07 | 0.3043 |
| 128 | 0.8585 | 0.8662 | 129.22 | 2817.60 | 0.3162 |

정리된 공개 TSV는 [pope_agg_weighted_latency_results.tsv](pope_agg_weighted_latency_results.tsv)다. 실행 직후의 runtime TSV는 `$EXP2/exp2_results.tsv`에 생성된다.

## 9. 진행 상태 확인

```bash
# nohup wrapper log
tail -f "$EXP2/pope_gqa.nohup.log"

# worker가 기록하는 실제 진행 log (GPU 번호에 따라 파일명이 달라짐)
tail -f /tmp/worker_exp2_g1.log

# 실행 중인 inference 확인
ps -ef | grep -E 'worker_exp2|llava.eval.model_vqa' | grep -v grep

# GPU 확인
nvidia-smi

# 현재 생성된 결과
column -ts $'\t' "$EXP2/exp2_results.tsv" | less -S
```

worker는 answer line 수가 목표 문항 수보다 작으면 같은 파일에 이어서 생성한다. 완료된 answer는 다시 inference하지 않고 평가 단계로 넘어간다.

## 10. 산출물

```text
reproduced_results/agg/
├── exp2_results.tsv                   # 최종 집계
├── pope_gqa.nohup.log
├── textvqa_sqa.nohup.log
└── attn_stats/
    └── AGG-<M2><s|w>_<dataset>.jsonl # 샘플별 선택 통계

term_project/exp_runner/
├── exp2_answers/<dataset>/
│   └── AGG-<M2><s|w>.jsonl           # 모델 생성 답변
├── exp2_logs/
│   └── AGG-..._eval.txt               # evaluator 원문 출력
└── exp2_locks/
    └── AGG-..._<dataset>/             # 완료/실행 lock
```

ID의 `s`는 `simple_avg`, `w`는 `weighted_avg`다. `exp2_results.tsv`의 주요 열은 다음과 같다.

| 열 | 의미 |
|---|---|
| `BENCH` | `pope`, `gqa`, `textvqa`, `sqa` |
| `M2` | Stage-2 최종 visual token 수 |
| `METHOD` | cluster representative merge 방식 |
| `SELECT` | `attngain` |
| `DIVERSE` | `greedygain` |
| `METRIC`, `VALUE` | benchmark metric과 측정값 |
| `METRIC2`, `VALUE2` | 보조 metric. POPE latency job에서는 `AvgAcc` |
| `GEN` | 생성 문항 수/전체 문항 수 |
| `M1_MEAN/STD/MIN/MAX` | AGG가 자동 결정한 Stage-1 token 수 통계 |
| `INFER_SEC` | latency worker에서 기록한 inference wall-clock seconds |
| `LATENCY_SEC_PER_Q` | `INFER_SEC / 생성 문항 수` |

## 11. 공개 결과와 비교

재현이 끝나면 모든 행에서 `GEN`의 분자와 분모가 같아야 한다. 공개 결과와 key 기준으로 비교할 수 있다.

```bash
cd "$REPO_ROOT"
python - <<'PY'
import csv
import os

reference = "vispruner_md/exp2/exp2_results.tsv"
reproduced = os.path.join(os.environ["EXP2"], "exp2_results.tsv")

def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return {
        (r["ID"], r["BENCH"]): r
        for r in rows
        if r["SELECT"] == "attngain" and r["DIVERSE"] == "greedygain"
    }

ref, got = load(reference), load(reproduced)
for key in sorted(ref):
    if key not in got:
        print("MISSING", key)
        continue
    a, b = float(ref[key]["VALUE"]), float(got[key]["VALUE"])
    print(key, f"reference={a:.4f}", f"reproduced={b:.4f}", f"diff={b-a:+.4f}")
PY
```

동일 checkpoint, 질문 파일, 이미지, evaluator, package version을 사용하면 공개값과 같거나 부동소수점/환경 차이 범위에서 매우 가까워야 한다.

## 12. 중단 후 재개 및 재실행

answer JSONL이 남아 있으면 worker가 line 수를 확인하고 이어서 실행한다.

```bash
# 같은 명령을 다시 nohup으로 시작하면 미완료 answer부터 재개
```

특정 job을 처음부터 다시 실행하려면 해당 benchmark의 answer, stats, lock, 기존 결과 행을 함께 관리해야 한다. 가장 안전한 방법은 새 `EXP2` 디렉터리를 사용하고 `term_project/exp_runner/exp2_answers`, `exp2_logs`, `exp2_locks`를 별도 보관한 후 비우는 것이다. 기존 공개 TSV를 직접 수정하지 않는다.

## 13. 자주 발생하는 문제

### 다른 `llava` package가 import됨

증상: AGG 함수 또는 CLI option이 없다는 ImportError가 발생한다.

```bash
cd "$REPO_ROOT/term_project"
PYTHONPATH="$PWD" conda run -n vispruner python -c \
  "import llava; print(llava.__file__)"
```

출력 경로가 현재 clone의 `term_project/llava`여야 한다. `pip install -e .`를 다시 실행하고 오래된 `PYTHONPATH`를 제거한다. worker는 현재 `term_project`를 `PYTHONPATH` 맨 앞에 자동 추가한다.

### CLIP vision tower를 찾지 못함

`MODEL/config.json`의 `mm_vision_tower`가 다른 서버의 절대경로를 가리키는지 확인한다. Hugging Face ID 또는 현재 서버의 CLIP 디렉터리로 바꾼다.

### job이 즉시 건너뛰어짐

`term_project/exp_runner/exp2_locks/<ID>_<benchmark>/`가 이미 존재하면 중복 실행 방지를 위해 skip한다. 동일 job을 의도적으로 다시 실행할 때만 해당 lock과 연결된 answer/stats/result 행을 정리한다.

### GQA metric이 비어 있음

`GQA_EVAL_DIR/eval/eval.py`와 `GQA_Q_PATH`가 올바른지 확인한다. evaluator 출력에 `accuracy:` 행이 있어야 worker가 값을 읽을 수 있다.

### CUDA out of memory

다른 GPU process를 종료하거나 빈 GPU index를 사용한다. worker의 첫 번째 인자는 `CUDA_VISIBLE_DEVICES`에 지정할 물리 GPU 번호다.

### 결과가 공개값과 다름

다음 순서로 확인한다.

1. checkpoint가 `liuhaotian/llava-v1.5-7b`인지 확인한다.
2. 질문 파일 line 수와 split을 확인한다.
3. PyTorch/transformers 버전을 확인한다.
4. job TSV의 `M2`, merge method, `attngain`, `greedygain`을 확인한다.
5. evaluator와 annotation 버전을 확인한다.
6. `GEN`이 전체 문항 수와 같은지 확인한다.

## 14. 최소 단일 설정 실행

전체 24개 설정 전에 한 개만 확인하려면 임시 job TSV를 만든다.

```bash
printf '%s\n' \
  $'# ID\tPHASE\tBENCH\tM2\tMETHOD\tSELECT\tDIVERSE\tSTAGE1' \
  $'AGG-32s\trepro\tpope\t32\tsimple_avg\tattngain\tgreedygain\t576' \
  > /tmp/agg_smoke_job.tsv

cd "$REPO_ROOT/term_project"
nohup env MODEL="$MODEL" EXP2="$EXP2" \
  POPE_QF="$POPE_QF" POPE_IMG="$POPE_IMG" POPE_ANN="$POPE_ANN" \
  conda run -n vispruner --no-capture-output \
  bash exp_runner/workers/worker_exp2.sh 1 /tmp/agg_smoke_job.tsv \
  > "$EXP2/agg_32s_pope.nohup.log" 2>&1 &
```

이 검증도 POPE 전체 8,910문항을 실행한다. 짧은 기능 검증은 앞의 `exp2_selection_smoke.py`를 사용한다.

# 04. 실행 로그 (재현 절차 / 명령어 / 트러블슈팅)

모든 명령은 `Term_project/VisPruner_run` 기준, conda 환경 `vispruner` 활성화 상태.

## 0. 환경 활성화

```bash
source /home/jhlee/miniconda3/etc/profile.d/conda.sh
conda activate vispruner
cd /home/jhlee/CLUST_KETI/SKKU_Works/Y1_S1/Advanced_ML_DL/Term_project/VisPruner_run
export CUDA_VISIBLE_DEVICES=2
```

## 1. 코드 독립 사본 구성

```bash
rsync -a --exclude='.git' --exclude='assets' \
  Advanced_ML_DL/VisPruner/ Term_project/VisPruner_run/
mkdir -p Term_project/VisPruner_run/{models,playground/data/eval/pope}
```

## 2. conda 환경 + 의존성 (소요: pip ~수 분)

```bash
conda create -n vispruner python=3.10 -y
conda activate vispruner
pip install -e .           # torch2.1.2 / transformers4.37.2 등 설치 (exit 0)
```

## 3. 모델 다운로드 (소요: LLaVA ~3분, CLIP ~2분)

```bash
huggingface-cli download liuhaotian/llava-v1.5-7b \
  --local-dir models/llava-v1.5-7b --local-dir-use-symlinks False         # 13GB
huggingface-cli download openai/clip-vit-large-patch14-336 \
  --local-dir models/clip-vit-large-patch14-336 --local-dir-use-symlinks False \
  --exclude "*.h5" "*.msgpack" "flax*" "tf_*"                              # 1.6GB
```

`config.json` 의 vision tower를 로컬 경로로 패치 (독립 실행):
```bash
cp models/llava-v1.5-7b/config.json models/llava-v1.5-7b/config.json.orig
python -c "import json;p='models/llava-v1.5-7b/config.json';c=json.load(open(p));\
c['mm_vision_tower']='<...>/models/clip-vit-large-patch14-336';json.dump(c,open(p,'w'),indent=2)"
```

## 4. POPE 데이터 구성

```bash
# (1) POPE annotation 3종
BASE=https://raw.githubusercontent.com/AoiDragon/POPE/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco
for f in coco_pope_{adversarial,popular,random}.json; do
  curl -sSL -o playground/data/eval/pope/coco/$f $BASE/$f; done
# (2) llava_pope_test.jsonl 생성 (random→popular→adversarial 순, 전역 question_id, LLaVA 접미사)
#     → 총 8910 문항
# (3) 필요한 COCO val2014 500장만 개별 다운로드
cat /tmp/pope_imgs.txt | xargs -P16 -I{} sh -c \
  'curl -sS -o playground/data/eval/pope/val2014/{} http://images.cocodataset.org/val2014/{}'
```

## 5. Sanity check (300 subset)

```bash
python -m llava.eval.model_vqa_loader --model-path models/llava-v1.5-7b \
  --question-file playground/data/eval/pope/llava_pope_sanity.jsonl \
  --image-folder playground/data/eval/pope/val2014 \
  --answers-file .../sanity/n_576/r_0.5.jsonl \
  --visual_token_num 576 --important_ratio 0.5 --temperature 0 --conv-mode vicuna_v1
```

## 6. 트러블슈팅 기록

### [문제] POPE 정확도가 비정상적으로 낮음 (0.72, 모델이 거의 "No"만 답)
- 증상: 프루닝 없는 576 baseline 조차 sanity F1 0.61 / Acc 0.72 (논문 ~0.86).
- 1차 분리: 모델을 직접 호출(`Describe the image`)하면 **정상**으로 상세 묘사 →
  모델/가중치/이미지 파이프라인은 정상. POPE 단답 프롬프트에서만 "No" 편향.
- 2차 분리: 동일 이미지에 `"... single word or phrase."` 접미사 유무 비교 →
  접미사+greedy 일 때만 오답 "No", 접미사 없으면 정답 "Yes" (피처가 미세 손상된 정황).
- 3차 분리: 비전 타워 feature 추출을 `output_attentions=True/False` 로 직접 비교 시도 →
  **`CUDA error: an illegal memory access was encountered`** 발생(비동기 보고).
- `CUDA_LAUNCH_BLOCKING=1` 로 재실행 → 에러 없이 정상, 두 경로 피처 **diff=0** 확인.
- 원인 특정: `vt dtype=float32`, model/proj `float16`. `builder.py`가
  `device_map='auto'`(평가 기본)에서 비전 타워를 fp16으로 캐스팅하지 않아 dtype 혼용 →
  VisPruner attention/argsort/matmul 커널에서 비동기 메모리 접근 위반 → 피처 무음 손상.

### [해결] `llava/model/builder.py` 패치
```python
if device_map != 'auto':
    vision_tower.to(device=device_map, dtype=torch.float16)
else:
    vision_tower.to(dtype=next(model.parameters()).dtype)   # 추가
```
- 검증: sanity(300) Acc **0.717 → 0.837** 정상화, CUDA 에러 소멸.

### [문제] dtype 수정 후에도 전체 8910 실행 중 간헐 크래시
- 증상: n=576 전체 실행이 7701/8910에서 동일 CUDA illegal memory access로 크래시
  (dtype 수정으로 빈도 급감했으나 잔여 간헐 결함 존재).
- 해결(견고화):
  1. `CUDA_LAUNCH_BLOCKING=1` 안정모드 (검증: 피처 diff=0, 무에러).
  2. `model_vqa_loader`에 **resume** 추가 — 이미 답한 question_id 건너뛰고 append.
  3. 드라이버에 세팅별 **자동 재시도 루프**(8910 채울 때까지).
- 결과: 128/64/32 는 안정모드로 **각 1회 완주**(크래시 0). n=576 은 1차 크래시분이
  불안정모드 답변과 섞여 신뢰 불가 → 답변 삭제 후 **안정모드 전량 재생성**
  (F1 0.8024 → 0.8587 정상화).

### [기타]
- `do_sample=False` 인데 `temperature=0` 경고 → 무해(greedy decoding 정상).
- 디스크 99% → COCO 전체 zip 대신 필요 이미지 500장만 받아 회피.
- 백그라운드 실행은 하니스 추적 방식 사용(수동 nohup/setsid는 셸 종료 시 같이 죽음).
  `pkill -f <패턴>` 은 래퍼 셸 자신을 매칭해 종료시킬 수 있으니 지양.

## 7. 전체 재현 실행

```bash
bash run_pope_all.sh      # n ∈ {576,128,64,32}, r=0.5, 각 8910문항 추론+eval_pope
```
단일 세팅만:
```bash
bash scripts/v1_5/eval/pope_local.sh 128 0.5
```

## 8. 단계별 소요 시간

| 단계 | 소요 |
|------|------|
| conda 환경 + pip install -e . | ~수 분 |
| LLaVA-1.5-7B 다운로드 (13GB) | ~3분 |
| CLIP 다운로드 (1.6GB) | ~2분 |
| POPE 데이터 구성 (이미지 500장) | ~1분 |
| Sanity (300문항) 추론 | ~1–2분 |
| 본 실험 1세팅 (8910문항) 추론 | (03 문서에 기록) |

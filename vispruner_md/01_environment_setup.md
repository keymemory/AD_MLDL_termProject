# 01. 실험 환경 셋업

## 1. 하드웨어 / 시스템

| 항목 | 값 |
|------|-----|
| GPU | NVIDIA RTX A6000 49GB × 3 (실험은 `CUDA_VISIBLE_DEVICES=2` 단일 GPU 사용) |
| NVIDIA Driver | 535.183.01 |
| CUDA (driver 지원) | 12.2 |
| OS | Linux 5.14.0 (Ubuntu OEM) |
| 작업 루트 | `/home/jhlee/CLUST_KETI/SKKU_Works/Y1_S1/Advanced_ML_DL/Term_project` |

> 시스템 `nvcc`는 10.1로 매우 오래되었으나, conda 환경에 PyTorch가 자체 CUDA 12.1 런타임(휠)을 포함하므로 빌드/실행에 영향 없음.

## 2. Python / 패키지 환경

- conda 환경: **`vispruner`** (`python=3.10`) — `/home/jhlee/miniconda3/envs/vispruner`
- VisPruner `pyproject.toml`의 `pip install -e .` 로 설치 (원본 논문 코드 의존성 그대로)

| 패키지 | 버전 |
|--------|------|
| torch | 2.1.2+cu121 |
| torchvision | 0.16.2+cu121 |
| transformers | 4.37.2 |
| tokenizers | 0.15.1 |
| accelerate | 0.21.0 |
| sentencepiece | 0.1.99 |
| timm | 0.6.13 |
| numpy | 1.26.4 |
| scikit-learn | 1.2.2 |

설치 명령:
```bash
conda create -n vispruner python=3.10 -y
conda activate vispruner
cd Term_project/VisPruner_run
pip install -e .
```

## 3. 모델 다운로드

| 모델 | 출처 | 로컬 경로 | 용량 | 방법 |
|------|------|-----------|------|------|
| LLaVA-1.5-7B | `liuhaotian/llava-v1.5-7b` (HF) | `VisPruner_run/models/llava-v1.5-7b` | 13 GB | `huggingface-cli download` |
| CLIP ViT-L/14-336 | `openai/clip-vit-large-patch14-336` (HF) | `VisPruner_run/models/clip-vit-large-patch14-336` | 1.6 GB | `huggingface-cli download` |

- LLaVA-1.5-7B의 `config.json` 내 `mm_vision_tower`가 기본값 `openai/clip-vit-large-patch14-336`(HF 원격)로 되어 있어, **Term_project 단독 실행 보장을 위해 로컬 CLIP 경로로 패치**함.
  - `config.json.orig` 로 원본 백업 후 `mm_vision_tower` → 로컬 절대경로로 수정.
- 모델의 핵심 설정: `mm_vision_select_layer = -2`, `mm_vision_select_feature = patch`, `image_aspect_ratio = pad`.

## 4. 데이터셋 (POPE)

VisPruner 코드(`scripts/v1_5/eval/pope.sh`)는 LLaVA POPE 평가 포맷을 그대로 사용한다.

필요 파일:
- `playground/data/eval/pope/llava_pope_test.jsonl` — 질문 파일 (LLaVA 포맷)
- `playground/data/eval/pope/coco/coco_pope_{random,popular,adversarial}.json` — 정답 annotation
- `playground/data/eval/pope/val2014/` — COCO val2014 이미지

원본 EVAL.md는 Google Drive `eval.zip`을 요구하나, 독립 재구성을 위해:

1. **POPE annotation**: POPE 공식 repo(`AoiDragon/POPE`, commit `e3e3926`)의 `output/coco/` 3개 json을 raw 다운로드.
   - random 2910, popular 3000, adversarial 3000 = **총 8910 문항**
2. **`llava_pope_test.jsonl`**: 위 3개 json으로부터 직접 생성.
   - 전역 유일 `question_id`(1..8910) 부여, `text`에 LLaVA 표준 접미사 `" Answer the question using a single word or phrase."` 추가, `category` 필드 추가, `label` 보존.
   - 생성 순서 = `coco_pope_{random,popular,adversarial}` 파일 순서와 동일 (eval_pope.py의 순서 기반 정렬과 일치 보장).
3. **COCO val2014 이미지**: 전체 zip(6.2GB) 대신 POPE에서 실제 참조하는 **고유 500장만** `images.cocodataset.org`에서 개별 다운로드 (총 79MB, 디스크 절약).

## 5. 디스크 현황

- `/home` 파티션: 2.0T 중 가용 **약 28GB** (사용률 99%) — HF 캐시(~95GB)가 큰 비중.
- 전체 데이터 zip 대신 필요 이미지 500장만 받아 디스크 부담 최소화.

## 6. 최종 디렉토리 구조 (Term_project 단독 실행 가능)

```
Term_project/
├── develop_test.md
├── vispruner_md/                      # 결과 문서 (본 폴더)
└── VisPruner_run/                     # VisPruner 코드 사본 (.git/assets 제외) — 독립 실행
    ├── llava/                         # 핵심 패키지 (model, eval ...)
    ├── scripts/v1_5/eval/pope_local.sh   # 로컬 경로 전용 실행 스크립트(추가)
    ├── run_pope_all.sh                # 4개 세팅 일괄 실행 드라이버(추가)
    ├── models/
    │   ├── llava-v1.5-7b/             # LLM 체크포인트 (config는 로컬 CLIP 경로로 패치)
    │   └── clip-vit-large-patch14-336/   # 비전 인코더
    └── playground/data/eval/pope/
        ├── llava_pope_test.jsonl      # 8910 질문
        ├── coco/coco_pope_*.json      # 정답 annotation
        ├── val2014/                   # COCO 이미지 500장
        └── answers/                   # 추론 결과 출력
```

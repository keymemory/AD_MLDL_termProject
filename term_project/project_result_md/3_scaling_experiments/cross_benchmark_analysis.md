# Cross-Benchmark Analysis: M1 Scaling Law — 4벤치마크 비교

> 작성: 2026-05-26 | Exp B(POPE/GQA) + Exp C(TextVQA) + Exp D(ScienceQA) 통합
> 업데이트: 2026-05-29 | **Exp W-B/C/D (weighted_avg) 결과 추가** → Section 8

---

## 0. 벤치마크 데이터 특성

### 0.1 기본 통계

| 항목 | POPE | GQA | TextVQA | ScienceQA |
|---|---|---|---|---|
| **문항 수** | **8,910** | **12,578** | **5,000** | **4,241** |
| **고유 이미지 수** | 500 | 398 | 3,166 | 2,017 (+ 2,224 텍스트 전용) |
| **이미지당 평균 문항** | 17.8 | 31.6 | 1.58 | 2.10 (이미지 문항 기준) |
| **평가 split** | test (COCO val2014) | testdev_balanced | val (v0.5.1) | test |
| **이미지 출처** | COCO val2014 | GQA scene graph | Flickr / Open Images | 교과서·과학 자료 |

### 0.2 문항 유형 및 답변 형식

| 항목 | POPE | GQA | TextVQA | ScienceQA |
|---|---|---|---|---|
| **답변 형식** | Binary (Yes/No) | 개방형 단어/구 | 개방형 단어/구 | 다지선다 (2–5지) |
| **정답 레이블** | yes/no 1:1 균형 | 단일 정답 | 10명 어노테이터 | 단일 정답 (index) |
| **질문 구조** | "Is there a X?" | 공간·속성·비교 등 | "What does X say?" | 과학 개념·추론 |
| **선택지 수** | — | — | — | 2지(2,228) / 3지(971) / 4지(1,004) / 5지(38) |

### 0.3 세부 구성

**POPE (8,910문항 = 3 subset × ~3,000)**

| Subset | 문항 수 | Yes | No | 특징 |
|---|---|---|---|---|
| random | 2,910 | 1,500 | 1,410 | COCO 객체 무작위 샘플 |
| popular | 3,000 | 1,500 | 1,500 | 자주 등장하는 객체 (hard negative) |
| adversarial | 3,000 | 1,500 | 1,500 | 공존 가능성 높은 객체 (가장 어려운 subset) |

**GQA (12,578문항)**

| 질문 유형 | 문항 수 | 비율 |
|---|---|---|
| 이진 (Is/Are/Does 등) | 5,339 | 42.4% |
| 개방형 (What/How/Who 등) | 7,239 | 57.6% |

- 이미지 1장당 평균 31.6문항 → 동일 이미지에 다양한 compositional 질문

**TextVQA (5,000문항)**

| 항목 | 값 |
|---|---|
| 고유 이미지 | 3,166장 (이미지당 평균 1.58 질문) |
| OCR token 제공 | ✅ (llava_textvqa_val_v051_ocr 버전) |
| 정답 어노테이터 | 10명 (majority vote 평가) |
| 주요 도메인 | 간판·메뉴판·책표지·패키지 등 텍스트 포함 자연 이미지 |

**ScienceQA (4,241문항)**

| 과목 | 문항 수 | 비율 |
|---|---|---|
| Natural Science (자연과학) | 2,252 | 53.1% |
| Language Science (언어·독해) | 1,100 | 25.9% |
| Social Science (사회과학) | 889 | 21.0% |

| 이미지 유무 | 문항 수 | 비율 |
|---|---|---|
| 이미지 포함 | 2,017 | 47.6% |
| **텍스트 전용** | **2,224** | **52.4%** |

- 학년 범위: Grade 1–12 (초·중·고)
- 질문 유형: closed choice 4,090 / yes-or-no 113 / true-or-false 38

### 0.4 시각 의존도 × 과제 난이도 분류

| 벤치마크 | 시각 의존도 | 필요 능력 | 이미지 복잡도 |
|---|---|---|---|
| POPE | ★★★★☆ | 객체 존재 여부 | 자연 사진 (COCO, 다양한 객체) |
| GQA | ★★★★★ | 공간·속성·관계 추론 | 자연 사진 (밀도 높은 scene graph) |
| TextVQA | ★★★★★ | 픽셀 수준 문자 판독 | 텍스트 포함 이미지 (세밀한 디테일) |
| ScienceQA | ★★☆☆☆ | 과학 지식 + 다이어그램 이해 | 교과서 다이어그램·차트 (52%는 이미지 없음) |

---

## 0.5 데이터 출처 (원본 논문 + 다운로드 링크)

### 0.5.1 벤치마크별 원본 논문

| 벤치마크 | 논문 제목 | 저자 | 학회 / 연도 | arXiv | 공식 사이트 |
|---|---|---|---|---|---|
| **POPE** | Evaluating Object Hallucination in Large Vision-Language Models | Li et al. | EMNLP **2023** | [2305.10355](https://arxiv.org/abs/2305.10355) | [GitHub](https://github.com/RUCAIBox/POPE) |
| **GQA** | GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering | Hudson & Manning | CVPR **2019** | [1902.09506](https://arxiv.org/abs/1902.09506) | [cs.stanford.edu/~dorarad/gqa](https://cs.stanford.edu/people/dorarad/gqa/) |
| **TextVQA** | Towards VQA Models That Can Read | Singh et al. | CVPR **2019** | [1904.08920](https://arxiv.org/abs/1904.08920) | [textvqa.org](https://textvqa.org/) |
| **ScienceQA** | Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering | Lu et al. | NeurIPS **2022** | [2209.09513](https://arxiv.org/abs/2209.09513) | [scienceqa.github.io](https://scienceqa.github.io/) |

### 0.5.2 실제 다운로드 경로 (본 실험 기준)

#### POPE
| 구성 요소 | 다운로드 경로 | 비고 |
|---|---|---|
| **질문·레이블 파일** | [POPE GitHub (AoiDragon)](https://github.com/AoiDragon/POPE/tree/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco) | `coco_pope_{adversarial\|popular\|random}.json` |
| **이미지 (COCO val2014)** | [cocodataset.org](https://cocodataset.org/#download) → `2014 Val images` | [직접 링크](http://images.cocodataset.org/zips/val2014.zip) (6.2 GB) |
| **eval 패키지 (LLaVA 공용)** | [Google Drive (eval.zip)](https://drive.google.com/file/d/1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy/view?usp=sharing) | VisPruner/LLaVA 공통 eval 구조 포함 |
| **이미지 원본 논문** | Lin et al., ECCV 2014 ([arXiv 1405.0312](https://arxiv.org/abs/1405.0312)) | MS-COCO |

#### GQA
| 구성 요소 | 다운로드 경로 | 비고 |
|---|---|---|
| **질문·답변 파일** | [cs.stanford.edu/~dorarad/gqa/download.html](https://cs.stanford.edu/people/dorarad/gqa/download.html) | `testdev_balanced_questions.json` 사용 |
| **평가 스크립트** | [cs.stanford.edu/~dorarad/gqa/evaluate.html](https://cs.stanford.edu/people/dorarad/gqa/evaluate.html) | eval.py (v1.2 패치 필요) |
| **이미지 (Visual Genome)** | [visualgenome.org/api/v0/api_home.html](https://visualgenome.org/api/v0/api_home.html) | Krishna et al., IJCV 2017 ([arXiv 1602.07332](https://arxiv.org/abs/1602.07332)) |

#### TextVQA
| 구성 요소 | 다운로드 경로 | 비고 |
|---|---|---|
| **어노테이션 (v0.5.1)** | [dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json](https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json) | Facebook AI (Meta) 직접 호스팅 |
| **이미지** | [dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip) | Open Images v3 기반 |
| **HuggingFace (본 실험 사용)** | [huggingface.co/datasets/facebook/textvqa](https://huggingface.co/datasets/facebook/textvqa) | Arrow 포맷 로컬 캐시 → 이미지 추출 (`setup_textvqa.py`) |
| **OCR 보강 질문 파일** | `eval.zip` 내 `llava_textvqa_val_v051_ocr.jsonl` | VisPruner eval 패키지 포함 |

#### ScienceQA
| 구성 요소 | 다운로드 경로 | 비고 |
|---|---|---|
| **공식 원본** | [github.com/lupantech/ScienceQA](https://github.com/lupantech/ScienceQA) → `data/scienceqa/` | `images/`, `problems.json`, `pid_splits.json` |
| **HuggingFace (본 실험 사용)** | [huggingface.co/datasets/derek-thomas/ScienceQA](https://huggingface.co/datasets/derek-thomas/ScienceQA) | `split="test"` 로드 → 이미지·메타데이터 추출 (`setup_scienceqa.py`) |
| **LLaVA 질문 파일** | `eval.zip` 내 `llava_test_CQM-A.json` | VisPruner eval 패키지 포함 |

### 0.5.3 평가 도구 (eval 스크립트)

| 벤치마크 | 평가 스크립트 | 평가 기준 |
|---|---|---|
| **POPE** | `llava/eval/eval_pope.py` | F1 score (Yes/No binary, adversarial/popular/random 3-subset 평균) |
| **GQA** | `llava/eval/eval_gqa.py` | Exact match Accuracy |
| **TextVQA** | `llava/eval/eval_textvqa.py` | Soft accuracy (10명 어노테이터 중 ≥2명 일치) |
| **ScienceQA** | `llava/eval/eval_science_qa.py` | Accuracy (전체 4,241) + IMG-Accuracy (이미지 포함 2,017문항만) |

### 0.5.4 본 실험 로컬 경로

| 벤치마크 | Split | 로컬 경로 | 문항 수 |
|---|---|---|---|
| POPE | test (COCO val2014) | `VisPruner/playground/data/eval/pope/coco/*.json` | 8,910 |
| GQA | testdev_balanced | `VisPruner/playground/data/eval/gqa/data/` | 12,578 |
| TextVQA | val v0.5.1 (OCR augmented) | `/data1/heejung/datasets/textvqa_val_images/` | 5,000 |
| ScienceQA | test | `/data1/heejung/datasets/scienceqa/images/test/{pid}/image.png` | 4,241 |

---

## 0.6 Two-Stage K-Means 프로세스: 직관적 이해

### 0.6.1 왜 Two-Stage인가? — 한 문장 요약

> **"LLM이 중요하다고 본 M1개 시각 토큰을 먼저 고른 뒤(Stage 1), 그 중에서 비슷한 것끼리 묶어 M2개로 압축한다(Stage 2)."**

### 0.6.2 단계별 직관적 설명

```
[원본 이미지]
    ↓  ViT(CLIP) 패치 인코딩
[576개 visual token]  ← 이미지를 24×24 격자로 나눈 것 (각 패치 = 1 토큰)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Stage 1: Attention Pruning (주목도 필터링)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  LLM(Vicuna)의 CLS 토큰 어텐션 → 각 시각 토큰의 "중요도" 점수 계산
  + 공간적 다양성 보정 (너무 한 곳에 몰리지 않게)
  → 상위 M1개 토큰 선택

  비유: 시험 공부할 때 교과서 576페이지 중
        "출제자(LLM)가 자주 본 M1페이지"만 추리는 작업

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Stage 2: Spherical K-Means Merging (의미 기반 압축)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  M1개 토큰을 K=M2 클러스터로 그룹화 (K-Means)
  각 클러스터의 평균(centroid)이 최종 시각 토큰 1개
  → 최종 M2개 토큰만 LLM에 입력

  비유: 추린 M1페이지를 M2개 "테마(클러스터)"로 요약 정리
        비슷한 내용끼리 묶어 대표 요약본 1개씩 만들기
```

### 0.6.3 M1 Scaling의 의미

| M1 값 | 효과 | 설명 |
|---|---|---|
| **M1 = M2** | Stage 1 = identity | K-Means 입력이 이미 M2개 → 사실상 K-Means no-op |
| **M1 < 최고점** | 품질 상승 | 더 많은 후보 → K-Means centroid가 이미지 더 잘 표현 |
| **M1 = 최적점** | 최적 (= peak) | 충분히 많되 이질적 토큰 유입 최소 — 성능 곡선의 꼭짓점 |
| **M1 > 최적점** | 품질 하락 | 배경·노이즈 토큰 유입 → centroid 품질 저하 |
| **M1 = 576 (전체)** | Stage 1 없음 | 무작위 초기화와 유사 → 최저 성능 |

### 0.6.4 Task별 효과 차이의 직관

```
[객체 인식 / 관계 추론 (POPE, GQA)]
  → 시각 정보의 "의미(semantic)"가 중요
  → K-Means 평균화: 비슷한 의미 토큰 집합 = 유효한 semantic 압축
  → Two-Stage 유익 ✅

[OCR / 텍스트 인식 (TextVQA)]
  → 픽셀 수준의 정밀한 공간 정보가 중요
  → K-Means 평균화: 인접 토큰 평균 = 공간 디테일 희석
  → Two-Stage 유해 ❌

[과학 지식 추론 (ScienceQA)]
  → 문항의 52%는 이미지가 없어 시각 토큰 무관
  → 이미지 포함 문항에서만 미미한 이득
  → Two-Stage 중립 ➖
```

---

## 1. 4벤치마크 M1 Scaling Curve 요약

*ScienceQA는 텍스트 전용 문항(52.4%) 포함 → **SQA Δ Acc**: 전체 4,241문항 / **SQA Δ IMG-Acc**: 이미지 포함 2,017문항만*

### M2=64 시리즈

*각 벤치마크의 Stage1-only (M2=64) 대비 Δ*
*(절대값 기준: POPE F1=0.8095, GQA=55.59%, TextVQA=55.73%, SQA Acc=69.91%, SQA IMG-Acc=68.96%)*

| M1 | M1:M2 | POPE Δ F1 | GQA Δ Acc | TextVQA Δ Acc | SQA Δ Acc (전체) | SQA Δ IMG-Acc |
|---:|---:|---|---|---|---|---|
| 64 | 1.0× | 0.000 | 0.00% | 0.00% | 0.00% | 0.00% |
| **96** | **1.5×** | +0.007 | +0.48% | **−1.84%** | **+0.24%** | **+0.50% ★** |
| 128 | 2.0× | +0.013 | +1.07% | −1.70% | −0.02% | −0.05% |
| 192 | 3.0× | +0.021 | +1.45% | −2.86% | −0.12% | −0.24% |
| **256** | **4.0×** | **+0.027 ★** | **+2.01% ★** | −3.14% | −0.09% | −0.19% |
| 384 | 6.0× | +0.022 | +1.67% | −4.33% | −0.02% | −0.05% |
| 576 | 9.0× | −0.001 | +1.05% | −5.48% | −0.21% | −0.44% |

### M2=32 시리즈

*각 벤치마크의 Stage1-only (M2=32) 대비 Δ*
*(절대값 기준: POPE F1=0.7400, GQA=51.58%, TextVQA=53.83%, SQA Acc=69.61%, SQA IMG-Acc=68.32%)*

| M1 | M1:M2 | POPE Δ F1 | GQA Δ Acc | TextVQA Δ Acc | SQA Δ Acc (전체) | SQA Δ IMG-Acc |
|---:|---:|---|---|---|---|---|
| 32 | 1.0× | 0.000 | 0.00% | 0.00% | 0.00% | 0.00% |
| 48 | 1.5× | +0.012 | +1.53% | −0.55% | +0.51% | +1.09% |
| **64** | **2.0×** | +0.036 | +1.94% | −0.98% | **+0.66%** | **+1.39% ★** |
| 96 | 3.0× | +0.042 | +3.00% | −1.08% | +0.18% | +0.40% |
| 128 | 4.0× | +0.047 | +3.29% | −1.62% | +0.49% | +1.04% |
| 192 | 6.0× | +0.060 | +3.66% | −2.67% | 0.00% | 0.00% |
| 256 | 8.0× | +0.057 | **+4.18% ★** | −3.30% | +0.11% | +0.25% |
| 288 | 9.0× | +0.060 | +3.91% | −3.42% | +0.23% | +0.50% |
| **384** | **12.0×** | **+0.063 ★** | +4.06% | −4.69% | +0.04% | +0.10% |
| 576 | 18.0× | +0.022 | +2.36% | −6.05% | −0.05% | −0.10% |

---

## 2. 벤치마크별 특성 × Scaling 패턴

| 벤치마크 | 문항 수 | 과제 유형 | 시각 의존도 | Scaling 패턴 | M2=64 최대 Δ | M2=32 최대 Δ |
|---|---|---|---|---|---|---|
| **POPE** | 8,910 | 객체 환각 탐지 (Yes/No) | 높음 (전체 이미지) | 유니모달, 최적점 M1:M2=4× | +0.027 F1 | +0.063 F1 |
| **GQA** | 12,578 | 구성적 시각 추론 | 높음 (공간·관계) | 유니모달, 최적점 M1:M2=4× | +2.01% | +4.18% |
| **TextVQA** | 5,000 | OCR·텍스트 인식 | 매우 높음 (픽셀 정밀도) | **단조 감소** | **−1.84%** | **−0.55%** |
| **ScienceQA** | 4,241 (이미지 2,017) | 과학 지식 추론 | 낮음 (52% 텍스트 전용) | 약한 유니모달 | +0.50%† | +1.39%† |

*† IMG-Acc 기준 (이미지 포함 2,017문항만). 전체 Acc: M2=64 +0.24%, M2=32 +0.66%*

---

## 3. 핵심 발견 및 해석

### 3.1 Two-Stage K-Means의 Task-Type Dependency

```
[시각 의존도: 높음 + 의미적 이해]  → K-Means 유효 (POPE, GQA)
  단조증가 → 최고점(최적점) → 완만한 하락
  "semantic aggregation이 object·relation 이해에 기여"

[시각 의존도: 매우 높음 + 픽셀 정밀도]  → K-Means 유해 (TextVQA)
  즉시 단조 감소
  "feature 평균화가 문자 판독에 필요한 공간 정밀도를 희석"

[시각 의존도: 낮음 + 혼합 모달리티]  → K-Means 중립 (ScienceQA)
  거의 평탄, 미미한 양/음 변동
  "텍스트 전용 문항(52%)이 시각 효과를 희석"
```

### 3.2 최적점(Sweet Spot) M1:M2 비율의 Task 의존성

> **최적점(sweet spot)** = 성능이 가장 높은 M1:M2 비율. 이 비율보다 M1이 크거나 작으면 성능이 떨어짐.

| 패턴 | Task 유형 | M2=64 최적점 (최고 M1:M2) | M2=32 최적점 (최고 M1:M2) |
|---|---|---|---|
| 유니모달 이득 | 의미 이해 (POPE, GQA) | **4×** | **8–12×** |
| 단조 감소 | 픽셀 정밀도 (TextVQA) | 없음 (M1=M2 그대로가 최고) | 없음 |
| 거의 평탄 | 혼합 모달 (ScienceQA) | 1.5× (+0.24%) | 2× (+0.66%) |

> **M2가 작을수록 최적점 비율이 올라가는 현상은 의미 이해 task에서만 관찰됨**

### 3.3 Stage1 vs K-Means의 기여 분리

| 벤치마크 | Stage1 only | KMeans only | Best Two-Stage | Stage1 기여 | K-Means 기여 |
|---|---|---|---|---|---|
| POPE (M2=64) | 0.8047 | 0.8083 | **0.8364** | seed 품질 결정 | +0.028 |
| GQA (M2=64) | 56.03% | 56.64% | **57.60%** | seed 품질 결정 | +1.57% |
| TextVQA (M2=64) | **55.73%** | 50.25% | 55.73% | OCR 최적 | K-Means 불필요 |
| ScienceQA (M2=64) | 69.91% | 69.70% | **70.15%** | 미미 | +0.24% |

- **POPE/GQA**: Stage1이 K-Means seed 품질 결정 → 두 단계 시너지
- **TextVQA**: Stage1 attention pruning이 이미 최적 → K-Means는 정보 손실
- **ScienceQA**: 두 방법 모두 효과 미미 (텍스트 문항 완충)

### 3.4 토큰 효율성 관점

| 구성 | 토큰 수 | GQA | TextVQA | 비고 |
|---|---|---|---|---|
| Stage1-only (M2=64) | 64 | 56.03% | 55.73% | 기준 |
| **Two-Stage M32-256** | **32** | **55.76%** ✅ | 50.53% | GQA: 절반 토큰으로 64토큰 수준 |
| Stage1-only (M2=32) | 32 | 52.04% | 53.83% | 단순 절반 비교 |

> GQA에서 32토큰 Two-Stage가 64토큰 Stage1-only를 초과하지만, TextVQA에서는 반대

---

## 4. 논문 Figure 제안

### Figure A: 4-벤치마크 M1 Scaling Curve (M2=64)
- x축: M1:M2 비율 (1×~9×)
- y축: 각 벤치마크의 normalized Δ (baseline=0%)
- 4개 curve가 선명하게 갈라지는 시각화
- 핵심 메시지: POPE/GQA (↑), TextVQA (↓), ScienceQA (→)

### Figure B: 최적점(Sweet Spot) Heat Map
- x축: M2 (32, 64), y축: 벤치마크
- 색상: Δ Acc (파란색=이득, 빨간색=손실)
- 각 셀에서 가장 높은 성능을 내는 M1:M2 비율 = 최적점

---

## 5. 논문 Conclusion 핵심 클레임

1. **M1 Scaling Law는 task-type에 따라 근본적으로 다름**
   - Semantic understanding: unimodal (M2의 4–12×에서 최고점)
   - Fine-grained visual (OCR): monotone decrease
   - Mixed-modal (knowledge): near-flat

2. **K-Means merging의 조건부 효과**
   - 의미 집합화 유효 → POPE, GQA
   - 공간 정밀도 요구 → 해로움 (TextVQA)
   - 혼합 구성 → 중립 (ScienceQA)

3. **토큰 효율성 실용적 결론**
   - 의미 이해 task: Two-Stage M2=32 (256 pool) ≈ Stage1-only M2=64
   - OCR task: Two-Stage 미적용, Stage1만 사용 권장
   - 혼합 task: 단순 Stage1-only로도 충분

---

## 6. POPE Subset별 상세 분석

> POPE는 세 subset (adversarial/popular/random)으로 구성됨.
> **adversarial**이 가장 어렵고 (공존 가능한 객체 질문), **random**이 가장 쉬움.
> K-Means Two-Stage의 효과가 subset별로 다르게 나타남.

### 6.1 M2=64 시리즈 — Subset별 F1

*baseline: A-64 (Stage1-only, M2=64)*

| M1 | M1:M2 | Adversarial | Popular | Random | **Avg** |
|---:|---:|---|---|---|---|
| 64 (base) | 1.0× | 0.7876 | 0.8165 | 0.8209 | 0.8095 |
| 96 | 1.5× | 0.7972 | 0.8293 | 0.8222 | 0.8162 |
| 128 | 2.0× | 0.7977 | 0.8329 | 0.8376 | 0.8227 |
| 192 | 3.0× | 0.8108 | 0.8367 | 0.8433 | 0.8303 |
| **256** | **4.0×** | **0.8159** | **0.8406** | **0.8527 ★** | **0.8364 ★** |
| 384 | 6.0× | **0.8208 ★** | 0.8328 | 0.8401 | 0.8313 |
| 576 | 9.0× | 0.7927 | 0.8137 | 0.8186 | 0.8083 |

*★ 각 열 내 최댓값 (Avg 최적점 = M1=256; Adversarial 최적점 = M1=384)*

### 6.2 M2=32 시리즈 — Subset별 F1

*baseline: A-32 (Stage1-only, M2=32)*

| M1 | M1:M2 | Adversarial | Popular | Random | **Avg** |
|---:|---:|---|---|---|---|
| 32 (base) | 1.0× | 0.7248 | 0.7463 | 0.7488 | 0.7400 |
| 48 | 1.5× | 0.7374 | 0.7597 | 0.7574 | 0.7515 |
| 64 | 2.0× | 0.7638 | 0.7807 | 0.7824 | 0.7756 |
| 96 | 3.0× | 0.7617 | 0.7912 | 0.7934 | 0.7821 |
| 128 | 4.0× | 0.7689 | 0.7903 | 0.8011 | 0.7867 |
| 192 | 6.0× | 0.7817 | 0.8073 | 0.8106 | 0.7998 |
| 256 | 8.0× | 0.7819 | 0.8009 | 0.8091 | 0.7973 |
| 288 | 9.0× | 0.7796 | 0.8088 | 0.8113 | 0.7999 |
| **384** | **12.0×** | **0.7869 ★** | **0.8097 ★** | **0.8117 ★** | **0.8027 ★** |
| 576 | 18.0× | 0.7447 | 0.7719 | 0.7698 | 0.7621 |

### 6.3 Subset별 핵심 인사이트

#### Adversarial이 항상 가장 어렵다
| 구성 | Adversarial | Popular | Random | 격차 (Adv vs Rnd) |
|---|---|---|---|---|
| M2=64 baseline | 0.7876 | 0.8165 | 0.8209 | −0.033 |
| M2=64 최적점 (M1=256) | 0.8159 | 0.8406 | 0.8527 | −0.037 |
| M2=32 baseline | 0.7248 | 0.7463 | 0.7488 | −0.024 |
| M2=32 최적점 (M1=384) | 0.7869 | 0.8097 | 0.8117 | −0.025 |

- Adversarial은 최고점에서도 Random 대비 −0.035 이상 낮음
- Two-Stage를 통해 모든 subset이 고르게 개선되나, Adversarial 개선폭이 큼

#### Adversarial 이득이 상대적으로 큼 (M2=64)
| M1 | Δ Adv | Δ Pop | Δ Rnd |
|---|---|---|---|
| 96 (1.5×) | +0.010 | +0.013 | +0.001 |
| 128 (2.0×) | +0.010 | +0.016 | +0.017 |
| 192 (3.0×) | +0.023 | +0.020 | +0.022 |
| 256 (4.0×) | +0.028 | +0.024 | +0.032 |
| 384 (6.0×) | **+0.033** | +0.016 | +0.019 |
| 576 (9.0×) | +0.005 | −0.003 | −0.002 |

- M1=384에서 Adversarial 이득(+0.033)이 가장 큼
- 즉, Two-Stage는 "공존 가능한 객체로 인한 환각(adversarial hallucination)"에 특히 효과적
- K-Means semantic aggregation이 adversarial 오류를 줄이는 메커니즘 시사

#### Adversarial 최적점 M1이 다르다 (M2=64)
- Adversarial: 최적점 M1=**384** (6×)
- Popular / Random / Avg: 최적점 M1=**256** (4×)
- → Adversarial 환각 억제에는 더 많은 pool 토큰이 유익함

---

## 8. simple_avg vs weighted_avg 병합 방법 비교

> **Exp W-B (POPE/GQA) + W-C (TextVQA) + W-D (ScienceQA)** — 2026-05-28~29 실행
> 동일한 M1·M2 조합에서 클러스터 병합 방법만 변경: **simple_avg** (단순 평균) vs **weighted_avg** (CLS 어텐션 점수 가중 평균)

### 8.1 실험 구성

| 항목 | simple_avg | weighted_avg |
|---|---|---|
| 병합 공식 | centroid = mean(tokens in cluster) | centroid = Σ(attn_score × token) / Σ(attn_score) |
| 가중치 기준 | 없음 | Stage 1에서 사용한 CLS 어텐션 점수 |
| 직관 | 클러스터 내 동등 기여 | 중요한 토큰에 더 높은 가중치 |

*비교 가능한 M1 범위: M2=64 → M1=96~576 (POPE simple_avg는 M1=256/384/576만 존재), M2=32 → M1=48~576*

---

### 8.2 TextVQA: weighted_avg 뚜렷한 우세

*baseline (no pruning): M2=64 → **55.73%**, M2=32 → **53.83%***

| M1 | M2=64 simple | M2=64 **weighted** | **Δ** | M2=32 simple | M2=32 **weighted** | **Δ** |
|---:|---|---|---|---|---|---|
| 48 | — | — | — | 53.28 | **53.88** | **+0.60** |
| 64 | — | — | — | 52.85 | **53.30** | +0.45 |
| 96 | 53.89 | **55.05** | **+1.16** | 52.75 | **53.07** | +0.32 |
| 128 | 54.03 | **54.82** | **+0.79** | 52.21 | **52.93** | +0.72 |
| 192 | 52.87 | **53.92** | **+1.05** | 51.16 | **52.30** | **+1.14** |
| 256 | 52.59 | **53.97** | **+1.38** | 50.53 | **52.13** | **+1.60** |
| 288 | — | — | — | 50.41 | **51.91** | **+1.50** |
| 384 | 51.40 | **52.96** | **+1.56** | 49.14 | **51.46** | **+2.32** |
| 576 | 50.25 | **52.31** | **+2.06 ★** | 47.78 | **50.12** | **+2.34 ★** |

> **weighted_avg가 모든 M1에서 일관되게 우세. M1이 클수록 이점 확대.**
> M2=64, M1=576: simple 50.25% → weighted 52.31% **(+2.06%p)**
> M2=32, M1=576: simple 47.78% → weighted 50.12% **(+2.34%p)**

---

### 8.3 POPE: 대형 M1에서 weighted_avg 소폭 우세

*baseline: M2=64 → **0.8095**, M2=32 → **0.7400***

| M1 | M2=64 simple | M2=64 **weighted** | **Δ** | M2=32 simple | M2=32 **weighted** | **Δ** |
|---:|---|---|---|---|---|---|
| 48 | — | — | — | 0.7515 | 0.7477 | −0.0038 |
| 96 | — | **0.8137** | — | 0.7821 | 0.7818 | −0.0003 |
| 128 | — | **0.8260** | — | 0.7867 | **0.7914** | +0.0047 |
| 192 | — | **0.8350** | — | 0.7998 | **0.8045** | +0.0047 |
| 256 | 0.8364 | **0.8365** | +0.0001 | 0.7973 | **0.8065** | +0.0092 |
| 288 | — | — | — | 0.7999 | **0.8077** | +0.0078 |
| 384 | 0.8313 | **0.8381** | **+0.0068** | 0.8027 | **0.8061** | +0.0034 |
| 576 | 0.8083 | **0.8217** | **+0.0134 ★** | 0.7621 | **0.7742** | **+0.0121 ★** |

> M1이 클수록 weighted 이점 확대 (M1=576에서 최대 +0.013).
> M1이 작을 때(M2=32, M1=48)는 weighted_avg가 소폭 불리.

---

### 8.4 GQA: 방법 간 차이 미미

*baseline: M2=64 → **55.59%**, M2=32 → **51.58%***

| M1 | M2=64 simple | M2=64 **weighted** | **Δ** | M2=32 simple | M2=32 **weighted** | **Δ** |
|---:|---|---|---|---|---|---|
| 48 | — | — | — | 53.11 | **53.23** | +0.12 |
| 96 | 56.07 | 56.07 | 0.00 | 54.58 | **54.64** | +0.06 |
| 128 | — | **56.92** | — | 54.87 | **55.14** | +0.27 |
| 192 | 57.04 | **57.20** | +0.16 | **55.24** | 55.13 | −0.11 |
| 256 | **57.60** | 57.54 | −0.06 | **55.76** | 55.49 | −0.27 |
| 288 | — | — | — | 55.49 | **55.79** | +0.30 |
| 384 | 57.26 | **57.44** | +0.18 | **55.64** | 55.48 | −0.16 |
| 576 | 56.64 | **56.76** | +0.12 | 53.94 | **54.17** | +0.23 |

> 차이가 ±0.3%p 이내로 혼재 → 두 방법 간 유의미한 차이 없음.

---

### 8.5 ScienceQA: 실질적 차이 없음

*baseline: M2=64 → **69.91% / IMG 68.96%**, M2=32 → **69.61% / IMG 68.32%***

| M1 | M2=64 simple | M2=64 **weighted** | **Δ** | M2=32 simple | M2=32 **weighted** | **Δ** |
|---:|---|---|---|---|---|---|
| 48 | — | — | — | 70.12 | 69.89 | −0.23 |
| 64 | — | — | — | **70.27** | **70.43** | **+0.16** |
| 96 | 70.15 | 69.89 | −0.26 | 69.79 | 69.94 | +0.15 |
| 128 | 69.89 | **70.03** | +0.14 | 70.10 | **70.15** | +0.05 |
| 192 | 69.79 | **69.91** | +0.12 | 69.61 | **69.96** | +0.35 |
| 256 | 69.82 | 69.77 | −0.05 | 69.72 | **70.03** | +0.31 |
| 288 | — | — | — | **69.84** | 69.75 | −0.09 |
| 384 | **69.89** | 69.87 | −0.02 | 69.65 | 69.72 | +0.07 |
| 576 | 69.70 | 69.68 | −0.02 | 69.56 | 69.51 | −0.05 |

> 전체적으로 ±0.35%p 이내의 노이즈 수준 차이. 텍스트 전용 문항(52.4%)이 효과를 희석.

---

### 8.6 핵심 관찰 및 해석

| 벤치마크 | weighted_avg 효과 | 평균 Δ (M2=64) | 평균 Δ (M2=32) | 해석 |
|---|---|---|---|---|
| **TextVQA** | **뚜렷한 우세** | **+1.3%p** | **+1.3%p** | 어텐션 기반 가중치가 텍스트 픽셀 정보를 집중 보존 |
| **POPE** | 소폭 우세 | +0.007 | +0.005 | 중요 시각 토큰 강조 → 환각 억제 미미한 기여 |
| **GQA** | 거의 동일 | ±0.08%p | ±0.08%p | 공간·관계 추론에서 두 방법 동등 |
| **ScienceQA** | 차이 없음 | ±0.07%p | ±0.09%p | 텍스트 전용 문항 완충 + 다이어그램의 낮은 토큰 민감도 |

**핵심 결론:**

> **weighted_avg의 이점은 task의 "공간 정밀도 요구도"에 비례한다.**
>
> - **TextVQA** (픽셀 수준 문자 판독): 클러스터 내 CLS 주목 토큰을 더 많이 반영 → 단순 평균으로 희석되던 텍스트 특징 보존 → 큰 이득
> - **POPE** (객체 존재 여부): 중요 영역 집중 약간 유리 → 작은 이득
> - **GQA / ScienceQA**: 의미적 집합화 자체가 이미 충분 → 가중 여부 무관

> **M1이 클수록(공격적 압축일수록) weighted_avg 이점이 커지는 이유:**
> M1이 크면 클러스터당 더 많은 토큰이 병합됨 → 단순 평균에서 정보 손실 증가 → 어텐션 가중치의 선택적 보존 효과 더 부각

---

## 7. 참고 문헌 (BibTeX)

```bibtex
@inproceedings{li2023pope,
  title     = {Evaluating Object Hallucination in Large Vision-Language Models},
  author    = {Li, Yifan and Du, Yifan and Zhou, Kun and Wang, Jinpeng and Zhao, Wayne Xin and Wen, Ji-Rong},
  booktitle = {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2023},
  url       = {https://arxiv.org/abs/2305.10355}
}

@inproceedings{hudson2019gqa,
  title     = {{GQA}: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering},
  author    = {Hudson, Drew A. and Manning, Christopher D.},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2019},
  url       = {https://arxiv.org/abs/1902.09506}
}

@inproceedings{singh2019textvqa,
  title     = {Towards {VQA} Models That Can Read},
  author    = {Singh, Amanpreet and Natarajan, Vivek and Shah, Meet and Jiang, Yu and Chen, Xinlei and Batra, Dhruv and Parikh, Devi and Rohrbach, Marcus},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2019},
  url       = {https://arxiv.org/abs/1904.08920}
}

@inproceedings{lu2022scienceqa,
  title     = {Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering},
  author    = {Lu, Pan and Mishra, Swaroop and Xia, Tanglin and Qiu, Liang and Chang, Kai-Wei and Zhu, Song-Chun and Tafjord, Oyvind and Clark, Peter and Kalyan, Ashwin},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2022},
  url       = {https://arxiv.org/abs/2209.09513}
}

@inproceedings{lin2014coco,
  title     = {Microsoft {COCO}: Common Objects in Context},
  author    = {Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C. Lawrence},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2014},
  url       = {https://arxiv.org/abs/1405.0312}
}

@article{krishna2017visualgenome,
  title   = {Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations},
  author  = {Krishna, Ranjay and Zhu, Yuke and Groth, Oliver and Johnson, Justin and Hata, Kenji and Kravitz, Joshua and Chen, Stephanie and Kalantidis, Yannis and Li, Li-Jia and Shamma, David A and Bernstein, Michael S and Li, Fei-Fei},
  journal = {International Journal of Computer Vision (IJCV)},
  year    = {2017},
  url     = {https://arxiv.org/abs/1602.07332}
}

@article{zhang2025vispruner,
  title   = {Beyond Text-Visual Attention: Exploiting Visual Cues for Effective Token Pruning in {VLMs}},
  author  = {Zhang, Qizhe and Cheng, Aosong and Lu, Ming and Zhang, Renrui and Zhuo, Zhiyong and Cao, Jiajun and Guo, Shaobo and She, Qi and Zhang, Shanghang},
  journal = {arXiv preprint arXiv:2412.01818},
  note    = {ICCV 2025},
  year    = {2025},
  url     = {https://arxiv.org/abs/2412.01818}
}
```

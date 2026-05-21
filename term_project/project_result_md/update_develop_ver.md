# [실험 프롬프트 v2] Two-Stage Framework 체계적 실험

## 목표
구현 완료된 Two-Stage Framework(experiments/term_project/)으로
제안 논문 Table 1에 명시된 **전체 실험 조건**을 수행한다.

## 환경
- 구현 코드: `experiments/term_project/`
- 데이터셋: `experiments/dataset/` (pope, gqa, textvqa, vqav2 등 모두 준비됨)
- 모델: LLaVA-1.5-7B (로컬)
- GPU: RTX A6000 (48GB), 디스크 ~500GB 여유
- 추론: fp16, greedy decoding, temperature=0, CUDA_LAUNCH_BLOCKING=1
- 참고 코드: `experiments/PACT/` (PACT baseline용), `experiments/VisPruner/` (원본)

---

## 논문 Table 1 실험 조건 (전체 충족 필수)

```
Backbone VLM        : LLaVA-1.5-7B
Pruning Baselines   : FastV (ECCV'24)           ← 직접 구현+실행
Merging Baselines   : ToMe (ICLR'23)            ← 논문 숫자 인용
Pruning+Merging     : SparseVLM (ICML'25)       ← 논문 숫자 인용
                      PACT (CVPR'25)            ← 직접 실행 (코드 있음)
Base Method         : VisPruner (ICCV'25)       ← 직접 실행
Retained Tokens     : 128, 64, 32
Benchmarks          : VQAv2, GQA, POPE          ← 3개 필수
                      + TextVQA (추가)
Metric              : Accuracy (%)
Efficiency Metrics  : Token reduction ratio, Inference latency
```

---

## 실험 1: 제안 방법 비교 (핵심)

3가지 변형을 비교한다:
- **(A) VisPruner only**: enable_clustering=False
- **(B) Ours simple**: enable_clustering=True, merge_method=simple_avg
- **(C) Ours weighted**: enable_clustering=True, merge_method=weighted_avg

### 실행 목록

| ID | Method | enable_clustering | stage1(M1) | final(M2) | merge_method | 벤치마크 |
|----|--------|:-:|--:|--:|---|---|
| A-128 | VisPruner only | OFF | - | 128 | - | POPE, GQA, VQAv2, TextVQA |
| A-64 | VisPruner only | OFF | - | 64 | - | POPE, GQA, VQAv2, TextVQA |
| A-32 | VisPruner only | OFF | - | 32 | - | POPE, GQA, VQAv2, TextVQA |
| B-128 | Ours simple | ON | 192 | 128 | simple_avg | POPE, GQA, VQAv2, TextVQA |
| B-64 | Ours simple | ON | 128 | 64 | simple_avg | POPE, GQA, VQAv2, TextVQA |
| B-32 | Ours simple | ON | 64 | 32 | simple_avg | POPE, GQA, VQAv2, TextVQA |
| C-128 | Ours weighted | ON | 192 | 128 | weighted_avg | POPE, GQA, VQAv2, TextVQA |
| C-64 | Ours weighted | ON | 128 | 64 | weighted_avg | POPE, GQA, VQAv2, TextVQA |
| C-32 | Ours weighted | ON | 64 | 32 | weighted_avg | POPE, GQA, VQAv2, TextVQA |

r = 0.5 공통. 총 9세팅 × 4벤치마크 = 36회.
실행 순서: POPE(빠름) → GQA → TextVQA → VQAv2(무거움)

### VQAv2 실행 관련
- VQAv2 test-dev(107K)는 EvalAI 채점 필요 → **val set을 사용하여 로컬 채점**
- val set이면 question type(yes/no, number, other)별 분석도 가능
- `experiments/dataset/vqav2/`에 데이터 확인 후 val set 기반으로 세팅
- val set이 없으면 다운로드: https://visualqa.org/download.html
- 로컬 채점 스크립트: LLaVA eval 방식 또는 VQAv2 공식 eval 코드 사용

---

## 실험 2: Baseline 직접 실행

### 2-A: FastV 직접 구현 및 실행

FastV는 구현이 간단하다:
- LLM의 **2번째 레이어 이후** text-visual attention 기반으로 visual token pruning
- 낮은 attention score를 받는 visual token을 제거

구현 방법:
1. `experiments/term_project/` 내에 FastV 로직 구현
2. LLaVA의 language model forward에서 layer 2의 attention을 추출
3. visual token 중 attention score 하위를 제거하여 지정 토큰 수만 보존
4. 128, 64, 32 토큰으로 POPE, GQA, VQAv2에서 실행

| ID | Method | Tokens | 벤치마크 |
|----|--------|--------|---------|
| F-128 | FastV | 128 | POPE, GQA, VQAv2 |
| F-64 | FastV | 64 | POPE, GQA, VQAv2 |
| F-32 | FastV | 32 | POPE, GQA, VQAv2 |

FastV 구현이 어려우면 VisPruner 논문 Table 1 숫자를 인용하고 사유 기록.

### 2-B: PACT 직접 실행

`experiments/PACT/` 코드를 사용하여 LLaVA-1.5-7B에서 실행.

1. PACT 코드의 README/eval 스크립트를 분석하여 실행 방법 파악
2. 모델 경로를 로컬 LLaVA-1.5-7B로 설정
3. 128, 64, 32 토큰으로 POPE, GQA, VQAv2에서 실행

| ID | Method | Tokens | 벤치마크 |
|----|--------|--------|---------|
| P-128 | PACT | 128 | POPE, GQA, VQAv2 |
| P-64 | PACT | 64 | POPE, GQA, VQAv2 |
| P-32 | PACT | 32 | POPE, GQA, VQAv2 |

PACT 코드가 LLaVA-1.5-7B에서 동작하지 않으면 에러 내용 기록 후 논문 숫자 인용.

### 2-C: 논문 숫자 인용 (ToMe, SparseVLM)

직접 실행하지 않고 VisPruner 논문 Table 1 숫자를 인용한다.
결과표에 출처를 "(논문 인용)"으로 명시.

**POPE 기준 인용값:**
| Method | 128 tokens | 64 tokens | 32 tokens |
|--------|-----------|----------|----------|
| ToMe | 62.8 | 52.5 | 39.0 |
| SparseVLM | 80.5 | 75.1 | 67.9 |

**GQA 기준 인용값:**
| Method | 128 tokens | 64 tokens | 32 tokens |
|--------|-----------|----------|----------|
| ToMe | 52.4 | 48.6 | 43.6 |
| SparseVLM | 56.0 | 52.7 | 48.3 |

**VQAv2 기준 인용값:**
| Method | 128 tokens | 64 tokens | 32 tokens |
|--------|-----------|----------|----------|
| ToMe | 63.0 | 57.1 | 46.8 |
| SparseVLM | 73.8 | 68.2 | 58.6 |

---

## 실험 3: Ablation Study

### 3-A: Important/Diverse Ratio
M1=128, M2=64, clustering ON(simple), POPE + GQA:

| ID | r | Important | Diverse |
|----|:---:|---:|---:|
| R-30 | 0.3 | 38 | 90 |
| R-50 | 0.5 | 64 | 64 |
| R-70 | 0.7 | 90 | 38 |

### 3-B: Clustering 유무 직접 비교
POPE + GQA:

**64토큰:**
| ID | M1 | M2 | Clustering |
|----|---:|---:|---|
| Ab-off64 | 64 | 64 | OFF |
| Ab-on64s | 128 | 64 | ON(simple) |
| Ab-on64w | 128 | 64 | ON(weighted) |

**32토큰:**
| ID | M1 | M2 | Clustering |
|----|---:|---:|---|
| Ab-off32 | 32 | 32 | OFF |
| Ab-on32s | 64 | 32 | ON(simple) |
| Ab-on32w | 64 | 32 | ON(weighted) |

### 3-C: Stage1 토큰 수(M1) 민감도
M2=64, clustering ON(simple), r=0.5, POPE:

| ID | M1 | M1/M2 |
|----|---:|:---:|
| M-96 | 96 | 1.5× |
| M-128 | 128 | 2× |
| M-192 | 192 | 3× |

---

## 실험 4: Question-Type-Aware 분석

### 4-A: POPE 카테고리별 (추가 실험 불필요)
실험 1의 POPE 결과를 random/popular/adversarial로 분리 집계.
eval_pope.py가 카테고리별 점수를 출력하므로 정리만 수행.

### 4-B: VQAv2 Question Type별 분석 (핵심)
실험 1의 VQAv2 결과를 **yes/no, number, other** question type별로 분리 집계.

분석 포인트:
- yes/no: 객체 존재 확인 → 적은 토큰으로 충분할 것으로 예상
- number(counting): 세밀한 공간 정보 → 공격적 압축에 취약
- other: 다양한 시각 정보 필요 → 중간

이 분석 결과가 제안 논문의 **task-aware policy의 근거**가 된다.
각 question type에서 VisPruner only vs Ours의 성능 차이를 비교하여
clustering이 어떤 유형의 질문에 더 효과적인지 정량적으로 보여야 한다.

---

## 실험 5: 효율성 측정

실험 1 중 대표 세팅(A-64, B-64, C-64, A-32, B-32)에 대해 측정.
벤치마크: POPE 100개 샘플 (warmup 10개 제외 후 평균)

| 측정 항목 | 방법 |
|---------|------|
| Token Reduction Ratio | (576 − M2) / 576 × 100% |
| Inference Latency | torch.cuda.Event 기반 per-sample 시간 (초) |
| GPU Memory | torch.cuda.max_memory_allocated() |
| Clustering Overhead | Stage 2 단독 소요 시간 (전체 대비 비율) |

---

## 실행 순서 (권장)

```
Phase A: 제안 방법 (실험 1)
  1. A시리즈 POPE → GQA → TextVQA → VQAv2
  2. B시리즈 POPE → GQA → TextVQA → VQAv2
  3. C시리즈 POPE → GQA → TextVQA → VQAv2

Phase B: Baseline (실험 2)
  4. FastV 구현 + POPE → GQA → VQAv2
  5. PACT 실행 POPE → GQA → VQAv2

Phase C: 분석 (실험 3~5)
  6. Ablation 3-A (ratio) POPE → GQA
  7. Ablation 3-B (clustering 유무) POPE → GQA
  8. Ablation 3-C (M1 민감도) POPE
  9. 효율성 측정 (실험 5) POPE
  10. Question-type 분석 (실험 4) — 기존 결과에서 집계
```

각 단계 완료 시 중간 결과를 md에 기록하고 진행 상황을 알려줘.

---

## 에러 대응
- 에러 발생 시 원인 분석 → 스스로 수정 → 재실행
- FastV 구현 실패 시: 논문 숫자 인용으로 대체, 사유 기록
- PACT 실행 실패 시: 논문 숫자 인용으로 대체, 사유 기록
- VQAv2 val set 문제 시: 다운로드 재시도 또는 subset(1000개)으로 대체
- 해결 불가 시 해당 실험만 스킵, 사유 기록 후 다음 진행

---

## 산출물

모든 결과를 `experiments/test_result_md/`에 작성:

### test_result_md/02_main_results.md
- **종합 비교표** (논문 Table 1 대응):
  모든 method × 모든 벤치마크 × 모든 토큰 수를 하나의 표로 정리

| Method | Tokens | POPE | GQA | VQAv2 | TextVQA |
|--------|--------|------|-----|-------|---------|
| FastV | 128/64/32 | | | | |
| ToMe (인용) | 128/64/32 | | | | |
| SparseVLM (인용) | 128/64/32 | | | | |
| PACT | 128/64/32 | | | | |
| VisPruner | 128/64/32 | | | | |
| **Ours (simple)** | 128/64/32 | | | | |
| **Ours (weighted)** | 128/64/32 | | | | |

- VisPruner 대비 Ours의 개선폭 분석
- 토큰 수별 성능 추세 분석

### test_result_md/03_ablation_results.md
- 3-A: Ratio(0.3/0.5/0.7) 결과표 + 분석
- 3-B: Clustering 유무 비교표 + 분석
- 3-C: M1 민감도 결과표 + 분석
- Simple vs Weighted 비교 분석

### test_result_md/04_question_type_analysis.md
- POPE random/popular/adversarial별 성능표
- **VQAv2 yes-no / number / other별 성능표** (핵심)
- 카테고리별 clustering 효과 분석
- Task-aware policy에 대한 시사점

### test_result_md/05_efficiency_results.md
- 토큰 수별 reduction ratio, latency, GPU memory 표
- Clustering overhead 분석
- VisPruner only 대비 추가 overhead 정량화

### test_result_md/06_experiment_log.md
- 실행 명령어 전체 기록
- FastV 구현 과정 기록
- PACT 실행 과정 기록
- 에러 및 해결 과정
- 실험별 소요 시간
- 스킵한 실험과 사유
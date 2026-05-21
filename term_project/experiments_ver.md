# [실험 프롬프트] Two-Stage Framework 체계적 실험

## 목표
구현 완료된 Two-Stage Framework(experiments/term_project/)으로 제안 논문 Table 1에 따른 전체 실험을 수행한다.

## 환경
- 구현 코드: `experiments/term_project/`
- 데이터셋: `experiments/dataset/` (pope, gqa, textvqa, vqav2 등)
- 모델: LLaVA-1.5-7B (로컬)
- GPU: RTX A6000 (48GB)
- 추론: fp16, greedy decoding, temperature=0, CUDA_LAUNCH_BLOCKING=1

---

## 실험 전체 구조 (제안 논문 Table 1 대응)

```
실험 1: 제안 방법 3가지 변형 비교 (핵심 실험)
         → VisPruner only / Ours(clustering) / Ours(clustering + task-aware)
실험 2: 기존 baseline 방법 비교
         → FastV, ToMe, SparseVLM, PACT 숫자 수집/실행
실험 3: Ablation Study
         → (a) important/diverse ratio  (b) clustering 유무  (c) simple vs weighted avg
실험 4: Question-Type-Aware 분석
         → POPE 카테고리별 + VQAv2 question type별
실험 5: 효율성 측정
         → Token reduction ratio, Inference latency
```

---

## 실험 1: 제안 방법 3가지 변형 비교 (핵심)

3가지 변형을 비교한다:
- **(A) VisPruner only**: Stage 1만 수행 (enable_clustering=False)
- **(B) Ours (VisPruner + Clustering)**: Stage 1 + Stage 2 수행
- **(C) Ours Full (VisPruner + Clustering + Task-Aware)**: 실험 4에서 찾은 최적 하이퍼파라미터 적용

벤치마크: **POPE → GQA → TextVQA** 순서로 진행 (빠른 것부터)
토큰 수: **128, 64, 32**
important_ratio r = 0.5 (기본)

### 실행 목록

| ID | Method | enable_clustering | stage1_tokens(M1) | visual_token_num(M2) | merge_method | 벤치마크 |
|----|--------|:-:|--:|--:|---|---|
| A-128 | VisPruner only | OFF | - | 128 | - | POPE, GQA, TextVQA |
| A-64 | VisPruner only | OFF | - | 64 | - | POPE, GQA, TextVQA |
| A-32 | VisPruner only | OFF | - | 32 | - | POPE, GQA, TextVQA |
| B-128s | Ours simple | ON | 192 | 128 | simple_avg | POPE, GQA, TextVQA |
| B-64s | Ours simple | ON | 128 | 64 | simple_avg | POPE, GQA, TextVQA |
| B-32s | Ours simple | ON | 64 | 32 | simple_avg | POPE, GQA, TextVQA |
| B-128w | Ours weighted | ON | 192 | 128 | weighted_avg | POPE, GQA, TextVQA |
| B-64w | Ours weighted | ON | 128 | 64 | weighted_avg | POPE, GQA, TextVQA |
| B-32w | Ours weighted | ON | 64 | 32 | weighted_avg | POPE, GQA, TextVQA |

총 9세팅 × 3벤치마크 = 27회 실험

---

## 실험 2: 기존 Baseline 방법 비교

제안 논문 Table 1에 명시된 baseline들과 비교한다.

### 2-A: VisPruner 논문 Table 1에서 숫자 인용 (직접 실행 불필요)

아래 숫자는 VisPruner 논문 Table 1(LLaVA-1.5-7B)에서 가져온다.
결과 정리 시 출처를 명시하고, 실험 1의 결과와 나란히 비교표를 작성한다.

| Method | 128 tokens | 64 tokens | 32 tokens |
|--------|-----------|----------|----------|
| **FastV (ECCV'24)** | POPE 59.6, GQA 49.6, VQAv2 61.8 | POPE 48.0, GQA 46.1, VQAv2 55.0 | POPE 32.5, GQA 41.5, VQAv2 43.4 |
| **ToMe (ICLR'23)** | POPE 62.8, GQA 52.4, VQAv2 63.0 | POPE 52.5, GQA 48.6, VQAv2 57.1 | POPE 39.0, GQA 43.6, VQAv2 46.8 |
| **SparseVLM (ICML'25)** | POPE 80.5, GQA 56.0, VQAv2 73.8 | POPE 75.1, GQA 52.7, VQAv2 68.2 | POPE 67.9, GQA 48.3, VQAv2 58.6 |
| **PACT (CVPR'25)** | 논문에서 직접 확인 필요 | | |
| **VisPruner (ICCV'25)** | POPE 84.6, GQA 58.2, VQAv2 75.8 | POPE 80.4, GQA 55.4, VQAv2 72.7 | POPE 72.7, GQA 52.2, VQAv2 67.7 |

### 2-B: PACT 직접 실행 (선택적)

`experiments/PACT/` 코드가 있으므로 실행 가능하면 POPE + GQA에서 128/64/32 토큰으로 실행.
실행 불가 시 PACT 논문에서 숫자를 인용하고 사유 기록.

### 2-C: VQAv2 실험 (선택적)

VQAv2는 107,394문항으로 매우 무겁고 채점에 EvalAI 서버 제출이 필요하다.
- 실행 가능하면: 실험 1의 9개 세팅 중 **64토큰 3개(A-64, B-64s, B-64w)만** VQAv2로 추가 실행
- 실행 불가 시: 스킵하고 사유 기록. POPE + GQA + TextVQA 결과로 비교표 작성

---

## 실험 3: Ablation Study

### 3-A: Important/Diverse Ratio 분석

최종 M2=64, clustering ON(simple_avg), M1=128 고정.
벤치마크: POPE + GQA

| ID | important_ratio (r) | Important 수 | Diverse 수 |
|----|:---:|---:|---:|
| R-30 | 0.3 | 128×0.3=38 | 128×0.7=90 |
| R-50 | 0.5 | 128×0.5=64 | 128×0.5=64 |
| R-70 | 0.7 | 128×0.7=90 | 128×0.3=38 |

### 3-B: Clustering 유무 직접 비교

동일한 최종 토큰 수에서 clustering ON vs OFF 비교.
벤치마크: POPE + GQA

**64토큰 비교:**
| ID | M1 | M2 | Clustering | 설명 |
|----|---:|---:|---|---|
| C-off64 | 64 | 64 | OFF | VisPruner만으로 64개 |
| C-on64s | 128 | 64 | ON(simple) | 1단계 128개 → 2단계 64개 |
| C-on64w | 128 | 64 | ON(weighted) | 1단계 128개 → 2단계 64개 |

**32토큰 비교:**
| ID | M1 | M2 | Clustering | 설명 |
|----|---:|---:|---|---|
| C-off32 | 32 | 32 | OFF | VisPruner만으로 32개 |
| C-on32s | 64 | 32 | ON(simple) | 1단계 64개 → 2단계 32개 |
| C-on32w | 64 | 32 | ON(weighted) | 1단계 64개 → 2단계 32개 |

### 3-C: Stage1 토큰 수(M1) 민감도 분석

최종 M2=64 고정, clustering ON(simple_avg), r=0.5.
벤치마크: POPE

| ID | M1 | M2 | M1/M2 비율 |
|----|---:|---:|:---:|
| M-96 | 96 | 64 | 1.5× |
| M-128 | 128 | 64 | 2× |
| M-192 | 192 | 64 | 3× |

→ M1을 얼마나 크게 잡아야 clustering 효과가 최대인지 확인

### 3-D: Simple Average vs Weighted Average

실험 1의 B-시리즈(simple) vs B-시리즈(weighted) 결과를 비교하면 자동 완성.
추가 실험 불필요, 결과 정리 시 분석만 수행.

---

## 실험 4: Question-Type-Aware 분석

### 4-A: POPE 카테고리별 분석 (추가 실험 불필요)

실험 1의 POPE 결과를 **random / popular / adversarial** 카테고리별로 분리 집계.
eval_pope.py가 카테고리별 점수를 출력하므로 그 결과를 정리하면 된다.

분석 포인트:
- adversarial(가장 어려운)에서 clustering 효과가 더 큰지
- diverse 토큰 비율이 adversarial 성능에 미치는 영향

### 4-B: VQAv2 Question Type별 분석 (선택적)

VQAv2는 yes/no, number, other 3가지 question type을 제공한다.
실험 2-C에서 VQAv2를 실행했다면, question type별 정확도를 분리 집계.

Task-aware tuning 탐색:
- 각 question type별로 최적의 (M1, r, M2) 조합이 다를 수 있음
- yes/no: 적은 토큰으로 충분 → 공격적 압축 가능
- number(counting): 세밀한 공간 정보 필요 → 보수적 압축
- other: 중간

VQAv2 실행이 불가하면 이 실험은 스킵하고, POPE 카테고리 분석으로 대체한다.

---

## 실험 5: 효율성 측정

실험 1의 세팅 중 대표적인 것들(A-64, B-64s, B-64w, A-32, B-32s)에 대해 측정.
벤치마크: POPE (가장 빠름)

### 측정 항목

| 항목 | 측정 방법 |
|------|----------|
| Token Reduction Ratio | (576 - M2) / 576 × 100% |
| Inference Latency | POPE 100개 샘플 평균 추론 시간 (초/문항). torch.cuda.Event로 측정 |
| GPU Memory | nvidia-smi 또는 torch.cuda.max_memory_allocated() |
| Clustering Overhead | Stage 2 단독 소요 시간 (전체 latency에서 차지하는 비율) |

측정 시 첫 10개는 warmup으로 제외하고, 이후 100개의 평균을 사용한다.

---

## 실행 순서 (권장)

```
1. 실험 1: A시리즈(VisPruner only) POPE → GQA → TextVQA
2. 실험 1: B시리즈(Ours simple) POPE → GQA → TextVQA
3. 실험 1: B시리즈(Ours weighted) POPE → GQA → TextVQA
4. 실험 3-A: Ratio ablation (POPE, GQA)
5. 실험 3-B: Clustering 유무 비교 (POPE, GQA)
6. 실험 3-C: M1 민감도 (POPE)
7. 실험 5: 효율성 측정 (POPE)
8. 실험 4-A: POPE 카테고리 분석 (기존 결과에서 집계)
9. [선택] 실험 2-B: PACT 실행
10. [선택] 실험 2-C + 4-B: VQAv2
```

각 단계 완료 시마다 중간 결과를 기록하고 진행 상황을 알려줘.

---

## 에러 대응
- 에러 발생 시 원인 분석 → 스스로 수정 → 재실행
- 해결 불가 시 해당 실험만 스킵하고 사유 기록 후 다음 진행
- 수정 내역은 모두 로그에 기록

---

## 산출물

모든 결과를 `experiments/test_result_md/`에 작성:

### test_result_md/02_baseline_comparison.md
- 실험 1 결과: VisPruner vs Ours(simple) vs Ours(weighted) 비교표
- 실험 2 결과: 기존 baseline(FastV, ToMe, SparseVLM, PACT, VisPruner)과의 종합 비교표
- 토큰 수(128/64/32)별 성능 추세 분석
- 제안 방법의 개선 정도 분석

### test_result_md/03_ablation_results.md
- 3-A: Important/diverse ratio(0.3/0.5/0.7) 결과표 + 분석
- 3-B: Clustering 유무 비교 결과표 + 분석
- 3-C: M1 민감도 결과표 + 분석
- 3-D: Simple vs weighted 비교 분석

### test_result_md/04_question_type_analysis.md
- 4-A: POPE random/popular/adversarial별 성능표
- 4-B: VQAv2 question type별 성능표 (실행한 경우)
- 카테고리/question type별 clustering 효과 분석

### test_result_md/05_efficiency_results.md
- 토큰 수별 reduction ratio, inference latency, GPU memory 표
- Clustering overhead 분석
- VisPruner only 대비 추가 overhead 정량화

### test_result_md/06_experiment_log.md
- 실행한 명령어 전체 기록 (복붙 가능하게)
- 에러 발생 및 해결 과정
- 실험별 소요 시간
- 스킵한 실험과 사유

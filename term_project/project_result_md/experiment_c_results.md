# Experiment C: TextVQA M1 Scaling Law — 결과 및 분석

> 최초 작성: 2026-05-25 | 최종 업데이트: 2026-05-26 (전체 완료)

---

## 1. M2=64 시리즈 결과

*baseline: TextVQA Acc=55.73% (TC-64-base, Stage1-only)*

| M1 | M1:M2 | TextVQA Acc | Δ Acc |
|---:|---:|---|---|
| 64 | 1.0× | 55.73% | — |
| 96 | 1.5× | 53.89% | **−1.84%** |
| 128 | 2.0× | 54.03% | −1.70% |
| 192 | 3.0× | 52.87% | −2.86% |
| 256 | 4.0× | 52.59% | −3.14% |
| 384 | 6.0× | 51.40% | −4.33% |
| 576 | 9.0× *(KMeans only)* | **50.25%** | **−5.48%** |

- **sweet spot 없음** — M1이 클수록 **단조 감소**
- M1=M2=64 (1.0×): Stage1-only와 완전 동일 → K-Means identity 확인
- M1=576 (KMeans only): 최저, baseline 대비 −5.48%

---

## 2. M2=32 시리즈 결과

*baseline: TextVQA Acc=53.83% (TC-32-base, Stage1-only)*

| M1 | M1:M2 | TextVQA Acc | Δ Acc |
|---:|---:|---|---|
| 32 | 1.0× | 53.83% | — |
| 48 | 1.5× | 53.28% | −0.55% |
| 64 | 2.0× | 52.85% | −0.98% |
| 96 | 3.0× | 52.75% | −1.08% |
| 128 | 4.0× | 52.21% | −1.62% |
| 192 | 6.0× | 51.16% | −2.67% |
| 256 | 8.0× | 50.53% | −3.30% |
| 288 | 9.0× | 50.41% | −3.42% |
| 384 | 12.0× | 49.14% | −4.69% |
| 576 | 18.0× *(KMeans only)* | **47.78%** | **−6.05%** |

- 동일하게 **단조 감소** — M2 작을수록 하락폭 더 큼 (18× 시 −6.05%)

---

## 3. 핵심 인사이트

### 3.1 M2별 비교

| M2 | baseline | 1.0× (identity) | 최저 (KMeans only) | 최저 Δ |
|---|---|---|---|---|
| 64 | 55.73% | 55.73% (=baseline) | 50.25% (9×) | **−5.48%** |
| 32 | 53.83% | 53.83% (=baseline) | 47.78% (18×) | **−6.05%** |

> **M2가 작을수록 하락폭 더 큼** — Exp B의 이득 방향과 정반대

### 3.2 Scaling Curve 형태 — 3-벤치마크 비교

```
POPE (M2=64):    단조증가 → peak(4×,+0.027) → 하강  ← 유니모달 ↑
GQA  (M2=64):    단조증가 → peak(4×,+2.01%) → 하강  ← 유니모달 ↑
TextVQA(M2=64):  즉시 하락 (1.5×부터 −1.84%)        ← 단조감소 ↓
```

| 벤치마크 | 특성 | Scaling 패턴 | Two-Stage 최대 효과 |
|---|---|---|---|
| POPE | 객체 환각 (Yes/No) | 유니모달, peak 4–12× | **+0.063** (F1) |
| GQA | 구성 시각 추론 | 유니모달, peak 4–8× | **+4.18%** |
| **TextVQA** | **OCR·텍스트 인식** | **단조 감소** | **−1.84% ~ −6.05%** |

### 3.3 Stage1의 기여 정량화

| 구성 | M2=64 Acc | M2=32 Acc |
|---|---|---|
| KMeans only (M1=576) | 50.25% | 47.78% |
| **Stage1 only (baseline)** | **55.73%** | **53.83%** |
| Best Two-Stage | 55.73% (=baseline) | 53.83% (=baseline) |

- TextVQA에서 Best Two-Stage = Stage1-only = M1=M2 (K-Means no-op)
- **Stage1 > KMeans-only: +5.48% / +6.05%** → Stage1 attention-based pruning이 OCR에 적합
- K-Means는 오히려 해롭고, Stage1 pruning만으로도 우수

### 3.4 K-Means가 OCR에 해로운 이유

```
객체·관계 이해 (POPE/GQA):
  → K-Means: 의미적으로 유사한 토큰 집합화 → semantic aggregation 유효
  → M1 확장: 더 다양한 seed → centroid 품질↑ (어느 수준까지)

OCR·텍스트 인식 (TextVQA):
  → 글자 형태·픽셀 위치가 critical
  → K-Means: 인접 토큰 평균 → 공간 정밀도 희석
  → M1 확장: 더 많은 이질 토큰 유입 → 중심점 smearing 심화
  → M1=M2: 형식적 identity → 영향 없음 (Stage1과 동일)
```

---

## 4. 논문 서술 초안

### Figure 캡션 (3-벤치마크 Scaling Curve)

> **Figure X.** Performance as a function of M1:M2 ratio across three benchmarks (M2=64). POPE (object hallucination, blue) and GQA (compositional reasoning, orange) exhibit unimodal curves peaking at M1=4×M2 (+0.027 F1, +2.01% Acc respectively), while TextVQA (OCR/text recognition, red) decreases monotonically from the baseline, reaching −5.5% at M1=9×M2. This divergence reveals that K-Means centroid merging benefits semantic aggregation tasks but degrades fine-grained spatial tasks where per-token precision is critical.

### Results 섹션 핵심 문장

> In sharp contrast to the unimodal gains on POPE and GQA, TextVQA accuracy **decreases monotonically** as M1 increases, with a maximum drop of −5.5% (M2=64, M1=9×M2) and −6.1% (M2=32, M1=18×M2). The M1=M2 configuration (identity K-Means) yields no degradation relative to Stage 1, confirming that the loss originates from the merging step rather than candidate token selection. This task-type dependency is consistent with the nature of each benchmark: while POPE and GQA require semantic-level understanding amenable to centroid aggregation, TextVQA demands pixel-level spatial fidelity for character recognition that averaging operations inherently compromise.

---

## 5. 실험 이력

| 일시 | 내용 |
|---|---|
| 2026-05-25 21:08 | HF Arrow 이미지 3,166장 추출, annotation 5,000개 생성 |
| 2026-05-25 21:08 | Experiment C 2-GPU nohup 런치 (19개 job) |
| 2026-05-26 05:47 | 전체 19개 job 완료 (5,000/5,000 each) |

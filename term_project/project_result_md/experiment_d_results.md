# Experiment D: ScienceQA M1 Scaling Law — 결과 및 분석

> 최초 작성: 2026-05-26 | 최종 업데이트: 2026-05-26 (전체 완료)
>
> ScienceQA 특성: 4,241문항 중 **2,017개(47.6%)만 이미지 포함**, 나머지 2,224개(52.4%)는 텍스트 전용
>
> **⚠️ 텍스트 전용 문항은 추론 시 images=None 처리 → M1/M2 값 무관하게 동일 결과**
> → 전체 Acc는 희석됨. **IMG-Acc(이미지 포함 2,017문항만)** 가 M1 스케일링 실질 지표

---

## 1. M2=64 시리즈 결과

*baseline: IMG-Acc=68.96% / 전체 Acc=69.91% (TD-64-base, Stage1-only)*

| M1 | M1:M2 | 전체 Acc | Δ | **IMG-Acc** | **Δ (이미지)** |
|---:|---:|---|---|---|---|
| 64 | 1.0× | 69.91% | — | 68.96% | — |
| **96** | **1.5×** | **70.15%** | +0.24% | **69.46% ★** | **+0.50%** |
| 128 | 2.0× | 69.89% | −0.02% | 68.91% | −0.05% |
| 192 | 3.0× | 69.79% | −0.12% | 68.72% | −0.24% |
| 256 | 4.0× | 69.82% | −0.09% | 68.77% | −0.19% |
| 384 | 6.0× | 69.89% | −0.02% | 68.91% | −0.05% |
| 576 | 9.0× *(KMeans only)* | 69.70% | −0.21% | 68.52% | −0.44% |

- Peak: M1=96 (1.5×), IMG-Acc +0.50%

---

## 2. M2=32 시리즈 결과

*baseline: IMG-Acc=68.32% / 전체 Acc=69.61% (TD-32-base, Stage1-only)*

| M1 | M1:M2 | 전체 Acc | Δ | **IMG-Acc** | **Δ (이미지)** |
|---:|---:|---|---|---|---|
| 32 | 1.0× | 69.61% | — | 68.32% | — |
| 48 | 1.5× | 70.12% | +0.51% | 69.41% | +1.09% |
| **64** | **2.0×** | **70.27%** | +0.66% | **69.71% ★** | **+1.39%** |
| 96 | 3.0× | 69.79% | +0.18% | 68.72% | +0.40% |
| 128 | 4.0× | 70.10% | +0.49% | 69.36% | +1.04% |
| 192 | 6.0× | 69.61% | 0.00% | 68.32% | 0.00% |
| 256 | 8.0× | 69.72% | +0.11% | 68.57% | +0.25% |
| 288 | 9.0× | 69.84% | +0.23% | 68.82% | +0.50% |
| 384 | 12.0× | 69.65% | +0.04% | 68.42% | +0.10% |
| 576 | 18.0× *(KMeans only)* | 69.56% | −0.05% | 68.22% | −0.10% |

- Peak: M1=64 (2×), IMG-Acc **+1.39%** (전체 Acc +0.66% 대비 2.1× 더 큰 효과)
- 텍스트 전용 문항 52.4%가 스케일링 효과를 절반 이상 희석시킴

---

## 3. 핵심 인사이트

### 3.1 M2별 Sweet Spot 비교 (IMG-Acc 기준)

| M2 | Peak M1 (비율) | 전체 Acc Max Gain | **IMG-Acc Max Gain** |
|---|---|---|---|
| 64 | M1=96, **1.5×** | +0.24% | **+0.50%** |
| 32 | M1=64, **2.0×** | +0.66% | **+1.39%** |

### 3.2 희석 효과 정량화

```
텍스트 전용 문항 비율: 52.4% (2,224 / 4,241)
이미지 포함 문항:      47.6% (2,017 / 4,241)

전체 Acc Max Gain (M2=32):  +0.66%
IMG-Acc Max Gain (M2=32):   +1.39%

희석 비율: 0.66 / 1.39 = 0.47 ≈ 이미지 문항 비율(0.476)
→ 텍스트 전용 문항이 정확히 스케일링 효과를 절반으로 희석
```

### 3.3 Stage1 기여 정량화 (IMG-Acc 기준)

| 구성 | M2=64 IMG-Acc | M2=32 IMG-Acc |
|---|---|---|
| KMeans only (M1=576) | 68.52% | 68.22% |
| Stage1 only (baseline) | 68.96% | 68.32% |
| Best Two-Stage | **69.46%** | **69.71%** |

- KMeans only < Stage1 only (−0.44% / −0.10%) → Stage1 attention pruning이 여전히 우세
- Best Two-Stage > Stage1 only (+0.50% / +1.39%) → 이미지 문항에서는 Two-Stage 유효

---

## 4. 논문 서술 초안

### Results 핵심 문장

> ScienceQA, which contains 52% text-only questions, shows a nearly flat scaling curve with marginal gains of +0.24% (M2=64) and +0.66% (M2=32) at small M1:M2 ratios. The muted response reflects the mixed nature of the benchmark: visual tokens are irrelevant for text-only questions, diluting any K-Means effect. The slight positive gain at low M1 ratios suggests that the 48% image-containing science questions do benefit modestly from centroid merging, consistent with the pattern observed on POPE and GQA, but at a reduced scale due to the text-only dilution.

---

## 5. 실험 이력

| 일시 | 내용 |
|---|---|
| 2026-05-26 05:47 | HF 데이터셋 이미지 2,017장 추출, problems.json / pid_splits.json 생성 |
| 2026-05-26 05:55 | Experiment D 2-GPU nohup 런치 (19개 job) |
| 2026-05-26 | 전체 19개 job 완료 (4,241/4,241 each) |

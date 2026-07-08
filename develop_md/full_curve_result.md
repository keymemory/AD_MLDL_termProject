# 전 구간 성능 곡선 — 실측값 전량 정리 (묶음 A)

> **출처**: 모든 성능값은 `term_project/exp_runner/results_phase2A.tsv` (210행) 에서 읽음.
> τ/k 값은 `term_project/exp_runner/exp_jobs_phase2_A.tsv` (job 정의) 와 ID로 조인해 매핑.
> floor%/cap% 도 results_phase2A.tsv 의 FLOOR/CAP 컬럼(실측). **추정·생성값 없음. 없는 조합은 `미측정`.**
> 지표: POPE=F1, GQA/TextVQA=Acc. 병합: weighted=weighted_avg, simple=simple_avg. r=0.5 고정.

## 데이터 커버리지 (실측 210개)
- **energy**: 3 데이터셋 × 3 M2(32/64/128) × 7 τ(0.3~0.9) × 2 병합 = **126개 전부 실측** ✅
- **statistical**: 3 데이터셋 × **2 M2(32/64만)** × 7 k(0.2~0.8) × 2 병합 = **84개 실측** ✅
- **statistical M2=128 = 미측정** (붕괴로 실험 제외). 이론상 42개 조합 없음 → 아래 표에서 `미측정`, 말미에 목록.

---

# POPE  (지표: F1)

## POPE — energy

| M2 | 병합 | τ=0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|----|------|-------|-----|-----|-----|-----|-----|-----|
| 32 | weighted | 0.7719 | 0.7479 | 0.7554 | 0.7818 | 0.8011 | 0.8071 | 0.8048 |
| 32 | simple | 0.7719 | 0.7495 | 0.7563 | 0.7844 | 0.7878 | 0.7970 | 0.7980 |
| 64 | weighted | 0.8334 | 0.8242 | 0.8132 | 0.8145 | 0.8321 | 0.8364 | 0.8416 |
| 64 | simple | 0.8334 | 0.8242 | 0.8144 | 0.8098 | 0.8214 | 0.8331 | 0.8332 |
| 128 | weighted | 0.8596 | 0.8599 | 0.8533 | 0.8443 | 0.8499 | 0.8551 | 0.8569 |
| 128 | simple | 0.8596 | 0.8599 | 0.8533 | 0.8442 | 0.8502 | 0.8567 | 0.8571 |

**M2별 best (energy):**
- M2=32: best τ=0.8 (weighted), F1=0.8071 · avg_M1=263 · floor%=0.0 cap%=0.6 adapt%=99.4
- M2=64: best τ=0.9 (weighted), F1=0.8416 · avg_M1=375 · floor%=0.0 cap%=73.9 adapt%=26.1
- M2=128: best τ=0.4 (weighted), F1=0.8599 · avg_M1=128 · floor%=100.0 cap%=0.0 adapt%=0.0

**건강구간 (energy, floor%/cap%/adapt%; selection 산물이라 병합 무관 → weighted 기준):**

| M2 | 항목 | τ=0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|----|------|-------|-----|-----|-----|-----|-----|-----|
| 32 | floor% | 99.4 | 67.3 | 11.5 | 0.4 | 0.0 | 0.0 | 0.0 |
| 32 | cap% | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.6 | 73.9 |
| 32 | adapt% | 0.6 | 32.7 | 88.5 | 99.6 | 100.0 | 99.4 | 26.1 |
| 64 | floor% | 100.0 | 99.4 | 72.3 | 10.4 | 0.0 | 0.0 | 0.0 |
| 64 | cap% | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.6 | 73.9 |
| 64 | adapt% | 0.0 | 0.6 | 27.7 | 89.6 | 100.0 | 99.4 | 26.1 |
| 128 | floor% | 100.0 | 100.0 | 100.0 | 85.6 | 19.9 | 0.0 | 0.0 |
| 128 | cap% | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.6 | 73.9 |
| 128 | adapt% | 0.0 | 0.0 | 0.0 | 14.4 | 80.1 | 99.4 | 26.1 |

## POPE — statistical

| M2 | 병합 | k=0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 |
|----|------|-------|-----|-----|-----|-----|-----|-----|
| 32 | weighted | 0.7973 | 0.7794 | 0.7705 | 0.7664 | 0.7659 | 0.7419 | 0.7577 |
| 32 | simple | 0.7896 | 0.7850 | 0.7752 | 0.7629 | 0.7624 | 0.7422 | 0.7556 |
| 64 | weighted | 0.8225 | 0.8151 | 0.8109 | 0.8059 | 0.8189 | 0.8134 | 0.8224 |
| 64 | simple | 0.8254 | 0.8098 | 0.8123 | 0.8048 | 0.8194 | 0.8133 | 0.8225 |
| 128 | weighted | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 |
| 128 | simple | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 |

**M2별 best (statistical):**
- M2=32: best k=0.2 (weighted), F1=0.7973 · avg_M1=123 · floor%=0.0 cap%=0.0 adapt%=100.0
- M2=64: best k=0.2 (simple), F1=0.8254 · avg_M1=123 · floor%=1.6 cap%=0.0 adapt%=98.4
- M2=128: 미측정 (statistical 실험 제외)

---

# GQA  (지표: Acc)

## GQA — energy

| M2 | 병합 | τ=0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|----|------|-------|-----|-----|-----|-----|-----|-----|
| 32 | weighted | 51.85 | 52.45 | 53.11 | 54.09 | 55.44 | 55.84 | 56.21 |
| 32 | simple | 51.93 | 52.46 | 53.27 | 54.52 | 55.17 | 55.53 | 55.07 |
| 64 | weighted | 55.65 | 55.92 | 55.82 | 56.69 | 56.89 | 57.50 | 57.57 |
| 64 | simple | 55.65 | 55.94 | 55.76 | 56.20 | 57.04 | 57.31 | 57.39 |
| 128 | weighted | 58.39 | 58.15 | 58.03 | 58.54 | 58.49 | 58.52 | 58.69 |
| 128 | simple | 58.39 | 58.15 | 58.03 | 58.50 | 58.54 | 58.67 | 58.59 |

**M2별 best (energy):**
- M2=32: best τ=0.9 (weighted), Acc=56.21 · avg_M1=378 · floor%=0.0 cap%=79.9 adapt%=20.1
- M2=64: best τ=0.9 (weighted), Acc=57.57 · avg_M1=378 · floor%=0.0 cap%=80.0 adapt%=20.0
- M2=128: best τ=0.9 (weighted), Acc=58.69 · avg_M1=378 · floor%=0.0 cap%=80.0 adapt%=20.0

**건강구간 (energy, floor%/cap%/adapt%; selection 산물이라 병합 무관 → weighted 기준):**

| M2 | 항목 | τ=0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|----|------|-------|-----|-----|-----|-----|-----|-----|
| 32 | floor% | 98.6 | 67.1 | 8.2 | 0.0 | 0.0 | 0.0 | 0.0 |
| 32 | cap% | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.8 | 79.9 |
| 32 | adapt% | 1.4 | 32.9 | 91.8 | 100.0 | 100.0 | 99.2 | 20.1 |
| 64 | floor% | 100.0 | 98.3 | 68.0 | 7.4 | 0.0 | 0.0 | 0.0 |
| 64 | cap% | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.8 | 80.0 |
| 64 | adapt% | 0.0 | 1.7 | 32.0 | 92.6 | 100.0 | 99.2 | 20.0 |
| 128 | floor% | 100.0 | 100.0 | 100.0 | 79.4 | 14.0 | 0.1 | 0.0 |
| 128 | cap% | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.8 | 80.0 |
| 128 | adapt% | 0.0 | 0.0 | 0.0 | 20.6 | 86.0 | 99.1 | 20.0 |

> ⚠️ 위 floor 표는 weighted 실행 기준. 아래 조합은 **병합본이 독립 실행이라 floor%가 다르게 측정**됨(실측 그대로, 판정엔 영향 미미):
>   - M2=64 τ=0.6: weighted floor%=7.4 / simple floor%=9.4 (경계 구간, 독립 실행 차이)

## GQA — statistical

| M2 | 병합 | k=0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 |
|----|------|-------|-----|-----|-----|-----|-----|-----|
| 32 | weighted | 54.81 | 54.48 | 53.80 | 53.82 | 53.36 | 53.20 | 52.82 |
| 32 | simple | 54.98 | 54.13 | 53.92 | 53.91 | 53.01 | 53.39 | 52.76 |
| 64 | weighted | 56.52 | 56.19 | 56.32 | 56.12 | 55.85 | 56.19 | 55.89 |
| 64 | simple | 56.89 | 56.16 | 56.21 | 55.86 | 55.89 | 56.11 | 55.84 |
| 128 | weighted | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 |
| 128 | simple | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 |

**M2별 best (statistical):**
- M2=32: best k=0.2 (simple), Acc=54.98 · avg_M1=125 · floor%=0.4 cap%=0.0 adapt%=99.6
- M2=64: best k=0.2 (simple), Acc=56.89 · avg_M1=125 · floor%=1.8 cap%=0.0 adapt%=98.2
- M2=128: 미측정 (statistical 실험 제외)

---

# TEXTVQA  (지표: Acc)

## TEXTVQA — energy

| M2 | 병합 | τ=0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|----|------|-------|-----|-----|-----|-----|-----|-----|
| 32 | weighted | 52.23 | 53.18 | 53.63 | 53.49 | 52.89 | 52.39 | 51.40 |
| 32 | simple | 52.24 | 53.28 | 53.69 | 53.31 | 51.76 | 50.98 | 49.96 |
| 64 | weighted | 53.88 | 54.96 | 55.72 | 54.85 | 54.63 | 53.97 | 53.49 |
| 64 | simple | 53.88 | 54.92 | 55.80 | 54.87 | 54.43 | 53.44 | 52.03 |
| 128 | weighted | 56.29 | 55.89 | 56.54 | 56.69 | 56.67 | 55.51 | 54.68 |
| 128 | simple | 56.29 | 55.89 | 56.54 | 56.58 | 56.53 | 55.43 | 54.22 |

**M2별 best (energy):**
- M2=32: best τ=0.5 (simple), Acc=53.69 · avg_M1=48 · floor%=22.3 cap%=0.0 adapt%=77.7
- M2=64: best τ=0.5 (simple), Acc=55.80 · avg_M1=66 · floor%=82.3 cap%=0.0 adapt%=17.7
- M2=128: best τ=0.6 (weighted), Acc=56.69 · avg_M1=129 · floor%=92.5 cap%=0.0 adapt%=7.5

**건강구간 (energy, floor%/cap%/adapt%; selection 산물이라 병합 무관 → weighted 기준):**

| M2 | 항목 | τ=0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|----|------|-------|-----|-----|-----|-----|-----|-----|
| 32 | floor% | 98.8 | 73.8 | 22.3 | 2.6 | 0.1 | 0.0 | 0.0 |
| 32 | cap% | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.2 | 42.7 |
| 32 | adapt% | 1.2 | 26.2 | 77.7 | 97.4 | 99.9 | 99.8 | 57.3 |
| 64 | floor% | 100.0 | 99.5 | 82.3 | 33.8 | 4.9 | 0.2 | 0.0 |
| 64 | cap% | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.2 | 42.7 |
| 64 | adapt% | 0.0 | 0.5 | 17.7 | 66.2 | 95.1 | 99.6 | 57.3 |
| 128 | floor% | 100.0 | 100.0 | 100.0 | 92.5 | 49.9 | 7.4 | 0.0 |
| 128 | cap% | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.2 | 42.7 |
| 128 | adapt% | 0.0 | 0.0 | 0.0 | 7.5 | 50.1 | 92.4 | 57.3 |

## TEXTVQA — statistical

| M2 | 병합 | k=0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 |
|----|------|-------|-----|-----|-----|-----|-----|-----|
| 32 | weighted | 53.23 | 53.44 | 53.40 | 53.45 | 53.41 | 52.86 | 53.14 |
| 32 | simple | 52.60 | 52.92 | 53.01 | 52.95 | 53.35 | 53.04 | 53.22 |
| 64 | weighted | 54.58 | 54.90 | 54.83 | 55.03 | 55.46 | 55.32 | 55.32 |
| 64 | simple | 54.53 | 54.43 | 54.53 | 55.18 | 55.41 | 55.34 | 55.32 |
| 128 | weighted | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 |
| 128 | simple | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 |

**M2별 best (statistical):**
- M2=32: best k=0.5 (weighted), Acc=53.45 · avg_M1=75 · floor%=2.1 cap%=0.0 adapt%=97.9
- M2=64: best k=0.6 (weighted), Acc=55.46 · avg_M1=73 · floor%=46.5 cap%=0.0 adapt%=53.5
- M2=128: 미측정 (statistical 실험 제외)

---

## M2별 best τ 한 줄 요약 (energy)
- **POPE**: M2=32 best τ=0.8, F1=0.8071 / M2=64 best τ=0.9, F1=0.8416 / M2=128 best τ=0.4, F1=0.8599
- **GQA**: M2=32 best τ=0.9, Acc=56.21 / M2=64 best τ=0.9, Acc=57.57 / M2=128 best τ=0.9, Acc=58.69
- **TEXTVQA**: M2=32 best τ=0.5, Acc=53.69 / M2=64 best τ=0.5, Acc=55.80 / M2=128 best τ=0.6, Acc=56.69

---

## 미측정(누락) 조합 목록
- 총 **42개** 조합 미측정.
- 전부 **statistical × M2=128** 조합 (붕괴로 실험 제외). 상세:
  - pope statistical M2=128 weighted_avg k=0.2
  - pope statistical M2=128 weighted_avg k=0.3
  - pope statistical M2=128 weighted_avg k=0.4
  - pope statistical M2=128 weighted_avg k=0.5
  - pope statistical M2=128 weighted_avg k=0.6
  - pope statistical M2=128 weighted_avg k=0.7
  - pope statistical M2=128 weighted_avg k=0.8
  - pope statistical M2=128 simple_avg k=0.2
  - pope statistical M2=128 simple_avg k=0.3
  - pope statistical M2=128 simple_avg k=0.4
  - pope statistical M2=128 simple_avg k=0.5
  - pope statistical M2=128 simple_avg k=0.6
  - pope statistical M2=128 simple_avg k=0.7
  - pope statistical M2=128 simple_avg k=0.8
  - gqa statistical M2=128 weighted_avg k=0.2
  - gqa statistical M2=128 weighted_avg k=0.3
  - gqa statistical M2=128 weighted_avg k=0.4
  - gqa statistical M2=128 weighted_avg k=0.5
  - gqa statistical M2=128 weighted_avg k=0.6
  - gqa statistical M2=128 weighted_avg k=0.7
  - gqa statistical M2=128 weighted_avg k=0.8
  - gqa statistical M2=128 simple_avg k=0.2
  - gqa statistical M2=128 simple_avg k=0.3
  - gqa statistical M2=128 simple_avg k=0.4
  - gqa statistical M2=128 simple_avg k=0.5
  - gqa statistical M2=128 simple_avg k=0.6
  - gqa statistical M2=128 simple_avg k=0.7
  - gqa statistical M2=128 simple_avg k=0.8
  - textvqa statistical M2=128 weighted_avg k=0.2
  - textvqa statistical M2=128 weighted_avg k=0.3
  - textvqa statistical M2=128 weighted_avg k=0.4
  - textvqa statistical M2=128 weighted_avg k=0.5
  - textvqa statistical M2=128 weighted_avg k=0.6
  - textvqa statistical M2=128 weighted_avg k=0.7
  - textvqa statistical M2=128 weighted_avg k=0.8
  - textvqa statistical M2=128 simple_avg k=0.2
  - textvqa statistical M2=128 simple_avg k=0.3
  - textvqa statistical M2=128 simple_avg k=0.4
  - textvqa statistical M2=128 simple_avg k=0.5
  - textvqa statistical M2=128 simple_avg k=0.6
  - textvqa statistical M2=128 simple_avg k=0.7
  - textvqa statistical M2=128 simple_avg k=0.8


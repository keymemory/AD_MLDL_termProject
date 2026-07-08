# VLM 시각 토큰 압축 — 전체 실험 결과 종합 (마스터)

> LLaVA-1.5-7B 시각 토큰(576) 2단계 압축 연구의 **모든 실험 결과**를 한 문서로 종합.
> Phase 1(adaptive 구현·검증·E2) + Phase 2 Part 1(데이터셋 확장·건강구간) + Part 2 묶음 A(전 구간 곡선).
> 백본: LLaVA-1.5-7B / greedy(temp=0) / fp16 / CUDA_LAUNCH_BLOCKING=1. 원자료: `results*.tsv`.

---

## 0. 연구 개요

CLIP-ViT-L/14-336이 이미지를 576 visual token으로 만들어 LLM에 주입 → O(n²) 비용. **토큰을 M2(32/64/128)로
압축하되 성능 유지**가 과제. 기존 VisPruner는 important 개수를 고정(M1=2×M2, r=0.5)하는데, 본 연구는
**important 개수를 이미지마다 자동 결정(adaptive selection)** + **Spherical K-Means 병합(Two-Stage)** 을 제안.

**최종 핵심 기여**: adaptive의 최적 강도(τ)가 **태스크 의미구조에 따라 질적으로 다르다**
(GQA는 τ↑, TextVQA는 중간 τ, POPE는 상황별) — Part 2에서 입증.

---

## 1. 방법론

```
576 → [Stage1 VisPruner] [CLS]attn important(M1·r) + cosine diverse(M1·(1-r)) → M1개
    → [Stage2 Spherical K-Means] M1 → M2 병합(simple/weighted avg) → M2개 → LLM
```
- **selection_method**: `topk`(기존 VisPruner, M1 고정) / `energy`(누적 attn 질량≥τ) / `statistical`(μ+kσ 이상치)
- **M1 자동결정**: `M1 = clamp(round(n_imp/0.5), floor=M2, cap=384)`
- **floor 의미**: M1=M2이면 Stage2 병합 생략, important(소수)+diverse(다수)로 채움 (결과 해석의 열쇠)
- 신규 코드: `adaptive_selection.py`, `spherical_kmeans.py`, `llava_arch.py::encode_images` 분기

---

## 2. Phase 1 — adaptive 구현·검증·E2 본실험

### 2-1. 회귀 검증 (회귀 안전성) ✅
- `selection_method="topk"` default = 기존 VisPruner **비트동일**: POPE 300 subset + **full 8910 모두 text_diff=0**,
  F1 = 0.8446517939988288 (구코드 ↔ 신코드 16자리 동일).

### 2-2. E2 본실험 (POPE+GQA, M2=64, energy τ/statistical k 스윕)
| 설정 | POPE F1 | GQA Acc | AVG_M1 |
|---|---|---|---|
| topk baseline (M1=128) | 0.8227 | 56.66 | 128 |
| energy τ=0.7 weighted | 0.8269 | 57.59 | ~163 |
| energy τ=0.8 weighted | **0.8398** | 57.41 | ~263 |
| statistical k=0.3 simple | 0.8129 | — | ~97 |

**Phase 1 결론**: energy τ=0.8이 baseline 상회(0.8398)하나 **M1이 2배(263)**. 같은 M1=128로 맞추면(τ*≈0.64)
energy ≈ baseline. → **적응의 가치는 "무조건 우월"이 아니라 토큰-성능 trade-off를 부드럽게 탐색 + 고정을 특수해로 포함.**

---

## 3. Phase 2 Part 1 — 데이터셋 확장 + 건강 τ/k

### 3-1. MME·MMBench 추가 (topk 작동 검증) ✅
- MME: parquet 이미지 추출 + `mme_eval.py`(perception+cognition). topk M2=128 → **MME total 1752.57**.
- MMBench: `model_vqa_mmbench.py` selection 인자 + `mmbench_eval.py`. topk M2=128 → **Acc 72.13**.
- worker.sh에 mme/mmbench 케이스 통합(11컬럼 호환).

### 3-2. ★ M2별 건강 τ/k (probe, adapt≥90%)
| M2 | energy 건강 τ | statistical 건강 k |
|---|---|---|
| 32 | 0.6 / 0.7 / 0.8 | 0.2~0.6 |
| 64 | 0.7 / 0.8 | 0.2 / 0.3 |
| 128 | 0.8 (단일) | **없음(전 구간 붕괴)** |
→ **floor=M2라 M2 클수록 건강구간 급감.** M2=128 statistical은 붕괴 → 본실험 제외.

---

## 4. Phase 2 Part 2 묶음 A — 전 구간 성능 곡선 (완주 210/210)

POPE/GQA/TextVQA × energy(τ 0.3~0.9)/statistical(k 0.2~0.8) × simple/weighted × M2(32/64/128). M2=128 stat 제외.

### 4-1. energy weighted 곡선 (τ별, ★=건강구간)
**POPE (F1)**
| M2 | 0.3 | 0.5 | 0.7★ | 0.8★ | 0.9 |
|---|---|---|---|---|---|
| 32 | 0.772 | 0.755 | 0.801 | **0.807** | 0.805 |
| 64 | 0.833 | 0.813 | 0.832 | 0.836 | **0.842** |
| 128 | **0.860** | 0.853 | 0.850 | 0.855 | 0.857 |

**GQA (Acc)**
| M2 | 0.3 | 0.5 | 0.7★ | 0.8★ | 0.9 |
|---|---|---|---|---|---|
| 32 | 51.85 | 53.11 | 55.44 | 55.84 | **56.21** |
| 128 | 58.39 | 58.03 | 58.49 | 58.52 | **58.69** |

**TextVQA (Acc)**
| M2 | 0.3 | 0.5 | 0.6 | 0.7★ | 0.8★ | 0.9 |
|---|---|---|---|---|---|---|
| 32 | 52.23 | **53.63** | 53.49 | 52.89 | 52.39 | 51.40 |
| 128 | 56.29 | 56.54 | **56.69** | 56.67 | 55.51 | 54.68 |

### 4-2. ★★ 핵심 발견 — 태스크별 최적 selection이 정반대 (M2=32 강압축)
τ를 0.3(floor)→0.9(cap)로 올릴 때:
- **GQA(공간추론)**: 51.85 → 56.21 **단조 증가(+4.4)** → important 토큰 많이 보존할수록 좋다.
- **TextVQA(OCR)**: 52.23 → **53.63(τ0.5)** → 51.40 → **중간이 최고, 높은 τ는 글자 병합 손상.**
- **POPE(환각)**: 강압축은 건강 τ 우위(0.807@τ0.8), 약압축(M2=128)은 **floor(diverse 다수)가 최고(0.860@τ0.3)**.

→ **"최적 압축 방식(τ)이 태스크 의미구조에 따라 질적으로 다르다"** = 본 연구의 핵심 노벨티.
→ **압축이 강할수록(M2=32) 태스크 차이 극대화** (M2=128은 곡선 완만).

### 4-3. best τ/k (POPE 건강구간 기준)
| 수식 | M2=32 | M2=64 | M2=128 |
|---|---|---|---|
| energy | τ=0.8 | τ=0.8 | τ=0.8 |
| statistical | k=0.2 | k=0.2 | (제외) |
⚠️ **TextVQA는 τ=0.8이 나쁨**(중간 τ 최적) → 묶음 B는 **데이터셋별 best** 권장.

---

## 5. 종합 핵심 발견 (3줄 요약)

1. **회귀 안전**: topk = 기존 VisPruner 비트동일(full 8910 F1 16자리 일치).
2. **적응의 본질**: 같은 M1에선 적응≈고정, **적응은 τ 하나로 토큰-성능 trade-off를 부드럽게 탐색하고 고정을 특수해로 포함.**
3. **★ 태스크별 차이(노벨티)**: GQA는 τ↑(important 중요), TextVQA는 중간 τ(OCR 병합 손상), POPE는 상황별
   (약압축 floor·diverse). **압축이 강할수록 이 차이가 결정적.**

---

## 6. 남은 작업 (묶음 B — 최종 비교표)
- 7개 데이터셋(POPE/GQA/TextVQA/VQAv2/SQA/MME/MMBench) × {topk, energy best, statistical best} × {simple, weighted} × M2.
- 기존 topk 9행 유지 + adaptive 추가. **best τ는 데이터셋별 사용 권장(§4-3).**
- MMBench adaptive는 `model_vqa_mmbench.py` .meta 집계 보강 필요.

---

## 7. 인프라 · 산출물
- **실험 러너**: `exp_runner/worker.sh`(11컬럼 락기반+resume+채점), `launch_phase2A.sh`, cron+flock detach.
- **채점기**: `eval_pope.py`, GQA `eval.py`, `eval_textvqa`, `vqa_eval.py`, `mme_eval.py`, `mmbench_eval.py`.
- **분석**: `probe_dist.py`(분포), `analyze_healthy.py`(건강구간), `generate_phase2A_report.py`(곡선·best).
- **원자료**: `results.tsv`(기존 topk), `results_phase1.tsv`(E2), `results_phase2A.tsv`(210행), `results_phase2.tsv`(MME/MMBench 검증).
- **문서**: `phase1_final_report.md`, `phase2_part1_report.md`, `phase2_healthy_range.md`,
  `phase2_part2_A_FINAL.md`, `phase2_part2_A_comprehensive.md`, (본 문서) `EXPERIMENTS_MASTER_SUMMARY.md`.

> 협업: 클로드 웹에서 설계 → Claude Code가 구현·실험 → `develop_md/*.md`로 핸드오프. 단계별 확인 게이트.

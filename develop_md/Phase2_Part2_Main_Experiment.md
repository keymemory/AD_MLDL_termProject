# Phase 2 — Part 2: 본 실험 (stage1+2 전체 성능 매트릭스) — Claude Code 지시서

> **목적**: adaptive selection(energy/statistical) + K-Means 병합(simple/weighted)을 여러 데이터셋 ×
> M2(32/64/128)에서 돌려 **실제 성능(F1/Acc)**을 측정하고, 기존 결과(VisPruner/Ours-S/Ours-W)와 비교한다.
> **미팅 결론**("stage1+2 완전한 결과물 뽑고 다시 논의")에 답하는 핵심 실험.

---

## 0. 작업 환경 규칙 (먼저 읽고 전체 적용)

- **GPU 최대 활용**: 사용 가능한 GPU를 최대한 다 써서 병렬화. `nvidia-smi` 확인,
  **타 사용자 점유 GPU는 건드리지 마라**.
- **확인 질문 금지**: yes/no를 나에게 묻지 마라. 멈추지 말고 쭉 수행. 파괴적 작업은 백업 후 진행+기록.
- **오류 자동 처리**: 오류 시 멈추지 말고 분석·수정 후 `develop_md/`에 기록(무엇이 왜, 어떻게 고쳤는지).
- **문서화**: 각 파트·중간 결과·최종 결과를 `develop_md/`에 단계별 문서화.
- **백그라운드 안전 실행**: 본 실험은 며칠 걸린다. **detach(cron+flock 등 Part 1에서 검증된 방식)로
  백그라운드 실행**하고, VSCode/SSH 종료와 무관하게 완주하게 하라. resume(중단 시 이어쓰기) 보장.
- **회귀 안전**: topk default = 기존 VisPruner 비트동일 유지.

---

## 1. 배경 — 지금까지의 맥락 (Claude Code 참고)

### 1-1. 연구
LLaVA-1.5-7B 시각 토큰(576) 2단계 압축. Stage 1(VisPruner 선택)으로 M1개 보존 → Stage 2(Spherical
K-Means)로 최종 M2개(32/64/128) 병합. 변형: VisPruner(Stage1만) / Ours-S(simple) / Ours-W(weighted).

### 1-2. selection 방식 (Phase 1에서 구현 완료)
- **topk**: 기존 VisPruner. M1 고정(M2의 2배).
- **energy**: 누적 [CLS] attention 질량 ≥ τ 최소 토큰 = important. M1 자동.
- **statistical**: μ+kσ 이상치 = important. M1 자동.
- M1 = `clamp(round(n_imp/0.5), floor=M2, cap=384)`, r=0.5 고정.

### 1-3. Part 1에서 확정된 것 — ★ 각 M2별 건강 τ/k (가장 중요)
probe로 확인한 결과, **floor=M2라 M2가 커질수록 건강구간이 좁아진다**:

| M2 | energy 건강 τ (adapt≥90%) | statistical 건강 k (adapt≥90%) |
|---|---|---|
| 32 | 0.6 / 0.7 / 0.8 | 0.2 / 0.3 / 0.4 / 0.5 / 0.6 |
| 64 | 0.7 / 0.8 | 0.2 / 0.3 |
| 128 | 0.8 (단일) | **없음 (전 구간 붕괴 → 제외)** |

**이 표가 본 실험의 핵심 전제다. 각 M2마다 건강 τ/k가 다르므로 단일값 고정은 금지.**

### 1-4. 기존 결과 (보고서 표 2 — topk 기반, 이미 있음)
| M2 | Method | POPE | GQA | TextVQA | VQAv2 | SQA |
|---|---|---|---|---|---|---|
| 128 | VisPruner | 84.47 | 58.28 | 56.76 | 72.18 | 68.86 |
| 128 | Ours-S | 85.37 | 58.26 | 54.77 | 72.08 | 68.86 |
| 128 | Ours-W | 85.24 | 58.27 | 55.27 | 72.17 | 68.62 |
| 64 | VisPruner | 80.95 | 55.59 | 55.68 | 68.88 | 68.57 |
| 64 | Ours-S | 82.27 | 56.66 | 54.08 | 69.77 | 68.86 |
| 64 | Ours-W | 82.14 | 56.26 | 54.33 | 70.44 | 69.16 |
| 32 | VisPruner | 74.00 | 51.58 | 53.55 | 63.47 | 68.32 |
| 32 | Ours-S | 77.56 | 53.52 | 53.38 | 65.46 | 69.31 |
| 32 | Ours-W | 77.53 | 54.03 | 53.61 | 65.88 | 69.46 |

→ 이 9행(topk)은 **유지**하고, 여기에 **energy/statistical 행을 추가**해 같은 표에서 비교한다.

---

## 2. 본 실험 구성 — 두 묶음

### 묶음 A — 전 구간 세밀 스윕 (정당성용)

**목적**: "왜 건강구간 τ/k가 최적인지"를 성능 곡선으로 증명. **붕괴 구간(τ=0.3 등)도 성능 측정**해야
"양 끝은 실제로 성능이 낮다 → 가운데가 최적"이 데이터로 입증된다(probe 분포만으론 부족, 성능까지 필요).

- **데이터셋**: POPE, GQA, TextVQA (3개) — 강점 2개 + 약점(OCR) 1개. "강한 것만 골랐다"는 의심 차단.
- **energy**: τ = 0.3 / 0.4 / 0.5 / 0.6 / 0.7 / 0.8 / 0.9 (전 7개, 붕괴 포함)
- **statistical**: k = 0.2 / 0.3 / 0.4 / 0.5 / 0.6 / 0.7 / 0.8 (전 7개, 붕괴 포함)
- **병합**: simple_avg, weighted_avg
- **M2**: 32 / 64 / 128
- **제외**: M2=128 statistical (Part 1에서 전 구간 붕괴 확인 → 분포로 이미 증명, 성능 측정 생략)

### 묶음 B — 비교표 채우기 (best값, 전 데이터셋)

**목적**: 보고서 표 2 형식의 최종 비교표. 기존 topk + adaptive best를 모든 데이터셋에서.

- **데이터셋**: POPE, GQA, TextVQA, VQAv2, SQA, MME, MMBench (7개 전부)
- **selection**:
  - 기존 topk: VisPruner / Ours-S / Ours-W (※ 5개 기존 데이터셋은 표 2 값 재활용 가능,
    MME·MMBench는 새로 측정)
  - energy: 각 M2 **건강 best τ** (M2=32→τ=0.7, M2=64→τ=0.8, M2=128→τ=0.8) × simple/weighted
    - best τ는 묶음 A의 POPE 성능에서 가장 좋은 건강 τ로 확정(묶음 A 먼저 끝나면 그 값 사용)
  - statistical: 각 M2 **건강 best k** (M2=32→k 묶음A 최적, M2=64→k 묶음A 최적) × simple/weighted
    - M2=128 statistical 제외
- **M2**: 32 / 64 / 128

> **best τ/k 확정 방법**: 묶음 A의 POPE(대표 데이터셋) 성능이 가장 높은 건강 τ/k를 그 M2의 best로.
> 묶음 A를 먼저 완주 → best 확정 → 묶음 B 실행. (또는 동시에 큐에 넣되 묶음 B는 건강 best만)

---

## 3. 실행 방법 (기존 인프라 사용)

### 3-1. job tsv 작성 (11컬럼)
컬럼: `ID BENCH M2 CLUST M1 METHOD R SELMETHOD ETAU SK SROBUST`

예시 (묶음 A energy, POPE M2=64):
```
A-en03-64s  pope  64  1  128  simple_avg    0.5  energy  0.3  2.0  0
A-en07-64s  pope  64  1  128  simple_avg    0.5  energy  0.7  2.0  0
A-en08-64w  pope  64  1  128  weighted_avg  0.5  energy  0.8  2.0  0
...
```
예시 (묶음 A statistical, GQA M2=32):
```
A-st02-32s  gqa  32  1  64  simple_avg    0.5  statistical  0.5  0.2  0
A-st05-32w  gqa  32  1  64  weighted_avg  0.5  statistical  0.5  0.5  0
...
```
예시 (묶음 B best, MME M2=64 energy best τ=0.8):
```
B-en08-mme-64s  mme  64  1  128  simple_avg  0.5  energy  0.8  2.0  0
```

> **주의**:
> - CLUST=1, METHOD=simple_avg/weighted_avg, SELMETHOD=energy/statistical일 때 M1은 어차피 수식이
>   자동 결정하므로 tsv의 M1 컬럼은 placeholder(예: M2의 2배). 실제 M1은 .meta에 평균 기록됨.
> - statistical일 때 ETAU는 placeholder, energy일 때 SK는 placeholder.
> - **M2=128 statistical job은 생성하지 마라.**

### 3-2. MMBench adaptive 집계 보강 (Part 1 메모 반영)
Part 1에서 확인된 것: `model_vqa_mmbench.py`가 `_adaptive_log` 집계를 안 해서 adaptive selection 시
AVG_M1/floor/cap이 "-"로 기록됨. **묶음 B에서 MMBench로 energy/statistical을 쓰려면**
`model_vqa_mmbench.py`에도 `.meta` 집계(model_vqa_loader 방식)를 먼저 추가하라.

### 3-3. 실행
```bash
cd /home/jhlee/CLUST_KETI/AD_MLDL_termProject/term_project
# 묶음 A job tsv 작성 → 병렬 실행 (백그라운드 detach)
bash exp_runner/launch.sh exp_runner/exp_jobs_phase2_A.tsv   # 또는 검증된 cron+flock 패턴
# 완주 후 best τ/k 확정 → 묶음 B job tsv 작성 → 실행
```
- resume(25회 retry)·flock 동시쓰기 보호 그대로.
- 결과는 `results_phase2.tsv`에 13컬럼(+SELMETHOD/AVG_M1/AVG_R)으로 누적.

---

## 4. 결과 정리 (실험 후)

### 4-1. 묶음 A — 전 구간 성능 곡선
데이터셋(POPE/GQA/TextVQA) × M2(32/64/128) 별로, τ(또는 k)에 따른 성능 곡선 표/그래프:
```
POPE M2=64 energy:
  τ    0.3   0.4   0.5   0.6   0.7   0.8   0.9
  F1   ?     ?     ?     ?     ?(★)  ?(★)  ?
  adapt 0%   0%   19%   89%  100%  100%  28%
```
- **검증 포인트**: 건강구간(adapt≥90%)의 성능이 붕괴구간보다 높은가? → "건강구간=최적" 입증.
- 붕괴구간(τ=0.3 등)은 floor에 깔려 topk와 유사한 성능일 것으로 예상(확인).

### 4-2. 묶음 B — 최종 비교표 (보고서 표 2 확장형)
```
M2   Method              POPE  GQA  TextVQA  VQAv2  SQA  MME  MMBench
128  VisPruner(topk)     84.47 ...
     Ours-S(topk)        85.37 ...
     Ours-W(topk)        85.24 ...
     Ours-energy-S       ?     ...   ← 추가
     Ours-energy-W       ?     ...   ← 추가
     (statistical 제외)
64   VisPruner(topk)     80.95 ...
     ...
     Ours-energy-S/W     ?
     Ours-statistical-S/W ?         ← 추가 (M2=64 건강 best k)
32   ...                            ← energy + statistical 둘 다 추가
```
- **핵심 비교**: 같은 M2·데이터셋에서 **고정 selection(topk) vs adaptive(energy/statistical)** 직접 비교.
- adaptive가 topk 대비 우위인지, 어느 M2·데이터셋에서 그런지.

### 4-3. 문서화
- `develop_md/phase2_part2_result.md`: 묶음 A 곡선 + 묶음 B 비교표 + 해석.
- 실패/이상 케이스(붕괴 구간 성능, 예상과 다른 결과)도 기록 — "왜 그런지" 분석 포함.

---

## 5. done 조건

1. ✅ (필요시) MMBench adaptive 집계 보강
2. ✅ 묶음 A 완주: POPE/GQA/TextVQA × energy(τ 7개)/statistical(k 7개) × simple/weighted × M2(32/64/128),
   M2=128 statistical 제외
3. ✅ best τ/k 확정 (묶음 A POPE 성능 기준)
4. ✅ 묶음 B 완주: 7개 데이터셋 × topk + adaptive best × simple/weighted × M2
5. ✅ 결과 정리: 전 구간 성능 곡선 + 최종 비교표 + 해석 (`develop_md/`)

---

## 6. 핵심 체크포인트 (잊지 말 것)

- **각 M2의 건강 τ/k는 §1-3 표를 따른다.** 단일값 고정 금지.
- **M2=128 statistical 제외** (붕괴, 분포로 이미 증명).
- **붕괴 구간도 성능 측정**(묶음 A) — "건강구간이 왜 최적인지" 정당화용.
- **기존 topk 9행(표 2)은 유지**, 그 위에 adaptive 추가 비교.
- 백그라운드 detach + resume으로 며칠 걸려도 완주.
- 전 과정 문서화, 오류 자동 수정+기록.

> 묶음 A부터 백그라운드 실행하고, 진행 상황과 묶음 A 완료(+best τ/k 확정)를 보고해줘.
> 묶음 B는 best 확정 후 이어서 실행하거나, 동일 큐에 건강 best만 넣어 함께 돌려도 된다.
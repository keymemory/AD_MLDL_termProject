# 전체 실험 상세 정리 — 이유·목적·방법론·수식/구조·실험조건·결과

> LLaVA-1.5-7B 시각 토큰 2단계 압축 연구의 **모든 실험을 각각 6개 항목**(이유/목적/방법론/수식·구조/조건/결과)으로 정리.
> 공통 환경: LLaVA-1.5-7B(Vicuna-7B + CLIP-ViT-L/14-336) / greedy(temp=0) / fp16 / CUDA_LAUNCH_BLOCKING=1 / conda `vispruner`.
> 원자료: `results.tsv`(기존 topk), `results_phase1.tsv`(E2), `results_phase2A.tsv`(묶음A), `results_phase2B.tsv`(묶음B), `results_phase3_merge.tsv`(taskaware).

---

## 공통 방법론 (모든 실험의 토대)

```
이미지 → CLIP-ViT-L/14-336 → 576 visual token
  ① Stage1 (VisPruner): [CLS]-attention 상위 important + cosine 다양성 diverse → M1개 보존
  ② Stage2 (Spherical K-Means): M1개를 M2개로 병합(대표벡터) → M2개(32/64/128) → LLM
```
- **selection_method**: `topk`(기존 VisPruner, M1=2×M2 고정) / `energy`(누적 [CLS]attn 질량≥τ) / `statistical`(μ+kσ 이상치)
- **M1 자동결정**: `M1 = clamp(round(n_imp / 0.5), floor=M2, cap=384)`, r=0.5
- **merge_method**: `simple_avg` / `weighted_avg`([CLS]attn 가중) / `taskaware`(Phase3)
- **floor의 의미**: M1=M2이면 Stage2 병합 생략, important(소수)+diverse(다수)로 채움

---

# 실험 1 — 회귀 검증 (Regression, topk = VisPruner 비트동일)

**이유** — adaptive selection(energy/statistical)을 새로 추가하면서, 신규 코드가 기존 VisPruner 경로를
훼손하지 않았음을 먼저 증명해야 이후 모든 비교가 신뢰를 얻는다.

**목적** — `selection_method="topk"` default가 기존 VisPruner와 **출력까지 완전히 동일(비트동일)**함을 확인.

**방법론** — 코드 수정 전 구코드로 POPE를 추론해 answers를 스냅샷 → 신규 코드(topk default)로 동일 추론 →
question_id별 답변 텍스트를 대조(answer_id는 랜덤이라 제외). 2단계(300 subset → full 8910).

**수식/구조** — topk 경로는 `M1=2×M2`, `important=int(M1·r)`, diverse는 cosine bipartite 제거 — 기존 로직 그대로.
분기 `if selection_method=="topk"` 안에서 한 글자도 안 바꿈.

**실험 조건** — POPE 300 subset + full 8910, M2=128, greedy.

**결과** — **text_diff = 0 / 8910**(전부 일치), F1 = **0.8446517939988288** 구코드 ↔ 신코드 **소수점 16자리 동일**.
→ 회귀 안전성 확립. 이후 adaptive 실험의 baseline 신뢰 확보.

---

# 실험 2 — E2 본실험 (adaptive selection 첫 성능 측정)

**이유** — adaptive selection이 실제로 작동하고, 고정(topk)과 비교해 성능이 어떤지 첫 정량 확인이 필요.

**목적** — energy τ / statistical k 스윕으로 (a) adaptive가 고정을 능가하는지, (b) 같은 토큰 예산에서 어떤지 규명.

**방법론** — POPE·GQA(M2=64)에서 energy τ={0.6,0.7,0.8}, statistical k={0.3,0.4,0.5} × simple/weighted 측정.
고정 baseline(topk M1=128)과 비교. `.meta`로 AVG_M1 기록.

**수식/구조** — energy: n_imp = (누적 attn 질량 ≥ τ 최소 토큰), M1=clamp(round(n_imp/0.5), M2, 384).

**실험 조건** — POPE(8910)·GQA(12578), M2=64, energy 0.6~0.8 / statistical 0.3~0.5.

**결과** (POPE F1):
| 설정 | POPE F1 | GQA Acc | AVG_M1 |
|---|---|---|---|
| topk baseline (M1=128) | 0.8227 | 56.66 | 128 |
| energy τ=0.7 weighted | 0.8269 | 57.59 | 163 |
| energy τ=0.8 weighted | **0.8398** | 57.41 | 263 |
| statistical k=0.3 simple | 0.8129 | — | 97 |
→ energy τ=0.8이 baseline 상회(0.8398)하나 **M1이 2배(263)**. τ* 역산(실험8)으로 같은 M1=128 맞추면 energy≈baseline.
**핵심 해석**: 적응의 가치는 "무조건 우월"이 아니라 **토큰-성능 trade-off를 τ 하나로 부드럽게 탐색 + 고정을 특수해로 포함**.

---

# 실험 3 — 데이터셋 확장 (MME·MMBench 추가)

**이유** — 교수님 피드백(데이터셋 확대) + 5개 벤치만으론 일반성 부족. classification 계열(MME/MMBench) 추가로 보강.

**목적** — MME·MMBench를 기존 실험 인프라(worker.sh)에 통합하고, topk로 정상 작동을 검증.

**방법론** — MME: parquet에서 이미지 추출(카테고리별 경로 체계 상이 처리) + `mme_eval.py`(parquet GT 직접 채점).
MMBench: `model_vqa_mmbench.py`에 selection 인자 + `mmbench_eval.py`(A/B/C/D 매칭). worker.sh 11컬럼 통합.

**수식/구조** — MME score = Σ_category (acc + acc_plus)×100 (perception 10 + cognition 4).
MMBench = 선택지 정확도(single-pred-prompt).

**실험 조건** — MME 2374문항(1187 이미지×2), MMBench dev 4377문항. topk M2=128.

**결과** — MME total **1752.57**, MMBench Acc **72.13** (topk M2=128). 정상 채점·기록 확인 → 통합 성공.

---

# 실험 4 — 건강구간 probe (M2별 건강 τ/k)

**이유** — floor=M2이므로 M2가 커지면 같은 τ라도 floor에 깔려 adaptive가 topk와 같아질 위험. 본실험 전 각 M2의
"수식이 실제로 적응하는" τ/k 구간을 알아야 한다.

**목적** — M2=32/64/128 각각에서 energy τ(0.3~0.9)/statistical k(0.2~0.8)의 adapt%(floor/cap 미도달 비율)를 측정.

**방법론** — `probe_dist.py`(LLM 디코딩 생략, vision tower + selection만 → 수십 배 빠름)로 POPE 300 subset에서
이미지별 n_imp/M1/floor/cap 분포 측정. 검증: probe k=2.0 = 실제 추론과 일치.

**수식/구조** — adapt% = 100 − floor% − cap%. 건강 기준 adapt ≥ 90%.

**실험 조건** — POPE 300 subset, 42조합(3 M2 × (7 τ + 7 k)).

**결과**:
| M2 | energy 건강 τ | statistical 건강 k |
|---|---|---|
| 32 | 0.6 / 0.7 / 0.8 | 0.2~0.6 |
| 64 | 0.7 / 0.8 | 0.2 / 0.3 |
| 128 | 0.8 (단일) | **없음(전 구간 붕괴)** |
→ M2 클수록 건강구간 급감, **M2=128 statistical은 붕괴** → 본실험 제외. 각 M2별 건강 τ/k를 본실험 전제로 확정.

---

# 실험 5 — 묶음 A (전 구간 성능 곡선)

**이유** — "왜 건강구간이 최적인지"를 성능 곡선으로 증명하려면 붕괴 구간(floor/cap)도 성능을 측정해야 한다(분포만으론 부족).

**목적** — POPE/GQA/TextVQA × τ/k 전 구간 성능으로 (a) 건강구간 우위 검증, (b) **태스크별 최적 τ 차이** 규명.

**방법론** — energy τ(0.3~0.9)/statistical k(0.2~0.8) × simple/weighted × M2(32/64/128), M2=128 stat 제외. 210 job.
cron+flock detach, GPU 병렬. 완주 210/210(4.5일).

**실험 조건** — POPE 8910 / GQA 12578 / TextVQA 5000, 210 job.

**결과** — energy weighted 최적 τ (M2=32 강압축 기준):
| 데이터셋 | 최적 τ | 패턴 | 핵심 수치 |
|---|---|---|---|
| POPE | 0.8 (8.2×) / 약압축은 floor | 압축률 의존 | M2=32: 0.807, M2=128 floor τ0.3: 0.860 |
| **GQA** | **0.9/0.8 (단조↑)** | **important 친화** | M2=32: 51.85(floor)→56.21(τ0.9), **+4.4** |
| **TextVQA** | **0.5 (볼록)** | **병합 손상** | M2=32: τ0.5=53.63 최고, τ0.8=52.39 하락 |
→ **★ 핵심 노벨티: 최적 τ가 태스크별로 정반대**(GQA↑, TextVQA 중간, POPE 압축률의존). 압축 강할수록 차이 극대화.
상세 곡선: `phase2A_m1m2_ratio_by_dataset.md`, `phase2_part2_A_FINAL.md`.

---

# 실험 6 — 묶음 B (7개 벤치 최종 비교표)

**이유** — 묶음 A best τ/k로 나머지 데이터셋의 adaptive를 측정해 7개 벤치 최종 비교표를 완성.

**목적** — 고정 selection(topk) vs adaptive(energy/statistical)를 전 데이터셋·M2에서 직접 비교.

**방법론** — VQAv2/SQA/MME/MMBench × energy τ=0.8 / statistical k=0.2 × simple/weighted × M2. MME/MMBench는
topk 기준선(VisPruner/Ours-S/Ours-W)도 측정. 58 job, cron detach. MMBench는 `model_vqa_mmbench.py` 집계 보강.

**실험 조건** — VQAv2 6000 / SQA 4241 / MME 2374 / MMBench 4377.

**결과** (energy τ=0.8 weighted vs 기존 topk 최고):
| 벤치 | M2=32 adaptive | M2=32 topk 최고 | Δ | M2=128 adaptive | M2=128 topk |
|---|---|---|---|---|---|
| VQAv2 | **68.28** | 65.88 | **+2.40** | 72.63 | 72.18 |
| MMBench | **70.87** | 69.43 | **+1.44** | 72.61 | 72.38 |
| SQA | 68.62 | 68.57(A)/69.16(C) | ≈ | 69.01 | — |
| MME | 1670.77 | 1648.40 | +22 | 1751.97 | **1773.19**(topk 우위) |
→ **강압축(M2=32)에서 adaptive가 topk 크게 상회**(VQAv2 +2.4, MMBench +1.4) — 묶음 A POPE/GQA와 일관.
약압축(M2=128)은 대체로 대등. **MME만 topk(VisPruner) 우위**(1773 vs 1751) — MME에 OCR 하위태스크가 있어
병합이 불리한 것으로 추정(TextVQA와 유사). 최종 비교표는 작업 C에서 완성 예정.

---

# 실험 7 — Phase3 실험 A (Task-Aware / Merge-Distortion 병합)

**이유** — 묶음 A에서 발견한 **TextVQA(OCR) 병합 손상**의 해법. "발견 → 해법"으로 노벨티 확장.

**목적** — 병합 시 정보 손실이 큰 토큰(글자)을 자동 보존해, OCR 성능을 회복하되 의미 태스크는 안 해치는지 검증.

**방법론** — K-Means 병합 후 각 토큰의 병합 왜곡(centroid와 방향차)을 계산, 통계적 이상치인 토큰은 병합 제외·원본
보존. 나머지는 weighted 병합. **M2 불변**. k_d(0.5~2.0) 스윕. baseline = 묶음A energy τ=0.8 weighted.

**수식/구조**:
```
Distortion(i) = 1 − cos(x_i, c_{cluster(i)})          # centroid와 방향 차이
preserve = { i : Distortion(i) ≥ μ_d + k_d·σ_d }       # 왜곡 이상치 → 원본 보존
최종 = preserve p개(원본) + (M2−p)개(병합)  = M2 불변,  p ≤ M2·0.5 상한
```

**실험 조건** — TextVQA/POPE/GQA × k_d(0.5/1.0/1.5/2.0) × M2(32/64/128), energy τ=0.8. 36 job.

**결과** (best Δ vs weighted baseline):
| 데이터셋 | M2=32 | M2=64 | M2=128 | 판정 |
|---|---|---|---|---|
| **TextVQA**(핵심) | **+0.15** | **+0.25** | **+0.38** | ✅ 모든 M2 회복 |
| POPE(대조) | −0.007 | +0.003 | +0.003 | ✅ 중립 |
| GQA(대조) | −0.55 | +0.11 | +0.29 | 🟡 M2=32만 근소 하락 |
→ **부분 성공**: TextVQA OCR 모든 M2 회복(preserve된 글자 토큰 보존 효과), POPE 중립, GQA 대체로 중립.
가설("병합 손상 토큰 자동 보존 → OCR 회복") 지지. 개선폭은 작지만(+0.15~0.38) 일관. 상세: `phase3_taskaware_merge_result.md`.

---

# 실험 8 — Phase3 실험 B (τ* 역산)

**이유** — "왜 고정값 M1=128인가"에 답해 기존 고정 실험의 정당성을 완결.

**목적** — 데이터셋 평균 M1이 128(=기존 고정 topk M1)이 되는 energy τ*를 역산 → 고정=adaptive의 특수해 증명.

**방법론** — 묶음 A의 (τ, AVG_M1) 측정점을 선형 보간해 AVG_M1=2×M2 되는 τ* 산출. **추가 추론 불필요**(분포 재활용).

**수식/구조** — τ* = interp{(τ, AVG_M1)}에서 AVG_M1 = 2×M2 되는 τ.

**실험 조건** — POPE/GQA/TextVQA × M2(32/64/128), 묶음 A 데이터.

**결과**:
| 데이터셋 | τ* (M1=128, M2=64) | M2별 τ* (M1=2×M2) |
|---|---|---|
| POPE | 0.646 | 0.52 / 0.65 / 0.79 |
| GQA | 0.637 | 0.52 / 0.64 / 0.78 |
| TextVQA | 0.692 | 0.55 / 0.69 / 0.83 |
→ **고정 M1=128이 데이터셋별로 다른 τ*(0.637~0.692), M2별로도 다름(0.52~0.83)** → 고정값의 임의성.
검증: τ* 지점 성능(≈0.815) ≈ 고정 topk(0.8227), 실험2와 일관. → **고정 = adaptive의 τ=τ* 특수해** 수치 증명.
상세: `phase3_tau_star_result.md`.

---

## 종합 — 실험 흐름과 결론

| # | 실험 | 핵심 결론 |
|---|---|---|
| 1 | 회귀 검증 | topk = VisPruner 비트동일(F1 16자리) — 신뢰 토대 |
| 2 | E2 | 같은 M1에선 적응≈고정, 적응은 τ로 trade-off 탐색 |
| 3 | 데이터셋 확장 | MME/MMBench 통합(1752.57 / 72.13) |
| 4 | 건강구간 probe | M2 클수록 건강구간 급감, M2=128 stat 붕괴 |
| 5 | **묶음 A** | **★ 최적 τ 태스크별 정반대**(GQA↑/TextVQA중간/POPE압축률의존) |
| 6 | 묶음 B | 강압축서 adaptive > topk(VQAv2 +2.4), MME만 topk 우위 |
| 7 | **task-aware 병합** | **★ OCR 회복(TextVQA 전 M2 +), 발견→해법 확장** |
| 8 | τ* 역산 | 고정 = adaptive의 특수해(τ* 데이터셋별 상이) |

**최종 노벨티**: (정당성) 회귀·τ*로 고정을 adaptive의 특수해로 흡수 + (기여1) **최적 압축이 태스크 의미구조에
따라 질적으로 다르다** + (기여2) **task-aware 병합으로 OCR 손상을 자동 회복**(발견→해법).

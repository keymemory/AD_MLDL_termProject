# Adaptive Visual Token Selection — 종합 보고서 (Phase 2 Part 2 묶음 A)

> LLaVA-1.5-7B 시각 토큰 2단계 압축에서 **adaptive selection(energy/statistical)** 의 전 구간 성능을
> 측정·분석한 종합 문서. 목적·방법론·실험 설계·현재 결과(POPE 완료분)·해석을 self-contained로 정리.
> 데이터 출처: `results_phase2A.tsv`(86/210 jobs, 진행 중), `phase2_part2_A_partial.md`(자동 곡선표).

---

## 1. 연구 배경 및 목적

### 1-1. 문제
VLM(LLaVA-1.5-7B)은 CLIP-ViT-L/14-336이 이미지를 **576개 visual token**으로 변환해 LLM에 주입한다.
LLM self-attention은 시퀀스 길이에 O(n²)라, 576 토큰이 비용·지연을 키운다. 다수가 배경·중복이라
**시각 토큰을 줄이되 성능을 지키는 것**이 과제다.

### 1-2. 기존 방법과 한계
- **VisPruner(ICCV'25)**: [CLS] attention으로 important 토큰, cosine 유사도로 diverse 토큰을 선택
  (pruning-only). 그러나 important **개수를 고정 비율(r=0.5, M1=2×M2)** 로 잡는다 — 이미지 복잡도와
  무관하게 동일 개수를 important로 본다.
- **본 연구의 Two-Stage**: ① Stage1에서 M1개 보존 → ② Stage2(Spherical K-Means)로 M2개로 병합.
  "넓게 보존 후 의미 단위 병합"으로 동일 토큰 수 대비 정보 밀도를 높인다(simple/weighted avg).

### 1-3. 이번 핵심 질문 — adaptive selection
important **개수를 이미지마다 자동 결정**하면(고정 대신) 더 나은가? 두 수식을 제안:
- **energy**: 내림차순 [CLS] attention 누적 질량이 τ를 넘는 최소 토큰 = important (적분 관점)
- **statistical**: μ+kσ(또는 median+k·MAD) 이상치 = important (전역 통계 관점)

**목적**: adaptive(energy/statistical)가 고정(topk=VisPruner)보다 우위인지, 어떤 M2·데이터셋·압축률에서
그러한지를 **전 구간 τ/k 성능 곡선**으로 규명한다.

---

## 2. 방법론

### 2-1. 파이프라인
```
576 토큰 → [Stage1] [CLS]attn important(M1·r) + cosine diverse(M1·(1-r)) → M1개
        → [Stage2] Spherical K-Means: M1 → M2 병합(simple/weighted avg) → M2개 → LLM
```

### 2-2. important 개수 자동 결정 (energy/statistical)
- n_imp = (energy) 누적질량≥τ 최소 개수 / (statistical) μ+kσ 초과 개수 — **이미지마다 다름**
- **M1 = clamp(round(n_imp / 0.5), floor=M2, cap=384)**, r=0.5 고정
  - floor=M2: n_imp가 너무 작으면 M1이 M2 밑으로 → M2로 클램프(병합할 토큰 부족 방지)
  - cap=384: n_imp 폭주 시 배경·노이즈 과다 → 384로 클램프
- **topk(기존)**: n_imp를 안 쓰고 M1=2×M2 고정. (selection_method=topk, 회귀 시 VisPruner와 비트동일)

### 2-3. ★ floor/cap의 의미 (결과 해석의 열쇠)
- **floor에 걸리면(M1=M2) Stage2 병합이 일어나지 않는다**(조건 `M2 < M1`이 거짓). 이때는 Stage1만으로
  M2개를 고르며, important=수식상 소수 + **diverse 다수**로 채워진다.
- cap에 걸리면 M1=384로 고정(과다 보존).
- 즉 floor/cap 구간은 "수식이 적응하지 못하고 경계값에 깔린" 상태다.

---

## 3. 실험 설계 (묶음 A — 전 구간 세밀 스윕)

### 3-1. 왜 붕괴 구간까지 측정하나
"건강구간(adapt≥90%)이 왜 최적인지"를 **성능 곡선**으로 증명하려면, 붕괴 구간(floor/cap) 성능도 측정해
"양 끝은 실제로 낮다 → 가운데가 최적"을 데이터로 보여야 한다(분포만으론 부족).

### 3-2. 매트릭스 (210 job)
| 축 | 값 |
|---|---|
| 데이터셋 | POPE, GQA, TextVQA (강점 2 + 약점 OCR 1) |
| energy τ | 0.3 / 0.4 / 0.5 / 0.6 / 0.7 / 0.8 / 0.9 (붕괴 포함) |
| statistical k | 0.2 / 0.3 / 0.4 / 0.5 / 0.6 / 0.7 / 0.8 (붕괴 포함) |
| 병합 | simple_avg, weighted_avg |
| M2 | 32 / 64 / 128 |
| 제외 | **M2=128 statistical** (Part1서 전 구간 붕괴 확인) |

### 3-3. Part1에서 확정한 건강구간 (probe, adapt≥90%)
| M2 | energy 건강 τ | statistical 건강 k |
|---|---|---|
| 32 | 0.6 / 0.7 / 0.8 | 0.2~0.6 |
| 64 | 0.7 / 0.8 | 0.2 / 0.3 |
| 128 | 0.8 (단일) | 없음(제외) |
→ **floor=M2라 M2가 커질수록 건강구간이 좁아진다**(M2=128 statistical은 붕괴). 각 M2별 건강 τ/k를 본실험 전제로.

### 3-4. 실행 인프라
- `exp_runner/worker.sh` 락기반(11컬럼 job tsv) + resume(answers 완주 시 추론 skip, 25회 retry).
- **cron+flock detach**(Part1 검증): VSCode/SSH 종료와 무관하게 며칠 완주. GPU 0·1 각 worker 2개.
- 결과 `results_phase2A.tsv`(15컬럼: 성능 + AVG_M1/AVG_R/floor%/cap%).

---

## 4. 결과 (현재 86/210, POPE 70개 전체 완료)

### 4-1. best τ/k (건강구간 내 POPE 성능 최고 → 묶음B용)
| 수식 | M2=32 | M2=64 | M2=128 |
|---|---|---|---|
| **energy** | τ=0.8 (0.807) | τ=0.8 (0.836) | τ=0.8 (0.857) |
| **statistical** | k=0.2 (0.797) | k=0.2 (0.825) | (제외) |
(weighted 기준 POPE F1)

### 4-2. POPE energy 곡선 (τ별 F1 / floor%)
| M2 | τ=0.3 | 0.4 | 0.5 | 0.6 | 0.7★ | 0.8★ | 0.9 |
|---|---|---|---|---|---|---|---|
| 32 (w) | 0.772 | 0.748 | 0.755 | 0.782 | 0.801 | **0.807** | 0.805 |
| 64 (w) | 0.833 | 0.824 | 0.813 | 0.815 | 0.832 | 0.836 | **0.842** |
| 128 (w) | **0.860** | 0.860 | 0.853 | 0.844 | 0.850 | 0.855 | 0.857 |
| _floor%(M2=128)_ | 100 | 100 | 100 | 86 | 20 | 0 | 0 |
(★ = Part1 건강구간. M2=64는 cap에 가까운 τ=0.9가 최고, M2=128은 floor τ=0.3이 최고)

### 4-3. ★ 핵심 발견 — floor 붕괴 구간이 건강구간만큼/더 높다 (POPE)
지시서의 가정("붕괴구간 성능이 낮아 건강구간=최적")이 **POPE에서는 성립하지 않는다**:
- **POPE M2=128: τ=0.3(floor 100%) = 0.860 > τ=0.8(건강) = 0.857**
- POPE M2=64: τ=0.3(floor) 0.833 ≈ τ=0.8(건강) 0.836, 오히려 τ=0.9(cap 근처) 0.842가 최고
- POPE M2=32: τ=0.3(floor) 0.772 < τ=0.8(건강) 0.807 — **여기선 건강구간 우위**

---

## 5. 해석 및 함의

### 5-1. 왜 POPE에서 floor가 강한가
floor(M1=M2)이면 **Stage2 병합이 생략**되고, Stage1에서 important=energy 소수(n_imp~6) + **diverse 다수**로
M2개를 채운다. 즉 **diverse(배경/맥락) 토큰 비율이 매우 높아진다.** POPE는 "그럴듯하지만 없는 객체"
거부가 핵심인데, 배경 맥락이 많을수록 환각 거부에 유리하다 → floor가 오히려 강하다.

### 5-2. 압축률 의존성 (Phase1 E2와 일관)
- **M2=32(강압축)**: 건강 τ=0.8이 floor를 명확히 이김(0.807 vs 0.772). 토큰이 극히 적을 때는
  "넓게 보존 후 병합"이 "소수 선택"보다 정보 손실이 작다.
- **M2=128(약압축)**: floor(diverse 다수)가 이미 충분 → adaptive 병합의 추가 이점이 작거나 역전.
- → **adaptive의 가치는 압축이 강할수록(작은 M2) 커진다.**

### 5-3. 정직한 포지셔닝
POPE만 보면 "adaptive가 topk를 압도한다"고 말할 수 없다. adaptive의 진짜 가치는 (a) **단일 τ/k로
토큰-성능 trade-off를 부드럽게 탐색**하고 (b) **고정을 특수해로 포함**하며 (c) **강압축에서 우위**인 데 있다.
- ⏳ **GQA(공간추론)·TextVQA(OCR)는 important 토큰이 더 중요한 태스크**라, floor(diverse 위주)보다
  건강구간(important 보존+병합)이 뚜렷이 우위일 가능성이 있다. 이게 확인되면 "태스크 의미구조에 따라
  최적 selection이 질적으로 다르다"는 본 연구의 노벨티가 강화된다.

---

## 6. 진행 상황 및 다음 단계

- **현재**: 86/210 (40.9%). POPE 완료, GQA M2=32 진행 중. GPU 0·1(각 worker 2개), cron+flock detach.
  ETA 약 2~2.5일.
- **다음(자동)**: 완주 시 `phase2_part2_A_result.md` 자동 생성(전체 곡선 + best τ/k).
- **묶음 B(별도)**: best τ/k(energy τ=0.8, statistical k=0.2)로 7개 데이터셋(POPE/GQA/TextVQA/VQAv2/SQA/
  MME/MMBench) 비교표. 기존 topk 9행 유지 + adaptive 추가. (MMBench는 `model_vqa_mmbench.py`에
  .meta 집계 보강 필요 — Part1 메모.)
- **관전 포인트**: GQA/TextVQA에서 §4-3의 floor 반전이 사라지고 건강구간이 우위인가? → 태스크별
  최적 selection 차이(노벨티) 입증 여부.

> 산출물: `results_phase2A.tsv`(원자료), `phase2_part2_A_partial.md`(자동 곡선표),
> `exp_runner/generate_phase2A_report.py`(곡선·best 생성기), `phase2_part1_report.md`(MME/MMBench·건강구간).

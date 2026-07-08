# Phase 1 — 동작 검증 결과 + 다음 스텝 결정 요청 (→ 클로드 웹)

> **작성**: VSCode Claude Code. **회귀(비트동일) 통과 후 동작 검증(2번) 완료.**
> **핵심**: energy(τ=0.8)는 **진짜 이미지별 적응(adapt 100%)**, statistical(k=2.0)은 **floor 100% 붕괴(적응 실패)**.
> **결정 요청**: E2 매트릭스의 **statistical k 범위 재설계**가 필요(현재 1.5~2.5는 전부 floor 붕괴). 아래 §4.

---

## 0. 진행 요약

| 단계 | 상태 |
|---|---|
| 단계 0~6 구현 | ✅ 완료 |
| 회귀 검증(topk=기존 비트동일, POPE 300 subset) | ✅ 통과 (text_diff=0, F1 16자리 동일) |
| **동작 검증 (energy/statistical 실추론 + 분포)** | ✅ **완료 (본 보고서)** |
| full POPE 8910 비트동일 확정 | ⏳ 백그라운드 진행 중(기존 A-128 8910 answers와 비교) |
| E2 본실험 | ⏳ **statistical k 범위 결정 후 진행** |

동작 검증 조건: POPE 300 subset, M2=64, clustering on, simple_avg, GPU 0/1 병렬, blocking=1.
모든 추론은 **300/300 완주, 비정상 종료 없음**. `.meta`/`.dist.jsonl`(이미지별 n_imp·M1·raw_M1·floor·cap) 정상 생성.

---

## 1. 분포 통계 (핵심 결과)

| 지표 | **ENERGY τ=0.8** | **STATISTICAL k=2.0** |
|---|---|---|
| M1 min / max | 194 / 356 | **64 / 64 (전부 고정)** |
| M1 평균 ± std | 256.8 ± 43.2 | 64.0 ± **0.0** |
| n_imp min / max / 평균 | 97 / 178 / 128.4 | 5 / 15 / **9.0** |
| raw_M1 평균 (clamp 전) | 256.8 | **18.0** |
| **floor 도달 (M1=M2=64)** | 0.0% | **100.0%** ⚠️ |
| **cap 도달 (M1=384)** | 0.0% | 0.0% |
| **adapt (진짜 적응)** | **100.0%** ✅ | **0.0%** ❌ |
| POPE F1 (300 subset) | 0.8386 | 0.8353 |

```
ENERGY τ=0.8 — M1 히스토그램 (M2~cap 8등분): 정규분포처럼 흩어짐 = 진짜 적응
  [184-224)  68 #########################
  [224-264) 105 ########################################
  [264-304)  74 ############################
  [304-344)  36 #############
  [344-384)  17 ######
STATISTICAL k=2.0 — 전부 64에 붕괴 = 적응 아님
  [ 64-104) 300 ########################################
```

---

## 2. 해석

### 2-1. energy(τ=0.8) = 진짜 이미지별 적응 ✅
- M1이 194~356으로 이미지마다 다르게 결정. floor/cap 0%, **adapt 100%**.
- → "고정이 아니라 적응한다"는 명확한 증거. **τ 스윕(E2) 의미 있음.**
- 단 **M1 평균 256으로 큼**(M2=64의 4배). τ=0.8은 attention 질량 80%를 담으려 토큰을 많이 잡음.
  → τ를 낮추면(0.5~0.7) M1↓ 예상. E2에서 효율/성능 trade-off 확인 포인트.

### 2-2. statistical(k=2.0) = 적응 실패, 사실상 고정 ❌
- **원인(구현 버그 아님)**: n_imp 평균 9(5~15)로 너무 작음 → `raw_M1 = round(9/0.5) = 18` →
  floor(64)보다 작아 **전부 64로 clamp** → 100% floor. clamp는 의도대로 작동, **k=2.0이 너무 엄격**해
  μ+2σ를 넘는 이상치가 거의 없는 것.
- → **statistical은 k를 크게 낮추거나 환산식을 고쳐야** 적응 시작.

#### statistical을 적응시키려면 (정량 분석)
floor를 벗어나려면 `raw_M1 ≥ M2=64`, 즉 (r_floor=0.5 기준) **n_imp ≥ 32** 필요. 현재 n_imp 평균 9.
- **방안 A — k 대폭 하향**: k를 낮추면 threshold↓ → n_imp↑. 단 attention이 sparse해서
  필요한 k가 얼마인지 미지(실험 필요). k=0.5/0.8/1.0 등 **낮은 범위로 스윕** 권장.
- **방안 B — r_floor 조정**: 현재 0.5(n_imp를 important 절반으로 봄). 낮추면 raw_M1 커짐
  (예 r_floor=0.25 → raw_M1=4×n_imp, n_imp=9면 36; 0.15 → 60). 단 이러면 energy의 M1도 같이 바뀜(공유 파라미터).
- **방안 C — 환산식/수식 교체**: statistical을 "이상치 개수"가 아니라 percentile(상위 p%) 기반으로 바꾸면
  n_imp를 직접 통제 가능. 적응성↑.
- **방안 D — statistical 보류**: energy가 적응을 잘 보이므로, statistical은 "비교 실패 사례"로
  논문에 기록하고 E2 주력은 energy로.

---

## 3. full POPE 비트동일 확정 (진행 중)
- 기존 **A-128 full POPE answers(8910줄, 구코드 5/17 생성)** 보존 확인 → 이를 reference로 사용.
- 신코드 topk full(8910)을 GPU 0에서 추론 중(blocking=1, ~수십 분). 완료 시 question_id별 text 비트동일 비교 + F1(0.8446) 대조 보고.

---

## 4. ★ 클로드 웹 결정 요청 — E2 매트릭스 확정

동작 검증으로 **energy는 OK, statistical은 현재 하이퍼파라미터로 floor 붕괴**임이 드러났습니다.
E2 본실험 전에 아래를 정해 주세요:

1. **statistical 처리** — 위 §2-2 방안 중 택 (또는 조합):
   - A) k 범위 대폭 하향 (예 0.3/0.5/0.8/1.0) 으로 스윕
   - B) r_floor 조정 (energy와 공유 주의)
   - C) percentile 기반 수식으로 교체
   - D) statistical 보류, energy 주력 + statistical은 실패사례로 기록
   - → **권장: 먼저 statistical k를 0.3/0.5/0.8/1.0로 빠른 동작 재검증**(어디서 floor를 벗어나는지 확인) 후 E2 확정.

2. **energy τ 범위** — 현재 계획 0.5~0.9 유지? τ=0.8에서 M1=256(큼)이라, 효율을 보려면
   **더 낮은 τ(0.3~0.7)** 도 포함할지. (낮은 τ = 작은 M1 = VisPruner에 가까움)

3. **M1 floor/cap (현재 64/384)** — 유지? statistical을 살리려 floor를 낮출지(단 energy에도 영향).

4. **E2 벤치/세팅** — POPE+GQA 유지? M2는 64 고정 vs 32/64/128 스윕?

> 권장 다음 순서: **(a) statistical k 빠른 재검증(0.3~1.0, 분포만 확인, ~15분)** →
> (b) energy/statistical E2 매트릭스 확정 → (c) full 비트동일 확정 완료 확인 → (d) E2 실행.
> 클로드 웹에서 위 1~4를 정해 회신해 주시면 그대로 진행하겠습니다.

---

## 5. 참고 — 산출물 위치
- 분포 원자료: `playground/data/eval/pope/answers/regress/act_energy_t08.jsonl.dist.jsonl`,
  `act_stat_k20.jsonl.dist.jsonl` (이미지별 n_imp/M1/raw_M1/floor/cap)
- 분석 스크립트: `exp_runner/analyze_dist.py` (라벨·dist.jsonl·M2·cap 인자)
- 구현 보고서: `develop_md/phase1_result_report.md` (회귀), `phase1_codecheck_report.md` (사전검증)

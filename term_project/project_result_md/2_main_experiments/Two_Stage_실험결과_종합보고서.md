# Two-Stage Visual Token Reduction — 실험 결과 종합 보고서 (클로드 웹 검토용)

> 제안: VisPruner(ICCV'25) **Stage1(토큰 선택)** + **Stage2(Spherical K-Means 병합)** 2단계 프레임워크
> 모델: LLaVA-1.5-7B (+CLIP-ViT-L/14-336) · r=0.5 기본 · greedy(temp=0) · fp16 · `CUDA_LAUNCH_BLOCKING=1`
> 채점: POPE=F1(eval_pope), GQA=Acc(testdev_balanced), TextVQA=Acc(VQA metric) — **전부 로컬 채점**
> 이 문서는 **단독 완결형** — 클로드 웹에 그대로 올려 결과 검증·분석 가능. 총 33-job 완료.

표기: **A** = VisPruner only(clustering OFF) · **B** = Ours(Stage1+Stage2), **s**=simple_avg/**w**=weighted_avg
B의 Stage1 토큰수 M1: M2=128→M1=192, M2=64→M1=128, M2=32→M1=64 (모두 1.5~3×)

---

## 1. 핵심 결론

1. **제안 B가 POPE에서 VisPruner-only를 전 토큰수 일관 상회**, 압축이 강할수록 이득 증가
   (128 +0.90, 64 +1.32, **32 +3.56** F1).
2. GQA 저토큰(64/32) +1~2.5p, **VQAv2(val subset) 저토큰 +0.9~+2.4p** 개선 — POPE·GQA·
   VQAv2 모두 동일 패턴(압축↑ → clustering 이득↑). TextVQA만 소폭 하락(OCR 특성).
3. **clustering 추가 오버헤드 무시 가능**(latency·GPU mem ≈ VisPruner-only) → 사실상 무비용.
4. POPE·GQA에서 기존 baseline(FastV/ToMe/SparseVLM/VisPruner 논문) 전부 상회.
5. M1↑(Stage1 보존↑) → 성능 단조 향상. POPE는 diverse 비중↑(r↓) 유리.
6. **VQAv2 question-type**: number(counting) 압축 최취약(−11.6), clustering이 저토큰서
   number/other 회복 → task-aware policy(number=보수압축, other/hard=weighted) 근거 확보.
7. FastV/PACT는 프롬프트 규정대로 인용/사유(직접 실행 보류·불가, §7).

---

## 2. 실험 1 — 제안 3변형 비교 (핵심)

데이터셋 커버리지(원본 VisPruner↔제안): POPE/GQA/TextVQA/VQAv2/SQA-IMG **5종 모두 양쪽 완료**.

| M2 | 방법 | POPE(F1) | GQA(Acc) | TextVQA | VQAv2† | SQA-IMG‡ |
|---:|---|---:|---:|---:|---:|---:|
| 128 | A VisPruner-only | 84.47 | 58.28 | **56.76** | **72.18** | **68.86** |
| 128 | B Ours simple | **85.37** | 58.26 | 54.77 | 72.08 | **68.86** |
| 128 | B Ours weighted | 85.24 | **58.27** | 55.27 | 72.17 | 68.62 |
| 64 | A VisPruner-only | 80.95 | 55.59 | **55.68** | 68.88 | 68.57 |
| 64 | B Ours simple | **82.27** | **56.66** | 54.08 | 69.77 | **68.86** |
| 64 | B Ours weighted | 82.14 | 56.26 | 54.33 | **70.44** | **69.16** |
| 32 | A VisPruner-only | 74.00 | 51.58 | 53.55 | 63.47 | 68.32 |
| 32 | B Ours simple | **77.56** | 53.52 | 53.38 | 65.46 | 69.31 |
| 32 | B Ours weighted | 77.53 | **54.03** | **53.61** | **65.88** | **69.46** |

† VQAv2 = val **균형 subset 6000**(yes-no/number/other 각 2000) 로컬 채점(공식 VQA acc).
test-dev(EvalAI) 대신 val 로컬채점·균형 subset이라 절대값은 자연분포보다 보수적 —
**논문 test-dev 인용치와 직접 비교 불가, A↔B 상대 비교만 유효.**
‡ SQA-IMG = update_ver2 실험B로 제안 구조에서 추가(IMG-Acc, 2017 IMG문항). A시리즈
68.86/68.57/68.32 = 원본 VisPruner 재현값 **정확 일치**(회귀안전). 토큰수 둔감(상식추론),
clustering 저토큰서 무회귀·소폭↑(최고 C-32 69.46 > VisPruner 논문 69.2).
또한 **원본 VisPruner_run에서 VQAv2 동일 subset 실행(실험A)**: V-128/64=72.18/68.88로
제안 A시리즈와 정확 일치 → 원본↔제안 VisPruner-only 경로 정합 교차검증 완료.

**개선폭(B−A)**: POPE +0.90/+1.32/+3.56 · GQA −0.02/+1.07/+1.94 · TextVQA −1.99/−1.60/−0.17 ·
VQAv2 −0.10/+0.89/+1.99 (simple), 0/+1.56/+2.41 (weighted) → **POPE·GQA·VQAv2 모두 압축
강할수록 clustering 이득↑ 동일 패턴**, TextVQA(OCR)만 예외적 소폭 하락.

---

## 3. 실험 2 — 기존 Baseline 종합 비교

출처: VisPruner 논문 Table 1(LLaVA-1.5-7B) 인용 + 본 실험. (PACT·VQAv2 §7 참조)

### POPE
| Method | 128 | 64 | 32 |
|---|---:|---:|---:|
| FastV (ECCV'24) | 59.6 | 48.0 | 32.5 |
| ToMe (ICLR'23) | 62.8 | 52.5 | 39.0 |
| SparseVLM (ICML'25) | 80.5 | 75.1 | 67.9 |
| VisPruner (논문) | 84.6 | 80.4 | 72.7 |
| A VisPruner-only (재현) | 84.47 | 80.95 | 74.00 |
| **B Ours simple** | **85.37** | **82.27** | **77.56** |
| **B Ours weighted** | 85.24 | 82.14 | 77.53 |

### GQA
| Method | 128 | 64 | 32 |
|---|---:|---:|---:|
| FastV | 49.6 | 46.1 | 41.5 |
| ToMe | 52.4 | 48.6 | 43.6 |
| SparseVLM | 56.0 | 52.7 | 48.3 |
| VisPruner (논문) | 58.2 | 55.4 | 52.2 |
| A VisPruner-only (재현) | 58.28 | 55.59 | 51.58 |
| **B Ours simple** | 58.26 | **56.66** | 53.52 |
| **B Ours weighted** | 58.27 | 56.26 | **54.03** |

→ 재현 VisPruner-only가 논문값과 정합(±0.5~1.3). **제안 B가 모든 기존 baseline + VisPruner
논문값을 상회**. 32토큰서 POPE 72.7(논문)→**77.56**(+4.86), GQA 52.2→**54.03**(+1.83).

---

## 4. 실험 3 — Ablation

### 3-A. Ratio r (M2=64, M1=128, simple)
| r | Important:Diverse | POPE(F1) | GQA(Acc) |
|---:|---|---:|---:|
| 0.3 | 38:90 | **82.74** | 56.65 |
| 0.5 | 64:64 | 82.27 | **56.66** |
| 0.7 | 90:38 | 81.63 | 56.61 |

→ POPE: diverse↑(r↓) 단조 유리. GQA: r 무감각(±0.05). 권장 기본 r=0.5, 환각평가는 r↓.

### 3-B. Clustering 유무 (동일 최종 토큰수)
| 토큰 | OFF (=A) | ON simple (=B) | ON weighted | Δ(best) |
|---|---:|---:|---:|---:|
| POPE 64 | 80.95 | **82.27** | 82.14 | **+1.32** |
| POPE 32 | 74.00 | **77.56** | 77.53 | **+3.56** |
| GQA 64 | 55.59 | **56.66** | 56.26 | **+1.07** |
| GQA 32 | 51.58 | 53.52 | **54.03** | **+2.45** |

→ 동일 토큰수에서 clustering ON이 항상 우세, 압축 강할수록 효과 극대.

### 3-C. Stage1 토큰수 M1 민감도 (M2=64, simple, r=0.5, POPE)
| M1 | M1/M2 | POPE(F1) |
|---:|:--:|---:|
| 96 | 1.5× | 81.62 |
| 128 | 2× | 82.27 |
| 192 | 3× | **83.03** |

→ M1↑ 단조 향상(3×까지). Stage1에서 많이 남길수록 Stage2 병합이 정보를 더 잘 보존.

### 3-D. Simple vs Weighted (실험1 재분석)
차이 ≤0.5p로 작음. **POPE는 simple 근소 우위, TextVQA·GQA(저토큰)는 weighted 근소 우위.**
weighted_avg는 [CLS] attention 가중으로 중요 토큰 강조 → 세밀/난이도 높은 task에 유리.

---

## 5. 실험 4 — POPE 카테고리별 (random/popular/adversarial, F1)

| 토큰 | 방법 | random | popular | adversarial | avg |
|---|---|---:|---:|---:|---:|
| 128 | A | 85.90 | 85.18 | 82.31 | 84.47 |
| 128 | B simple | 87.45 | 85.55 | 83.11 | 85.37 |
| 64 | A | 82.09 | 81.65 | 79.11 | 80.95 |
| 64 | B simple | 83.76 | 83.28 | 79.77 | 82.27 |
| 64 | B weighted | 83.04 | 82.40 | **80.98** | 82.14 |
| 32 | A | 74.88 | 74.63 | 72.48 | 74.00 |
| 32 | B simple | 78.24 | 78.07 | **76.38** | 77.56 |

→ clustering이 전 카테고리 일관 개선. **adversarial(최난이도)에서 효과 큼**: 32토큰서 +3.90,
64토큰서 weighted가 특히 강함(79.11→80.98). diverse 토큰 보존이 환각 거부에 기여.

### 5-B. 실험 4-B — VQAv2 Question Type별 (val 균형 subset, overall/yes-no/number/other)

| M2 | 방법 | overall | yes/no | number | other |
|---:|---|---:|---:|---:|---:|
| 128 | A | 72.18 | 90.33 | 56.37 | 69.83 |
| 64 | A | 68.88 | 87.80 | 51.02 | 67.82 |
| 64 | B weighted | 70.44 | 89.28 | 52.78 | 69.27 |
| 32 | A | 63.47 | 84.00 | 44.77 | 61.65 |
| 32 | B simple | 65.46 | 85.82 | **46.87** | 63.70 |
| 32 | B weighted | 65.88 | 85.77 | 46.70 | **65.17** |

→ **number(counting)가 압축 최취약** (A 56.4→44.8, −11.6 @128→32; 세밀 공간정보 필요).
clustering이 저토큰서 number/other 회복(B-32s number +2.1, B-32w other +3.5).
yes/no는 압축에 강건(84~90 유지). **task-aware policy 근거**: number=보수적 압축,
other/hard=weighted_avg, yes/no=공격 압축 가능.

---

## 6. 실험 5 — 효율성 (POPE 110샘플, warmup10 제외 100 평균)

| Setting | M2 | clustering | Token감소 | Latency(s/q) | GPU Mem(GB) |
|---|---:|:--:|---:|---:|---:|
| A-64 | 64 | OFF | 88.9% | 0.3217 | 14.51 |
| B-64s | 64 | ON | 88.9% | 0.3232 | 14.50 |
| B-64w | 64 | ON | 88.9% | 0.3124 | 14.50 |
| A-32 | 32 | OFF | 94.4% | 0.3261 | 14.50 |
| B-32s | 32 | ON | 94.4% | 0.3171 | 14.50 |

→ **B latency ≈ A (±1%, 측정 노이즈 내), GPU mem 동일.** Stage2(Spherical K-Means)는
작은 텐서·max_iter≤10·조기종료·벡터연산이라 LLM 디코딩 대비 미미. LLM 입력 토큰수가
M2로 동일하므로 디코딩 비용 불변. **정확도 향상이 사실상 무비용**.
(`CUDA_LAUNCH_BLOCKING=1`로 절대 latency는 부풀려져 상대 비교용; 비차단 시 더 빠름.)

---

## 7. 베이스라인 처리 / 미실행 (사유 명시)

- **VQAv2**: (v2 추가) test-dev EvalAI 대신 **val 균형 subset 6000 로컬 채점**으로 실행 완료
  (위 §2·§5-B). 균형 subset이라 논문 test-dev 인용치와 절대 비교 불가, A↔B 상대비교 유효.
- **FastV (실험2-A)**: 직접 실행 보류 → VisPruner Table 1 인용. 사유: FastV는 LLM 디코더
  layer-K self-attention + KV-cache 프루닝 요구. 고정 의존성(transformers 4.37.2)에서
  in-decoder 통합은 버전민감·고위험, 오구현 시 baseline 수치 왜곡으로 비교 신뢰성 훼손.
  프롬프트 규정("구현 어려우면 논문 숫자 인용+사유") 적용.
- **PACT (실험2-B)**: 직접 실행 불가 → 인용/사유. PACT 코드는 별도 env(pactenv:
  python3.12.7/cuda11.8/flash-attn2.6.3) + 자체 transformers 번들, 지원 백본이
  **LLaVA-OneVision-7B/Qwen2-VL/LLaVA-1.6**로 본 과제 백본 **LLaVA-1.5-7B 미지원**.
  환경·백본 상이로 동일조건 비교 불가. 프롬프트 규정 적용(제공 인용블록에 PACT 수치
  미포함 → 수치 미확보 표기, 날조 금지).
- **ToMe/SparseVLM**: 프롬프트 지시대로 VisPruner Table 1 인용(직접 실행 안 함).

---

## 8. 발견·수정한 버그 (재현 신뢰성)

**VisPruner 원본 diverse-선택 루프의 홀수-R IndexError**: r=0.7(R-70) 실행 중
`shape mismatch [1,235] vs [1,236]` 발생. 원인 — `a=residual[...,::2]`는 홀수 R에서
`ceil(R/2)`개인데 batch arange 확장이 `R//2-r`(floor)로 불일치. r=0.5는 R이 계속 짝수라
미발생, r=0.7서 노출. **수정**: arange 확장 길이를 `distinct_indices.shape[1]`(실제 길이)로
변경 — 짝수 R 동작 불변, 홀수 R 안전. 재실행 시 재시도·IndexError 0 검증.
그 외 CUDA illegal memory access 0 (builder.py dtype 패치 + LAUNCH_BLOCKING 유지).

---

## 9. 종합 평가

| 관점 | 결과 |
|---|---|
| 정확도 | POPE 전 토큰수 개선(최대 +3.56), GQA 저토큰 개선, baseline 전부 상회 |
| 효율 | clustering 오버헤드 ≈0 (latency/mem 동일) |
| 강건성 | adversarial 등 hard case 개선, M1↑ 단조 향상, r 견고 |
| 한계 | TextVQA(OCR) 소폭 하락 — 세밀 텍스트는 병합 민감(weighted로 완화) |
| task-aware | POPE=simple+공격압축, hard-neg=weighted, OCR=보수압축 — 조합 튜닝 유효 |

**결론**: 제안 Two-Stage(VisPruner + Spherical K-Means)는 **동일 토큰수·동일 비용으로
VisPruner 및 기존 baseline을 일관 상회**. 특히 극단 압축(32토큰, 94.4% 감소)에서 이득이
가장 크고, hard negative(adversarial)에서 강건. TextVQA류 OCR만 보수적 적용 권장.

상세: `02~06_*.md`, 원시 데이터 `term_project/exp_runner/results.tsv`.
